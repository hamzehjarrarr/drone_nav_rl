"""Fly pilots (learned policy or the classical reference) on given worlds and record results.

Shared by the evaluation script and the Streamlit app.

How success is judged: a flight succeeds when the drone has passed every waypoint and held
a hover at the last one (within 0.3 m, slower than 0.5 m/s, for 2 s) according to its own
onboard position estimate. In ideal mode the estimate is exact, so this is the true
position. In realistic mode the estimate drifts over a flight, as visual-inertial odometry
does, and no policy can see that drift from onboard sensors. The true end error is
therefore reported separately (`hover_rms`, `true_final_error`), as is success judged on the
true position (`true_success`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from sim.baseline import StraightLinePilot
from sim.curriculum import stage_index
from sim.env import EnvConfig, VecDroneEnv
from sim.math3d import rot_to_quat
from sim.params import DEFAULT_PROFILE
from sim.sensors import ray_directions_body
from sim.world import World

Pilot = Callable[[VecDroneEnv, np.ndarray], np.ndarray]


def policy_pilot(policy) -> Pilot:
    """Wrap a NumpyPolicy: acts on the onboard observation only."""
    return lambda env, obs_a: policy(obs_a)


def reference_pilot(cruise_speed: float = 3.0) -> Pilot:
    """The classical straight-line pilot. Note: it is given the TRUE state."""
    pilots: dict[int, StraightLinePilot] = {}

    def act(env: VecDroneEnv, obs_a: np.ndarray) -> np.ndarray:
        # built per environment so it uses that environment's drone profile
        pilot = pilots.setdefault(id(env), StraightLinePilot(env.p, cruise_speed=cruise_speed))
        ar = np.arange(env.n)
        target = env.world.waypoints[ar, env.wp_idx]
        final = env.wp_idx == env.world.n_waypoints - 1
        return pilot(env.state.pos, env.state.vel, env.state.rot, target, final)

    return act


@dataclass
class RunOptions:
    realistic: bool = False
    wind: float = 0.0
    gust: float = 0.0
    push: tuple[float, float, float] | None = None
    push_time: float = 3.0  # s
    seed: int = 0
    record: bool = False  # keep per-step trajectories (for the 3D viewer)
    profile: str = DEFAULT_PROFILE  # drone profile the policy was trained for


def make_env(n: int, seed: int = 0, profile: str = DEFAULT_PROFILE) -> VecDroneEnv:
    return VecDroneEnv(
        EnvConfig(n_envs=n, world_pool_size=1, profile=profile),
        seed=seed,
        stage=stage_index("dense_obstacles"),
    )


def fly(worlds: list[World], pilot: Pilot, opt: RunOptions | None = None) -> list[dict]:
    """Fly one episode per world (all in parallel). Returns one result dict per world."""
    opt = opt or RunOptions()
    n = len(worlds)
    env = make_env(n, opt.seed, opt.profile)
    stage = stage_index("realism" if opt.realistic else "dense_obstacles")
    obs_a, _ = env.load_worlds(
        worlds, stage_ids=np.full(n, stage), realism=1.0 if opt.realistic else 0.0
    )

    # scenario conditions override whatever the stage sampled
    rng = np.random.default_rng(opt.seed + 12345)
    heading = rng.uniform(0, 2 * np.pi, n)
    env.wind_base[:] = np.stack([np.cos(heading), np.sin(heading), np.zeros(n)], 1) * opt.wind
    env.gust[:] = 0.0
    env.gust_std[:] = opt.gust
    if opt.push is not None:
        env.schedule_push(int(opt.push_time / env.cfg.policy_dt), np.asarray(opt.push))

    n_rays = env.n_rays
    ray_range = np.where(env.realism > 0.5, env.cfg.rays.real_max_range, env.cfg.rays.max_range)
    finished = np.zeros(n, bool)
    res = [
        dict(success=False, true_success=False, collision=False, out_of_bounds=False, timeout=False)
        for _ in range(n)
    ]
    est_hold = np.zeros(n)  # hover time at the final waypoint by the onboard estimate
    true_err = np.full(n, np.nan)  # true distance to the final waypoint when the hold completes
    path = np.zeros(n)
    min_clear = np.full(n, np.inf)
    hover_err: list[list[float]] = [[] for _ in range(n)]
    smooth = np.zeros(n)
    prev_a = np.zeros((n, 4))
    prev_pos = env.state.pos.copy()
    frames: list[list[dict]] = [[] for _ in range(n)]

    def record(i: int, a_i: np.ndarray, prox_i: np.ndarray, mask_i: np.ndarray) -> None:
        dist = (1.0 - prox_i) * ray_range[i]
        frames[i].append(
            {
                "t": round(float(env.t[i] * env.cfg.policy_dt), 3),
                "pos": env.state.pos[i].round(3).tolist(),
                "quat": rot_to_quat(env.state.rot[i]).round(4).tolist(),
                "speed": round(float(np.linalg.norm(env.state.vel[i])), 2),
                "motors": env.state.motor_thrust[i].round(2).tolist(),
                "wp": int(env.wp_idx[i]),
                "rays": np.where(mask_i > 0.5, dist, -1.0).round(2).tolist(),
            }
        )

    if opt.record:
        for i in range(n):
            record(i, prev_a[i], obs_a[i, -2 * n_rays : -n_rays], obs_a[i, -n_rays:])

    max_steps = int(env.max_steps.max()) + 2
    for _ in range(max_steps):
        a = np.clip(pilot(env, obs_a), -1.0, 1.0)
        obs_a, _, _, done, _, info = env.step(a, auto_reset=False)
        live = ~finished
        step_len = np.linalg.norm(env.state.pos - prev_pos, axis=1)
        path += step_len * live
        prev_pos = env.state.pos.copy()
        min_clear = np.where(live, np.minimum(min_clear, info["min_clearance"]), min_clear)
        smooth += np.sum((a - prev_a) ** 2, axis=1) * live
        prev_a = a
        # success by the onboard estimate (equals the true position in ideal mode)
        ar = np.arange(n)
        final_wp = env.world.waypoints[ar, env.world.n_waypoints - 1]
        at_final = env.wp_idx == env.world.n_waypoints - 1
        est_d = np.linalg.norm(env.state.pos + env.est_bias - final_wp, axis=1)
        speed = np.linalg.norm(env.state.vel, axis=1)
        holding = at_final & (est_d < env.cfg.final_radius) & (speed < env.cfg.final_max_speed)
        est_hold = np.where(live & holding, est_hold + env.cfg.policy_dt, 0.0)
        for i in np.flatnonzero(live & holding):
            hover_err[i].append(float(np.linalg.norm(env.state.pos[i] - final_wp[i])))
        est_done = live & (est_hold >= env.cfg.hold_time) & ~done
        for i in np.flatnonzero(est_done):  # finished by its own estimate (true pos may be off)
            finished[i] = True
            res[i]["success"] = True
            res[i]["steps"] = int(env.t[i])
            res[i]["waypoints_reached"] = int(env.world.n_waypoints[i])
            true_err[i] = float(np.linalg.norm(env.state.pos[i] - final_wp[i]))
        if opt.record:
            for i in np.flatnonzero(live):
                record(i, a[i], obs_a[i, -2 * n_rays : -n_rays], obs_a[i, -n_rays:])
        for i in np.flatnonzero(live & done):
            finished[i] = True
            for k in ("success", "collision", "out_of_bounds", "timeout"):
                res[i][k] = bool(info[k][i])
            res[i]["true_success"] = bool(info["success"][i])
            if info["success"][i]:
                true_err[i] = float(np.linalg.norm(env.state.pos[i] - final_wp[i]))
            # a timeout while holding by the estimate counts once the 2 s hold is complete
            if info["timeout"][i] and est_hold[i] >= env.cfg.hold_time:
                res[i]["success"], res[i]["timeout"] = True, False
            res[i]["steps"] = int(env.t[i])
            res[i]["waypoints_reached"] = int(info["waypoints_reached"][i])
        if finished.all():
            break

    for i, w in enumerate(worlds):
        steps = res[i].get("steps", int(env.t[i]))
        t = steps * env.cfg.policy_dt
        pts = np.vstack([w.start[None], w.waypoints[: w.n_waypoints]])
        route = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        res[i].update(
            time=t,
            route_length=route,
            path_length=float(path[i]),
            avg_speed=float(path[i] / max(t, 1e-6)),
            route_efficiency=float(route / max(path[i], 1e-6)),
            min_clearance=float(min_clear[i]),
            hover_rms=float(np.sqrt(np.mean(np.square(hover_err[i])))) if hover_err[i] else None,
            smoothness=float(smooth[i] / max(steps, 1)),
            n_waypoints=int(w.n_waypoints),
            true_final_error=None if np.isnan(true_err[i]) else float(true_err[i]),
            estimate_drift=float(np.linalg.norm(env.est_bias[i])),
        )
        if opt.record:
            res[i]["frames"] = frames[i]
            res[i]["ray_dirs"] = _ray_dirs(env)
            res[i]["ray_range"] = float(ray_range[i])
    return res


def _ray_dirs(env: VecDroneEnv) -> list[list[float]]:
    d, _, _ = ray_directions_body(env.cfg.rays)
    return d.round(4).tolist()


def summarize(results: list[dict]) -> dict:
    """Aggregate metrics over episodes."""
    ok = [r for r in results if r["success"]]
    n = max(len(results), 1)

    def mean(key: str, rows: list[dict]) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "episodes": len(results),
        "success_rate": len(ok) / n,
        "true_success_rate": sum(r.get("true_success", r["success"]) for r in results) / n,
        "collision_rate": sum(r["collision"] for r in results) / n,
        "timeout_rate": sum(r["timeout"] for r in results) / n,
        "avg_speed": mean("avg_speed", ok),
        "route_efficiency": mean("route_efficiency", ok),
        "hover_rms": mean("hover_rms", ok),
        "true_final_error": mean("true_final_error", ok),
        "estimate_drift": mean("estimate_drift", results),
        "min_clearance": mean("min_clearance", ok),
        "smoothness": mean("smoothness", ok),
        "time": mean("time", ok),
    }