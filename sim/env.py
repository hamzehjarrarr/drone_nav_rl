"""Vectorized drone navigation environment.

One `VecDroneEnv` simulates n drones at once. The policy acts at 50 Hz with
collective-thrust + body-rate (CTBR) actions; physics runs at 250 Hz underneath,
through a simulated firmware rate controller.

step() returns separate observations for the actor (onboard sensing only) and the
critic (privileged ground truth), plus rewards and termination flags. Finished
environments are reset automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.actions import action_to_ctbr
from sim.curriculum import STAGES, Stage
from sim.dynamics import QuadState, step_dynamics
from sim.math3d import rot_z, so3_exp
from sim.params import DEFAULT_PROFILE, RANDOMIZED_FIELDS, ParamBatch, load_profile, sample_params
from sim.rate_controller import RateController
from sim.sensors import RayConfig, sense
from sim.world import (
    World,
    WorldBatch,
    WorldSpec,
    distance_to_obstacles,
    generate_world,
    out_of_arena,
)


@dataclass(frozen=True)
class RewardConfig:
    progress: float = 1.0  # per metre of progress towards the active waypoint
    waypoint_bonus: float = 2.0
    success_bonus: float = 10.0
    hover_shaping: float = 0.05  # per step, near the final waypoint at low speed
    hover_precision: float = 0.1  # per step, sharp bonus within ~0.3 m of the final waypoint
    clearance: float = 0.4  # per step at the worst clearance
    clearance_margin: float = 1.0  # m, from the drone's collision radius
    collision: float = 20.0
    # flying at obstacles faster than the drone could brake in time: the safe closing speed
    # at clearance d is sqrt(2 * brake_accel * d); the excess is penalized (squared)
    speed_safety: float = 0.05
    brake_accel: float = 4.0  # m/s^2, well inside what the drone can do, leaves reaction time
    over_speed: float = 0.02  # * (speed - cruise_cap)^2 above the cap
    cruise_cap: float = 5.0  # m/s
    attitude: float = 1.0  # * (cos 60 deg - cos tilt) when tilted past 60 deg
    heading: float = 0.02  # per step, for pointing the nose (and front sensor) where it flies
    smoothness: float = 0.02  # * |a_t - a_{t-1}|^2
    body_rate: float = 0.0005  # * |omega|^2
    tilt: float = 0.02  # * (1 - cos(tilt))
    time: float = 0.01  # per step


@dataclass(frozen=True)
class EnvConfig:
    n_envs: int = 256
    profile: str = DEFAULT_PROFILE  # drone profile: built-in name or path to a .toml file
    sim_dt: float = 0.004  # 250 Hz physics
    substeps: int = 5  # -> 50 Hz policy
    rays: RayConfig = field(default_factory=RayConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    waypoint_radius: float = 1.0  # pass-through radius of intermediate waypoints
    final_radius: float = 0.3
    final_max_speed: float = 0.5
    hold_time: float = 2.0
    min_avg_speed: float = 0.8  # timeout budget = route length / this + slack
    world_pool_size: int = 300  # pre-generated worlds per stage
    replay_fraction: float = 0.2  # share of episodes drawn from earlier stages
    # light low-pass filter between the policy and the flight controller (part of the
    # deployed control chain too): applied = a * action + (1 - a) * previous applied
    action_filter: float = 0.7

    @property
    def policy_dt(self) -> float:
        return self.sim_dt * self.substeps


class WorldPool:
    """Keeps a pool of generated worlds per stage and refreshes it slowly."""

    def __init__(self, size: int, rng: np.random.Generator) -> None:
        self.size = size
        self.rng = rng
        self.pools: dict[int, list[World]] = {}

    def _spec(self, stage: Stage, corridor: bool) -> WorldSpec:
        lo, hi = stage.waypoints
        jitter = self.rng.uniform(0.5, 1.0)
        return WorldSpec(
            n_trees=int(round(stage.trees * jitter)),
            n_walls=int(round(stage.walls * jitter)),
            n_blocks=int(round(stage.blocks * jitter)),
            n_waypoints=int(self.rng.integers(lo, hi + 1)),
            corridor=corridor,
            hover_only=stage.hover_only,
        )

    def _make(self, stage: Stage) -> World:
        corridor = self.rng.random() < stage.corridor_prob
        return generate_world(self._spec(stage, corridor), self.rng)

    def sample(self, stage_id: int) -> World:
        stage = STAGES[stage_id]
        pool = self.pools.setdefault(stage_id, [])
        if len(pool) < self.size:
            pool.append(self._make(stage))
            return pool[-1]
        if self.rng.random() < 0.05:  # refresh slowly so worlds keep changing
            pool[int(self.rng.integers(len(pool)))] = self._make(stage)
        return pool[int(self.rng.integers(len(pool)))]


class VecDroneEnv:
    def __init__(self, cfg: EnvConfig | None = None, seed: int = 0, stage: int = 0) -> None:
        self.cfg = cfg or EnvConfig()
        self.p, self.ranges = load_profile(self.cfg.profile)
        self.n = self.cfg.n_envs
        self.rng = np.random.default_rng(seed)
        self.pool = WorldPool(self.cfg.world_pool_size, np.random.default_rng(seed + 1))
        self.stage = stage
        self.n_rays = self.cfg.rays.n_rays

        n = self.n
        self.state = QuadState.zeros(n)
        self.params: ParamBatch = sample_params(n, self.rng, self.p, self.ranges, randomize=False)
        self.rc = RateController(self.p, gain_scale=np.ones(n))
        self.rc.reset(n)
        self.world = WorldBatch.empty(n)
        self.start = np.zeros((n, 3))

        self.stage_id = np.zeros(n, int)
        self.soft = np.zeros(n, bool)
        self.realism = np.zeros(n)
        self.wp_idx = np.zeros(n, int)
        self.prev_dist = np.zeros(n)
        self.prev_clear = np.full(n, -1.0)  # clearance at the previous step (-1: just reset)
        self.hold = np.zeros(n)
        self.t = np.zeros(n, int)
        self.max_steps = np.ones(n, int)
        self.prev_action = np.zeros((n, 4))
        self.filtered_action = np.zeros((n, 4))
        self.action_hist = np.zeros((n, 3, 4))
        self.delay = np.zeros(n, int)
        self.wind_base = np.zeros((n, 3))
        self.gust = np.zeros((n, 3))
        self.gust_std = np.zeros(n)
        self.sag = np.zeros(n)
        self.est_bias = np.zeros((n, 3))
        self.push_step = np.full(n, -1)
        self.push_force = np.zeros((n, 3))
        self.ep_return = np.zeros(n)
        self.ep_collisions = np.zeros(n, int)
        self.reset_all()

    # ------------------------------------------------------------------ dims
    @property
    def actor_obs_dim(self) -> int:
        return 9 + 3 + 3 + 3 + 3 + 3 + 1 + 4 + 2 * self.n_rays

    @property
    def critic_obs_dim(self) -> int:
        return self.actor_obs_dim + 20

    # ------------------------------------------------------------------ reset
    def set_stage(self, stage: int) -> None:
        self.stage = int(np.clip(stage, 0, len(STAGES) - 1))

    def _pick_stage(self, k: int) -> np.ndarray:
        ids = np.full(k, self.stage)
        if self.stage > 0:
            replay = self.rng.random(k) < self.cfg.replay_fraction
            ids[replay] = self.rng.integers(0, self.stage, size=replay.sum())
        return ids

    def reset_all(self) -> tuple[np.ndarray, np.ndarray]:
        self._reset_idx(np.arange(self.n))
        return self._observe()

    def _reset_idx(
        self,
        idx: np.ndarray,
        worlds: list[World] | None = None,
        stage_ids: np.ndarray | None = None,
        realism: np.ndarray | float | None = None,
    ) -> None:
        if len(idx) == 0:
            return
        k = len(idx)
        stage_ids = self._pick_stage(k) if stage_ids is None else stage_ids
        self.stage_id[idx] = stage_ids
        stages = [STAGES[s] for s in stage_ids]

        # dynamics parameters
        rand = np.array([s.randomize_dynamics for s in stages])
        fresh = sample_params(k, self.rng, self.p, self.ranges, randomize=True)
        nominal = sample_params(k, self.rng, self.p, self.ranges, randomize=False)
        for name in RANDOMIZED_FIELDS:
            arr = getattr(self.params, name)
            src_r, src_n = getattr(fresh, name), getattr(nominal, name)
            sel = rand.reshape((-1,) + (1,) * (src_r.ndim - 1))
            arr[idx] = np.where(sel, src_r, src_n)
        self.rc.gain_scale[idx] = np.where(rand, self.rng.uniform(0.8, 1.2, k), 1.0)
        self.rc.reset(self.n, idx)

        if realism is None:  # each episode is realistic with the stage's probability
            share = np.array([s.realism for s in stages])
            realism = (self.rng.random(k) < share).astype(float)
        realism = np.broadcast_to(np.asarray(realism, float), (k,))
        self.realism[idx] = realism
        self.soft[idx] = [s.soft_collisions for s in stages]

        # worlds and start state
        for j, i in enumerate(idx):
            w = worlds[j] if worlds is not None else self.pool.sample(int(stage_ids[j]))
            self.world.set(i, w)
            self.start[i] = w.start
            self.state.pos[i] = w.start
            self.state.rot[i] = rot_z(np.array(w.start_yaw))
        self.state.vel[idx] = 0.0
        self.state.omega[idx] = 0.0
        self.state.motor_thrust[idx] = self.p.hover_thrust_per_motor

        rnd = np.array([s.random_init for s in stages])
        if rnd.any():
            r_idx = idx[rnd]
            m = len(r_idx)
            tilt_axis = self.rng.normal(size=(m, 3))
            tilt_axis[:, 2] = 0.0
            tilt_axis /= np.linalg.norm(tilt_axis, axis=1, keepdims=True) + 1e-9
            angle = self.rng.uniform(0, np.deg2rad(35.0), m)
            self.state.rot[r_idx] = self.state.rot[r_idx] @ so3_exp(tilt_axis * angle[:, None])
            self.state.vel[r_idx] = self.rng.normal(0.0, 0.7, (m, 3))
            self.state.omega[r_idx] = self.rng.normal(0.0, 1.0, (m, 3))

        # route bookkeeping
        self.wp_idx[idx] = 0
        first = self.world.waypoints[idx, 0]
        self.prev_dist[idx] = np.linalg.norm(first - self.state.pos[idx], axis=1)
        self.prev_clear[idx] = -1.0
        self.hold[idx] = 0.0
        self.t[idx] = 0
        self.max_steps[idx] = [self._time_budget(i) for i in idx]
        self.prev_action[idx] = 0.0
        self.filtered_action[idx] = 0.0
        self.action_hist[idx] = 0.0
        self.delay[idx] = np.where(realism > 0.5, self.rng.integers(0, 3, k), 0)

        # wind, battery sag, state-estimate error
        wind_max = np.array([s.wind_max for s in stages])
        heading = self.rng.uniform(0, 2 * np.pi, k)
        speed = self.rng.uniform(0, 1, k) * wind_max
        self.wind_base[idx] = np.stack(
            [np.cos(heading) * speed, np.sin(heading) * speed, np.zeros(k)], axis=1
        )
        self.gust[idx] = 0.0
        self.gust_std[idx] = [s.gust_std for s in stages]
        self.sag[idx] = self.rng.uniform(0.0, 0.08, k) * realism
        self.est_bias[idx] = 0.0
        self.push_step[idx] = -1
        self.ep_return[idx] = 0.0
        self.ep_collisions[idx] = 0

    def _time_budget(self, i: int) -> int:
        n_wp = self.world.n_waypoints[i]
        pts = np.vstack([self.start[i][None], self.world.waypoints[i, :n_wp]])
        length = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        seconds = length / self.cfg.min_avg_speed + self.cfg.hold_time + 4.0
        return int(seconds / self.cfg.policy_dt)

    def load_worlds(
        self,
        worlds: list[World],
        stage_ids: np.ndarray | None = None,
        realism: np.ndarray | float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reset every env onto given worlds (evaluation and the app). Pass realism (0 or 1)
        to fix the sensing mode instead of drawing it from the stage."""
        assert len(worlds) == self.n
        sid = np.full(self.n, self.stage) if stage_ids is None else stage_ids
        self._reset_idx(np.arange(self.n), worlds=worlds, stage_ids=sid, realism=realism)
        return self._observe()

    def schedule_push(self, step: int, force: np.ndarray) -> None:
        """Apply a world-frame force (N) for 0.1 s starting at the given policy step."""
        self.push_step[:] = step
        self.push_force[:] = force

    # ------------------------------------------------------------------ step
    def step(self, action: np.ndarray, auto_reset: bool = True):
        cfg = self.cfg
        a = np.clip(action, -1.0, 1.0)
        k = cfg.action_filter
        self.filtered_action = k * a + (1.0 - k) * self.filtered_action

        # actuation latency (realistic mode): apply an action from 0-2 steps ago
        self.action_hist = np.roll(self.action_hist, 1, axis=1)
        self.action_hist[:, 0] = self.filtered_action
        applied = self.action_hist[np.arange(self.n), self.delay]
        thrust, rates = action_to_ctbr(applied, self.p)

        # wind gusts: Ornstein-Uhlenbeck process, 2 s correlation time
        tau = 2.0
        dtp = cfg.policy_dt
        self.gust += -self.gust * dtp / tau + self.gust_std[:, None] * np.sqrt(
            2 * dtp / tau
        ) * self.rng.normal(size=(self.n, 3)) * np.array([1.0, 1.0, 0.3])
        wind = self.wind_base + self.gust
        thrust_scale = 1.0 - self.sag * np.minimum(self.t / np.maximum(self.max_steps, 1), 1.0)
        pushing = (
            (self.push_step >= 0)
            & (self.t >= self.push_step)
            & (self.t < self.push_step + int(0.1 / dtp) + 1)
        )
        ext = np.where(pushing[:, None], self.push_force, 0.0)

        for _ in range(cfg.substeps):
            f = self.rc(thrust, rates, self.state.omega, cfg.sim_dt)
            step_dynamics(
                self.state,
                f,
                self.params,
                cfg.sim_dt,
                wind=wind,
                thrust_scale=thrust_scale,
                external_force=ext,
            )
        self.t += 1

        # state-estimate drift (realistic mode): random walk on position
        self.est_bias += (
            self.rng.normal(0.0, 0.05 * np.sqrt(dtp), (self.n, 3)) * self.realism[:, None]
        )

        reward, done, truncated, info = self._reward_and_done(a)
        self.prev_action = a
        self.ep_return += reward

        obs_a, obs_c = self._observe()
        info["terminal_critic_obs"] = obs_c.copy()
        info["episode_return"] = self.ep_return.copy()
        info["episode_length"] = self.t.copy()
        info["stage_id"] = self.stage_id.copy()

        if auto_reset and done.any():
            idx = np.flatnonzero(done)
            self._reset_idx(idx)
            obs_a, obs_c = self._observe()
        return obs_a, obs_c, reward, done, truncated, info

    def _reward_and_done(self, a: np.ndarray):
        cfg, w = self.cfg, self.cfg.reward
        s = self.state
        n = self.n
        ar = np.arange(n)
        r = np.zeros(n)

        n_wp = self.world.n_waypoints
        target = self.world.waypoints[ar, self.wp_idx]
        dist = np.linalg.norm(target - s.pos, axis=1)
        is_final = self.wp_idx == n_wp - 1
        speed = np.linalg.norm(s.vel, axis=1)

        # progress towards the active waypoint
        r += w.progress * (self.prev_dist - dist)

        # intermediate waypoint reached -> next one
        reached = (~is_final) & (dist < cfg.waypoint_radius)
        r += w.waypoint_bonus * reached
        self.wp_idx = np.where(reached, self.wp_idx + 1, self.wp_idx)
        new_target = self.world.waypoints[ar, self.wp_idx]
        dist = np.where(reached, np.linalg.norm(new_target - s.pos, axis=1), dist)
        self.prev_dist = dist
        is_final = self.wp_idx == n_wp - 1

        # final hover
        near = is_final & (dist < cfg.final_radius) & (speed < cfg.final_max_speed)
        self.hold = np.where(near, self.hold + cfg.policy_dt, 0.0)
        slow = np.exp(-(speed**2))
        r += w.hover_shaping * is_final * np.exp(-(dist**2) / 0.5) * slow
        r += w.hover_precision * is_final * np.exp(-(dist**2) / 0.05) * slow
        success = self.hold >= cfg.hold_time
        r += w.success_bonus * success

        # safety
        d_obs = (
            distance_to_obstacles(
                s.pos, self.world.cyl, self.world.cyl_mask, self.world.box, self.world.box_mask
            )
            - self.p.collision_radius
        )
        clear = np.clip((w.clearance_margin - d_obs) / w.clearance_margin, 0.0, 1.0)
        r -= w.clearance * clear**2
        collided = d_obs <= 0.0
        ground = s.pos[:, 2] < 0.05
        outside = out_of_arena(s.pos)
        flipped = s.rot[:, 2, 2] < 0.0
        r -= w.collision * collided * np.where(self.soft, 0.1, 1.0)
        self.ep_collisions += collided

        # speed safety: closing speed towards the nearest surface vs. what can still be braked
        closing = np.where(self.prev_clear >= 0, (self.prev_clear - d_obs) / cfg.policy_dt, 0.0)
        self.prev_clear = d_obs
        v_safe = np.sqrt(2.0 * w.brake_accel * np.maximum(d_obs, 0.0))
        r -= w.speed_safety * np.maximum(closing - v_safe, 0.0) ** 2
        r -= w.over_speed * np.maximum(speed - w.cruise_cap, 0.0) ** 2
        r -= w.attitude * np.maximum(0.5 - s.rot[:, 2, 2], 0.0)

        # face the direction of travel: the front sectors are the fine ones (and the only
        # ones the realistic depth camera has), so looking ahead is what makes avoidance work
        v_h = s.vel[:, :2]
        speed_h = np.linalg.norm(v_h, axis=1)
        nose = s.rot[:, :2, 0]
        nose = nose / np.maximum(np.linalg.norm(nose, axis=1, keepdims=True), 1e-6)
        align = np.sum(nose * v_h, axis=1) / np.maximum(speed_h, 1e-6)
        r += w.heading * np.clip(speed_h / 2.0, 0.0, 1.0) * align

        # smoothness, attitude, time
        r -= w.smoothness * np.sum((a - self.prev_action) ** 2, axis=1)
        r -= w.body_rate * np.sum(s.omega**2, axis=1)
        r -= w.tilt * (1.0 - s.rot[:, 2, 2])
        r -= w.time

        hard_fail = (collided & ~self.soft) | ground | outside | flipped
        r -= w.collision * (ground | outside | flipped)
        timeout = self.t >= self.max_steps
        done = success | hard_fail | timeout
        truncated = timeout & ~success & ~hard_fail

        info = {
            "success": success,
            "collision": (collided & ~self.soft) | ground,
            "out_of_bounds": outside | flipped,
            "timeout": truncated,
            "waypoints_reached": self.wp_idx + success.astype(int),
            "n_waypoints": n_wp.copy(),
            "min_clearance": d_obs,
        }
        return r, done, truncated, info

    # ------------------------------------------------------------------ observations
    def _observe(self) -> tuple[np.ndarray, np.ndarray]:
        s = self.state
        n = self.n
        ar = np.arange(n)
        realism = self.realism[:, None]
        rot_t = np.transpose(s.rot, (0, 2, 1))

        est_pos = s.pos + self.est_bias + self.rng.normal(0, 0.02, (n, 3)) * realism
        est_vel = s.vel + self.rng.normal(0, 0.05, (n, 3)) * realism
        gyro = s.omega + self.rng.normal(0, 0.02, (n, 3)) * realism

        n_wp = self.world.n_waypoints
        wp1 = self.world.waypoints[ar, self.wp_idx]
        wp2 = self.world.waypoints[ar, np.minimum(self.wp_idx + 1, n_wp - 1)]
        is_final = (self.wp_idx == n_wp - 1).astype(float)

        def body_vec(p: np.ndarray) -> np.ndarray:
            return np.einsum("nij,nj->ni", rot_t, p - est_pos)

        def body_rel(p: np.ndarray) -> np.ndarray:
            d = body_vec(p)
            norm = np.linalg.norm(d, axis=1, keepdims=True)
            return d / np.maximum(norm, 10.0)  # unit-ish beyond 10 m, linear inside

        prox, mask, _ = sense(s.rot, s.pos, self.world, self.cfg.rays, self.realism, self.rng)

        actor = np.concatenate(
            [
                s.rot.reshape(n, 9),
                np.einsum("nij,nj->ni", rot_t, est_vel) / 5.0,
                gyro / 5.0,
                body_rel(wp1),
                body_rel(wp2),
                np.clip(body_vec(wp1), -2.0, 2.0) / 2.0,  # fine resolution near the target
                is_final[:, None],
                self.prev_action,
                prox,
                mask.astype(float),
            ],
            axis=1,
        )

        d_obs = distance_to_obstacles(
            s.pos, self.world.cyl, self.world.cyl_mask, self.world.box, self.world.box_mask
        )
        nom = self.p
        priv = np.concatenate(
            [
                s.pos / 15.0,
                s.vel / 5.0,
                (self.wind_base + self.gust) / 5.0,
                (self.params.mass / nom.mass - 1.0)[:, None],
                (self.params.thrust_coeff / nom.thrust_coeff - 1.0)[:, None],
                (self.params.motor_tau / nom.motor_tau - 1.0)[:, None],
                np.clip(d_obs, -1.0, 5.0)[:, None] / 5.0,
                (self.wp_idx / np.maximum(n_wp, 1))[:, None],
                (self.t / np.maximum(self.max_steps, 1))[:, None],
                (1.0 - self.sag * self.t / np.maximum(self.max_steps, 1))[:, None],
                self.realism[:, None],
                (wp1 - s.pos) / 10.0,
            ],
            axis=1,
        )
        critic = np.concatenate([actor, priv], axis=1)
        return actor.astype(np.float32), critic.astype(np.float32)

    # ------------------------------------------------------------------ helpers
    def snapshot(self) -> dict:
        """Lightweight per-env state for logging / visualization (copies)."""
        return {
            "pos": self.state.pos.copy(),
            "rot": self.state.rot.copy(),
            "vel": self.state.vel.copy(),
            "motor_thrust": self.state.motor_thrust.copy(),
            "wp_idx": self.wp_idx.copy(),
        }