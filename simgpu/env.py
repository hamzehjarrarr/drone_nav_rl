"""GPU version of sim.env.VecDroneEnv: the same environment, held in PyTorch tensors.

Differences from the NumPy version, none of which change what the drone experiences:
  * Worlds are generated on the CPU (with the same generator) into a pool per stage,
    then uploaded to the GPU once. A few pool entries are regenerated every rollout,
    so worlds keep changing.
  * Resets happen for all finished environments at once with masked updates.
  * Random numbers come from PyTorch, so a seed gives different (but equally
    distributed) worlds, noise and wind than the NumPy version.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from sim.curriculum import STAGES
from sim.env import EnvConfig, WorldPool
from sim.params import load_profile
from sim.world import MAX_BOX, MAX_CYL, MAX_WP, World
from simgpu.core import (
    Nominal,
    ParamsT,
    RateControllerT,
    StateT,
    action_to_ctbr,
    distance_to_obstacles,
    out_of_arena,
    rot_z,
    so3_exp,
    step_dynamics,
)
from simgpu.sensors import SectorScanT


def _worlds_to_arrays(worlds: list[World]) -> dict[str, np.ndarray]:
    return {
        "cyl": np.stack([w.cyl for w in worlds]),
        "cyl_mask": np.stack([w.cyl_mask for w in worlds]),
        "box": np.stack([w.box for w in worlds]),
        "box_mask": np.stack([w.box_mask for w in worlds]),
        "wps": np.stack([w.waypoints for w in worlds]),
        "n_wp": np.array([w.n_waypoints for w in worlds]),
        "start": np.stack([w.start for w in worlds]),
        "yaw": np.array([w.start_yaw for w in worlds]),
    }


class TorchDroneEnv:
    """Vectorized drone navigation on a GPU. Same observations, rewards and curriculum
    as sim.env.VecDroneEnv; step() takes and returns torch tensors on `device`."""

    def __init__(
        self,
        cfg: EnvConfig | None = None,
        seed: int = 0,
        stage: int = 0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        pool_size: int = 1024,
        refresh_per_rollout: int = 16,
        k_cyl: int = 16,
    ) -> None:
        self.cfg = cfg or EnvConfig()
        self.p, self.ranges = load_profile(self.cfg.profile)
        self.n = self.cfg.n_envs
        self.device = torch.device(device)
        self.dtype = dtype
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)
        self.stage = int(stage)
        self.n_rays = self.cfg.rays.n_rays
        self.nom = Nominal(self.p, self.device, dtype)
        self.scanner = SectorScanT(self.cfg.rays, self.device, dtype, k_cyl=k_cyl)
        self.refresh_per_rollout = refresh_per_rollout

        # per-stage settings as lookup tables indexed by stage id
        def table(attr, dt=dtype):
            vals = [float(getattr(s, attr)) for s in STAGES]
            return torch.tensor(vals, dtype=dt, device=self.device)

        self.st_random_init = table("random_init", torch.bool)
        self.st_randomize = table("randomize_dynamics", torch.bool)
        self.st_wind = table("wind_max")
        self.st_gust = table("gust_std")
        self.st_realism = table("realism")
        self.st_soft = table("soft_collisions", torch.bool)

        # world pools, (stage, pool index, ...) on the device; filled when a stage is reached
        self.pool_size = pool_size
        self._maker = WorldPool(1, np.random.default_rng(seed + 1))
        s_count, ps = len(STAGES), pool_size
        z = self._zeros
        self.pool = {
            "cyl": z(s_count, ps, MAX_CYL, 4),
            "cyl_mask": torch.zeros(s_count, ps, MAX_CYL, dtype=torch.bool, device=self.device),
            "box": z(s_count, ps, MAX_BOX, 6),
            "box_mask": torch.zeros(s_count, ps, MAX_BOX, dtype=torch.bool, device=self.device),
            "wps": z(s_count, ps, MAX_WP, 3),
            "n_wp": torch.ones(s_count, ps, dtype=torch.long, device=self.device),
            "start": z(s_count, ps, 3),
            "yaw": z(s_count, ps),
        }
        self.pool_ready = [False] * s_count
        self._ensure_pools(self.stage)

        n = self.n
        self.state = StateT(z(n, 3), z(n, 3), z(n, 3, 3), z(n, 3), z(n, 4))
        self.params = ParamsT(
            mass=z(n), inertia=z(n, 3), thrust_coeff=z(n), motor_gain=z(n, 4), kappa=z(n),
            motor_tau=z(n), drag_scale=z(n), rotor_drag_scale=z(n), ground_effect_scale=z(n),
        )  # fmt: skip
        self.rc = RateControllerT(self.nom, n)
        self.world = {k: torch.zeros_like(v[0, :1]).expand(n, *v.shape[2:]).clone()
                      for k, v in self.pool.items()}  # fmt: skip
        long0 = lambda: torch.zeros(n, dtype=torch.long, device=self.device)  # noqa: E731
        self.stage_id, self.wp_idx, self.t, self.max_steps = long0(), long0(), long0(), long0()
        self.delay, self.push_step = long0(), long0() - 1
        self.soft = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.realism, self.prev_dist, self.hold = z(n), z(n), z(n)
        self.prev_action, self.filtered_action = z(n, 4), z(n, 4)
        self.action_hist = z(n, 3, 4)
        self.wind_base, self.gust = z(n, 3), z(n, 3)
        self.est_bias, self.push_force = z(n, 3), z(n, 3)
        self.gust_std, self.sag, self.ep_return = z(n), z(n), z(n)
        self.ep_collisions = long0()
        self.reset_all()

    # ------------------------------------------------------------------ helpers
    def _zeros(self, *shape):
        return torch.zeros(*shape, dtype=self.dtype, device=self.device)

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, dtype=self.dtype, device=self.device)

    def _randn(self, *shape):
        return torch.randn(*shape, generator=self.gen, dtype=self.dtype, device=self.device)

    def _randint(self, high, n):
        return torch.randint(0, high, (n,), generator=self.gen, device=self.device)

    def _scale(self, frac: float, *shape):
        return 1.0 + frac * (2.0 * self._rand(*shape) - 1.0)

    @property
    def actor_obs_dim(self) -> int:
        return 9 + 3 + 3 + 3 + 3 + 1 + 4 + 2 * self.n_rays

    @property
    def critic_obs_dim(self) -> int:
        return self.actor_obs_dim + 20

    # ------------------------------------------------------------------ world pools
    def _write_pool(self, stage: int, slots: np.ndarray, worlds: list[World]) -> None:
        arr = _worlds_to_arrays(worlds)
        idx = torch.as_tensor(slots, device=self.device)
        for k, v in arr.items():
            dst = self.pool[k]
            dst[stage, idx] = torch.as_tensor(v, device=self.device).to(dst.dtype)

    def _ensure_pools(self, up_to: int) -> None:
        for s in range(up_to + 1):
            if not self.pool_ready[s]:
                worlds = [self._maker._make(STAGES[s]) for _ in range(self.pool_size)]
                self._write_pool(s, np.arange(self.pool_size), worlds)
                self.pool_ready[s] = True

    def refresh_pools(self) -> None:
        """Regenerate a few worlds (mostly of the current stage) so the pool keeps changing."""
        rng = self._maker.rng
        for _ in range(self.refresh_per_rollout):
            s = self.stage
            if s > 0 and rng.random() < 0.2:  # some replay-stage worlds too
                s = int(rng.integers(self.stage))
            slot = np.array([int(rng.integers(self.pool_size))])
            self._write_pool(s, slot, [self._maker._make(STAGES[s])])

    def set_stage(self, stage: int) -> None:
        self.stage = int(np.clip(stage, 0, len(STAGES) - 1))
        self._ensure_pools(self.stage)

    # ------------------------------------------------------------------ reset
    def reset_all(self):
        self._reset(torch.ones(self.n, dtype=torch.bool, device=self.device))
        return self._observe()

    def load_worlds(self, worlds: list[World], stage_ids: np.ndarray | None = None):
        """Reset every env onto given worlds (tests and evaluation)."""
        assert len(worlds) == self.n
        arr = _worlds_to_arrays(worlds)
        w = {k: torch.as_tensor(v, device=self.device).to(self.pool[k].dtype)
             for k, v in arr.items()}  # fmt: skip
        sid = np.full(self.n, self.stage) if stage_ids is None else np.asarray(stage_ids)
        self._reset(
            torch.ones(self.n, dtype=torch.bool, device=self.device),
            world=w,
            stage_ids=torch.as_tensor(sid, device=self.device),
        )
        return self._observe()

    def _reset(self, m: torch.Tensor, world: dict | None = None, stage_ids=None) -> None:
        """Reset the environments where mask m is True (all others are left unchanged)."""
        n, cfg, nom = self.n, self.cfg, self.nom

        def put(obj, name, new):
            old = getattr(obj, name)
            mm = m.view(-1, *([1] * (old.dim() - 1)))
            setattr(obj, name, torch.where(mm, new.to(old.dtype), old))

        if stage_ids is None:
            sid = torch.full((n,), self.stage, dtype=torch.long, device=self.device)
            if self.stage > 0:
                replay = self._rand(n) < cfg.replay_fraction
                sid = torch.where(replay, self._randint(self.stage, n), sid)
        else:
            sid = stage_ids.long()
        put(self, "stage_id", sid)

        # dynamics parameters (nominal, or randomized in the later stages)
        rand = self.st_randomize[sid]
        rg, p = self.ranges, self.p

        def dr(nominal, frac, *shape):
            nominal = torch.as_tensor(nominal, dtype=self.dtype, device=self.device)
            fresh = nominal * self._scale(frac, n, *shape)
            r = rand.view(-1, *([1] * len(shape)))
            return torch.where(r, fresh, nominal.expand_as(fresh))

        prm = self.params
        ge_on = 1.0 if p.ground_effect else 0.0
        put(prm, "mass", dr(p.mass, rg.mass))
        put(prm, "inertia", dr(p.inertia, rg.inertia, 3))
        put(prm, "thrust_coeff", dr(p.thrust_coeff, rg.thrust_coeff))
        put(prm, "motor_gain", dr(1.0, rg.per_motor_thrust, 4))
        put(prm, "kappa", dr(p.kappa, rg.kappa))
        put(prm, "motor_tau", dr(p.motor_tau, rg.motor_tau))
        put(prm, "drag_scale", dr(1.0, rg.drag))
        put(prm, "rotor_drag_scale", dr(1.0, rg.rotor_drag))
        put(prm, "ground_effect_scale", ge_on * dr(1.0, rg.ground_effect))
        put(self.rc, "gain_scale", torch.where(rand, self._scale(0.2, n), 1.0))
        put(self.rc, "integral", torch.zeros_like(self.rc.integral))

        realism = self.st_realism[sid]
        put(self, "realism", realism)
        put(self, "soft", self.st_soft[sid])

        # worlds and start state
        if world is None:
            pidx = self._randint(self.pool_size, n)
            world = {k: v[sid, pidx] for k, v in self.pool.items()}
        for k in self.world:
            old = self.world[k]
            mm = m.view(-1, *([1] * (old.dim() - 1)))
            self.world[k] = torch.where(mm, world[k], old)
        s = self.state
        rot = rot_z(world["yaw"])
        rnd = self.st_random_init[sid]
        axis = self._randn(n, 3)
        axis[:, 2] = 0.0
        axis = axis / (axis.norm(dim=1, keepdim=True) + 1e-9)
        angle = self._rand(n) * math.radians(35.0)
        tilted = rot @ so3_exp(axis * angle[:, None])
        put(s, "pos", world["start"])
        put(s, "rot", torch.where(rnd[:, None, None], tilted, rot))
        put(s, "vel", torch.where(rnd[:, None], 0.7 * self._randn(n, 3), 0.0))
        put(s, "omega", torch.where(rnd[:, None], self._randn(n, 3), 0.0))
        put(s, "motor_thrust", torch.full((n, 4), nom.hover_per_motor, dtype=self.dtype,
                                          device=self.device))  # fmt: skip

        # route bookkeeping and time budget
        wps, n_wp = self.world["wps"], self.world["n_wp"]
        put(self, "wp_idx", torch.zeros_like(self.wp_idx))
        put(self, "prev_dist", (wps[:, 0] - s.pos).norm(dim=1))
        put(self, "hold", torch.zeros_like(self.hold))
        put(self, "t", torch.zeros_like(self.t))
        pts = torch.cat([world["start"][:, None], world["wps"]], 1)
        seg = (pts[:, 1:] - pts[:, :-1]).norm(dim=-1)
        used = torch.arange(MAX_WP, device=self.device)[None, :] < world["n_wp"][:, None]
        length = (seg * used).sum(1)
        seconds = length / cfg.min_avg_speed + cfg.hold_time + 4.0
        put(self, "max_steps", (seconds / cfg.policy_dt).long())
        put(self, "prev_action", torch.zeros_like(self.prev_action))
        put(self, "filtered_action", torch.zeros_like(self.filtered_action))
        put(self, "action_hist", torch.zeros_like(self.action_hist))
        put(self, "delay", torch.where(realism > 0.5, self._randint(3, n), 0))
        del n_wp

        # wind, battery sag, estimate drift, pushes
        heading = self._rand(n) * 2 * math.pi
        speed = self._rand(n) * self.st_wind[sid]
        wind = torch.stack([torch.cos(heading) * speed, torch.sin(heading) * speed,
                            torch.zeros_like(speed)], 1)  # fmt: skip
        put(self, "wind_base", wind)
        put(self, "gust", torch.zeros_like(self.gust))
        put(self, "gust_std", self.st_gust[sid])
        put(self, "sag", self._rand(n) * 0.08 * realism)
        put(self, "est_bias", torch.zeros_like(self.est_bias))
        put(self, "push_step", torch.full_like(self.push_step, -1))
        put(self, "ep_return", torch.zeros_like(self.ep_return))
        put(self, "ep_collisions", torch.zeros_like(self.ep_collisions))

    def schedule_push(self, step: int, force) -> None:
        self.push_step[:] = step
        self.push_force[:] = torch.as_tensor(force, dtype=self.dtype, device=self.device)

    # ------------------------------------------------------------------ step
    def step(self, action, auto_reset: bool = True):
        cfg = self.cfg
        a = torch.as_tensor(action, device=self.device).to(self.dtype).clamp(-1.0, 1.0)
        k = cfg.action_filter
        self.filtered_action = k * a + (1.0 - k) * self.filtered_action

        self.action_hist = torch.roll(self.action_hist, 1, dims=1)
        self.action_hist[:, 0] = self.filtered_action
        ar = torch.arange(self.n, device=self.device)
        applied = self.action_hist[ar, self.delay]
        thrust, rates = action_to_ctbr(applied, self.nom)

        tau, dtp = 2.0, cfg.policy_dt
        gscale = torch.tensor([1.0, 1.0, 0.3], dtype=self.dtype, device=self.device)
        self.gust = (
            self.gust
            - self.gust * dtp / tau
            + self.gust_std[:, None] * math.sqrt(2 * dtp / tau) * self._randn(self.n, 3) * gscale
        )
        wind = self.wind_base + self.gust
        frac = torch.clamp(self.t / torch.clamp(self.max_steps, min=1), max=1.0).to(self.dtype)
        thrust_scale = 1.0 - self.sag * frac
        pushing = (
            (self.push_step >= 0)
            & (self.t >= self.push_step)
            & (self.t < self.push_step + int(0.1 / dtp) + 1)
        )
        ext = torch.where(pushing[:, None], self.push_force, 0.0)

        for _ in range(cfg.substeps):
            f = self.rc(thrust, rates, self.state.omega, cfg.sim_dt)
            step_dynamics(self.state, f, self.params, self.nom, cfg.sim_dt, wind, thrust_scale, ext)
        self.t = self.t + 1
        self.est_bias = self.est_bias + (
            self._randn(self.n, 3) * 0.05 * math.sqrt(dtp) * self.realism[:, None]
        )

        reward, done, truncated, info = self._reward_and_done(a)
        self.prev_action = a
        self.ep_return = self.ep_return + reward

        obs_a, obs_c = self._observe()
        info["terminal_critic_obs"] = obs_c
        info["episode_return"] = self.ep_return.clone()
        info["episode_length"] = self.t.clone()
        info["stage_id"] = self.stage_id.clone()

        if auto_reset:
            self._reset(done)
            idx = done.nonzero().squeeze(1)
            if idx.numel() > 0:  # fresh observations only for the envs that were reset
                new_a, new_c = self._observe(idx)
                obs_a, obs_c = obs_a.clone(), obs_c.clone()
                obs_a[idx], obs_c[idx] = new_a, new_c
        return obs_a, obs_c, reward, done, truncated, info

    def _reward_and_done(self, a):
        cfg, w = self.cfg, self.cfg.reward
        s = self.state
        ar = torch.arange(self.n, device=self.device)
        wps, n_wp = self.world["wps"], self.world["n_wp"]

        target = wps[ar, self.wp_idx]
        dist = (target - s.pos).norm(dim=1)
        is_final = self.wp_idx == n_wp - 1
        speed = s.vel.norm(dim=1)

        r = w.progress * (self.prev_dist - dist)
        reached = (~is_final) & (dist < cfg.waypoint_radius)
        r = r + w.waypoint_bonus * reached
        self.wp_idx = torch.where(reached, self.wp_idx + 1, self.wp_idx)
        new_dist = (wps[ar, self.wp_idx] - s.pos).norm(dim=1)
        dist = torch.where(reached, new_dist, dist)
        self.prev_dist = dist
        is_final = self.wp_idx == n_wp - 1

        near = is_final & (dist < cfg.final_radius) & (speed < cfg.final_max_speed)
        self.hold = torch.where(near, self.hold + cfg.policy_dt, 0.0)
        r = r + w.hover_shaping * is_final * torch.exp(-(dist**2) / 0.5) * torch.exp(-(speed**2))
        success = self.hold >= cfg.hold_time
        r = r + w.success_bonus * success

        wd = self.world
        d_obs = (
            distance_to_obstacles(s.pos, wd["cyl"], wd["cyl_mask"], wd["box"], wd["box_mask"])
            - self.nom.collision_radius
        )
        clear = torch.clamp((w.clearance_margin - d_obs) / w.clearance_margin, 0.0, 1.0)
        r = r - w.clearance * clear**2
        collided = d_obs <= 0.0
        ground = s.pos[:, 2] < 0.05
        outside = out_of_arena(s.pos)
        flipped = s.rot[:, 2, 2] < 0.0
        r = r - w.collision * collided * torch.where(self.soft, 0.1, 1.0)
        self.ep_collisions = self.ep_collisions + collided

        v_h = s.vel[:, :2]
        speed_h = v_h.norm(dim=1)
        nose = s.rot[:, :2, 0]
        nose = nose / torch.clamp(nose.norm(dim=1, keepdim=True), min=1e-6)
        align = (nose * v_h).sum(1) / torch.clamp(speed_h, min=1e-6)
        r = r + w.heading * torch.clamp(speed_h / 2.0, 0.0, 1.0) * align

        r = r - w.smoothness * ((a - self.prev_action) ** 2).sum(1)
        r = r - w.body_rate * (s.omega**2).sum(1)
        r = r - w.tilt * (1.0 - s.rot[:, 2, 2])
        r = r - w.time

        hard_fail = (collided & ~self.soft) | ground | outside | flipped
        r = r - w.collision * (ground | outside | flipped)
        timeout = self.t >= self.max_steps
        done = success | hard_fail | timeout
        truncated = timeout & ~success & ~hard_fail

        info = {
            "success": success,
            "collision": (collided & ~self.soft) | ground,
            "out_of_bounds": outside | flipped,
            "timeout": truncated,
            "waypoints_reached": self.wp_idx + success.long(),
            "n_waypoints": n_wp.clone(),
            "min_clearance": d_obs,
        }
        return r, done, truncated, info

    # ------------------------------------------------------------------ observations
    def _observe(self, idx: torch.Tensor | None = None):
        def g(x):
            return x if idx is None else x[idx]

        s, wd = self.state, self.world
        pos, vel, rot, omega = g(s.pos), g(s.vel), g(s.rot), g(s.omega)
        n = pos.shape[0]
        realism = g(self.realism)
        rr = realism[:, None]

        est_pos = pos + g(self.est_bias) + self._randn(n, 3) * 0.02 * rr
        est_vel = vel + self._randn(n, 3) * 0.05 * rr
        gyro = omega + self._randn(n, 3) * 0.02 * rr

        wps, n_wp, wp_idx = g(wd["wps"]), g(wd["n_wp"]), g(self.wp_idx)
        ar = torch.arange(n, device=self.device)
        wp1 = wps[ar, wp_idx]
        wp2 = wps[ar, torch.minimum(wp_idx + 1, n_wp - 1)]
        is_final = (wp_idx == n_wp - 1).to(self.dtype)

        def body_rel(pt):
            d = torch.einsum("nji,nj->ni", rot, pt - est_pos)
            return d / torch.clamp(d.norm(dim=1, keepdim=True), min=10.0)

        cyl, cm, box, bm = g(wd["cyl"]), g(wd["cyl_mask"]), g(wd["box"]), g(wd["box_mask"])
        prox, mask, _ = self.scanner.sense(rot, pos, cyl, cm, box, bm, realism, self.gen)

        actor = torch.cat(
            [
                rot.reshape(n, 9),
                torch.einsum("nji,nj->ni", rot, est_vel) / 5.0,
                gyro / 5.0,
                body_rel(wp1),
                body_rel(wp2),
                is_final[:, None],
                g(self.prev_action),
                prox,
                mask.to(self.dtype),
            ],
            1,
        )
        d_obs = distance_to_obstacles(pos, cyl, cm, box, bm)
        nom, prm = self.p, self.params
        t, max_steps = g(self.t).to(self.dtype), torch.clamp(g(self.max_steps), min=1)
        priv = torch.cat(
            [
                pos / 15.0,
                vel / 5.0,
                (g(self.wind_base) + g(self.gust)) / 5.0,
                (g(prm.mass) / nom.mass - 1.0)[:, None],
                (g(prm.thrust_coeff) / nom.thrust_coeff - 1.0)[:, None],
                (g(prm.motor_tau) / nom.motor_tau - 1.0)[:, None],
                torch.clamp(d_obs, -1.0, 5.0)[:, None] / 5.0,
                (wp_idx / torch.clamp(n_wp, min=1)).to(self.dtype)[:, None],
                (t / max_steps)[:, None],
                (1.0 - g(self.sag) * t / max_steps)[:, None],
                realism[:, None],
                (wp1 - pos) / 10.0,
            ],
            1,
        )
        return actor, torch.cat([actor, priv], 1)