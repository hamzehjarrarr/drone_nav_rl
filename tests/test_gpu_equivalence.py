"""The PyTorch simulator (simgpu/) must match the NumPy reference (sim/).

Runs on the CPU in float64, so differences are rounding only.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sim.curriculum import stage_index  # noqa: E402
from sim.dynamics import QuadState, step_dynamics  # noqa: E402
from sim.env import EnvConfig, VecDroneEnv, WorldPool  # noqa: E402
from sim.math3d import so3_exp  # noqa: E402
from sim.params import QuadParams, sample_params  # noqa: E402
from sim.sensors import RayConfig, sector_scan, sense  # noqa: E402
from sim.world import WorldBatch, WorldSpec, generate_world  # noqa: E402
from simgpu import TorchDroneEnv  # noqa: E402
from simgpu.core import Nominal, ParamsT, StateT  # noqa: E402
from simgpu.core import step_dynamics as step_t  # noqa: E402
from simgpu.sensors import SectorScanT  # noqa: E402

F64 = torch.float64


def t(x):
    """Copy (never share memory: the NumPy simulator updates its arrays in place)."""
    x = np.array(x)
    return torch.tensor(x) if x.dtype == bool else torch.tensor(x, dtype=F64)


def test_dynamics_match_with_randomized_parameters():
    rng = np.random.default_rng(0)
    n = 32
    pb = sample_params(n, rng, randomize=True)
    s = QuadState.zeros(n)
    s.pos[:] = rng.uniform([-3, -3, 0.05], [3, 3, 4], (n, 3))  # some close to the ground
    s.vel[:] = rng.normal(0, 2, (n, 3))
    s.rot[:] = so3_exp(rng.normal(0, 0.5, (n, 3)))
    s.omega[:] = rng.normal(0, 2, (n, 3))
    s.motor_thrust[:] = rng.uniform(0, 8, (n, 4))
    wind = rng.normal(0, 2, (n, 3))
    scale = rng.uniform(0.9, 1.0, n)
    ext = rng.normal(0, 1, (n, 3))

    st = StateT(t(s.pos), t(s.vel), t(s.rot), t(s.omega), t(s.motor_thrust))
    prm = ParamsT(**{k: t(getattr(pb, k)) for k in ParamsT.__dataclass_fields__})
    nom = Nominal(QuadParams(), "cpu", F64)
    for _ in range(50):
        cmd = rng.uniform(0, 8.5, (n, 4))
        step_dynamics(s, cmd, pb, 0.004, wind=wind, thrust_scale=scale, external_force=ext)
        step_t(st, t(cmd), prm, nom, 0.004, t(wind), t(scale), t(ext))
    for a, b in [(s.pos, st.pos), (s.vel, st.vel), (s.rot, st.rot), (s.omega, st.omega)]:
        np.testing.assert_allclose(b.numpy(), a, atol=1e-9)


@pytest.mark.parametrize("realistic", [False, True])
def test_sector_scan_matches(realistic):
    rng = np.random.default_rng(1)
    n = 24
    wb = WorldBatch.empty(n)
    for i in range(n):
        spec = WorldSpec(n_trees=28, n_walls=3, n_blocks=2, n_waypoints=2)
        wb.set(i, generate_world(spec, rng))
    pos = np.column_stack([rng.uniform(-14, 14, (n, 2)), rng.uniform(0.3, 6, n)])
    rot = so3_exp(rng.normal(0, 0.4, (n, 3)))
    cfg = RayConfig(real_noise_k=0.0, real_dropout=0.0)
    realism = np.full(n, 1.0 if realistic else 0.0)
    _, mask_np, d_np = sense(rot, pos, wb, cfg, realism, rng)

    sc = SectorScanT(cfg, "cpu", F64)
    world = [t(wb.cyl), t(wb.cyl_mask), t(wb.box), t(wb.box_mask)]
    _, mask_t, d_t = sc.sense(t(rot), t(pos), *world, t(realism), torch.Generator())
    np.testing.assert_array_equal(mask_t.numpy(), mask_np)
    np.testing.assert_allclose(d_t.numpy(), d_np, atol=1e-9)
    assert not bool(sc.overflow)

    # plain level scan as well
    yaw = rng.uniform(-np.pi, np.pi, n)
    ref = sector_scan(pos, yaw, wb.cyl, wb.cyl_mask, wb.box, wb.box_mask, cfg, 5.0)
    got = sc.scan(t(pos), t(yaw), *world, 5.0)
    np.testing.assert_allclose(got.numpy(), ref, atol=1e-9)


@pytest.mark.parametrize("stage", ["routes_open", "dense_obstacles"])
def test_full_environment_matches(stage):
    """Same worlds, same actions: observations, rewards and episode ends agree."""
    n = 12
    sid = stage_index(stage)
    from sim.curriculum import STAGES

    pool = WorldPool(1, np.random.default_rng(5))
    worlds = [pool._make(STAGES[sid]) for _ in range(n)]
    cfg = EnvConfig(n_envs=n, world_pool_size=2)
    ref = VecDroneEnv(cfg, seed=0, stage=sid)
    gpu = TorchDroneEnv(cfg, seed=0, stage=sid, device="cpu", dtype=F64, pool_size=4)
    oa_n, oc_n = ref.load_worlds(worlds, stage_ids=np.full(n, sid))
    oa_t, oc_t = gpu.load_worlds(worlds, stage_ids=np.full(n, sid))
    np.testing.assert_allclose(oc_t.numpy(), oc_n, atol=1e-5)

    rng = np.random.default_rng(2)
    alive = np.ones(n, bool)
    for _ in range(120):
        a = np.clip(rng.normal(0.05, 0.3, (n, 4)), -1, 1)
        oa_n, oc_n, r_n, d_n, tr_n, _ = ref.step(a, auto_reset=False)
        oa_t, oc_t, r_t, d_t, tr_t, _ = gpu.step(torch.as_tensor(a), auto_reset=False)
        # compare only envs whose episode is still running (finished ones keep flying)
        np.testing.assert_allclose(r_t.numpy()[alive], r_n[alive], atol=1e-6)
        np.testing.assert_allclose(oc_t.numpy()[alive], oc_n[alive], atol=1e-5)
        np.testing.assert_array_equal(d_t.numpy()[alive], d_n[alive])
        alive &= ~d_n
        if not alive.any():
            break


def test_gpu_env_trains_one_update():
    from rl.ppo import PPOConfig
    from rl.ppo_gpu import GPUPPO

    env = TorchDroneEnv(EnvConfig(n_envs=64), seed=0, stage=3, pool_size=8)
    agent = GPUPPO(env.actor_obs_dim, env.critic_obs_dim, PPOConfig(rollout_steps=16))
    obs_a, obs_c = env.reset_all()
    batch, obs_a, obs_c, finished = agent.collect(env, obs_a, obs_c)
    assert batch["obs_a"].shape == (64 * 16, env.actor_obs_dim)
    assert torch.isfinite(batch["adv"]).all()
    stats = agent.update(batch)
    assert all(np.isfinite(v) for v in stats.values())
    assert all(isinstance(e["success"], bool) for e in finished)