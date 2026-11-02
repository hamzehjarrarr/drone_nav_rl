import numpy as np

from sim.baseline import StraightLinePilot
from sim.curriculum import STAGES, stage_index
from sim.env import EnvConfig, VecDroneEnv


def _env(stage: int, n: int = 16, seed: int = 0) -> VecDroneEnv:
    return VecDroneEnv(EnvConfig(n_envs=n, world_pool_size=10), seed=seed, stage=stage)


def test_observation_shapes_and_finiteness():
    for stage in range(len(STAGES)):
        env = _env(stage)
        oa, oc = env.reset_all()
        assert oa.shape == (16, env.actor_obs_dim)
        assert oc.shape == (16, env.critic_obs_dim)
        for _ in range(20):
            oa, oc, r, d, tr, info = env.step(np.random.default_rng(0).uniform(-1, 1, (16, 4)))
            assert np.isfinite(oa).all() and np.isfinite(oc).all() and np.isfinite(r).all()


def test_same_seed_is_deterministic():
    outs = []
    for _ in range(2):
        env = _env(stage_index("dense_obstacles"), seed=7)
        a = np.zeros((16, 4))
        total = 0.0
        for _ in range(30):
            _, _, r, _, _, _ = env.step(a)
            total += r.sum()
        outs.append(total)
    assert outs[0] == outs[1]


def test_baseline_pilot_flies_open_routes():
    env = _env(stage_index("routes_open"), n=16, seed=2)
    pilot = StraightLinePilot()
    successes = failures = 0
    for _ in range(1500):
        ar = np.arange(env.n)
        target = env.world.waypoints[ar, env.wp_idx]
        final = env.wp_idx == env.world.n_waypoints - 1
        a = pilot(env.state.pos, env.state.vel, env.state.rot, target, final)
        _, _, _, d, _, info = env.step(a)
        successes += int(info["success"][d].sum())
        failures += int((d & ~info["success"]).sum())
    assert successes >= 10
    assert successes / max(successes + failures, 1) > 0.9


def test_crashing_ends_episode_with_penalty():
    env = _env(stage_index("routes_open"), n=4)
    a = np.tile([-1.0, 0.0, 0.0, 0.0], (4, 1))  # zero thrust: falls to the ground
    done_seen = False
    for _ in range(200):
        _, _, r, d, tr, info = env.step(a)
        if d.any():
            assert info["collision"][d].all()
            assert (r[d] < -5).all()
            assert not tr[d].any()
            done_seen = True
            break
    assert done_seen


def test_realism_stage_sets_latency_and_masks():
    env = _env(stage_index("realism"), n=32)
    assert (env.realism[env.stage_id == stage_index("realism")] == 1.0).all()
    oa, _ = env.reset_all()
    n_rays = env.n_rays
    mask = oa[:, -n_rays:]
    real = env.realism > 0.5
    assert (mask[real].sum(axis=1) < n_rays).all()
    assert (mask[~real] == 1.0).all()


def test_action_filter_smooths_commands():
    env = _env(stage_index("routes_open"), n=2)
    env.step(np.zeros((2, 4)))
    env.step(np.ones((2, 4)))
    k = env.cfg.action_filter
    assert np.allclose(env.filtered_action, k)  # moves only part of the way to the command
    assert np.allclose(env.prev_action, 1.0)  # the policy still sees its own raw action


def test_target_offset_is_fine_grained_near_the_waypoint():
    env = _env(stage_index("routes_open"), n=2)
    env.state.rot[:] = np.eye(3)
    wp = env.world.waypoints[np.arange(2), env.wp_idx]
    env.state.pos[0] = wp[0] - [0.3, 0.0, 0.0]  # 0.3 m behind the waypoint
    env.state.pos[1] = wp[1] - [0.5, 0.0, 0.0]
    oa, _ = env._observe()
    k = 9 + 3 + 3 + 3 + 3  # attitude, velocity, gyro, two scaled waypoints
    assert np.allclose(oa[:, k], [0.15, 0.25], atol=1e-6)  # metres / 2, clearly apart


def test_speed_safety_penalizes_rushing_at_obstacles():
    from sim.env import RewardConfig

    rewards = {}
    for name, rc in {"on": RewardConfig(), "off": RewardConfig(speed_safety=0.0)}.items():
        env = VecDroneEnv(
            EnvConfig(n_envs=2, world_pool_size=2, reward=rc),
            seed=0,
            stage=stage_index("routes_open"),
        )
        env.world.cyl[:, 0] = [2.0, 0.0, 0.3, 6.0]  # nearer than ground and ceiling
        env.world.cyl_mask[:, 0] = True
        env.state.pos[:] = [0.0, 0.0, 4.0]
        env.state.rot[:] = np.eye(3)
        env.state.vel[:] = [[2.0, 0, 0], [8.0, 0, 0]]
        clear = 2.0 - 0.3 - env.p.collision_radius
        env.prev_clear[:] = clear + env.state.vel[:, 0] * env.cfg.policy_dt  # approaching
        r, *_ = env._reward_and_done(np.zeros((2, 4)))
        rewards[name] = r
    penalty = rewards["off"] - rewards["on"]
    v_safe = np.sqrt(2 * RewardConfig().brake_accel * clear)
    assert penalty[0] == 0.0  # 2 m/s at 1.5 m: can still brake
    assert np.isclose(penalty[1], RewardConfig().speed_safety * (8.0 - v_safe) ** 2)


def test_attitude_penalty_starts_past_60_degrees():
    from sim.env import RewardConfig
    from sim.math3d import so3_exp

    rewards = {}
    for name, rc in {"on": RewardConfig(), "off": RewardConfig(attitude=0.0)}.items():
        env = VecDroneEnv(
            EnvConfig(n_envs=2, world_pool_size=2, reward=rc),
            seed=0,
            stage=stage_index("routes_open"),
        )
        env.state.pos[:] = [0.0, 0.0, 4.0]
        env.state.rot[0] = so3_exp(np.array([np.deg2rad(50.0), 0.0, 0.0]))
        env.state.rot[1] = so3_exp(np.array([np.deg2rad(75.0), 0.0, 0.0]))
        r, *_ = env._reward_and_done(np.zeros((2, 4)))
        rewards[name] = r
    penalty = rewards["off"] - rewards["on"]
    assert penalty[0] == 0.0  # 50 degrees: allowed
    assert np.isclose(penalty[1], 0.5 - np.cos(np.deg2rad(75.0)))  # 75 degrees: penalized