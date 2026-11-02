import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.policy_numpy import NumpyPolicy  # noqa: E402
from rl.export import export  # noqa: E402
from rl.ppo import PPO, PPOConfig  # noqa: E402
from rl.train import main as train_main  # noqa: E402
from sim.env import EnvConfig, VecDroneEnv  # noqa: E402


def test_ppo_collect_and_update_run():
    env = VecDroneEnv(EnvConfig(n_envs=8, world_pool_size=4), seed=0)
    agent = PPO(env.actor_obs_dim, env.critic_obs_dim, PPOConfig(rollout_steps=16))
    obs_a, obs_c = env.reset_all()
    batch, obs_a, obs_c, _ = agent.collect(env, obs_a, obs_c)
    assert batch["obs_a"].shape == (128, env.actor_obs_dim)
    assert np.isfinite(batch["adv"]).all()
    stats = agent.update(batch)
    assert all(np.isfinite(v) for v in stats.values())


def test_train_resume_and_export(tmp_path):
    run = train_main(
        [
            "--run-name",
            "t",
            "--total-steps",
            "2048",
            "--n-envs",
            "16",
            "--rollout-steps",
            "64",
            "--runs-dir",
            str(tmp_path),
        ]
    )
    ckpt = run / "checkpoints" / "final.pt"
    assert ckpt.exists() and (run / "progress.csv").exists()

    # resume continues from the saved step count
    train_main(
        [
            "--run-name",
            "t2",
            "--total-steps",
            "4096",
            "--n-envs",
            "16",
            "--rollout-steps",
            "64",
            "--runs-dir",
            str(tmp_path),
            "--resume",
            str(ckpt),
        ]
    )

    out = export(ckpt, tmp_path / "policy.npz")  # raises if NumPy != PyTorch
    policy = NumpyPolicy(out)
    env = VecDroneEnv(EnvConfig(n_envs=4, world_pool_size=2), seed=1)
    obs_a, _ = env.reset_all()
    a = policy(obs_a.astype(np.float64))
    assert a.shape == (4, 4) and np.all(np.abs(a) <= 1.0)
