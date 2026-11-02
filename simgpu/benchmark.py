"""Measure training speed of the GPU simulator on this machine.

    python -m simgpu.benchmark                       # picks cuda > mps > cpu
    python -m simgpu.benchmark --device mps --envs 1024,2048,4096,8192

For each batch size, prints simulator-only speed and full training speed
(rollout + PPO update), in environment steps per second. Pick the --n-envs with the
best training speed that still fits in memory.
"""

from __future__ import annotations

import argparse
import time

import torch

from rl.ppo import PPOConfig
from rl.ppo_gpu import GPUPPO
from rl.train import pick_device
from sim.curriculum import stage_index
from sim.env import EnvConfig
from simgpu import TorchDroneEnv


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def measure(device: str, n: int, stage: str, rollout: int = 32) -> tuple[float, float]:
    env = TorchDroneEnv(
        EnvConfig(n_envs=n), seed=0, stage=stage_index(stage), device=device, pool_size=64
    )
    agent = GPUPPO(env.actor_obs_dim, env.critic_obs_dim, PPOConfig(rollout_steps=rollout),
                   device=device)  # fmt: skip
    a = torch.zeros(n, 4, device=device)
    with torch.no_grad():
        for _ in range(5):
            env.step(a)
        _sync(device)
        t = time.perf_counter()
        for _ in range(20):
            env.step(a)
        _sync(device)
    sim = n * 20 / (time.perf_counter() - t)

    obs_a, obs_c = env.reset_all()
    batch, obs_a, obs_c, _ = agent.collect(env, obs_a, obs_c)  # warm-up
    agent.update(batch)
    _sync(device)
    t = time.perf_counter()
    batch, obs_a, obs_c, _ = agent.collect(env, obs_a, obs_c)
    agent.update(batch)
    _sync(device)
    train = n * rollout / (time.perf_counter() - t)
    return sim, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--envs", default="1024,2048,4096,8192")
    ap.add_argument("--stages", default="routes_open,dense_obstacles")
    args = ap.parse_args()
    device = pick_device(args.device)
    print(f"device={device}   steps per second (simulator only / full training)")
    for stage in args.stages.split(","):
        for n in map(int, args.envs.split(",")):
            try:
                sim, train = measure(device, n, stage)
            except RuntimeError as err:  # out of memory
                print(f"  {stage:<16} n_envs={n:<6} failed: {str(err).splitlines()[0]}")
                break
            print(f"  {stage:<16} n_envs={n:<6} {sim:>10,.0f} / {train:>10,.0f}")


if __name__ == "__main__":
    main()