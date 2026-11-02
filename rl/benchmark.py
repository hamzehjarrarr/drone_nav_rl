"""Measure simulator and training throughput on this machine (week 1 benchmark).

    python -m rl.benchmark
    python -m rl.benchmark --devices cpu,mps

Prints environment steps per second for several batch sizes and the time of one
PPO update per device, so you can pick --n-envs and --device for training.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from rl.ppo import PPO, PPOConfig
from sim.curriculum import stage_index
from sim.env import EnvConfig, VecDroneEnv


def env_speed(n_envs: int, stage: str, steps: int = 100) -> float:
    env = VecDroneEnv(
        EnvConfig(n_envs=n_envs, world_pool_size=20), seed=0, stage=stage_index(stage)
    )
    a = np.zeros((n_envs, 4))
    for _ in range(20):
        env.step(a)
    t = time.perf_counter()
    for _ in range(steps):
        env.step(a)
    return n_envs * steps / (time.perf_counter() - t)


def update_time(device: str, n_envs: int = 256, rollout: int = 64) -> float:
    env = VecDroneEnv(EnvConfig(n_envs=n_envs, world_pool_size=20), seed=0)
    agent = PPO(
        env.actor_obs_dim, env.critic_obs_dim, PPOConfig(rollout_steps=rollout), device=device
    )
    obs_a, obs_c = env.reset_all()
    batch, *_ = agent.collect(env, obs_a, obs_c)
    t = time.perf_counter()
    agent.update(batch)
    return time.perf_counter() - t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="64,128,256,512")
    ap.add_argument("--devices", default="cpu")
    args = ap.parse_args()

    print("Simulator speed (environment steps per second)")
    for stage in ("routes_open", "dense_obstacles"):
        for n in map(int, args.envs.split(",")):
            print(f"  {stage:<16} n_envs={n:<5} {env_speed(n, stage):>10,.0f}")

    print("\nOne PPO update on 256 x 64 samples")
    for dev in args.devices.split(","):
        if dev == "mps" and not torch.backends.mps.is_available():
            print("  mps: not available")
            continue
        print(f"  {dev:<4} {update_time(dev):.2f} s")


if __name__ == "__main__":
    main()
