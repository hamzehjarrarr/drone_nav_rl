"""Train the drone policy with PPO and an automatic curriculum.

Examples (run from the repository root):
    python -m rl.train --run-name first_try
    python -m rl.train --run-name quick --total-steps 2_000_000 --stage hover
    python -m rl.train --run-name more --resume runs/first_try/checkpoints/latest.pt
    python -m rl.train --device mps        # Apple GPU for the network updates
    python -m rl.train --sim torch         # whole simulator on the GPU (CUDA or MPS)

Outputs go to runs/<run-name>/: progress.csv, TensorBoard logs and checkpoints/.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from rl.ppo import PPO, PPOConfig
from sim.curriculum import STAGES, stage_index
from sim.env import EnvConfig, VecDroneEnv
from sim.params import DEFAULT_PROFILE


def pick_device(name: str) -> str:
    if name == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        return "cpu"
    return name


def parse_stage(value: str) -> int:
    return int(value) if value.isdigit() else stage_index(value)


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--total-steps", type=float, default=300e6)
    ap.add_argument(
        "--sim",
        default="numpy",
        choices=["numpy", "torch"],
        help="numpy: CPU simulator (default). torch: GPU simulator in simgpu/",
    )
    ap.add_argument(
        "--n-envs", type=int, default=None, help="parallel drones (default 256, or 4096 with torch)"
    )
    ap.add_argument(
        "--stage",
        default=None,
        help="start stage (name or index); default hover, or the checkpoint's stage",
    )
    ap.add_argument("--no-curriculum", action="store_true", help="stay on the start stage")
    ap.add_argument(
        "--window",
        type=int,
        default=None,
        help="episodes of the current stage used to judge advancement (default: 400, or a "
        "quarter of --n-envs if larger, so a lucky streak cannot advance the stage)",
    )
    ap.add_argument(
        "--device",
        default=None,
        choices=["cpu", "mps", "cuda", "auto"],
        help="default cpu for --sim numpy, auto (cuda > mps > cpu) for --sim torch",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="", help="checkpoint to continue from")
    ap.add_argument("--save-every", type=int, default=50, help="updates between checkpoints")
    ap.add_argument(
        "--rollout-steps", type=int, default=None, help="per env per update (64, or 32 with torch)"
    )
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument(
        "--profile",
        default=None,
        help="drone profile: built-in name or .toml path (default kingfisher, or the checkpoint's)",
    )
    ap.add_argument("--pool-size", type=int, default=1024, help="worlds per stage (torch sim)")
    ap.add_argument("--minibatches", type=int, default=None, help="per epoch (4, or 8 with torch)")
    args = ap.parse_args(argv)
    gpu_sim = args.sim == "torch"
    if args.n_envs is None:
        args.n_envs = 4096 if gpu_sim else 256
    if args.rollout_steps is None:
        args.rollout_steps = 32 if gpu_sim else 64
    if args.minibatches is None:
        args.minibatches = 8 if gpu_sim else 4
    if args.device is None:
        args.device = "auto" if gpu_sim else "cpu"
    if args.window is None:
        args.window = max(400, args.n_envs // 4)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)

    run_dir = Path(args.runs_dir) / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    stage = parse_stage(args.stage) if args.stage else 0
    ckpt = torch.load(args.resume, map_location=device, weights_only=True) if args.resume else None
    profile = args.profile or (ckpt or {}).get("profile") or DEFAULT_PROFILE
    env_cfg = EnvConfig(n_envs=args.n_envs, profile=profile)
    ppo_cfg = PPOConfig(rollout_steps=args.rollout_steps, lr=args.lr, minibatches=args.minibatches)
    if gpu_sim:
        from rl.ppo_gpu import GPUPPO
        from simgpu import TorchDroneEnv

        env = TorchDroneEnv(
            env_cfg, seed=args.seed, stage=stage, device=device, pool_size=args.pool_size
        )
        agent = GPUPPO(env.actor_obs_dim, env.critic_obs_dim, ppo_cfg, device=device)
    else:
        env = VecDroneEnv(env_cfg, seed=args.seed, stage=stage)
        agent = PPO(env.actor_obs_dim, env.critic_obs_dim, ppo_cfg, device=device)

    global_step = 0
    if ckpt is not None:
        agent.load_state_dict(ckpt)
        global_step = int(ckpt.get("global_step", 0))
        stage = stage if args.stage else int(ckpt.get("stage", stage))
        env.set_stage(stage)
        print(f"resumed from {args.resume} at step {global_step:,}, stage {STAGES[stage].name}")

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "device": device,
                "ppo": asdict(ppo_cfg),
                "env": {
                    "sim": args.sim,
                    "n_envs": args.n_envs,
                    "profile": profile,
                    "drone": env.p.name,
                    "actor_obs_dim": env.actor_obs_dim,
                    "critic_obs_dim": env.critic_obs_dim,
                },
            },
            indent=2,
        )
    )

    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(run_dir / "tb"))
    except Exception:  # tensorboard is optional
        writer = None

    csv_path = run_dir / "progress.csv"
    new_file = not csv_path.exists()
    csv_file = csv_path.open("a", newline="")
    log = csv.writer(csv_file)
    if new_file:
        log.writerow(
            [
                "step",
                "stage",
                "success_rate",
                "collision_rate",
                "timeout_rate",
                "mean_return",
                "mean_length",
                "waypoint_frac",
                "steps_per_sec",
                "policy_loss",
                "value_loss",
                "kl",
                "action_std",
            ]
        )

    recent: deque[dict] = deque(maxlen=args.window)
    obs_a, obs_c = env.reset_all()
    update = 0
    total = int(args.total_steps)
    print(
        f"sim={args.sim}  device={device}  envs={args.n_envs}  drone={env.p.name}  "
        f"stage={STAGES[env.stage].name}"
    )

    def save(name: str) -> None:
        sd = agent.state_dict()
        sd.update(
            global_step=global_step,
            stage=env.stage,
            actor_obs_dim=env.actor_obs_dim,
            critic_obs_dim=env.critic_obs_dim,
            profile=str(profile),
        )
        torch.save(sd, ckpt_dir / name)

    while global_step < total:
        t0 = time.time()
        batch, obs_a, obs_c, finished = agent.collect(env, obs_a, obs_c)
        stats = agent.update(batch)
        global_step += args.n_envs * args.rollout_steps
        update += 1
        sps = args.n_envs * args.rollout_steps / (time.time() - t0)

        recent.extend(e for e in finished if e["stage"] == env.stage)
        if recent:
            succ = np.mean([e["success"] for e in recent])
            coll = np.mean([e["collision"] for e in recent])
            tout = np.mean([e["timeout"] for e in recent])
            ret = np.mean([e["return"] for e in recent])
            length = np.mean([e["length"] for e in recent])
            wpf = np.mean([e["waypoint_frac"] for e in recent])
        else:
            succ = coll = tout = ret = length = wpf = float("nan")

        log.writerow(
            [
                global_step,
                STAGES[env.stage].name,
                f"{succ:.4f}",
                f"{coll:.4f}",
                f"{tout:.4f}",
                f"{ret:.3f}",
                f"{length:.1f}",
                f"{wpf:.4f}",
                f"{sps:.0f}",
                f"{stats['policy_loss']:.5f}",
                f"{stats['value_loss']:.5f}",
                f"{stats['kl']:.5f}",
                f"{stats['action_std']:.4f}",
            ]
        )
        csv_file.flush()
        if writer is not None:
            for k, v in {
                "success_rate": succ,
                "collision_rate": coll,
                "timeout_rate": tout,
                "return": ret,
                "episode_length": length,
                "waypoint_frac": wpf,
                "steps_per_sec": sps,
                "stage": env.stage,
                **stats,
            }.items():
                writer.add_scalar(k, v, global_step)

        if update % 10 == 0:
            print(
                f"step {global_step:>12,}  stage {STAGES[env.stage].name:<22} "
                f"success {succ:5.2f}  collision {coll:5.2f}  return {ret:7.2f}  "
                f"{sps:,.0f} steps/s"
            )

        # curriculum: advance once the current stage is mastered
        stage_cfg = STAGES[env.stage]
        if (
            not args.no_curriculum
            and len(recent) >= args.window
            and succ >= stage_cfg.advance_success
            and env.stage < len(STAGES) - 1
        ):
            save(f"stage_{env.stage}_{stage_cfg.name}.pt")
            env.set_stage(env.stage + 1)  # the torch sim builds that stage's worlds here
            recent.clear()
            print(f"==> advancing to stage {env.stage}: {STAGES[env.stage].name}")

        if update % args.save_every == 0:
            save("latest.pt")

    save("latest.pt")
    save("final.pt")
    csv_file.close()
    if writer is not None:
        writer.close()
    print(f"done: {global_step:,} steps. Checkpoints in {ckpt_dir}")
    return run_dir


if __name__ == "__main__":
    main()