"""Export a trained actor to a plain NumPy .npz file for the app and evaluation.

    python -m rl.export runs/first_try/checkpoints/final.pt app/weights/policy.npz

After writing, the script checks that the NumPy forward pass reproduces the
PyTorch actor's actions on random inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from app.policy_numpy import NumpyPolicy
from rl.networks import Actor
from sim.curriculum import STAGES


def export(ckpt_path: str | Path, out_path: str | Path) -> Path:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    obs_dim = int(sd["actor_obs_dim"])
    hidden = tuple(sd["ppo_cfg"]["hidden"])
    actor = Actor(obs_dim, hidden=hidden)
    actor.load_state_dict(sd["actor"])
    actor.eval()

    linears = [m for m in actor.net if isinstance(m, torch.nn.Linear)]
    arrays: dict[str, np.ndarray] = {
        "obs_mean": actor.norm.mean.numpy().astype(np.float64),
        "obs_var": actor.norm.var.numpy().astype(np.float64),
        "obs_clip": np.array(actor.norm.clip),
        "n_layers": np.array(len(linears)),
    }
    for i, lin in enumerate(linears):
        arrays[f"w{i}"] = lin.weight.detach().numpy().astype(np.float64)
        arrays[f"b{i}"] = lin.bias.detach().numpy().astype(np.float64)
    stage = int(sd.get("stage", 0))
    meta = {
        "global_step": int(sd.get("global_step", 0)),
        "stage": stage,
        "stage_name": STAGES[stage].name,
        "hidden": list(hidden),
        "obs_dim": obs_dim,
        "profile": str(sd.get("profile", "kingfisher")),
    }
    arrays["meta"] = np.array(json.dumps(meta))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)

    # parity check against PyTorch
    policy = NumpyPolicy(out_path)
    x = np.random.default_rng(0).normal(size=(256, obs_dim)).astype(np.float32)
    with torch.no_grad():
        ref = torch.clamp(actor(torch.from_numpy(x)), -1, 1).numpy()
    err = np.abs(policy(x.astype(np.float64)) - ref).max()
    if err > 1e-4:
        raise RuntimeError(f"NumPy export does not match PyTorch (max error {err:.2e})")
    print(f"exported {ckpt_path} -> {out_path} (max parity error {err:.1e})")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("out", nargs="?", default="app/weights/policy.npz")
    args = ap.parse_args()
    export(args.checkpoint, args.out)


if __name__ == "__main__":
    main()