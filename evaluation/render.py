"""Figures for the README: learning curve and a top-down trajectory plot.

    python -m evaluation.render curve runs/first_try/progress.csv docs/learning_curve.png
    python -m evaluation.render flight --scenario forest_medium --out docs/flight.png

The hero GIF / video for the README is screen-recorded from the app's 3D viewer.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sim.world import ARENA_HALF  # noqa: E402


def learning_curve(csv_path: str | Path, out: str | Path) -> Path:
    rows = list(csv.DictReader(Path(csv_path).open()))
    steps = np.array([float(r["step"]) for r in rows]) / 1e6
    succ = np.array([float(r["success_rate"]) for r in rows])
    stages = [r["stage"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    ax.plot(steps, succ, color="#2563eb", lw=1.6)
    for i in range(1, len(stages)):
        if stages[i] != stages[i - 1]:
            ax.axvline(steps[i], color="#9ca3af", lw=0.8, ls="--")
            ax.text(
                steps[i],
                1.02,
                stages[i].replace("_", " "),
                rotation=30,
                fontsize=7,
                color="#4b5563",
                ha="left",
                va="bottom",
            )
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Success rate (current stage)")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def flight_plot(result: dict, world, out: str | Path, title: str = "") -> Path:
    pos = np.array([f["pos"] for f in result["frames"]])
    speed = np.array([f["speed"] for f in result["frames"]])
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    for k in np.flatnonzero(world.cyl_mask):
        x, y, r, _ = world.cyl[k]
        ax.add_patch(plt.Circle((x, y), r, color="#166534", alpha=0.8))
    for k in np.flatnonzero(world.box_mask):
        x0, y0, _, x1, y1, _ = world.box[k]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color="#57534e", alpha=0.85))
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=speed, cmap="viridis", s=3)
    wps = world.waypoints[: world.n_waypoints]
    ax.scatter(
        wps[:, 0],
        wps[:, 1],
        marker="o",
        s=90,
        facecolors="none",
        edgecolors="#ea580c",
        lw=2,
        label="waypoints",
    )
    ax.scatter(*world.start[:2], marker="^", s=60, color="#111827", label="start")
    fig.colorbar(sc, ax=ax, fraction=0.046, label="speed (m/s)")
    ax.set_xlim(-ARENA_HALF, ARENA_HALF)
    ax.set_ylim(-ARENA_HALF, ARENA_HALF)
    ax.set_aspect("equal")
    ax.set_title(title or ("success" if result["success"] else "failed"))
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("curve")
    c.add_argument("csv")
    c.add_argument("out")
    f = sub.add_parser("flight")
    f.add_argument("--scenario", default="forest_medium")
    f.add_argument("--policy", default="app/weights/policy.npz")
    f.add_argument("--reference", action="store_true", help="fly the classical pilot instead")
    f.add_argument("--realistic", action="store_true")
    f.add_argument("--index", type=int, default=0)
    f.add_argument("--out", default="docs/flight.png")
    args = ap.parse_args()

    if args.cmd == "curve":
        print(learning_curve(args.csv, args.out))
        return

    from app.policy_numpy import NumpyPolicy
    from evaluation.rollout import RunOptions, fly, policy_pilot, reference_pilot
    from evaluation.scenarios import scenario_by_name, scenario_worlds

    scn = scenario_by_name(args.scenario)
    world = scenario_worlds(scn, args.index + 1)[args.index]
    if args.reference:
        pilot, profile = reference_pilot(), "kingfisher"
    else:
        policy = NumpyPolicy(args.policy)
        pilot, profile = policy_pilot(policy), policy.profile
    res = fly(
        [world],
        pilot,
        RunOptions(
            realistic=args.realistic,
            wind=scn.wind,
            gust=scn.gust,
            push=scn.push,
            record=True,
            profile=profile,
        ),
    )[0]
    print(flight_plot(res, world, args.out, title=f"{scn.name}"))


if __name__ == "__main__":
    main()