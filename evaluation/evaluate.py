"""Evaluate pilots on the fixed test scenarios, in ideal and realistic mode.

    python -m evaluation.evaluate --policy app/weights/policy.npz --episodes 50
    python -m evaluation.evaluate --pilot reference --episodes 20

Writes results.json and a Markdown table (results.md) to --out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.policy_numpy import NumpyPolicy
from evaluation.rollout import RunOptions, fly, policy_pilot, reference_pilot, summarize
from evaluation.scenarios import SCENARIOS, scenario_worlds


def _fmt(v, pct: bool = False) -> str:
    if v is None:
        return "–"
    return f"{100 * v:.0f}%" if pct else f"{v:.2f}"


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--policy", default="app/weights/policy.npz")
    ap.add_argument("--pilot", default="policy", choices=["policy", "reference", "both"])
    ap.add_argument("--episodes", type=int, default=50, help="episodes per scenario and mode")
    ap.add_argument("--modes", default="ideal,realistic")
    ap.add_argument("--scenarios", default="all", help="comma-separated names or 'all'")
    ap.add_argument("--out", default="docs/results")
    ap.add_argument("--profile", default=None, help="drone profile (default: the policy's)")
    args = ap.parse_args(argv)

    pilots = {}
    profile = args.profile
    if args.pilot in ("policy", "both"):
        policy = NumpyPolicy(args.policy)
        pilots["policy"] = policy_pilot(policy)
        profile = profile or policy.profile
    if args.pilot in ("reference", "both"):
        pilots["reference"] = reference_pilot()
    scenarios = (
        SCENARIOS
        if args.scenarios == "all"
        else [s for s in SCENARIOS if s.name in args.scenarios.split(",")]
    )
    modes = args.modes.split(",")

    results: dict = {}
    for pname, pilot in pilots.items():
        for mode in modes:
            for scn in scenarios:
                worlds = scenario_worlds(scn, args.episodes)
                opt = RunOptions(
                    realistic=(mode == "realistic"),
                    wind=scn.wind,
                    gust=scn.gust,
                    push=scn.push,
                    profile=profile or "kingfisher",
                )
                summary = summarize(fly(worlds, pilot, opt))
                results.setdefault(pname, {}).setdefault(mode, {})[scn.name] = summary
                print(
                    f"{pname:<9} {mode:<9} {scn.name:<16} success {summary['success_rate']:.2f}"
                    f"  collision {summary['collision_rate']:.2f}"
                )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2))

    lines = [
        "# Evaluation results",
        "",
        f"{args.episodes} episodes per scenario and mode, fixed test seeds.",
        "",
        "**Success** = every waypoint passed and a 2 s hover held at the last one, judged by the",
        "drone's onboard position estimate. In ideal mode that estimate is exact. In realistic",
        "mode it drifts during a flight (like visual-inertial odometry), which no policy can see",
        "from onboard sensors, so **Final error** gives the true distance from the target at",
        "the end of the hover.",
        "",
    ]
    for pname, by_mode in results.items():
        for mode, by_scn in by_mode.items():
            lines += [
                f"## {pname} · {mode}",
                "",
                "| Scenario | Success | Collision | Avg speed (m/s) | Route eff. | "
                "Final error (m) | Min clearance (m) |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
            for name, s in by_scn.items():
                lines.append(
                    f"| {name} | {_fmt(s['success_rate'], True)} | "
                    f"{_fmt(s['collision_rate'], True)} | {_fmt(s['avg_speed'])} | "
                    f"{_fmt(s['route_efficiency'])} | {_fmt(s['true_final_error'])} | "
                    f"{_fmt(s['min_clearance'])} |"
                )
            lines.append("")
    (out / "results.md").write_text("\n".join(lines))
    print(f"wrote {out / 'results.md'}")
    return results


if __name__ == "__main__":
    main()