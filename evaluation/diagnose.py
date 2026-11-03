"""Why do episodes fail? Fly a policy (noise-free) on fresh worlds of one training stage and
break every failure down by cause, with what the drone was doing just before.

    python -m evaluation.diagnose --policy app/weights/policy.npz --stage sparse_obstacles
    python -m evaluation.diagnose --stage realism --realistic

Prints plain text only; nothing is written to disk.
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from app.policy_numpy import NumpyPolicy
from sim.curriculum import STAGES, stage_index
from sim.env import EnvConfig, VecDroneEnv, WorldPool
from sim.world import ARENA_HALF, ARENA_HEIGHT


def classify(env: VecDroneEnv, i: int, info: dict) -> str:
    if info["success"][i]:
        return "success"
    p = env.state.pos[i]
    if env.state.rot[i, 2, 2] < 0.0:
        return "flipped"
    if p[2] > ARENA_HEIGHT:
        return "ceiling"
    if abs(p[0]) > ARENA_HALF or abs(p[1]) > ARENA_HALF:
        return "left arena"
    if p[2] < 0.05:
        return "ground"
    if info["collision"][i]:
        return "hit obstacle"
    return "timeout"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="app/weights/policy.npz")
    ap.add_argument("--stage", default="sparse_obstacles")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--realistic", action="store_true", help="depth camera, drift, latency")
    args = ap.parse_args()

    policy = NumpyPolicy(args.policy)
    sid = stage_index(args.stage)
    n = args.episodes
    pool = WorldPool(1, np.random.default_rng(args.seed))
    worlds = [pool._make(STAGES[sid]) for _ in range(n)]
    env = VecDroneEnv(EnvConfig(n_envs=n, world_pool_size=1, profile=policy.profile),
                      seed=args.seed, stage=sid)  # fmt: skip
    realism = 1.0 if args.realistic else 0.0
    obs, _ = env.load_worlds(worlds, stage_ids=np.full(n, sid), realism=realism)
    n_rays = env.n_rays
    n_az = env.cfg.rays.n_azimuth
    front = np.zeros(n_rays, bool)  # the 8 fine front bins, all bands
    for band in range(env.cfg.rays.n_elevation):
        front[band * n_az : band * n_az + 8] = True

    hist = []  # last ~0.5 s per env: speed, altitude, tilt, nearest seen (front / any)
    done_at = np.full(n, -1)
    cause = [""] * n
    snap = [None] * n
    max_steps = int(env.max_steps.max()) + 2
    for t in range(max_steps):
        prox = obs[:, -2 * n_rays : -n_rays]
        rng_max = env.cfg.rays.real_max_range if args.realistic else env.cfg.rays.max_range
        seen = (1.0 - prox) * rng_max
        hist.append(
            np.stack(
                [
                    np.linalg.norm(env.state.vel, axis=1),
                    env.state.pos[:, 2],
                    np.degrees(np.arccos(np.clip(env.state.rot[:, 2, 2], -1, 1))),
                    seen[:, front].min(1),
                    seen.min(1),
                ],
                1,
            )
        )
        hist = hist[-25:]
        obs, _, _, done, _, info = env.step(policy(obs.astype(np.float64)), auto_reset=False)
        for i in np.flatnonzero(done & (done_at < 0)):
            done_at[i] = t
            cause[i] = classify(env, i, info)
            h = np.stack(hist)[:, i]  # (steps, 5)
            snap[i] = dict(
                speed=h[-1, 0],
                alt=env.state.pos[i, 2],
                tilt=h[:, 2].max(),
                front_seen_05s=h[0, 3],
                any_seen_05s=h[0, 4],
                wp=int(info["waypoints_reached"][i]),
                n_wp=int(info["n_waypoints"][i]),
            )
        if (done_at >= 0).all():
            break

    counts = Counter(cause)
    mode = "realistic" if args.realistic else "ideal"
    print(f"{args.stage} ({mode} sensing): {n} episodes, noise-free policy, fresh worlds")
    for k in ("success", "hit obstacle", "ground", "flipped", "ceiling", "left arena", "timeout"):
        print(f"  {k:<13} {counts.get(k, 0) / n:6.1%}")

    print("\nper failure type (means): speed at end, altitude, max tilt in last 0.5 s,")
    print("nearest obstacle seen 0.5 s before (front sectors / any sector), waypoints done")
    for k in ("hit obstacle", "ground", "flipped", "ceiling", "left arena", "timeout"):
        rows = [snap[i] for i in range(n) if cause[i] == k]
        if not rows:
            continue

        def m(key, rows=rows):
            return np.mean([r[key] for r in rows])

        wp = np.mean([r["wp"] / max(r["n_wp"], 1) for r in rows])
        print(
            f"  {k:<13} n={len(rows):<4} speed {m('speed'):4.1f} m/s  alt {m('alt'):4.1f} m  "
            f"tilt {m('tilt'):4.0f} deg  seen {m('front_seen_05s'):4.1f} / "
            f"{m('any_seen_05s'):4.1f} m  waypoints {wp:4.0%}"
        )
    ok = [snap[i] for i in range(n) if cause[i] == "success"]
    if ok:
        t_ok = np.mean([done_at[i] for i in range(n) if cause[i] == "success"]) * env.cfg.policy_dt
        print(f"\n  successful flights take {t_ok:.1f} s on average")


if __name__ == "__main__":
    main()