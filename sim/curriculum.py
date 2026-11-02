"""Curriculum stages. Each stage says what worlds look like and which effects are on.

Obstacle density rises in three steps (sparse, medium, dense). Collisions always end
the episode: an earlier version let the drone bump through obstacles in the sparse
stage, so it reached the dense stage without ever having learned to avoid anything.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    name: str
    waypoints: tuple[int, int] = (1, 1)  # inclusive range of waypoints per route
    trees: int = 0
    walls: int = 0
    blocks: int = 0
    corridor_prob: float = 0.0
    hover_only: bool = False  # single target right next to the start
    random_init: bool = False  # start from random attitudes / velocities
    soft_collisions: bool = False  # collisions penalized but not terminal (not used)
    randomize_dynamics: bool = False
    wind_max: float = 0.0  # m/s, steady wind
    gust_std: float = 0.0  # m/s, gust strength
    realism: float = 0.0  # 0 ideal sensing/estimation, 1 realistic
    advance_success: float = 0.8  # success rate needed to move to the next stage


_DENSE = dict(waypoints=(2, 5), trees=28, walls=3, blocks=2, corridor_prob=0.1)

STAGES: tuple[Stage, ...] = (
    Stage("hover", hover_only=True, random_init=True),
    Stage("single_waypoint", waypoints=(1, 1), random_init=True),
    Stage("routes_open", waypoints=(2, 5)),
    Stage("sparse_obstacles", waypoints=(2, 5), trees=10, walls=1, advance_success=0.75),
    Stage("medium_obstacles", waypoints=(2, 5), trees=18, walls=2, blocks=1, advance_success=0.7),
    Stage("dense_obstacles", **_DENSE, advance_success=0.65),
    Stage(
        "domain_randomization",
        **_DENSE,
        randomize_dynamics=True,
        wind_max=3.0,
        gust_std=1.0,
        advance_success=0.6,
    ),
    Stage(
        "realism",
        **_DENSE,
        randomize_dynamics=True,
        wind_max=3.0,
        gust_std=1.0,
        realism=1.0,
        advance_success=1.01,  # final stage: never advances
    ),
)


def stage_index(name: str) -> int:
    for i, s in enumerate(STAGES):
        if s.name == name:
            return i
    raise KeyError(f"unknown stage {name!r}; options: {[s.name for s in STAGES]}")