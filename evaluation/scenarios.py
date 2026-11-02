"""Fixed, seeded test scenarios. The policy never trains on these exact worlds."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.world import World, WorldSpec, generate_world

TEST_SEED_OFFSET = 10_000  # keeps test worlds disjoint from training seeds


@dataclass(frozen=True)
class Scenario:
    name: str
    spec: WorldSpec
    description: str
    wind: float = 0.0  # m/s steady wind
    gust: float = 0.0  # m/s gust strength
    push: tuple[float, float, float] | None = None  # N, applied mid-flight
    tags: tuple[str, ...] = field(default_factory=tuple)


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("open_field", WorldSpec(n_waypoints=4), "No obstacles: cruise speed and hover"),
    Scenario("forest_low", WorldSpec(n_trees=12, n_waypoints=4), "Sparse trees"),
    Scenario("forest_medium", WorldSpec(n_trees=24, n_waypoints=4), "Medium-density forest"),
    Scenario("forest_high", WorldSpec(n_trees=36, n_waypoints=4), "Dense forest"),
    Scenario(
        "walls_with_gaps",
        WorldSpec(n_walls=6, n_blocks=2, n_waypoints=4),
        "Walls and buildings; find the openings",
    ),
    Scenario("corridor", WorldSpec(corridor=True, n_waypoints=3), "2.4 m wide corridor"),
    Scenario(
        "wind_and_gusts",
        WorldSpec(n_trees=24, n_waypoints=4),
        "Medium forest in 3 m/s wind with gusts",
        wind=3.0,
        gust=1.0,
    ),
    Scenario(
        "push_recovery",
        WorldSpec(n_waypoints=3),
        "Hit by a sideways push mid-flight",
        push=(0.0, 12.0, 0.0),
    ),
)


def scenario_by_name(name: str) -> Scenario:
    for s in SCENARIOS:
        if s.name == name:
            return s
    raise KeyError(name)


def scenario_worlds(scn: Scenario, n: int, seed: int = 0) -> list[World]:
    """n reproducible worlds for one scenario."""
    base = TEST_SEED_OFFSET + 97 * seed + sum(map(ord, scn.name))
    return [generate_world(scn.spec, np.random.default_rng(base + i)) for i in range(n)]
