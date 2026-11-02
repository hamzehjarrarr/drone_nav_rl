"""Procedural worlds: arena, obstacles (vertical cylinders, axis-aligned boxes) and routes.

A world is stored per environment in fixed-size padded arrays so the simulator can
process all environments at once. Unused obstacle slots are masked out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ARENA_HALF = 15.0  # arena spans x, y in [-15, 15] m
ARENA_HEIGHT = 8.0  # z in [0, 8] m
MAX_CYL = 40
MAX_BOX = 12
MAX_WP = 6

GRID_RES = 0.5  # m, for the flyability check


@dataclass
class WorldSpec:
    """How to generate one world."""

    n_trees: int = 0  # vertical cylinders
    n_walls: int = 0  # long thin boxes (walls) with gaps between them
    n_blocks: int = 0  # building-like boxes
    n_waypoints: int = 3
    min_alt: float = 1.0
    max_alt: float = 4.0
    wp_clearance: float = 1.2  # min free radius around each waypoint
    min_leg: float = 4.0  # min distance between consecutive waypoints
    corridor: bool = False
    hover_only: bool = False  # one target within ~1 m of the start (hover training)


@dataclass
class World:
    cyl: np.ndarray  # (MAX_CYL, 4): x, y, radius, height
    cyl_mask: np.ndarray  # (MAX_CYL,) bool
    box: np.ndarray  # (MAX_BOX, 6): xmin, ymin, zmin, xmax, ymax, zmax
    box_mask: np.ndarray  # (MAX_BOX,) bool
    waypoints: np.ndarray  # (MAX_WP, 3)
    n_waypoints: int
    start: np.ndarray  # (3,)
    start_yaw: float


@dataclass
class WorldBatch:
    """Stacked worlds for n environments."""

    cyl: np.ndarray  # (n, MAX_CYL, 4)
    cyl_mask: np.ndarray  # (n, MAX_CYL)
    box: np.ndarray  # (n, MAX_BOX, 6)
    box_mask: np.ndarray  # (n, MAX_BOX)
    waypoints: np.ndarray  # (n, MAX_WP, 3)
    n_waypoints: np.ndarray  # (n,) int

    @classmethod
    def empty(cls, n: int) -> WorldBatch:
        return cls(
            cyl=np.zeros((n, MAX_CYL, 4)),
            cyl_mask=np.zeros((n, MAX_CYL), bool),
            box=np.zeros((n, MAX_BOX, 6)),
            box_mask=np.zeros((n, MAX_BOX), bool),
            waypoints=np.zeros((n, MAX_WP, 3)),
            n_waypoints=np.ones(n, int),
        )

    def set(self, i: int, w: World) -> None:
        self.cyl[i], self.cyl_mask[i] = w.cyl, w.cyl_mask
        self.box[i], self.box_mask[i] = w.box, w.box_mask
        self.waypoints[i], self.n_waypoints[i] = w.waypoints, w.n_waypoints


# --------------------------------------------------------------------------- geometry
def distance_to_obstacles(
    p: np.ndarray,
    cyl: np.ndarray,
    cyl_mask: np.ndarray,
    box: np.ndarray,
    box_mask: np.ndarray,
) -> np.ndarray:
    """Distance from points p (n, 3) to the nearest obstacle surface, ground or ceiling.

    Negative values mean the point is inside an obstacle.
    """
    big = 1e6
    # vertical cylinders (finite height)
    dxy = np.linalg.norm(p[:, None, :2] - cyl[:, :, :2], axis=-1) - cyl[:, :, 2]
    dz = p[:, None, 2] - cyl[:, :, 3]
    d_cyl = np.where(dz > 0, np.sqrt(np.maximum(dxy, 0) ** 2 + dz**2), dxy)
    d_cyl = np.where(cyl_mask, d_cyl, big).min(axis=1)
    # boxes (signed distance to an AABB)
    lo, hi = box[:, :, :3], box[:, :, 3:]
    q = np.maximum(lo - p[:, None, :], p[:, None, :] - hi)
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(q.max(axis=-1), 0.0)
    d_box = np.where(box_mask, outside + inside, big).min(axis=1)
    d_ground = p[:, 2]
    d_ceiling = ARENA_HEIGHT - p[:, 2]
    return np.minimum.reduce([d_cyl, d_box, d_ground, d_ceiling])


def out_of_arena(p: np.ndarray) -> np.ndarray:
    return (
        (np.abs(p[:, 0]) > ARENA_HALF) | (np.abs(p[:, 1]) > ARENA_HALF) | (p[:, 2] > ARENA_HEIGHT)
    )


# --------------------------------------------------------------------------- generation
def _occupancy(
    cyl: np.ndarray, cyl_mask: np.ndarray, box: np.ndarray, box_mask: np.ndarray, inflate: float
) -> np.ndarray:
    """2D occupancy grid (obstacles treated as full-height columns, inflated)."""
    n_cells = int(2 * ARENA_HALF / GRID_RES)
    c = (np.arange(n_cells) + 0.5) * GRID_RES - ARENA_HALF
    gx, gy = np.meshgrid(c, c, indexing="ij")
    occ = np.zeros((n_cells, n_cells), bool)
    for k in np.flatnonzero(cyl_mask):
        x, y, r, _ = cyl[k]
        occ |= (gx - x) ** 2 + (gy - y) ** 2 <= (r + inflate) ** 2
    for k in np.flatnonzero(box_mask):
        x0, y0, _, x1, y1, _ = box[k]
        occ |= (
            (gx >= x0 - inflate)
            & (gx <= x1 + inflate)
            & (gy >= y0 - inflate)
            & (gy <= y1 + inflate)
        )
    return occ


def _cell(p: np.ndarray) -> tuple[int, int]:
    n_cells = int(2 * ARENA_HALF / GRID_RES)
    i = int(np.clip((p[0] + ARENA_HALF) / GRID_RES, 0, n_cells - 1))
    j = int(np.clip((p[1] + ARENA_HALF) / GRID_RES, 0, n_cells - 1))
    return i, j


def reachable(occ: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    """Grid flood fill from a to b through free cells (4-connected)."""
    free = ~occ
    ia, ib = _cell(a), _cell(b)
    if not free[ia] or not free[ib]:
        return False
    seen = np.zeros_like(free)
    seen[ia] = True
    while True:
        grown = seen.copy()
        grown[1:, :] |= seen[:-1, :]
        grown[:-1, :] |= seen[1:, :]
        grown[:, 1:] |= seen[:, :-1]
        grown[:, :-1] |= seen[:, 1:]
        grown &= free
        if grown[ib]:
            return True
        if np.array_equal(grown, seen):
            return False
        seen = grown


def generate_world(spec: WorldSpec, rng: np.random.Generator, max_tries: int = 50) -> World:
    """Sample obstacles and a flyable route. Retries until the route is reachable."""
    for _ in range(max_tries):
        w = _try_generate(spec, rng)
        if w is not None:
            return w
    # fall back to an obstacle-free world so training never stalls
    return _try_generate(
        WorldSpec(n_waypoints=spec.n_waypoints, min_alt=spec.min_alt, max_alt=spec.max_alt), rng
    )  # type: ignore[return-value]


def _try_generate(spec: WorldSpec, rng: np.random.Generator) -> World | None:
    lim = ARENA_HALF - 1.5
    cyl = np.zeros((MAX_CYL, 4))
    cyl_mask = np.zeros(MAX_CYL, bool)
    box = np.zeros((MAX_BOX, 6))
    box_mask = np.zeros(MAX_BOX, bool)

    nt = min(spec.n_trees, MAX_CYL)
    if nt:
        cyl[:nt, 0:2] = rng.uniform(-lim, lim, size=(nt, 2))
        cyl[:nt, 2] = rng.uniform(0.15, 0.45, size=nt)  # trunk/canopy radius
        cyl[:nt, 3] = rng.uniform(4.0, ARENA_HEIGHT, size=nt)
        cyl_mask[:nt] = True

    k = 0
    for _ in range(min(spec.n_walls, MAX_BOX)):
        # a wall along x or y with random length; gaps appear between walls
        length = rng.uniform(4.0, 10.0)
        thick = 0.3
        cx, cy = rng.uniform(-lim + 2, lim - 2, size=2)
        h = rng.uniform(3.0, ARENA_HEIGHT)
        if rng.random() < 0.5:
            box[k] = [cx - length / 2, cy - thick / 2, 0.0, cx + length / 2, cy + thick / 2, h]
        else:
            box[k] = [cx - thick / 2, cy - length / 2, 0.0, cx + thick / 2, cy + length / 2, h]
        box_mask[k] = True
        k += 1
    for _ in range(min(spec.n_blocks, MAX_BOX - k)):
        sx, sy = rng.uniform(1.0, 3.0, size=2)
        cx, cy = rng.uniform(-lim, lim, size=2)
        h = rng.uniform(2.0, ARENA_HEIGHT)
        box[k] = [cx - sx / 2, cy - sy / 2, 0.0, cx + sx / 2, cy + sy / 2, h]
        box_mask[k] = True
        k += 1

    if spec.corridor:
        # two long parallel walls forming a corridor along x, 2.4 m wide
        k = 0
        box_mask[:] = False
        for side in (-1.2, 1.2):
            box[k] = [-lim, side - 0.15, 0.0, lim, side + 0.15, ARENA_HEIGHT]
            box_mask[k] = True
            k += 1

    occ = _occupancy(cyl, cyl_mask, box, box_mask, inflate=0.6)

    wp_lim = ARENA_HALF - 3.0  # keep routes away from the arena boundary

    def free_point(alt_lo: float, alt_hi: float) -> np.ndarray | None:
        for _ in range(200):
            if spec.corridor:
                p = np.array(
                    [
                        rng.uniform(-wp_lim, wp_lim),
                        rng.uniform(-0.3, 0.3),
                        rng.uniform(alt_lo, alt_hi),
                    ]
                )
            else:
                p = np.array([*rng.uniform(-wp_lim, wp_lim, size=2), rng.uniform(alt_lo, alt_hi)])
            d = distance_to_obstacles(
                p[None], cyl[None], cyl_mask[None], box[None], box_mask[None]
            )[0]
            if d > spec.wp_clearance or (spec.corridor and d > 0.8):
                return p
        return None

    start = free_point(1.0, 2.0)
    if start is None:
        return None
    if spec.hover_only:
        target = start + rng.uniform([-1.0, -1.0, -0.5], [1.0, 1.0, 0.5])
        target[2] = max(target[2], 1.0)
        waypoints = np.zeros((MAX_WP, 3))
        waypoints[0] = target
        yaw = float(rng.uniform(-np.pi, np.pi))
        return World(cyl, cyl_mask, box, box_mask, waypoints, 1, start, yaw)
    wps = []
    prev = start
    for _ in range(spec.n_waypoints):
        ok = False
        for _ in range(60):
            p = free_point(spec.min_alt, spec.max_alt)
            if p is None:
                return None
            if np.linalg.norm(p[:2] - prev[:2]) >= spec.min_leg and reachable(occ, prev, p):
                ok = True
                break
        if not ok:
            return None
        wps.append(p)
        prev = p

    waypoints = np.zeros((MAX_WP, 3))
    waypoints[: len(wps)] = wps
    first = wps[0] - start
    yaw = float(np.arctan2(first[1], first[0]) + rng.normal(0.0, 0.3))
    return World(cyl, cyl_mask, box, box_mask, waypoints, len(wps), start, yaw)
