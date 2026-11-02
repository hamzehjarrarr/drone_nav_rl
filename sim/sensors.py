"""Range sensor: an exact, gap-free "sector scan" around the drone.

Why not plain ray casting: with one ray per direction, thin obstacles slip between
neighbouring rays (a 0.3 m tree trunk 3 m away is narrower than the gap between two
rays 22.5 degrees apart), so the drone cannot see them until it is almost touching them.

Instead, the space around the drone is split into sectors: azimuth bins (fine in front,
coarse behind) x elevation bands. Each sector reports the distance to the CLOSEST
obstacle surface anywhere inside it (min-pooling), like the min-pooled depth images
used in learned drone navigation. Nothing inside a sector can be missed.

All obstacles are vertical prisms standing on the ground (cylinders, boxes, the arena
walls), which makes the computation exact and cheap:
  * In the horizontal plane, the nearest point of a convex shape inside an azimuth
    wedge is either the shape's overall nearest point (if its bearing lies in the
    wedge) or the first hit along one of the two wedge edges.
  * Vertically, a prism at horizontal range d covers elevation angles from
    atan2(-z, d) to atan2(h - z, d); it counts for a band if the two ranges overlap.
Ground and ceiling are handled per band. Distances are horizontal ranges.

In ideal mode the scan is level (gimbal-stabilized): sectors turn with the drone's
heading (yaw) but not with its roll and pitch. In realistic mode the camera is body-fixed:
when the drone tilts, each sector's view tilts with it (modelled to first order by shifting
each azimuth bin's elevation bands by the elevation of that body direction).

Modes, chosen per environment by `realism` in [0, 1]:
  * ideal (realism <= 0.5): all sectors valid, range 5 m, no noise.
  * realistic (> 0.5): only sectors inside the reference depth camera's field of view
    (RealSense D435, about 85 x 58 degrees) are valid; range 3 m; noise that grows
    with distance squared and random dropouts. Invalid sectors read 0 and are flagged
    in a mask, so the network input has the same shape in both modes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.world import ARENA_HALF, ARENA_HEIGHT

_BIG = 1e9


def _default_az_edges() -> tuple[float, ...]:
    # 8 fine bins (11.25 deg) over the front +-45 deg, 8 coarse bins (33.75 deg) behind
    front = np.linspace(-45.0, 45.0, 9)
    back = np.linspace(45.0, 315.0, 9)[1:-1]
    return tuple(float(x) for x in np.concatenate([front, back]))


@dataclass(frozen=True)
class RayConfig:
    # azimuth bin edges in degrees, counter-clockwise from the nose (left is positive);
    # bin j spans [edge j, edge j+1], the last bin wraps around to the first edge
    az_edges_deg: tuple[float, ...] = _default_az_edges()
    # elevation band edges in degrees (bands are between consecutive edges)
    el_edges_deg: tuple[float, ...] = (-29.0, -14.5, 0.0, 14.5, 29.0)
    max_range: float = 5.0  # ideal-mode range, m
    # realistic depth camera (RealSense D435)
    fov_h_deg: float = 85.2
    fov_v_deg: float = 58.0
    real_max_range: float = 3.0  # operating range upper bound, m
    real_noise_k: float = 0.02  # noise std = k * d^2 (stereo depth error grows ~d^2)
    real_dropout: float = 0.05  # probability a valid sector returns nothing

    @property
    def n_azimuth(self) -> int:
        return len(self.az_edges_deg)

    @property
    def n_elevation(self) -> int:
        return len(self.el_edges_deg) - 1

    @property
    def n_rays(self) -> int:
        return self.n_azimuth * self.n_elevation


def _az_bins(cfg: RayConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin start angles, widths and centres in radians (all wrapped to [-pi, pi))."""
    e = np.deg2rad(np.asarray(cfg.az_edges_deg))
    nxt = np.roll(e, -1)
    width = np.mod(nxt - e, 2 * np.pi)
    width[width == 0] = 2 * np.pi
    centre = np.mod(e + width / 2 + np.pi, 2 * np.pi) - np.pi
    return e, width, centre


def ray_directions_body(cfg: RayConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit directions of the sector centres in the heading frame (R, 3), plus their
    azimuth and elevation in radians. Index order: elevation band major, azimuth minor."""
    _, _, az_c = _az_bins(cfg)
    el = np.deg2rad(np.asarray(cfg.el_edges_deg))
    el_c = 0.5 * (el[:-1] + el[1:])
    a, e = np.meshgrid(az_c, el_c, indexing="xy")
    a, e = a.ravel(), e.ravel()
    d = np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], axis=-1)
    return d, a, e


def camera_fov_mask(cfg: RayConfig) -> np.ndarray:
    """Sectors whose centre lies inside the depth camera's field of view."""
    _, a, e = ray_directions_body(cfg)
    return (np.abs(a) <= np.deg2rad(cfg.fov_h_deg) / 2 + 1e-9) & (
        np.abs(e) <= np.deg2rad(cfg.fov_v_deg) / 2 + 1e-9
    )


def _arena_boxes(n: int) -> np.ndarray:
    """The four arena walls as huge boxes outside the arena, (n, 4, 6)."""
    h, far, top = ARENA_HALF, 1e4, ARENA_HEIGHT + 1.0
    walls = np.array(
        [
            [h, -far, 0.0, far, far, top],
            [-far, -far, 0.0, -h, far, top],
            [-far, h, 0.0, far, far, top],
            [-far, -far, 0.0, far, -h, top],
        ]
    )
    return np.broadcast_to(walls, (n, 4, 6))


def _select(near: np.ndarray, valid: np.ndarray, max_range: float):
    """Indices of the obstacles that can matter: valid and nearer than max_range.

    A sector's distance to an obstacle is never below that obstacle's overall nearest
    distance, so anything farther than max_range cannot change the (capped) result.
    Picking exactly those keeps the scan exact while skipping empty and distant slots.
    """
    ok = valid & (near <= max_range)
    k = int(ok.sum(axis=1).max()) if ok.size else 0
    if k == 0:
        return None, None
    order = np.argsort(np.where(ok, near, np.inf), axis=1, kind="stable")[:, :k]
    return order, np.take_along_axis(ok, order, axis=1)


def sector_scan(
    pos: np.ndarray,
    yaw: np.ndarray,
    cyl: np.ndarray,
    cyl_mask: np.ndarray,
    box: np.ndarray,
    box_mask: np.ndarray,
    cfg: RayConfig,
    max_range: float,
    tilt: np.ndarray | None = None,
) -> np.ndarray:
    """Exact min-pooled horizontal range per sector, (n, n_rays), capped at max_range.

    tilt: optional (n, n_az) elevation offsets in radians that shift each azimuth bin's
    bands up or down (used for the body-fixed camera, which tilts with the drone).
    """
    n = pos.shape[0]
    p = pos[:, :2]
    z = pos[:, 2]
    edges, width, _ = _az_bins(cfg)
    n_az = len(edges)
    el = np.deg2rad(np.asarray(cfg.el_edges_deg))
    band = np.broadcast_to(el[None, :, None], (n, len(el), n_az))  # (n, n_el+1, n_az)
    if tilt is not None:
        band = np.clip(band + tilt[:, None, :], -np.deg2rad(85.0), np.deg2rad(85.0))
    tan_lo, tan_hi = np.tan(band[:, :-1]), np.tan(band[:, 1:])  # (n, n_el, n_az)

    # ground below and ceiling above, per band and bin
    with np.errstate(divide="ignore", invalid="ignore"):
        ground = np.where(
            band[:, :-1] < 0, np.maximum(z, 0.0)[:, None, None] / np.tan(-band[:, :-1]), _BIG
        )
        ceil_ = np.where(
            band[:, 1:] > 0,
            np.maximum(ARENA_HEIGHT - z, 0.0)[:, None, None] / np.tan(band[:, 1:]),
            _BIG,
        )
    sector = np.minimum(ground, ceil_)  # (n, n_el, n_az)

    # world-frame edge directions (n, n_az, 2)
    ang = yaw[:, None] + edges[None, :]
    u = np.stack([np.cos(ang), np.sin(ang)], axis=-1)

    parts = []  # (near, bearing, inside, edge hits (n, n_az, k), valid, height) per type

    # ---- cylinders ----
    v_all = cyl[:, :, :2] - p[:, None, :]
    near_all = np.maximum(np.linalg.norm(v_all, axis=-1) - cyl[:, :, 2], 0.0)
    idx, ok = _select(near_all, cyl_mask, max_range)
    if idx is not None:
        c = np.take_along_axis(cyl, idx[:, :, None], axis=1)  # (n, k, 4)
        v = c[:, :, :2] - p[:, None, :]
        d_c = np.linalg.norm(v, axis=-1)
        r = c[:, :, 2]
        inside = d_c <= r
        uv = np.einsum("nkd,ncd->nkc", u, v)  # (n, n_az, k)
        disc = uv**2 - (d_c**2 - r**2)[:, None, :]
        sq = np.sqrt(np.maximum(disc, 0.0))
        t0, t1 = uv - sq, uv + sq
        t = np.where(t0 > 0, t0, np.where(t1 > 0, 0.0, _BIG))
        t = np.where(disc >= 0, t, _BIG)
        t = np.where(inside[:, None, :], 0.0, t)
        near = np.maximum(d_c - r, 0.0)
        parts.append((near, np.arctan2(v[..., 1], v[..., 0]), inside, t, ok, c[:, :, 3]))

    # ---- boxes and the four arena walls ----
    boxes_all = np.concatenate([box, _arena_boxes(n)], axis=1)
    bmask_all = np.concatenate([box_mask, np.ones((n, 4), bool)], axis=1)
    q_all = np.clip(p[:, None, :], boxes_all[:, :, :2], boxes_all[:, :, 3:5])
    near_all = np.linalg.norm(q_all - p[:, None, :], axis=-1)
    idx, ok = _select(near_all, bmask_all, max_range)
    if idx is not None:
        b = np.take_along_axis(boxes_all, idx[:, :, None], axis=1)  # (n, k, 6)
        lo, hi = b[:, :, :2], b[:, :, 3:5]
        w = np.clip(p[:, None, :], lo, hi) - p[:, None, :]
        near = np.linalg.norm(w, axis=-1)
        inside = near <= 1e-9
        eps = 1e-12
        us = np.where(np.abs(u) < eps, np.where(u < 0, -eps, eps), u)[:, :, None, :]
        t_lo = (lo[:, None] - p[:, None, None, :]) / us
        t_hi = (hi[:, None] - p[:, None, None, :]) / us
        t_in = np.minimum(t_lo, t_hi).max(axis=-1)
        t_out = np.maximum(t_lo, t_hi).min(axis=-1)
        t = np.where(t_out >= np.maximum(t_in, 0.0), np.maximum(t_in, 0.0), _BIG)
        t = np.where(inside[:, None, :], 0.0, t)
        parts.append((near, np.arctan2(w[..., 1], w[..., 0]), inside, t, ok, b[:, :, 5]))

    if parts:
        near = np.concatenate([q[0] for q in parts], axis=1)  # (n, O)
        bear = np.concatenate([q[1] for q in parts], axis=1)
        inside = np.concatenate([q[2] for q in parts], axis=1)
        t_edge = np.concatenate([q[3] for q in parts], axis=2)  # (n, n_az, O)
        valid = np.concatenate([q[4] for q in parts], axis=1)
        height = np.concatenate([q[5] for q in parts], axis=1)

        # per azimuth bin: nearest point if its bearing is inside the bin, else edge hits
        rel = np.mod(bear[:, None, :] - ang[:, :, None], 2 * np.pi)
        in_bin = (rel < width[None, :, None]) | inside[:, None, :]
        dist = np.where(in_bin, near[:, None, :], np.minimum(t_edge, np.roll(t_edge, -1, axis=1)))
        dist = np.where(valid[:, None, :], dist, _BIG)  # (n, n_az, O)

        # elevation bands: a prism of height h at range d spans angles atan2(-z, d) to
        # atan2(h - z, d); compare with tangents instead of angles (same result, cheaper)
        dd = dist[:, None]  # (n, 1, n_az, O)
        vis = (-z[:, None, None, None] <= dd * tan_hi[..., None]) & (
            (height - z[:, None])[:, None, None, :] >= dd * tan_lo[..., None]
        )
        vis |= dd <= 0.0
        sector = np.minimum(sector, np.where(vis, dd, _BIG).min(axis=-1))

    return np.minimum(sector, max_range).reshape(n, -1)


def heading(rot: np.ndarray) -> np.ndarray:
    """Yaw angle of the body x-axis projected on the horizontal plane."""
    return np.arctan2(rot[:, 1, 0], rot[:, 0, 0])


def sense(
    rot: np.ndarray,
    pos: np.ndarray,
    world,
    cfg: RayConfig,
    realism: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full sensor model.

    Returns (proximity (n, R) in [0, 1] where 1 = touching, valid mask (n, R),
    noise-free distances (n, R)).
    """
    cam = realism > 0.5
    tilt = None
    if cam.any():
        # the camera is bolted to the drone: when it pitches or rolls, each azimuth bin's
        # view tilts by the elevation of that body direction (first-order tilt model)
        _, _, az_c = _az_bins(cfg)
        dz = np.cos(az_c)[None, :] * rot[:, 2, 0:1] + np.sin(az_c)[None, :] * rot[:, 2, 1:2]
        tilt = np.where(cam[:, None], np.arcsin(np.clip(dz, -1.0, 1.0)), 0.0)
    dist = sector_scan(
        pos,
        heading(rot),
        world.cyl,
        world.cyl_mask,
        world.box,
        world.box_mask,
        cfg,
        cfg.max_range,
        tilt=tilt,
    )
    true_dist = dist.copy()

    mask = np.ones_like(dist, dtype=bool)
    rng_max = np.where(cam, cfg.real_max_range, cfg.max_range)[:, None]
    if cam.any():
        mask[cam] = camera_fov_mask(cfg)[None, :]
        noise = rng.normal(size=dist.shape) * cfg.real_noise_k * dist**2 * realism[:, None]
        dist = np.where(cam[:, None], dist + noise, dist)
        drop = rng.random(dist.shape) < cfg.real_dropout * realism[:, None]
        dist = np.where(cam[:, None] & drop, rng_max, dist)
    dist = np.clip(dist, 0.0, rng_max)
    proximity = np.where(mask, 1.0 - dist / rng_max, 0.0)
    return proximity, mask, np.minimum(true_dist, rng_max)