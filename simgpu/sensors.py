"""PyTorch port of the exact min-pooled sector scan in sim/sensors.py.

Same geometry and the same sensor modes. One difference for speed on a GPU: instead
of picking a different number of nearby trees each step (which would force the GPU
to wait for the CPU), it always keeps the `k_cyl` nearest trees within range. With the
default of 16 that is exact in every training world (the dense stage has about 3-8
trees within 5 m); `overflow` reports if any environment ever had more.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from sim.sensors import RayConfig, _az_bins, camera_fov_mask
from sim.world import ARENA_HALF, ARENA_HEIGHT

_BIG = 1e9


class SectorScanT:
    def __init__(self, cfg: RayConfig, device, dtype, k_cyl: int = 16) -> None:
        def t(x):
            return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)

        self.cfg = cfg
        self.k_cyl = k_cyl
        edges, width, az_c = _az_bins(cfg)
        self.edges, self.width, self.az_c = t(edges), t(width), t(az_c)
        self.el = t(np.deg2rad(np.asarray(cfg.el_edges_deg)))
        self.fov = torch.as_tensor(camera_fov_mask(cfg), device=device)
        h, far, top = ARENA_HALF, 1e4, ARENA_HEIGHT + 1.0
        self.walls = t(
            [
                [h, -far, 0.0, far, far, top],
                [-far, -far, 0.0, -h, far, top],
                [-far, h, 0.0, far, far, top],
                [-far, -far, 0.0, far, -h, top],
            ]
        )
        self.overflow = torch.zeros((), dtype=torch.bool, device=device)
        self.lim85 = math.radians(85.0)

    # ------------------------------------------------------------------ exact scan
    def scan(self, pos, yaw, cyl, cyl_mask, box, box_mask, max_range, tilt=None):
        n = pos.shape[0]
        p, z = pos[:, :2], pos[:, 2]
        n_az = self.edges.shape[0]
        band = self.el[None, :, None].expand(n, -1, n_az)
        if tilt is not None:
            band = torch.clamp(band + tilt[:, None, :], -self.lim85, self.lim85)
        lo_b, hi_b = band[:, :-1], band[:, 1:]
        tan_lo, tan_hi = torch.tan(lo_b), torch.tan(hi_b)

        zc = torch.clamp(z, min=0.0)[:, None, None]
        ground = torch.where(lo_b < 0, zc / torch.tan(-lo_b), _BIG)
        hc = torch.clamp(ARENA_HEIGHT - z, min=0.0)[:, None, None]
        ceil_ = torch.where(hi_b > 0, hc / torch.tan(hi_b), _BIG)
        sector = torch.minimum(ground, ceil_)  # (n, n_el, n_az)

        ang = yaw[:, None] + self.edges[None, :]
        u = torch.stack([torch.cos(ang), torch.sin(ang)], -1)  # (n, n_az, 2)

        # ---- cylinders: keep the k nearest within range ----
        v_all = cyl[:, :, :2] - p[:, None, :]
        near_all = torch.clamp(v_all.norm(dim=-1) - cyl[:, :, 2], min=0.0)
        ok_all = cyl_mask & (near_all <= max_range)
        k = min(self.k_cyl, cyl.shape[1])
        if k < cyl.shape[1]:
            self.overflow = self.overflow | (ok_all.sum(1) > k).any()
        key = torch.where(ok_all, near_all, torch.full_like(near_all, float("inf")))
        idx = torch.topk(key, k, dim=1, largest=False).indices
        c = torch.gather(cyl, 1, idx[:, :, None].expand(-1, -1, 4))
        ok_c = torch.gather(ok_all, 1, idx)
        v = c[:, :, :2] - p[:, None, :]
        d_c = v.norm(dim=-1)
        r = c[:, :, 2]
        inside_c = d_c <= r
        uv = torch.einsum("nkd,ncd->nkc", u, v)  # (n, n_az, k)
        disc = uv**2 - (d_c**2 - r**2)[:, None, :]
        sq = torch.sqrt(torch.clamp(disc, min=0.0))
        t0, t1 = uv - sq, uv + sq
        zero = torch.zeros_like(t0)
        t_c = torch.where(t0 > 0, t0, torch.where(t1 > 0, zero, _BIG))
        t_c = torch.where(disc >= 0, t_c, _BIG)
        t_c = torch.where(inside_c[:, None, :], zero, t_c)
        near_c = torch.clamp(d_c - r, min=0.0)
        bear_c = torch.atan2(v[..., 1], v[..., 0])

        # ---- boxes and the four arena walls (all slots, there are few) ----
        b = torch.cat([box, self.walls[None].expand(n, -1, -1)], 1)
        ok_b = torch.cat([box_mask, torch.ones(n, 4, dtype=torch.bool, device=pos.device)], 1)
        lo, hi = b[:, :, :2], b[:, :, 3:5]
        w = torch.minimum(torch.maximum(p[:, None, :], lo), hi) - p[:, None, :]
        near_b = w.norm(dim=-1)
        ok_b = ok_b & (near_b <= max_range)
        inside_b = near_b <= 1e-9
        eps = 1e-12
        us = torch.where(u.abs() < eps, torch.where(u < 0, -eps, eps), u)[:, :, None, :]
        t_lo = (lo[:, None] - p[:, None, None, :]) / us
        t_hi = (hi[:, None] - p[:, None, None, :]) / us
        t_in = torch.minimum(t_lo, t_hi).amax(-1)
        t_out = torch.maximum(t_lo, t_hi).amin(-1)
        t_in0 = torch.clamp(t_in, min=0.0)
        t_b = torch.where(t_out >= t_in0, t_in0, _BIG)
        t_b = torch.where(inside_b[:, None, :], torch.zeros_like(t_b), t_b)
        bear_b = torch.atan2(w[..., 1], w[..., 0])

        near = torch.cat([near_c, near_b], 1)
        bear = torch.cat([bear_c, bear_b], 1)
        inside = torch.cat([inside_c, inside_b], 1)
        t_edge = torch.cat([t_c, t_b], 2)
        valid = torch.cat([ok_c, ok_b], 1)
        height = torch.cat([c[:, :, 3], b[:, :, 5]], 1)

        rel = torch.remainder(bear[:, None, :] - ang[:, :, None], 2 * math.pi)
        in_bin = (rel < self.width[None, :, None]) | inside[:, None, :]
        edge = torch.minimum(t_edge, torch.roll(t_edge, -1, dims=1))
        dist = torch.where(in_bin, near[:, None, :], edge)
        dist = torch.where(valid[:, None, :], dist, _BIG)

        dd = dist[:, None]  # (n, 1, n_az, O)
        vis = (-z[:, None, None, None] <= dd * tan_hi[..., None]) & (
            (height - z[:, None])[:, None, None, :] >= dd * tan_lo[..., None]
        )
        vis = vis | (dd <= 0.0)
        sector = torch.minimum(sector, torch.where(vis, dd, _BIG).amin(-1))
        return torch.clamp(sector, max=max_range).reshape(n, -1)

    # ------------------------------------------------------------------ full model
    def sense(self, rot, pos, cyl, cyl_mask, box, box_mask, realism, gen):
        """(proximity, valid mask, noise-free distance), like sim.sensors.sense."""
        cfg = self.cfg
        cam = realism > 0.5
        dz = (
            torch.cos(self.az_c)[None, :] * rot[:, 2, 0:1]
            + torch.sin(self.az_c)[None, :] * rot[:, 2, 1:2]
        )
        tilt = torch.where(cam[:, None], torch.asin(torch.clamp(dz, -1.0, 1.0)), 0.0)
        yaw = torch.atan2(rot[:, 1, 0], rot[:, 0, 0])
        dist = self.scan(pos, yaw, cyl, cyl_mask, box, box_mask, cfg.max_range, tilt)
        true_dist = dist

        rng_max = torch.where(cam, cfg.real_max_range, cfg.max_range).to(dist.dtype)[:, None]
        mask = torch.where(cam[:, None], self.fov[None, :], True)
        noise = torch.randn(dist.shape, generator=gen, device=dist.device, dtype=dist.dtype)
        noisy = dist + noise * cfg.real_noise_k * dist**2 * realism[:, None]
        dist = torch.where(cam[:, None], noisy, dist)
        drop = torch.rand(dist.shape, generator=gen, device=dist.device, dtype=dist.dtype)
        drop = drop < cfg.real_dropout * realism[:, None]
        dist = torch.where(cam[:, None] & drop, rng_max.expand_as(dist), dist)
        dist = torch.minimum(torch.clamp(dist, min=0.0), rng_max)
        prox = torch.where(mask, 1.0 - dist / rng_max, 0.0)
        return prox, mask, torch.minimum(true_dist, rng_max)