"""Small batched rotation helpers (all functions work on leading batch dimensions)."""

from __future__ import annotations

import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """(..., 3) -> (..., 3, 3) skew-symmetric matrices."""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    o = np.zeros_like(x)
    return np.stack(
        [
            np.stack([o, -z, y], axis=-1),
            np.stack([z, o, -x], axis=-1),
            np.stack([-y, x, o], axis=-1),
        ],
        axis=-2,
    )


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """Rodrigues formula. phi: (..., 3) rotation vectors -> (..., 3, 3)."""
    theta = np.linalg.norm(phi, axis=-1, keepdims=True)[..., None]  # (...,1,1)
    k = skew(phi)
    eye = np.broadcast_to(np.eye(3), k.shape)
    small = theta < 1e-8
    safe = np.where(small, 1.0, theta)
    a = np.where(small, 1.0, np.sin(safe) / safe)
    b = np.where(small, 0.5, (1.0 - np.cos(safe)) / (safe * safe))
    return eye + a * k + b * (k @ k)


def orthonormalize(r: np.ndarray) -> np.ndarray:
    """Gram-Schmidt re-orthonormalization of rotation matrices (..., 3, 3)."""
    x = r[..., :, 0]
    y = r[..., :, 1]
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    y = y - np.sum(x * y, axis=-1, keepdims=True) * x
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=-1)


def rot_z(yaw: np.ndarray) -> np.ndarray:
    """Rotation matrices about +z for yaw angles (...,)."""
    c, s = np.cos(yaw), np.sin(yaw)
    o, i = np.zeros_like(yaw), np.ones_like(yaw)
    return np.stack(
        [np.stack([c, -s, o], -1), np.stack([s, c, o], -1), np.stack([o, o, i], -1)], -2
    )


def rot_to_quat(r: np.ndarray) -> np.ndarray:
    """Rotation matrices (..., 3, 3) -> quaternions (..., 4) as (x, y, z, w)."""
    m = r
    tr = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q = np.zeros(m.shape[:-2] + (4,))
    # robust branchless-ish construction
    qw = np.sqrt(np.maximum(0.0, 1.0 + tr)) / 2.0
    qx = np.sqrt(np.maximum(0.0, 1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2])) / 2.0
    qy = np.sqrt(np.maximum(0.0, 1.0 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2])) / 2.0
    qz = np.sqrt(np.maximum(0.0, 1.0 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2])) / 2.0
    qx = np.copysign(qx, m[..., 2, 1] - m[..., 1, 2])
    qy = np.copysign(qy, m[..., 0, 2] - m[..., 2, 0])
    qz = np.copysign(qz, m[..., 1, 0] - m[..., 0, 1])
    q[..., 0], q[..., 1], q[..., 2], q[..., 3] = qx, qy, qz, qw
    return q / np.linalg.norm(q, axis=-1, keepdims=True)
