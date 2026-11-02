"""Classical reference pilot: straight lines between waypoints, no obstacle avoidance.

A cascaded geometric controller (velocity -> attitude -> body rates) that outputs the
same normalized CTBR actions as the learned policy. It exists only as a sanity
reference: it shows what happens without learned avoidance.
"""

from __future__ import annotations

import numpy as np

from sim.actions import ctbr_to_action
from sim.params import GRAVITY, QuadParams


class StraightLinePilot:
    def __init__(self, params: QuadParams | None = None, cruise_speed: float = 3.0) -> None:
        self.p = params or QuadParams()
        self.cruise = cruise_speed
        self.k_pos = 1.2
        self.k_vel = 2.5
        self.k_att = 8.0
        self.max_tilt = np.deg2rad(35.0)

    def __call__(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        rot: np.ndarray,
        target: np.ndarray,
        is_final: np.ndarray,
    ) -> np.ndarray:
        """All inputs batched (n, ...). Returns normalized actions (n, 4)."""
        err = target - pos
        dist = np.linalg.norm(err, axis=1, keepdims=True)
        # slow down near the final waypoint, keep cruising through intermediate ones
        speed = np.where(is_final[:, None], np.minimum(self.cruise, self.k_pos * dist), self.cruise)
        v_des = err / np.maximum(dist, 1e-6) * speed
        acc = self.k_vel * (v_des - vel)
        acc[:, 2] += GRAVITY

        # limit tilt
        horiz = np.linalg.norm(acc[:, :2], axis=1)
        max_h = acc[:, 2] * np.tan(self.max_tilt)
        scale = np.where(horiz > max_h, max_h / np.maximum(horiz, 1e-6), 1.0)
        acc[:, :2] *= scale[:, None]

        z_des = acc / np.linalg.norm(acc, axis=1, keepdims=True)
        thrust = self.p.mass * np.sum(acc * rot[:, :, 2], axis=1)

        # desired yaw: face the direction of travel (keeps a forward camera useful)
        heading = np.arctan2(err[:, 1], err[:, 0])
        x_c = np.stack([np.cos(heading), np.sin(heading), np.zeros_like(heading)], axis=1)
        y_des = np.cross(z_des, x_c)
        y_des /= np.linalg.norm(y_des, axis=1, keepdims=True)
        x_des = np.cross(y_des, z_des)
        r_des = np.stack([x_des, y_des, z_des], axis=2)

        # attitude error e_R = 0.5 * vee(R_d^T R - R^T R_d)
        m = np.einsum("nji,njk->nik", r_des, rot) - np.einsum("nji,njk->nik", rot, r_des)
        e_r = 0.5 * np.stack([m[:, 2, 1], m[:, 0, 2], m[:, 1, 0]], axis=1)
        rates = -self.k_att * e_r
        rates = np.clip(rates, -np.asarray(self.p.body_rate_max), np.asarray(self.p.body_rate_max))
        return ctbr_to_action(thrust, rates, self.p)
