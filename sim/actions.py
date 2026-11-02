"""Mapping between normalized policy actions in [-1, 1]^4 and physical CTBR commands.

action[0] -> collective thrust, piecewise linear so that 0 means hover:
    -1 -> 0 N,  0 -> hover thrust,  +1 -> max collective thrust
action[1:4] -> body rates, scaled by the body-rate limits.
"""

from __future__ import annotations

import numpy as np

from sim.params import QuadParams


def action_to_ctbr(a: np.ndarray, p: QuadParams) -> tuple[np.ndarray, np.ndarray]:
    a = np.clip(a, -1.0, 1.0)
    hover = p.weight
    t_max = p.max_collective_thrust
    thrust = np.where(a[:, 0] >= 0, hover + a[:, 0] * (t_max - hover), hover * (1 + a[:, 0]))
    rates = a[:, 1:4] * np.asarray(p.body_rate_max)[None, :]
    return thrust, rates


def ctbr_to_action(thrust: np.ndarray, rates: np.ndarray, p: QuadParams) -> np.ndarray:
    hover = p.weight
    t_max = p.max_collective_thrust
    a0 = np.where(thrust >= hover, (thrust - hover) / (t_max - hover), thrust / hover - 1.0)
    a = np.concatenate([a0[:, None], rates / np.asarray(p.body_rate_max)[None, :]], axis=1)
    return np.clip(a, -1.0, 1.0)
