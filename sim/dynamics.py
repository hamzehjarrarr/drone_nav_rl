"""Vectorized 6-DoF quadrotor rigid-body dynamics.

Modelled: first-order motors, quadratic body drag, linear rotor drag (blade flapping and
induced drag), ground effect, wind, battery sag and external pushes.

All arrays carry a leading batch dimension n (one row per simulated drone).
Frames follow the conventions in sim/params.py (world z up, body FLU).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.math3d import orthonormalize, so3_exp
from sim.params import AIR_DENSITY, GRAVITY, ParamBatch


@dataclass
class QuadState:
    pos: np.ndarray  # (n, 3) world position, m
    vel: np.ndarray  # (n, 3) world velocity, m/s
    rot: np.ndarray  # (n, 3, 3) body-to-world rotation
    omega: np.ndarray  # (n, 3) body angular rate, rad/s
    motor_thrust: np.ndarray  # (n, 4) actual motor thrusts, N

    @classmethod
    def zeros(cls, n: int) -> QuadState:
        return cls(
            pos=np.zeros((n, 3)),
            vel=np.zeros((n, 3)),
            rot=np.broadcast_to(np.eye(3), (n, 3, 3)).copy(),
            omega=np.zeros((n, 3)),
            motor_thrust=np.zeros((n, 4)),
        )

    def copy(self) -> QuadState:
        return QuadState(
            self.pos.copy(),
            self.vel.copy(),
            self.rot.copy(),
            self.omega.copy(),
            self.motor_thrust.copy(),
        )


def ground_effect_factor(z: np.ndarray, rotor_radius: float, scale: np.ndarray) -> np.ndarray:
    """Thrust multiplier near the ground (Cheeseman-Bennett), (n,).

    1 / (1 - (R / 4z)^2), shifted so it fades to exactly 1 at five rotor radii and capped
    at one rotor radius above the ground. `scale` sets the strength (0 disables it).
    """
    r = rotor_radius
    zc = np.maximum(z, r)
    cb = 1.0 / (1.0 - (r / (4.0 * zc)) ** 2)
    cb_far = 1.0 / (1.0 - (1.0 / 20.0) ** 2)  # value at z = 5 R
    return 1.0 + scale * np.maximum(cb - cb_far, 0.0)


def step_dynamics(
    s: QuadState,
    motor_cmd: np.ndarray,
    params: ParamBatch,
    dt: float,
    wind: np.ndarray | None = None,
    thrust_scale: np.ndarray | None = None,
    external_force: np.ndarray | None = None,
) -> QuadState:
    """Advance the state by dt (in place) and return it.

    motor_cmd:      (n, 4) commanded per-motor thrusts, N
    wind:           (n, 3) world wind velocity, m/s
    thrust_scale:   (n,) multiplier on all motor thrust (battery sag), default 1
    external_force: (n, 3) extra world force, N (e.g. a push)
    """
    nom = params.nominal

    # ---- motors: first-order lag towards the command, then saturation ----
    cmd = np.clip(motor_cmd, nom.thrust_min, nom.thrust_max)
    alpha = 1.0 - np.exp(-dt / params.motor_tau)[:, None]
    s.motor_thrust += alpha * (cmd - s.motor_thrust)
    s.motor_thrust = np.clip(s.motor_thrust, nom.thrust_min, nom.thrust_max)

    f = s.motor_thrust * params.motor_gain * (params.thrust_coeff / nom.thrust_coeff)[:, None]
    if thrust_scale is not None:
        f = f * thrust_scale[:, None]
    f = f * ground_effect_factor(s.pos[:, 2], nom.rotor_radius, params.ground_effect_scale)[:, None]

    # ---- forces (world frame) ----
    collective = f.sum(axis=1)  # (n,)
    body_z = s.rot[:, :, 2]
    force = body_z * collective[:, None]
    force[:, 2] -= params.mass * GRAVITY

    air_vel = s.vel - (wind if wind is not None else 0.0)
    air_vel_b = np.einsum("nji,nj->ni", s.rot, air_vel)  # world -> body
    area = np.asarray(nom.frontal_area)
    cd = np.array([nom.drag_coeff[0], nom.drag_coeff[0], nom.drag_coeff[1]])
    drag_b = -0.5 * AIR_DENSITY * (cd * area)[None, :] * air_vel_b * np.abs(air_vel_b)
    drag_b *= params.drag_scale[:, None]
    # rotor drag: moving propellers create a drag force roughly proportional to airspeed;
    # at a few m/s it is larger than the body drag above
    drag_b -= np.asarray(nom.rotor_drag)[None, :] * params.rotor_drag_scale[:, None] * air_vel_b
    force += np.einsum("nij,nj->ni", s.rot, drag_b)
    if external_force is not None:
        force += external_force

    # ---- torques (body frame) ----
    xy = np.asarray(nom.rotor_xy)
    spin = np.asarray(nom.spin_sign)
    tau = np.stack(
        [
            f @ xy[:, 1],
            -(f @ xy[:, 0]),
            params.kappa * (f @ spin),
        ],
        axis=1,
    )
    j = params.inertia
    jw = j * s.omega
    omega_dot = (tau - np.cross(s.omega, jw)) / j

    # ---- semi-implicit Euler integration ----
    acc = force / params.mass[:, None]
    s.vel += acc * dt
    s.pos += s.vel * dt
    s.omega += omega_dot * dt
    s.rot = orthonormalize(s.rot @ so3_exp(s.omega * dt))
    return s