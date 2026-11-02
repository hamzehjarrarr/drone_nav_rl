"""Batched rotations, rigid-body dynamics, rate controller and geometry in PyTorch.

Line-for-line ports of sim/math3d.py, sim/dynamics.py, sim/rate_controller.py,
sim/actions.py and the geometry helpers in sim/world.py. Every function works on a
leading batch dimension n and keeps tensors on whatever device they are on.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sim.params import AIR_DENSITY, GRAVITY, QuadParams
from sim.world import ARENA_HALF, ARENA_HEIGHT


# --------------------------------------------------------------------------- rotations
def skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([o, -z, y], -1),
            torch.stack([z, o, -x], -1),
            torch.stack([-y, x, o], -1),
        ],
        -2,
    )


def so3_exp(phi: torch.Tensor) -> torch.Tensor:
    """Rodrigues formula, (..., 3) -> (..., 3, 3). Written in a float32-safe form."""
    theta = phi.norm(dim=-1, keepdim=True).unsqueeze(-1)
    k = skew(phi)
    eye = torch.eye(3, dtype=phi.dtype, device=phi.device).expand_as(k)
    small = theta < 1e-8
    safe = torch.where(small, torch.ones_like(theta), theta)
    a = torch.where(small, torch.ones_like(theta), torch.sin(safe) / safe)
    half = torch.where(small, torch.ones_like(theta), torch.sin(safe / 2) / (safe / 2))
    b = 0.5 * half * half  # = (1 - cos t) / t^2 without cancellation
    return eye + a * k + b * (k @ k)


def orthonormalize(r: torch.Tensor) -> torch.Tensor:
    x = r[..., :, 0]
    y = r[..., :, 1]
    x = x / x.norm(dim=-1, keepdim=True)
    y = y - (x * y).sum(-1, keepdim=True) * x
    y = y / y.norm(dim=-1, keepdim=True)
    z = torch.linalg.cross(x, y)
    return torch.stack([x, y, z], -1)


def rot_z(yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(yaw), torch.sin(yaw)
    o, i = torch.zeros_like(yaw), torch.ones_like(yaw)
    return torch.stack(
        [torch.stack([c, -s, o], -1), torch.stack([s, c, o], -1), torch.stack([o, o, i], -1)], -2
    )


# --------------------------------------------------------------------------- drone model
class Nominal:
    """The nominal drone profile as tensors (what the firmware rate controller knows)."""

    def __init__(self, p: QuadParams, device, dtype) -> None:
        def t(x):
            return torch.as_tensor(x, dtype=dtype, device=device)

        self.p = p
        self.mass = p.mass
        self.inertia = t(p.inertia)
        self.rotor_xy = t(p.rotor_xy)
        self.spin = t(p.spin_sign)
        self.rotor_radius = p.rotor_radius
        self.thrust_coeff = p.thrust_coeff
        self.motor_tau = p.motor_tau
        self.thrust_min = p.thrust_min
        self.thrust_max = p.thrust_max
        self.area = t(p.frontal_area)
        self.cd = t([p.drag_coeff[0], p.drag_coeff[0], p.drag_coeff[1]])
        self.rotor_drag = t(p.rotor_drag)
        self.mixer = t(p.mixer_matrix())
        self.body_rate_max = t(p.body_rate_max)
        self.weight = p.weight
        self.max_collective = p.max_collective_thrust
        self.hover_per_motor = p.hover_thrust_per_motor
        self.collision_radius = p.collision_radius


@dataclass
class StateT:
    pos: torch.Tensor  # (n, 3)
    vel: torch.Tensor  # (n, 3)
    rot: torch.Tensor  # (n, 3, 3)
    omega: torch.Tensor  # (n, 3)
    motor_thrust: torch.Tensor  # (n, 4)


@dataclass
class ParamsT:
    """Per-environment physical parameters (same fields as sim.params.ParamBatch)."""

    mass: torch.Tensor  # (n,)
    inertia: torch.Tensor  # (n, 3)
    thrust_coeff: torch.Tensor  # (n,)
    motor_gain: torch.Tensor  # (n, 4)
    kappa: torch.Tensor  # (n,)
    motor_tau: torch.Tensor  # (n,)
    drag_scale: torch.Tensor  # (n,)
    rotor_drag_scale: torch.Tensor  # (n,)
    ground_effect_scale: torch.Tensor  # (n,)


def ground_effect_factor(z: torch.Tensor, rotor_radius: float, scale: torch.Tensor):
    r = rotor_radius
    zc = torch.clamp(z, min=r)
    cb = 1.0 / (1.0 - (r / (4.0 * zc)) ** 2)
    cb_far = 1.0 / (1.0 - (1.0 / 20.0) ** 2)
    return 1.0 + scale * torch.clamp(cb - cb_far, min=0.0)


def step_dynamics(
    s: StateT,
    motor_cmd: torch.Tensor,
    prm: ParamsT,
    nom: Nominal,
    dt: float,
    wind: torch.Tensor,
    thrust_scale: torch.Tensor,
    external_force: torch.Tensor,
) -> None:
    """Advance the state by dt in place (same model as sim.dynamics.step_dynamics)."""
    cmd = torch.clamp(motor_cmd, nom.thrust_min, nom.thrust_max)
    alpha = (1.0 - torch.exp(-dt / prm.motor_tau))[:, None]
    mt = s.motor_thrust + alpha * (cmd - s.motor_thrust)
    s.motor_thrust = torch.clamp(mt, nom.thrust_min, nom.thrust_max)

    f = s.motor_thrust * prm.motor_gain * (prm.thrust_coeff / nom.thrust_coeff)[:, None]
    f = f * thrust_scale[:, None]
    f = f * ground_effect_factor(s.pos[:, 2], nom.rotor_radius, prm.ground_effect_scale)[:, None]

    collective = f.sum(1)
    force = s.rot[:, :, 2] * collective[:, None]
    force[:, 2] -= prm.mass * GRAVITY

    air_vel = s.vel - wind
    air_vel_b = torch.einsum("nji,nj->ni", s.rot, air_vel)
    drag_b = -0.5 * AIR_DENSITY * (nom.cd * nom.area)[None, :] * air_vel_b * air_vel_b.abs()
    drag_b = drag_b * prm.drag_scale[:, None]
    drag_b = drag_b - nom.rotor_drag[None, :] * prm.rotor_drag_scale[:, None] * air_vel_b
    force = force + torch.einsum("nij,nj->ni", s.rot, drag_b) + external_force

    tau = torch.stack(
        [
            f @ nom.rotor_xy[:, 1],
            -(f @ nom.rotor_xy[:, 0]),
            prm.kappa * (f @ nom.spin),
        ],
        1,
    )
    j = prm.inertia
    omega_dot = (tau - torch.linalg.cross(s.omega, j * s.omega)) / j

    s.vel = s.vel + force / prm.mass[:, None] * dt
    s.pos = s.pos + s.vel * dt
    s.omega = s.omega + omega_dot * dt
    s.rot = orthonormalize(s.rot @ so3_exp(s.omega * dt))


class RateControllerT:
    """Firmware body-rate loop (port of sim.rate_controller.RateController)."""

    def __init__(self, nom: Nominal, n: int, kp=(25.0, 25.0, 12.0), ki=(0.0, 0.0, 0.0)):
        dev, dt = nom.inertia.device, nom.inertia.dtype
        self.nom = nom
        self.kp = torch.tensor(kp, dtype=dt, device=dev)
        self.ki = torch.tensor(ki, dtype=dt, device=dev)
        self.gain_scale = torch.ones(n, dtype=dt, device=dev)
        self.integral = torch.zeros(n, 3, dtype=dt, device=dev)

    def __call__(self, thrust, rate_cmd, omega, dt):
        err = rate_cmd - omega
        self.integral = torch.clamp(self.integral + err * dt, -1.0, 1.0)
        kp = self.kp[None, :] * self.gain_scale[:, None]
        alpha_des = kp * err + self.ki[None, :] * self.integral
        j = self.nom.inertia
        tau = j * alpha_des + torch.linalg.cross(omega, j.expand_as(omega) * omega)
        wrench = torch.cat([thrust[:, None], tau], 1)
        f = wrench @ self.nom.mixer.T
        return torch.clamp(f, self.nom.thrust_min, self.nom.thrust_max)


def action_to_ctbr(a: torch.Tensor, nom: Nominal):
    a = torch.clamp(a, -1.0, 1.0)
    hover, t_max = nom.weight, nom.max_collective
    thrust = torch.where(a[:, 0] >= 0, hover + a[:, 0] * (t_max - hover), hover * (1 + a[:, 0]))
    return thrust, a[:, 1:4] * nom.body_rate_max[None, :]


# --------------------------------------------------------------------------- geometry
def distance_to_obstacles(p, cyl, cyl_mask, box, box_mask) -> torch.Tensor:
    big = 1e6
    dxy = (p[:, None, :2] - cyl[:, :, :2]).norm(dim=-1) - cyl[:, :, 2]
    dz = p[:, None, 2] - cyl[:, :, 3]
    d_cyl = torch.where(dz > 0, torch.sqrt(torch.clamp(dxy, min=0) ** 2 + dz**2), dxy)
    d_cyl = torch.where(cyl_mask, d_cyl, big).amin(1)
    lo, hi = box[:, :, :3], box[:, :, 3:]
    q = torch.maximum(lo - p[:, None, :], p[:, None, :] - hi)
    outside = torch.clamp(q, min=0.0).norm(dim=-1)
    inside = torch.clamp(q.amax(-1), max=0.0)
    d_box = torch.where(box_mask, outside + inside, big).amin(1)
    d = torch.minimum(d_cyl, d_box)
    d = torch.minimum(d, p[:, 2])
    return torch.minimum(d, ARENA_HEIGHT - p[:, 2])


def out_of_arena(p: torch.Tensor) -> torch.Tensor:
    return (p[:, 0].abs() > ARENA_HALF) | (p[:, 1].abs() > ARENA_HALF) | (p[:, 2] > ARENA_HEIGHT)