"""Quadrotor parameters, loaded from a drone profile (a small TOML file).

The default profile is the Agilicious "Kingfisher" (sim/profiles/kingfisher.toml). To train
for another drone, copy that file, fill in the drone's measured values, and pass it with
`python -m rl.train --profile path/to/drone.toml`.

Conventions used everywhere in the simulator:
  * World frame: x forward/east, y left/north, z up (gravity along -z).
  * Body frame: FLU -> x forward, y left, z up (thrust along +z body).
  * Rotor order: [front-right, back-left, back-right, front-left] (same as Agilicious).
  * Spin signs: rotors 0 and 1 share one direction, rotors 2 and 3 the other.
    Yaw torque = kappa * sum(spin_sign[i] * thrust[i]).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

GRAVITY = 9.81  # m/s^2
AIR_DENSITY = 1.204  # kg/m^3
PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
DEFAULT_PROFILE = "kingfisher"


@dataclass(frozen=True)
class QuadParams:
    """Nominal parameters of one quadrotor (defaults = Agilicious Kingfisher)."""

    name: str = "Agilicious Kingfisher"
    mass: float = 0.752  # kg
    inertia: tuple[float, float, float] = (0.0025, 0.0021, 0.0043)  # kg m^2 (diagonal)

    # Rotor positions in the body frame (x, y), metres. Order: FR, BL, BR, FL.
    rotor_xy: tuple[tuple[float, float], ...] = (
        (0.075, -0.10),
        (-0.075, 0.10),
        (-0.075, -0.10),
        (0.075, 0.10),
    )
    spin_sign: tuple[float, float, float, float] = (1.0, 1.0, -1.0, -1.0)
    rotor_radius: float = 0.0648  # m

    # Motors / propellers
    thrust_coeff: float = 1.562522e-6  # N / (rad/s)^2  (thrust = k * omega^2)
    kappa: float = 0.022  # Nm / N  (rotor drag torque per unit thrust)
    motor_tau: float = 0.033  # s, first-order motor time constant
    motor_omega_min: float = 150.0  # rad/s
    motor_omega_max: float = 2800.0  # rad/s
    thrust_min: float = 0.0  # N per motor
    thrust_max: float = 8.5  # N per motor

    # Body-rate limits the policy may command (roll, pitch, yaw), rad/s
    body_rate_max: tuple[float, float, float] = (10.0, 10.0, 4.0)

    # Aerodynamics
    frontal_area: tuple[float, float, float] = (1.5e-2, 1.5e-2, 3.0e-2)  # m^2
    drag_coeff: tuple[float, float] = (1.04, 1.04)  # quadratic body drag (horizontal, vertical)
    rotor_drag: tuple[float, float, float] = (0.26, 0.28, 0.42)  # N / (m/s), linear, body axes
    ground_effect: bool = True

    # Collision radius used for obstacle checks (rotor tip span), metres
    collision_radius: float = 0.18

    # ---- derived quantities -------------------------------------------------
    @property
    def weight(self) -> float:
        return self.mass * GRAVITY

    @property
    def hover_thrust_per_motor(self) -> float:
        return self.weight / 4.0

    @property
    def max_collective_thrust(self) -> float:
        return 4.0 * self.thrust_max

    @property
    def thrust_to_weight(self) -> float:
        return self.max_collective_thrust / self.weight

    @property
    def hover_omega(self) -> float:
        return float(np.sqrt(self.hover_thrust_per_motor / self.thrust_coeff))

    def allocation_matrix(self) -> np.ndarray:
        """Maps per-motor thrusts f (4,) to [collective thrust, tau_x, tau_y, tau_z]."""
        xy = np.asarray(self.rotor_xy)
        s = np.asarray(self.spin_sign)
        return np.stack(
            [
                np.ones(4),  # collective thrust
                xy[:, 1],  # tau_x = y * f
                -xy[:, 0],  # tau_y = -x * f
                self.kappa * s,  # tau_z
            ]
        )

    def mixer_matrix(self) -> np.ndarray:
        """Inverse of the allocation matrix: [T, tau] -> per-motor thrusts."""
        return np.linalg.inv(self.allocation_matrix())


@dataclass(frozen=True)
class RandomizationRanges:
    """Relative (±fraction) ranges for domain randomization around the nominal values."""

    mass: float = 0.10
    inertia: float = 0.15
    thrust_coeff: float = 0.10
    kappa: float = 0.10
    motor_tau: float = 0.20
    drag: float = 0.30
    rotor_drag: float = 0.30
    per_motor_thrust: float = 0.05  # independent mismatch between the four motors
    ground_effect: float = 0.50


def _resolve(profile: str | Path) -> Path:
    p = Path(profile)
    if p.suffix == ".toml" and p.exists():
        return p
    candidate = PROFILE_DIR / f"{profile}.toml"
    if candidate.exists():
        return candidate
    names = sorted(x.stem for x in PROFILE_DIR.glob("*.toml"))
    raise FileNotFoundError(f"drone profile {profile!r} not found; built-in profiles: {names}")


def load_profile(profile: str | Path = DEFAULT_PROFILE) -> tuple[QuadParams, RandomizationRanges]:
    """Read a drone profile (built-in name such as 'kingfisher', or a path to a .toml file)."""
    with _resolve(profile).open("rb") as fh:
        d = tomllib.load(fh)
    body, rot, mot, ctl, aero = d["body"], d["rotors"], d["motors"], d["control"], d["aero"]
    params = QuadParams(
        name=str(d.get("name", Path(profile).stem)),
        mass=float(body["mass"]),
        inertia=tuple(float(x) for x in body["inertia"]),
        collision_radius=float(body["collision_radius"]),
        rotor_xy=tuple((float(x), float(y)) for x, y in rot["positions_xy"]),
        spin_sign=tuple(float(s) for s in rot["spin_sign"]),
        rotor_radius=float(rot["radius"]),
        thrust_coeff=float(mot["thrust_coeff"]),
        kappa=float(mot["kappa"]),
        motor_tau=float(mot["time_constant"]),
        motor_omega_min=float(mot["omega_min"]),
        motor_omega_max=float(mot["omega_max"]),
        thrust_min=float(mot["thrust_min"]),
        thrust_max=float(mot["thrust_max"]),
        body_rate_max=tuple(float(x) for x in ctl["body_rate_max"]),
        frontal_area=tuple(float(x) for x in aero["frontal_area"]),
        drag_coeff=tuple(float(x) for x in aero["drag_coeff"]),
        rotor_drag=tuple(float(x) for x in aero["rotor_drag"]),
        ground_effect=bool(aero.get("ground_effect", True)),
    )
    ranges = RandomizationRanges(**{k: float(v) for k, v in d.get("randomization", {}).items()})
    return params, ranges


@dataclass
class ParamBatch:
    """Per-environment parameter arrays for a vectorized simulator (n environments)."""

    mass: np.ndarray  # (n,)
    inertia: np.ndarray  # (n, 3)
    thrust_coeff: np.ndarray  # (n,)
    motor_gain: np.ndarray  # (n, 4) multiplicative per-motor thrust mismatch
    kappa: np.ndarray  # (n,)
    motor_tau: np.ndarray  # (n,)
    drag_scale: np.ndarray  # (n,) quadratic body drag multiplier
    rotor_drag_scale: np.ndarray  # (n,) linear rotor drag multiplier
    ground_effect_scale: np.ndarray  # (n,) 0 disables ground effect
    nominal: QuadParams = field(default_factory=QuadParams)

    @property
    def n(self) -> int:
        return self.mass.shape[0]


RANDOMIZED_FIELDS = (
    "mass",
    "inertia",
    "thrust_coeff",
    "motor_gain",
    "kappa",
    "motor_tau",
    "drag_scale",
    "rotor_drag_scale",
    "ground_effect_scale",
)


def sample_params(
    n: int,
    rng: np.random.Generator,
    nominal: QuadParams | None = None,
    ranges: RandomizationRanges | None = None,
    randomize: bool = True,
) -> ParamBatch:
    """Draw n parameter sets. With randomize=False every env gets the nominal values."""
    nominal = nominal or QuadParams()
    ranges = ranges or RandomizationRanges()

    def scale(frac: float, shape: tuple[int, ...]) -> np.ndarray:
        if not randomize or frac == 0.0:
            return np.ones(shape)
        return rng.uniform(1.0 - frac, 1.0 + frac, size=shape)

    ge_on = 1.0 if nominal.ground_effect else 0.0
    return ParamBatch(
        mass=nominal.mass * scale(ranges.mass, (n,)),
        inertia=np.asarray(nominal.inertia)[None, :] * scale(ranges.inertia, (n, 3)),
        thrust_coeff=nominal.thrust_coeff * scale(ranges.thrust_coeff, (n,)),
        motor_gain=scale(ranges.per_motor_thrust, (n, 4)),
        kappa=nominal.kappa * scale(ranges.kappa, (n,)),
        motor_tau=nominal.motor_tau * scale(ranges.motor_tau, (n,)),
        drag_scale=scale(ranges.drag, (n,)),
        rotor_drag_scale=scale(ranges.rotor_drag, (n,)),
        ground_effect_scale=ge_on * scale(ranges.ground_effect, (n,)),
        nominal=nominal,
    )