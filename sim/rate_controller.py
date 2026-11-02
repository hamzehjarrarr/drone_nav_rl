"""Inner body-rate controller: (collective thrust, body rates) -> per-motor thrusts.

This plays the role of the flight-controller firmware rate loop. It only knows the
NOMINAL drone parameters, like real firmware, so randomized dynamics act as a
genuine model mismatch for the learned policy.
"""

from __future__ import annotations

import numpy as np

from sim.params import QuadParams


class RateController:
    def __init__(
        self,
        params: QuadParams | None = None,
        kp: tuple[float, float, float] = (25.0, 25.0, 12.0),
        ki: tuple[float, float, float] = (0.0, 0.0, 0.0),
        gain_scale: np.ndarray | None = None,
    ) -> None:
        self.p = params or QuadParams()
        self.j = np.asarray(self.p.inertia)
        self.kp = np.asarray(kp)
        self.ki = np.asarray(ki)
        self.mixer = self.p.mixer_matrix()
        self.gain_scale = gain_scale  # (n,) optional per-env gain randomization
        self._integral: np.ndarray | None = None

    def reset(self, n: int, idx: np.ndarray | None = None) -> None:
        if self._integral is None or self._integral.shape[0] != n:
            self._integral = np.zeros((n, 3))
        elif idx is not None:
            self._integral[idx] = 0.0
        else:
            self._integral[:] = 0.0

    def __call__(
        self,
        collective_thrust: np.ndarray,
        rate_cmd: np.ndarray,
        omega: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """collective_thrust (n,), rate_cmd (n,3), omega (n,3) -> motor thrusts (n,4)."""
        n = omega.shape[0]
        if self._integral is None or self._integral.shape[0] != n:
            self.reset(n)
        err = rate_cmd - omega
        self._integral = np.clip(self._integral + err * dt, -1.0, 1.0)
        kp = self.kp[None, :]
        if self.gain_scale is not None:
            kp = kp * self.gain_scale[:, None]
        alpha_des = kp * err + self.ki[None, :] * self._integral
        tau = self.j * alpha_des + np.cross(omega, self.j * omega)
        wrench = np.concatenate([collective_thrust[:, None], tau], axis=1)  # (n, 4)
        f = wrench @ self.mixer.T
        return np.clip(f, self.p.thrust_min, self.p.thrust_max)
