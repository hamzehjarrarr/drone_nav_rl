import numpy as np

from sim.dynamics import QuadState, ground_effect_factor, step_dynamics
from sim.math3d import orthonormalize, rot_to_quat, so3_exp
from sim.params import GRAVITY, QuadParams, sample_params
from sim.rate_controller import RateController

DT = 0.004


def _hover_state(n=1):
    p = QuadParams()
    s = QuadState.zeros(n)
    s.pos[:, 2] = 2.0
    s.motor_thrust[:] = p.hover_thrust_per_motor
    return p, s


def test_equal_hover_thrust_holds_position():
    p, s = _hover_state(3)
    pb = sample_params(3, np.random.default_rng(0), randomize=False)
    for _ in range(500):  # 2 s
        step_dynamics(s, np.full((3, 4), p.hover_thrust_per_motor), pb, DT)
    assert np.allclose(s.pos, [[0, 0, 2.0]] * 3, atol=1e-6)
    assert np.allclose(s.vel, 0.0, atol=1e-6)


def test_free_fall_without_thrust():
    _, s = _hover_state()
    s.motor_thrust[:] = 0.0
    pb = sample_params(1, np.random.default_rng(0), randomize=False)
    pb.drag_scale[:] = 0.0  # vacuum: compare against the textbook free fall
    pb.rotor_drag_scale[:] = 0.0
    for _ in range(125):  # 0.5 s
        step_dynamics(s, np.zeros((1, 4)), pb, DT)
    # semi-implicit Euler: v = -g t exactly; z close to 2 - g t^2 / 2
    assert np.isclose(s.vel[0, 2], -GRAVITY * 0.5, atol=1e-6)
    assert abs(s.pos[0, 2] - (2.0 - 0.5 * GRAVITY * 0.25)) < 0.02


def test_motor_lag_time_constant():
    p, s = _hover_state()
    pb = sample_params(1, np.random.default_rng(0), randomize=False)
    s.motor_thrust[:] = 0.0
    steps = int(round(p.motor_tau / DT))
    for _ in range(steps):
        step_dynamics(s, np.full((1, 4), 4.0), pb, DT)
    # after one time constant a first-order system reaches ~63 %
    assert 0.58 < s.motor_thrust[0, 0] / 4.0 < 0.68


def test_rate_controller_tracks_roll_and_yaw():
    p, s = _hover_state()
    pb = sample_params(1, np.random.default_rng(0), randomize=False)
    rc = RateController(p)
    cmd = np.array([[3.0, 0.0, 1.5]])
    for _ in range(150):  # 0.6 s
        f = rc(np.array([p.weight]), cmd, s.omega, DT)
        step_dynamics(s, f, pb, DT)
    assert np.allclose(s.omega, cmd, atol=0.15)


def test_rotation_stays_orthonormal():
    r = so3_exp(np.random.default_rng(1).normal(size=(10, 3)))
    for _ in range(100):
        r = orthonormalize(r @ so3_exp(np.full((10, 3), 0.05)))
    eye = np.einsum("nji,njk->nik", r, r)
    assert np.allclose(eye, np.eye(3), atol=1e-9)
    assert np.allclose(np.linalg.det(r), 1.0)


def test_quaternion_roundtrip_unit():
    r = so3_exp(np.random.default_rng(2).normal(size=(20, 3)))
    q = rot_to_quat(r)
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0)
    # rebuild matrix from quaternion and compare
    x, y, z, w = q.T
    rr = np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )
    assert np.allclose(rr, r, atol=1e-8)


def test_rotor_drag_slows_a_gliding_drone():
    p, s = _hover_state()
    pb = sample_params(1, np.random.default_rng(0), randomize=False)
    s.vel[0] = [4.0, 0.0, 0.0]
    step_dynamics(s, np.full((1, 4), p.hover_thrust_per_motor), pb, DT)
    acc = (s.vel[0, 0] - 4.0) / DT
    rotor = -p.rotor_drag[0] * 4.0 / p.mass
    body = -0.5 * 1.204 * p.drag_coeff[0] * p.frontal_area[0] * 16.0 / p.mass
    assert np.isclose(acc, rotor + body, rtol=1e-6)
    assert abs(rotor) > 5 * abs(body)  # at 4 m/s rotor drag dominates


def test_ground_effect_adds_lift_only_near_ground():
    p = sample_params(1, np.random.default_rng(0), randomize=False).nominal
    r = p.rotor_radius
    near = ground_effect_factor(np.array([1.0 * r]), r, np.ones(1))[0]
    mid = ground_effect_factor(np.array([2.0 * r]), r, np.ones(1))[0]
    far = ground_effect_factor(np.array([5.0 * r, 1.0, 3.0]), r, np.ones(3))
    assert near > mid > 1.0
    assert 1.03 < near < 1.10
    assert np.all(far == 1.0)
    assert ground_effect_factor(np.array([r]), r, np.zeros(1))[0] == 1.0