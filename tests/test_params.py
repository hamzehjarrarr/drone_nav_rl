import numpy as np
import pytest

from sim.params import GRAVITY, QuadParams, RandomizationRanges, load_profile, sample_params


def test_hover_thrust_and_thrust_to_weight():
    p = QuadParams()
    assert np.isclose(p.hover_thrust_per_motor * 4, p.mass * GRAVITY)
    # Kingfisher: 4 x 8.5 N against ~7.38 N weight -> roughly 4.6
    assert 4.0 < p.thrust_to_weight < 5.5
    # hover rotor speed must sit inside the motor's range
    assert p.motor_omega_min < p.hover_omega < p.motor_omega_max


def test_mixer_inverts_allocation():
    p = QuadParams()
    a, m = p.allocation_matrix(), p.mixer_matrix()
    assert np.allclose(a @ m, np.eye(4))


def test_pure_collective_thrust_gives_equal_motors():
    p = QuadParams()
    f = p.mixer_matrix() @ np.array([p.weight, 0.0, 0.0, 0.0])
    assert np.allclose(f, p.hover_thrust_per_motor)


def test_roll_torque_direction():
    # positive tau_x (rotation about +x, FLU) lifts the left side,
    # so the left rotors (y > 0) must produce more thrust than the right rotors
    p = QuadParams()
    f = p.mixer_matrix() @ np.array([p.weight, 0.05, 0.0, 0.0])
    left = f[[1, 3]].sum()
    right = f[[0, 2]].sum()
    assert left > right


def test_sampling_shapes_and_nominal_mode():
    rng = np.random.default_rng(0)
    b = sample_params(8, rng)
    assert b.n == 8
    assert b.inertia.shape == (8, 3)
    assert b.motor_gain.shape == (8, 4)
    assert np.all(np.abs(b.mass / QuadParams().mass - 1.0) <= 0.10 + 1e-12)

    nom = sample_params(4, rng, randomize=False)
    assert np.allclose(nom.mass, QuadParams().mass)
    assert np.allclose(nom.motor_gain, 1.0)


def test_builtin_profile_matches_defaults(tmp_path):
    params, ranges = load_profile("kingfisher")
    assert params == QuadParams()
    assert ranges == RandomizationRanges()
    # a custom profile file overrides values
    text = (load_profile.__globals__["PROFILE_DIR"] / "kingfisher.toml").read_text()
    custom = tmp_path / "heavy.toml"
    custom.write_text(text.replace("mass = 0.752", "mass = 1.2"))
    heavy, _ = load_profile(custom)
    assert heavy.mass == 1.2 and heavy.thrust_max == params.thrust_max


def test_unknown_profile_lists_options():
    with pytest.raises(FileNotFoundError, match="kingfisher"):
        load_profile("no_such_drone")


def test_new_randomized_fields():
    b = sample_params(16, np.random.default_rng(1))
    assert b.rotor_drag_scale.shape == (16,) and b.ground_effect_scale.shape == (16,)
    assert np.all(np.abs(b.rotor_drag_scale - 1.0) <= 0.3 + 1e-12)
    nom = sample_params(4, np.random.default_rng(1), randomize=False)
    assert np.allclose(nom.rotor_drag_scale, 1.0) and np.allclose(nom.ground_effect_scale, 1.0)