import numpy as np

from sim.math3d import so3_exp
from sim.sensors import RayConfig, camera_fov_mask, ray_directions_body, sector_scan, sense
from sim.world import (
    ARENA_HEIGHT,
    MAX_BOX,
    MAX_CYL,
    WorldBatch,
    WorldSpec,
    distance_to_obstacles,
    generate_world,
)

CFG = RayConfig()


def _world(n=1):
    return WorldBatch.empty(n)


def _scan(w, pos, yaw=0.0, max_range=5.0):
    return sector_scan(
        np.asarray([pos], float),
        np.array([yaw]),
        w.cyl,
        w.cyl_mask,
        w.box,
        w.box_mask,
        CFG,
        max_range,
    )[0]


def _sector(az_deg, el_deg):
    """Index of the sector containing a direction (azimuth, elevation in degrees)."""
    _, az, el = ray_directions_body(CFG)
    k = np.argmin(
        np.abs(np.angle(np.exp(1j * (az - np.deg2rad(az_deg))))) + np.abs(el - np.deg2rad(el_deg))
    )
    return k


def test_layout_has_fine_front_and_full_coverage():
    _, az, el = ray_directions_body(CFG)
    assert CFG.n_rays == 64 and len(az) == 64
    edges = np.asarray(CFG.az_edges_deg)
    widths = np.mod(np.roll(edges, -1) - edges, 360.0)
    assert np.isclose(widths.sum(), 360.0)
    assert np.allclose(widths[:8], 11.25)  # front +-45 deg in fine bins


def test_tree_straight_ahead_distance():
    w = _world()
    w.cyl[0, 0] = [3.0, 0.0, 0.5, 6.0]
    w.cyl_mask[0, 0] = True
    d = _scan(w, [0.0, 0.0, 2.0])
    k = _sector(0.0 - 5.0, 7.0)  # bin just right of the nose, band above horizontal
    assert np.isclose(d[k], 2.5, atol=1e-6)


def test_thin_tree_between_old_ray_directions_is_seen():
    # a 0.3 m trunk 3 m away at a bearing that 16 evenly spaced rays would miss
    w = _world()
    bearing = np.deg2rad(11.0)
    w.cyl[0, 0] = [3.15 * np.cos(bearing), 3.15 * np.sin(bearing), 0.15, 6.0]
    w.cyl_mask[0, 0] = True
    d = _scan(w, [0.0, 0.0, 2.0])
    assert d.min() < 3.05


def test_exact_against_fine_ray_sampling():
    rng = np.random.default_rng(0)
    w = _world()
    wl = generate_world(WorldSpec(n_trees=30, n_walls=3, n_waypoints=2), rng)
    wl.cyl[:, 3] = 7.5  # tall prisms: the scan is exact for these
    wl.box[:, 5] = 7.5
    w.set(0, wl)
    pos = wl.waypoints[0]
    d = _scan(w, pos, yaw=0.3)
    # every sector distance must be achieved by some point of an obstacle (or ground):
    # sample many 2D rays per azimuth bin through the middle band and compare
    edges = np.deg2rad(np.asarray(CFG.az_edges_deg))
    widths = np.mod(np.roll(edges, -1) - edges, 2 * np.pi)
    band = 2  # elevation 0..14.5 deg: no ground, obstacles only
    for j in range(len(edges)):
        best = 5.0
        for a in edges[j] + 0.3 + np.linspace(0, widths[j], 200):
            u = np.array([np.cos(a), np.sin(a)])
            for cx, cy, r, _ in wl.cyl[wl.cyl_mask]:
                v = np.array([cx, cy]) - pos[:2]
                disc = (u @ v) ** 2 - (v @ v - r * r)
                if disc >= 0 and u @ v - np.sqrt(disc) > 0:
                    best = min(best, u @ v - np.sqrt(disc))
        ours = d[band * len(edges) + j]
        assert ours <= best + 1e-6  # never misses anything a ray would see


def test_box_and_ground_bands():
    w = _world()
    w.box[0, 0] = [2.0, -1.0, 0.0, 3.0, 1.0, 6.0]
    w.box_mask[0, 0] = True
    d = _scan(w, [0.0, 0.0, 2.0])
    assert np.isclose(d[_sector(-5.0, 7.0)], 2.0, atol=1e-6)
    # lowest band (-29..-14.5 deg) sees the ground at 2 / tan(29 deg) = 3.6 m behind
    assert np.isclose(d[_sector(180.0, -21.0)], 2.0 / np.tan(np.deg2rad(29.0)), atol=1e-6)


def test_scan_turns_with_heading():
    w = _world()
    w.cyl[0, 0] = [0.0, 3.0, 0.5, 6.0]  # to the world's +y (left)
    w.cyl_mask[0, 0] = True
    ahead = _sector(-5.0, 7.0)
    assert _scan(w, [0, 0, 2.0], yaw=0.0)[ahead] == 5.0
    assert np.isclose(_scan(w, [0, 0, 2.0], yaw=np.pi / 2)[ahead], 2.5, atol=1e-6)


def test_camera_mode_masks_rear_sectors():
    fov = camera_fov_mask(CFG)
    assert fov.sum() == 32  # 8 front bins x 4 bands inside 85 x 58 degrees
    wb = _world(2)
    rng = np.random.default_rng(0)
    rot = np.stack([np.eye(3), np.eye(3)])
    pos = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
    prox, mask, _ = sense(rot, pos, wb, CFG, np.array([0.0, 1.0]), rng)
    assert mask[0].all()
    assert np.array_equal(mask[1], fov)
    assert np.all(prox[1][~fov] == 0.0)
    assert np.all((prox >= 0) & (prox <= 1))


def test_distance_to_obstacles_signs():
    cyl = np.zeros((2, MAX_CYL, 4))
    cm = np.zeros((2, MAX_CYL), bool)
    cyl[:, 0] = [0.0, 0.0, 1.0, 5.0]
    cm[:, 0] = True
    box = np.zeros((2, MAX_BOX, 6))
    bm = np.zeros((2, MAX_BOX), bool)
    p = np.array([[3.0, 0.0, 2.0], [0.5, 0.0, 2.0]])
    d = distance_to_obstacles(p, cyl, cm, box, bm)
    assert np.isclose(d[0], 2.0)
    assert d[1] < 0


def test_generated_routes_are_clear_and_inside_arena():
    rng = np.random.default_rng(3)
    for spec in (
        WorldSpec(n_trees=25, n_walls=3, n_waypoints=4),
        WorldSpec(corridor=True, n_waypoints=3),
        WorldSpec(hover_only=True),
    ):
        w = generate_world(spec, rng)
        pts = w.waypoints[: w.n_waypoints]
        d = distance_to_obstacles(
            pts, *[np.repeat(a[None], len(pts), 0) for a in (w.cyl, w.cyl_mask, w.box, w.box_mask)]
        )
        assert np.all(d > 0.5)
        assert np.all((pts[:, 2] >= 0.9) & (pts[:, 2] <= ARENA_HEIGHT))


def test_body_fixed_camera_tilts_with_the_drone():
    # pitching the nose down makes the camera see the ground closer in its upper bands
    wb = _world(2)
    rng = np.random.default_rng(0)
    level = np.eye(3)
    nose_down = so3_exp(np.array([0.0, np.deg2rad(20.0), 0.0]))  # +pitch about y: nose down
    rot = np.stack([level, nose_down])
    pos = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
    cfg = RayConfig(real_noise_k=0.0, real_dropout=0.0)
    _, _, d = sense(rot, pos, wb, cfg, np.ones(2), rng)
    ahead_low = _sector(-5.0, -7.0)  # front bin, band just below horizontal
    assert d[0, ahead_low] == 3.0  # level camera: ground beyond the 3 m range
    assert d[1, ahead_low] < 3.0  # pitched camera: ground now inside the range
    # ideal mode stays level whatever the attitude
    _, _, d_ideal = sense(rot, pos, wb, cfg, np.zeros(2), rng)
    assert np.allclose(d_ideal[0], d_ideal[1])