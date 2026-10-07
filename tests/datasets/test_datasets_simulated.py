"""L1 simulators: known-result tests (closed forms, hand geometry, spec spot values) and contracts."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from indoorloc.core import SampleTable
from indoorloc.datasets import dataset_info, list_datasets, load_dataset
from indoorloc.datasets.simulated import building as bld
from indoorloc.datasets.simulated import channel as ch
from indoorloc.datasets.simulated import deepmimo as dmx
from indoorloc.datasets.simulated import geometry as geo
from indoorloc.datasets.simulated import measurements as ms
from indoorloc.datasets.simulated import propagation as pr
from indoorloc.datasets.simulated.office import SyntheticOffice

C = 299_792_458.0
SMALL = dict(size=(16.0, 10.0), n_test=40, n_trajectories=2, trajectory_duration=20.0, grid_spacing=2.0,
             samples_per_point=2)


# --------------------------------------------------------------------------------- propagation
def test_friis_known_values():
    assert pr.free_space_path_loss(1.0, 2.4e9) == pytest.approx(40.05, abs=0.005)
    assert pr.free_space_path_loss(10.0, 2.4e9) == pytest.approx(60.05, abs=0.005)       # +20 dB per decade
    assert pr.free_space_path_loss(2.0, 5.0e9) - pr.free_space_path_loss(1.0, 5.0e9) == pytest.approx(6.0206, abs=1e-4)
    # log-distance with n = 2 is Friis; n = 3 adds 30 dB per decade
    assert pr.log_distance_path_loss(7.0, 2.4e9, exponent=2.0) == pytest.approx(pr.free_space_path_loss(7.0, 2.4e9))
    assert pr.log_distance_path_loss(10.0, 2.4e9, exponent=3.0) == pytest.approx(40.052 + 30.0, abs=1e-3)
    shadowed = pr.log_distance_path_loss(np.full(20000, 10.0), 2.4e9, exponent=3.0, shadowing_std_db=6.0,
                                         random_state=0)
    assert shadowed.mean() == pytest.approx(70.052, abs=0.15) and shadowed.std() == pytest.approx(6.0, abs=0.1)
    np.testing.assert_array_equal(shadowed, pr.log_distance_path_loss(np.full(20000, 10.0), 2.4e9, exponent=3.0,
                                                                      shadowing_std_db=6.0, random_state=0))


def test_3gpp_inh_office_spot_values():
    # TR 38.901 Table 7.4.1-1, evaluated by hand (fc in GHz, d3D in m)
    assert pr.inh_office_path_loss(10.0, 3.5e9, True) == pytest.approx(60.5814, abs=1e-4)   # 32.4 + 17.3 + 20 log10 3.5
    assert pr.inh_office_path_loss(10.0, 3.5e9, False) == pytest.approx(69.1473, abs=1e-4)  # 17.3+38.3+24.9 log10 3.5
    assert pr.inh_office_path_loss(10.0, 3.5e9, False, optional_nlos=True) == pytest.approx(75.1814, abs=1e-4)
    # at 1 m and 28 GHz the max(PL_LOS, PL'_NLOS) clause is active: NLOS = LOS = 61.3432 dB
    assert pr.inh_office_path_loss(1.0, 28e9, False) == pytest.approx(61.3432, abs=1e-4)
    assert pr.inh_office_path_loss(0.2, 3.5e9, True) == pr.inh_office_path_loss(1.0, 3.5e9, True)  # clamped at 1 m
    np.testing.assert_allclose(pr.inh_office_path_loss([10.0, 10.0], 3.5e9, [True, False]), [60.5814, 69.1473],
                               atol=1e-4)
    # Table 7.4.2-1 LOS probability
    np.testing.assert_allclose(pr.inh_office_los_probability([1.0, 4.0, 10.0]), [1.0, 0.551152, 0.287424], atol=1e-6)
    np.testing.assert_allclose(pr.inh_office_los_probability([3.0, 20.0, 60.0], variant="open"),
                               [1.0, 0.809074, 0.512658], atol=1e-6)
    assert pr.INH_OFFICE_SHADOW_STD_DB == {"los": 3.0, "nlos": 8.03, "nlos_optional": 8.29}


def test_cost231_multi_wall_model():
    fs = pr.free_space_path_loss(10.0, 1.8e9)
    # 2 light + 1 heavy walls, one floor: L_FS + 2 * 3.4 + 6.9 + 18.3 (COST 231 values at 1.8 GHz)
    assert pr.multi_wall_path_loss(10.0, 1.8e9, [2, 1], 1) == pytest.approx(fs + 6.8 + 6.9 + 18.3)
    # two floors: 2 ** ((2 + 2) / (2 + 1) - 0.46) * 18.3 = 33.524 dB
    assert pr.multi_wall_path_loss(10.0, 1.8e9, [0, 0], 2) - fs == pytest.approx(33.5236, abs=1e-4)
    assert pr.multi_wall_path_loss(10.0, 1.8e9, [0, 0], 0) == pytest.approx(fs)
    assert pr.multi_wall_path_loss(10.0, 1.8e9, 3, wall_loss_db=5.0) == pytest.approx(fs + 15.0)  # one wall type
    with pytest.raises(ValueError):
        pr.multi_wall_path_loss(10.0, 1.8e9, [1, 2, 3])


def test_sensitivity_and_noise_floor():
    x = np.array([[-50.0, -96.0, np.nan]], dtype=np.float32)
    out = pr.apply_sensitivity(x, -95.0)
    assert out.dtype == np.float32 and out[0, 0] == -50.0 and np.isnan(out[0, 1:]).all()
    assert pr.thermal_noise_dbm(20e6, noise_figure_db=0.0) == pytest.approx(-100.965, abs=1e-3)  # kT0 = -173.975 dBm/Hz


def test_correlated_shadowing_follows_gudmundson():
    dc, h = 3.0, 0.5
    fields = pr.correlated_gaussian_field((40, 40), h, dc, n_fields=120, random_state=0)
    assert fields.shape == (120, 40, 40)
    assert fields.var() == pytest.approx(1.0, abs=0.03)
    for lag in (1, 4, 8):
        rho_x = np.mean(fields[:, :, :-lag] * fields[:, :, lag:])
        rho_y = np.mean(fields[:, :-lag, :] * fields[:, lag:, :])
        target = np.exp(-lag * h / dc)
        assert rho_x == pytest.approx(target, abs=0.03) and rho_y == pytest.approx(target, abs=0.03)
    # diagonal lag: the correlation is isotropic (Euclidean distance)
    rho_d = np.mean(fields[:, :-4, :-4] * fields[:, 4:, 4:])
    assert rho_d == pytest.approx(np.exp(-4 * np.sqrt(2) * h / dc), abs=0.03)
    again = pr.correlated_gaussian_field((40, 40), h, dc, n_fields=120, random_state=0)
    np.testing.assert_array_equal(fields, again)


def test_shadowing_map_interpolates_the_grid():
    (smap,) = pr.ShadowingMap.generate((0, 0, 10, 5), std_db=4.0, decorrelation_distance=3.0, random_state=1)
    f, h = smap.field, smap.spacing
    nodes = np.array([[0.0, 0.0], [2 * h, 3 * h]])
    np.testing.assert_allclose(smap(nodes), 4.0 * np.array([f[0, 0], f[3, 2]]))
    mid = smap(np.array([[0.5 * h, 0.0]]))
    assert mid[0] == pytest.approx(4.0 * (f[0, 0] + f[0, 1]) / 2)


# ------------------------------------------------------------------------------------ geometry
WALLS = np.array([[1.0, 0.0, 1.0, 2.0],    # vertical at x = 1, y in [0, 2]
                  [2.0, 0.0, 2.0, 2.0],    # vertical at x = 2
                  [0.0, 3.0, 5.0, 3.0]])   # horizontal at y = 3


def test_wall_crossing_counts_on_hand_made_geometry():
    p = np.array([[0.0, 1.0], [0.0, 1.0], [1.5, 2.5], [0.0, 2.5], [1.0, -1.0], [0.0, 0.5]])
    q = np.array([[3.0, 1.0], [0.5, 1.0], [1.5, 4.0], [3.0, 2.5], [1.0, 5.0], [3.0, 3.5]])
    np.testing.assert_array_equal(geo.count_crossings(p, q, WALLS), [2, 0, 1, 0, 1, 2])
    # row 4 runs along x = 1 (grazing is not a crossing) and crosses y = 3; row 5 passes above the
    # end of the x = 2 wall (it is at y = 2.5 there)
    t, u = geo.intersection_params(p[:1], q[:1], WALLS)
    np.testing.assert_allclose(t[0, :2], [1 / 3, 2 / 3])
    np.testing.assert_allclose(u[0, :2], [0.5, 0.5])
    assert np.isnan(t[0, 2])
    np.testing.assert_allclose(geo.count_crossings(p, q, WALLS, weights=[3.0, 5.0, 7.0]), [8, 0, 7, 0, 7, 10])
    np.testing.assert_array_equal(geo.count_crossings(p[:1], q[:1], WALLS, exclude=[0]), [1])


def test_distances_and_mirror_images():
    np.testing.assert_allclose(geo.point_segment_distance([[0.0, 1.0], [1.0, 3.0]], WALLS[:1]), [[1.0], [1.0]])
    np.testing.assert_allclose(geo.mirror_points([[1.0, 2.0]], [[-5.0, 0.0, 5.0, 0.0]])[0, 0], [1.0, -2.0])
    np.testing.assert_allclose(geo.mirror_points([[0.0, 0.0]], [[1.0, 0.0, 1.0, 1.0]])[0, 0], [2.0, 0.0])
    d = geo.segment_segment_distance([[0.0, 1.0], [3.0, 0.0]], [[0.5, 1.0], [3.0, 5.0]], WALLS)
    np.testing.assert_allclose(d[0], [0.5, 1.5, 2.0])
    np.testing.assert_allclose(d[1], [2.0, 1.0, 0.0])
    assert geo.wrap_angle(np.pi) == pytest.approx(-np.pi) and geo.wrap_angle(3 * np.pi / 2) == pytest.approx(-np.pi / 2)


def test_floor_plan_counts_walls_per_storey_along_the_3d_ray():
    walls = np.array([[5.0, -5.0, 5.0, 5.0]] * 3)
    plan = bld.FloorPlan(walls=walls, wall_floor=[0, 1, 2], wall_type=[0, 1, 1], bounds=(0, -5, 10, 5), n_floors=3,
                         floor_height=3.5)
    a = np.array([[0.0, 0.0, 1.2]] * 3)
    b = np.array([[10.0, 0.0, 1.2], [10.0, 0.0, 4.7], [10.0, 0.0, 8.2]])
    # same storey: the ground-floor light wall; to storey 1 the ray passes x = 5 at z = 2.95 (storey 0);
    # to storey 2 it passes x = 5 at z = 4.7 (storey 1, a heavy wall)
    np.testing.assert_array_equal(plan.wall_crossings(a, b), [[1, 0], [1, 0], [0, 1]])
    np.testing.assert_array_equal(plan.floor_crossings(a, b), [0, 1, 2])
    np.testing.assert_allclose(plan.crossed_walls(a, b) @ np.array([5.0, 7.0, 11.0]), [5.0, 5.0, 7.0])  # per wall
    np.testing.assert_array_equal(plan.nlos_mask(a, np.array([[4.0, 0.0, 1.2]] * 3)), [False] * 3)
    np.testing.assert_array_equal(plan.floor_of([0.0, 3.49, 3.5, 99.0, -1.0]), [0, 0, 1, 2, 0])
    back = bld.FloorPlan.from_meta(plan.to_meta())
    np.testing.assert_array_equal(back.walls, plan.walls)
    np.testing.assert_array_equal(back.wall_type, plan.wall_type)


def test_office_floor_plan_layout():
    plan = bld.office_floor_plan(size=(40.0, 20.0), n_floors=2, room_width=4.0, random_state=0)
    n = 10  # rooms per side
    assert len(plan.walls_on(0)) == 4 + 4 * n + 2 * (n - 1)
    assert plan.wall_type[plan.wall_floor == 0].sum() == 4                # only the outer walls are heavy
    areas = (plan.rooms[:, 2] - plan.rooms[:, 0]) * (plan.rooms[:, 3] - plan.rooms[:, 1])
    assert areas[plan.room_floor == 0].sum() == pytest.approx(40.0 * 20.0)  # rooms + corridor tile the floor
    assert (plan.room_kind == "corridor").sum() == 2
    xy, fl = bld.random_points(plan, 300, random_state=1)
    assert (plan.room_of(xy, fl) >= 0).all() and plan.is_free(xy, fl, 0.3).all()
    # from outside through the outer wall into a room: exactly one heavy wall
    np.testing.assert_array_equal(plan.wall_crossings([[-1.0, 2.0, 1.0]], [[2.0, 2.0, 1.0]]), [[0, 1]])
    # door positions are random but reproducible
    again = bld.office_floor_plan(size=(40.0, 20.0), n_floors=2, room_width=4.0, random_state=0)
    np.testing.assert_array_equal(plan.walls, again.walls)


def test_anchor_layouts_and_reference_grid():
    plan = bld.office_floor_plan(size=(20.0, 10.0), n_floors=2, random_state=0)
    xyz, fl = bld.place_anchors(plan, 4, layout="perimeter", height=(2.6, 0.6), margin=0.5)
    np.testing.assert_allclose(xyz[:4, :2], [[0.5, 0.5], [19.5, 9.5], [19.5, 0.5], [0.5, 9.5]])
    np.testing.assert_allclose(xyz[:4, 2], [2.6, 0.6, 2.6, 0.6])
    np.testing.assert_allclose(xyz[4:, 2], np.array([2.6, 0.6, 2.6, 0.6]) + 3.5)
    np.testing.assert_array_equal(fl, [0] * 4 + [1] * 4)
    g_xyz, _ = bld.place_anchors(plan, 6, layout="grid", random_state=3)
    assert len(g_xyz) == 12 and plan.is_free(g_xyz[:6, :2], 0, 0.3).all()
    orient = bld.facing_centre(plan, xyz[:4, :2])
    np.testing.assert_allclose(orient[0], np.arctan2(4.5, 9.5))
    xy, gfl = bld.reference_grid(plan, 1.0, margin=0.3)
    assert plan.is_free(xy, gfl, 0.3).all() and set(gfl.tolist()) == {0, 1}
    steps = np.diff(np.unique(xy[:, 0]))
    np.testing.assert_allclose(steps[steps < 1.5], 1.0)


# ------------------------------------------------------------------------------- measurements
def test_noiseless_measurements_equal_geometry():
    anchors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 8.0]])
    pos = np.array([[3.0, 4.0], [6.0, 8.0]])
    r = ms.simulate_ranges(pos, anchors, noise_std=0.0)
    np.testing.assert_allclose(r, [[5.0, np.hypot(7, 4), 5.0], [10.0, np.hypot(4, 8), 6.0]])
    np.testing.assert_allclose(ms.simulate_tdoa(pos, anchors, noise_std=0.0), r[:, 1:] - r[:, :1])
    a = ms.simulate_aoa(pos, anchors, orientations=[0.0, np.pi, -np.pi / 2], noise_std=0.0)
    np.testing.assert_allclose(a[0], [np.arctan2(4, 3), geo.wrap_angle(np.arctan2(4, -7) - np.pi),
                                      geo.wrap_angle(np.arctan2(-4, 3) + np.pi / 2)])
    np.testing.assert_allclose(ms.bearings([[1.0, 1.0]], [[0.0, 0.0]], [np.pi / 2]), [[-np.pi / 4]])


def test_nlos_bias_is_positive_and_seeded():
    pos = np.zeros((20000, 2))
    anchors = np.array([[3.0, 4.0]])
    r = ms.simulate_ranges(pos, anchors, noise_std=0.0, nlos=True, nlos_bias_mean=0.5, random_state=0)
    assert (r >= 5.0).all() and (r - 5.0).mean() == pytest.approx(0.5, abs=0.02)
    r2 = ms.simulate_ranges(pos, anchors, noise_std=0.1, nlos=0.3, random_state=7)
    np.testing.assert_array_equal(r2, ms.simulate_ranges(pos, anchors, noise_std=0.1, nlos=0.3, random_state=7))
    assert np.mean(r2 - 5.0) == pytest.approx(0.3 * 0.5, abs=0.02)   # E[bias] = p * mean
    with pytest.raises(ValueError):
        ms.simulate_ranges(pos[:1], anchors, nlos=1.5)


def test_tdoa_noise_has_the_reference_anchor_covariance():
    # d_i = r_i - r_0 with i.i.d. ToA errors of variance s^2: Cov(d) = s^2 (I + 1 1^T) (Chan and Ho 1994)
    anchors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    pos = np.full((40000, 2), 3.0)
    truth = ms.ranges_to_tdoa(ms.distances(pos[:1], anchors))
    err = ms.simulate_tdoa(pos, anchors, noise_std=0.2, random_state=0) - truth
    np.testing.assert_allclose(np.cov(err.T), 0.04 * (np.eye(3) + 1.0), atol=0.003)


# ------------------------------------------------------------------------------------ channel
def _music(H, n_paths, grid):
    """Tiny MUSIC over antennas with subcarriers as snapshots (test helper, not library code)."""
    R = H @ H.conj().T / H.shape[1]
    _, vec = np.linalg.eigh(R)
    En = vec[:, :H.shape[0] - n_paths]
    A = ch.ula_steering(grid, H.shape[0])                                   # (G, M)
    spec = 1.0 / np.linalg.norm(En.conj().T @ A.T, axis=0) ** 2
    peaks = np.flatnonzero((spec[1:-1] > spec[:-2]) & (spec[1:-1] > spec[2:])) + 1
    return np.sort(grid[peaks[np.argsort(spec[peaks])[-n_paths:]]])


def test_single_path_csi_encodes_angle_and_delay():
    offsets = ch.subcarrier_offsets()
    theta, tau = np.deg2rad(25.0), 40e-9
    H = ch.ofdm_csi([[0.5 * np.exp(0.3j)]], [[tau]], [[theta]], offsets, n_antennas=4)[0]    # (M, K)
    dphi = np.angle(H[1:, :] / H[:-1, :])
    np.testing.assert_allclose(dphi, np.pi * np.sin(theta), atol=1e-12)                   # 2 pi (d / lambda) sin
    slope = np.angle(H[0, 1:] / H[0, :-1])[0]                                             # adjacent subcarriers
    assert -slope / (2 * np.pi * ch.OFDM_SPACING_HZ) == pytest.approx(tau, rel=1e-9)
    np.testing.assert_allclose(np.abs(H), 0.5)


def test_music_resolves_two_coherent_paths_over_subcarriers():
    offsets = ch.subcarrier_offsets()
    true = np.deg2rad([-20.0, 35.0])
    H = ch.ofdm_csi([[1.0, 0.8 * np.exp(1j)]], [[30e-9, 75e-9]], [true], offsets, n_antennas=8)[0]
    grid = np.deg2rad(np.arange(-90.0, 90.01, 0.1))
    np.testing.assert_allclose(_music(H, 2, grid), true, atol=np.deg2rad(0.5))


def test_image_method_reflection_geometry():
    wall = np.array([[-10.0, 0.0, 10.0, 0.0]])
    paths = ch.multipath([[0.0, 1.0]], [[4.0, 1.0]], wall, frequency_hz=2.4e9, reflection_coef=0.5)
    assert list(paths.kind) == ["los", "wall"]
    np.testing.assert_allclose(paths.delay[0] * C, [4.0, np.sqrt(20.0)])
    np.testing.assert_allclose(paths.azimuth[0], [np.pi, np.arctan2(-1.0, -2.0)])            # seen from the AP
    np.testing.assert_allclose(paths.departure_azimuth[0], [0.0, np.arctan2(-1.0, 2.0)])
    ratio = np.abs(paths.gain[0, 0]) / np.abs(paths.gain[0, 1])
    assert ratio == pytest.approx(np.sqrt(20.0) / 4.0 / 0.5)
    lam = C / 2.4e9
    assert np.abs(paths.gain[0, 0]) == pytest.approx(lam / (4 * np.pi * 4.0))                # Friis amplitude
    # a wall too short for the specular point, and a wall between the two ends
    short = ch.multipath([[0.0, 1.0]], [[4.0, 1.0]], [[3.0, 0.0, 10.0, 0.0]], frequency_hz=2.4e9)
    assert np.isnan(short.gain[0, 1]) and np.isnan(short.delay[0, 1])
    between = ch.multipath([[0.0, 1.0]], [[4.0, 1.0]], [[2.0, -5.0, 2.0, 5.0]], frequency_hz=2.4e9, wall_loss_db=10.0)
    assert np.isnan(between.gain[0, 1])
    assert np.abs(between.gain[0, 0]) == pytest.approx(lam / (4 * np.pi * 4.0) * 10 ** (-0.5))
    # heights: the unfolded 3-D length and the elevation of arrival
    up = ch.multipath([[0.0, 1.0, 1.0]], [[4.0, 1.0, 4.0]], wall, frequency_hz=2.4e9)
    np.testing.assert_allclose(up.delay[0] * C, [5.0, np.sqrt(29.0)])
    assert up.elevation[0, 0] == pytest.approx(np.arctan2(-3.0, 4.0))


def test_music_finds_the_direct_and_the_wall_reflected_path():
    wall = np.array([[-10.0, 0.0, 10.0, 0.0]])
    device, ap = np.array([[0.0, 1.5]]), np.array([[6.0, 2.5]])
    paths = ch.multipath(device, ap, wall, frequency_hz=5.18e9, reflection_coef=0.6)
    boresight = np.pi                                                          # the AP looks towards -x
    H = ch.paths_to_csi(paths, ch.subcarrier_offsets(), n_antennas=8, orientation=boresight)[0]
    truth = np.sort(geo.wrap_angle(paths.azimuth[0] - boresight))
    # theta grows towards the array axis (boresight + 90 deg = -y here): the direct path is 1 m and the
    # image of the device 4 m to that side over 6 m of range
    np.testing.assert_allclose(truth, [np.arctan2(1.0, 6.0), np.arctan2(4.0, 6.0)])
    grid = np.deg2rad(np.arange(-90.0, 90.01, 0.1))
    np.testing.assert_allclose(_music(H, 2, grid), truth, atol=np.deg2rad(0.5))


def test_ula_phase_matches_the_documented_antenna_positions():
    # far field: antenna m sits at (m - (M - 1) / 2) d along orientation + 90 deg, so the phase of
    # antenna m relative to antenna 0 is -2 pi (L_m - L_0) / lambda for the exact path lengths L_m
    f, M, orient, theta = 5.18e9, 4, np.deg2rad(30.0), np.deg2rad(20.0)
    lam = C / f
    dev = 2000.0 * np.array([np.cos(orient + theta), np.sin(orient + theta)])
    paths = ch.multipath(dev[None], np.zeros((1, 2)), frequency_hz=f)
    H = ch.paths_to_csi(paths, np.array([0.0]), n_antennas=M, orientation=orient)[0, :, 0]
    axis = np.array([np.cos(orient + np.pi / 2), np.sin(orient + np.pi / 2)])
    antennas = np.outer(np.arange(M) - (M - 1) / 2, 0.5 * lam * axis)
    L = np.linalg.norm(dev - antennas, axis=1)
    np.testing.assert_allclose(np.angle(H[1:] / H[0]), geo.wrap_angle(-2 * np.pi * (L[1:] - L[0]) / lam), atol=1e-3)


def test_multipath_csi_matches_the_sum_of_paths():
    wall = np.array([[-10.0, 0.0, 10.0, 0.0]])
    paths = ch.multipath([[0.0, 1.0]], [[4.0, 1.0]], wall, frequency_hz=5e9, scatterers=[[2.0, 3.0]])
    assert list(paths.kind) == ["los", "wall", "scatterer"]
    offsets = ch.subcarrier_offsets([-1, 1])
    H = ch.paths_to_csi(paths, offsets, n_antennas=2, orientation=np.pi)
    manual = sum(paths.gain[0, p] * ch.ula_steering(paths.azimuth[0, p] - np.pi, 2)[:, None]
                 * np.exp(-2j * np.pi * offsets * paths.delay[0, p]) for p in range(3))
    np.testing.assert_allclose(H[0], manual, rtol=1e-12)


# ----------------------------------------------------------------------------------- walks, IMU
@pytest.fixture(scope="module")
def two_storey():
    plan = bld.office_floor_plan(size=(24.0, 12.0), n_floors=2, random_state=0)
    return plan, bld.navigation_graph(plan)


def test_shortest_path_is_optimal_on_an_open_grid():
    plan = bld.FloorPlan(walls=np.zeros((0, 4)), wall_floor=[], wall_type=[], bounds=(0, 0, 5, 5))
    g = bld.navigation_graph(plan, spacing=1.0, margin=0.0)
    start = int(np.flatnonzero((g.nodes == [0.5, 0.5]).all(axis=1))[0])
    goal = int(np.flatnonzero((g.nodes == [4.5, 2.5]).all(axis=1))[0])
    path = bld.shortest_path(g, start, goal)
    length = np.sum(np.hypot(*np.diff(g.nodes[path], axis=0).T))
    assert length == pytest.approx(2 + 2 * np.sqrt(2))   # 2 diagonal + 2 straight steps (octile distance)


def test_random_walk_stays_in_free_space_and_is_self_consistent(two_storey):
    plan, graph = two_storey
    route = bld.random_walk(plan, 90.0, graph=graph, change_floor_prob=0.5, random_state=2)
    assert route.total_duration >= 90.0
    # continuity between primitives: heading and position at the end of k equal the start of k+1
    end = route.sample(route.t0[1:] - 1e-9)
    np.testing.assert_allclose(np.unwrap(np.r_[route.h0[0], route.h0[:-1] + route.w[:-1] * route.duration[:-1]]),
                               np.unwrap(np.r_[route.h0[0], route.h0[1:]]), atol=1e-9)
    np.testing.assert_allclose(end.pos, route.p0[1:], atol=1e-6)
    tr = route.sample(np.arange(0.0, 90.0, 0.001))
    for f in range(plan.n_floors):
        same = (tr.floor[:-1] == f) & (tr.floor[1:] == f)
        assert not geo.crossing_matrix(tr.pos[:-1][same], tr.pos[1:][same], plan.walls_on(f)).any()
    assert set(np.unique(tr.floor)) == {0, 1}
    # the yaw rate integrates to the heading; distance walked = steps x step length (+- one step)
    h = np.unwrap(tr.heading)
    integ = np.concatenate([[0.0], np.cumsum((tr.yaw_rate[1:] + tr.yaw_rate[:-1]) / 2 * 0.001)])
    assert np.max(np.abs(integ - (h - h[0]))) < 0.01
    dist = np.sum(np.hypot(*np.diff(tr.pos[:, :2], axis=0).T))
    assert abs(dist - tr.step[-1] * tr.step_length) <= tr.step_length + 0.05
    np.testing.assert_allclose(tr.speed[tr.walking], route.speed)
    assert set(np.round(tr.pos[:, 2], 6)) >= {1.2, 4.7}                                  # device height per storey


def test_imu_is_consistent_with_the_walk(two_storey):
    plan, graph = two_storey
    tr = bld.random_walk(plan, 40.0, graph=graph, change_floor_prob=0.0, random_state=5).sample(np.arange(0, 40, 0.01))
    imu = bld.synthesize_imu(tr, acc_noise_std=0.0, gyro_noise_std=0.0, gyro_bias_std=0.0)
    np.testing.assert_allclose(imu[:, 5], tr.yaw_rate)
    turning = tr.walking & (np.abs(tr.yaw_rate) > 0)
    sway = 0.15 * 2.0 * np.sin(np.pi * tr.step_phase)
    np.testing.assert_allclose(imu[turning, 1] - sway[turning], tr.speed[turning] * tr.yaw_rate[turning])
    az = imu[:, 2]
    peaks = np.flatnonzero((az[1:-1] > az[:-2]) & (az[1:-1] >= az[2:]) & (az[1:-1] > bld.GRAVITY + 1.0)) + 1
    assert abs(len(peaks) - tr.step[-1]) <= 1
    assert np.allclose(az[~tr.walking], bld.GRAVITY)
    noisy = bld.synthesize_imu(tr, random_state=3)
    np.testing.assert_array_equal(noisy, bld.synthesize_imu(tr, random_state=3))


# ----------------------------------------------------------------------------- SyntheticOffice
def test_registry_and_info():
    assert "synthetic_office" in list_datasets()
    info = dataset_info("synthetic_office")
    assert info["splits"] == ("train", "test", "trajectory") and info["crs"] == "local"
    train, test = load_dataset("synthetic_office", **SMALL)          # nothing is downloaded
    assert isinstance(train, SampleTable) and train.meta["split"] == "train" and test.meta["split"] == "test"
    with pytest.raises(ValueError, match="splits"):  # no validation split: 'validation' must not silently mean test
        load_dataset("synthetic_office", split="validation", **SMALL)


@pytest.mark.parametrize("modality,dtype,width", [("wifi_rssi", np.float32, 8), ("ble_rssi", np.float32, 12),
                                                  ("ranges", np.float64, 4), ("tdoa", np.float64, 4),
                                                  ("aoa", np.float64, 4), ("csi", np.complex64, None)])
def test_contract_and_determinism(modality, dtype, width):
    ds = SyntheticOffice(seed=11, modality=modality, **SMALL)
    for split in ("train", "test", "trajectory"):
        t = ds.load(split)
        assert t.X.dtype == dtype and t.pos.dtype == np.float64 and t.pos.shape[1] == 2
        if width:
            assert t.X.shape[1] == width and len(t.meta["feature_names"]) == width
        else:
            assert t.X.shape[1:] == (3 * 4, 1, 56) and t.meta["antenna_anchor"].shape == (12,)
        assert (t.groups["source"] == "simulated").all() and len(np.unique(t.ids)) == len(t)
        assert t.meta["anchors"].shape == (len(t.meta["anchor_floor"]), 2) and t.meta["crs"] == "local"
        assert t.meta["floor_plan"]["walls"].shape[1] == 4 and len(t.meta["config_sha256"]) == 64
        assert t.floor is not None and t.building is None
        if dtype == np.float32:
            assert np.nanmax(t.X) < 0 and not np.any(t.X == -110)                  # dBm; NaN, never a sentinel
        again = SyntheticOffice(seed=11, modality=modality, **SMALL).load(split)
        np.testing.assert_array_equal(t.X, again.X)
        np.testing.assert_array_equal(t.pos, again.pos)
        assert all(np.array_equal(t.groups[k], again.groups[k]) for k in t.groups)
    other = SyntheticOffice(seed=12, modality=modality, **SMALL).load("test")
    assert not np.array_equal(other.pos, t.pos if t.meta["split"] == "test" else ds.load("test").pos)


def test_splits_structure():
    ds = SyntheticOffice(seed=0, **SMALL)
    train, test, traj = ds.load("train"), ds.load("test"), ds.load("trajectory")
    k = SMALL["samples_per_point"]
    assert len(train) % k == 0 and np.array_equal(train.groups["point"], np.repeat(np.arange(len(train) // k), k))
    np.testing.assert_array_equal(train.pos[::k], train.pos[1::k])                   # repeated scans per point
    assert not np.array_equal(train.X[::k], train.X[1::k])                           # with fresh noise
    assert len(test) == SMALL["n_test"] and sorted(test.groups) == ["room", "source"]
    rate = traj.meta["rate_hz"]
    assert len(traj) == SMALL["n_trajectories"] * int(SMALL["trajectory_duration"] * rate)
    first = traj.groups["trajectory"] == 0
    np.testing.assert_allclose(np.diff(traj.groups["time"][first]), 1.0 / rate)
    steps = np.linalg.norm(np.diff(traj.pos[first], axis=0), axis=1)
    assert steps.max() <= 1.9 / rate                                                   # walking speed <= 1.8 m/s


def test_noiseless_geometric_modalities_equal_geometry():
    phys = {"nlos_bias_mean": 0.0, "nlos_std": 0.0}
    for mod in ("ranges", "tdoa", "aoa"):
        t = SyntheticOffice(seed=1, modality=mod, noise_std=0.0, physics=phys, **SMALL).load("test")
        anchors = t.meta["anchors"]
        if mod == "ranges":
            np.testing.assert_allclose(t.X, ms.distances(t.pos, anchors), atol=1e-12)
        elif mod == "tdoa":
            d = ms.distances(t.pos, anchors)
            np.testing.assert_allclose(t.X, d[:, 1:] - d[:, :1], atol=1e-12)
        else:
            np.testing.assert_allclose(t.X, ms.bearings(t.pos, anchors, t.meta["anchor_orientations"]), atol=1e-12)


def test_rssi_equals_the_multiwall_model_without_noise():
    t = SyntheticOffice(seed=2, dim=3, noise_std=0.0, shadowing_std_db=0.0, physics={"quantize_db": 0.0,
                        "sensitivity_dbm": -300.0}, n_floors=2, **SMALL).load("test")
    plan = bld.FloorPlan.from_meta(t.meta["floor_plan"])
    anchors = t.meta["anchors"]
    dev = np.repeat(t.pos, len(anchors), axis=0)
    ap = np.tile(anchors, (len(t), 1))
    pl = pr.multi_wall_path_loss(np.linalg.norm(dev - ap, axis=1), 2.437e9, plan.wall_crossings(dev, ap),
                                 plan.floor_crossings(dev, ap))
    np.testing.assert_allclose(t.X, (20.0 - pl).reshape(len(t), -1).astype(np.float32), atol=1e-4)
    np.testing.assert_allclose(t.pos[:, 2] % 3.5, 1.2)                                # device height on each storey


def test_rssi_physics_trends():
    t = SyntheticOffice(seed=3, n_floors=3, noise_std=0.0, shadowing_std_db=0.0, physics={"sensitivity_dbm": -75.0},
                        **SMALL).load("test")
    same = t.floor[:, None] == t.meta["anchor_floor"][None, :]
    assert np.nanmean(t.X[same]) > np.nanmean(t.X[~same]) + 15.0                       # floors attenuate
    assert np.isnan(t.X[~same]).mean() > 0.2 and np.nanmin(t.X) >= -75.0              # below sensitivity: NaN
    shadowed = SyntheticOffice(seed=3, n_floors=3, noise_std=0.0, physics={"sensitivity_dbm": -75.0},
                               **SMALL).load("test")
    residual = (shadowed.X - t.X)[same]
    assert 2.5 < np.nanstd(residual) < 5.5                                            # 4 dB shadowing


@pytest.mark.parametrize("path_loss", ["log_distance", "3gpp_inh"])
def test_rssi_equals_the_other_path_loss_models_without_noise(path_loss):
    t = SyntheticOffice(seed=4, dim=3, n_floors=2, path_loss=path_loss, noise_std=0.0, shadowing_std_db=0.0,
                        physics={"quantize_db": 0.0, "sensitivity_dbm": -300.0}, **SMALL).load("test")
    assert t.meta["simulation"]["path_loss"] == path_loss
    plan = bld.FloorPlan.from_meta(t.meta["floor_plan"])
    anchors = t.meta["anchors"]
    dev, ap = np.repeat(t.pos, len(anchors), axis=0), np.tile(anchors, (len(t), 1))
    d = np.linalg.norm(dev - ap, axis=1)
    kf = plan.floor_crossings(dev, ap)
    floor_term = 18.3 * kf                                    # one slab: 1 ** (3 / 2 - 0.46) * 18.3 dB (COST 231)
    assert set(kf.tolist()) == {0, 1}
    if path_loss == "log_distance":                           # Friis at d0 = 1 m, then n = 3
        pl = 20 * np.log10(4 * np.pi * 2.437e9 / C) + 30.0 * np.log10(d)
    else:                                                     # TR 38.901 Table 7.4.1-1, LOS from the geometry
        los = (plan.wall_crossings(dev, ap).sum(axis=1) == 0) & (kf == 0)
        assert los.any() and (~los).any()
        pl_los = 32.4 + 17.3 * np.log10(d) + 20 * np.log10(2.437)
        pl = np.where(los, pl_los, np.maximum(pl_los, 17.30 + 38.3 * np.log10(d) + 24.9 * np.log10(2.437)))
    np.testing.assert_allclose(t.X, (20.0 - pl - floor_term).reshape(len(t), -1), atol=1e-4)


def test_walks_and_test_points_do_not_depend_on_the_modality():
    wifi = SyntheticOffice(seed=5, modality="wifi_rssi", **SMALL)
    imu = SyntheticOffice(seed=5, modality="trajectory", **SMALL)       # the task's name for the IMU modality
    ranges = SyntheticOffice(seed=5, modality="ranges", **SMALL)
    np.testing.assert_array_equal(wifi.load("test").pos, ranges.load("test").pos)
    a, b = wifi.load("trajectory"), imu.load("trajectory")
    assert b.meta["modality"] == "imu" and b.X.shape[1] == 6 and b.meta["channels"][5] == "gyr_z"
    assert b.X.dtype == np.float32 and b.pos.dtype == np.float64 and b.meta["rate_hz"] == 50.0
    again = SyntheticOffice(seed=5, modality="imu", **SMALL).load("trajectory")          # deterministic
    np.testing.assert_array_equal(b.X, again.X)
    np.testing.assert_array_equal(b.groups["step"], again.groups["step"])
    on_scan = np.isclose(b.groups["time"] * 1.0 % 1.0, 0.0) | np.isclose(b.groups["time"] % 1.0, 1.0)
    np.testing.assert_allclose(b.pos[on_scan], a.pos, atol=1e-9)                       # join on (trajectory, time)
    np.testing.assert_array_equal(b.groups["trajectory"][on_scan], a.groups["trajectory"])
    assert (np.diff(b.groups["step"][b.groups["trajectory"] == 0]) >= 0).all()
    with pytest.raises(ValueError, match="trajectory"):
        imu.load("train")


def test_csi_dataset_keeps_inter_antenna_phase():
    t = SyntheticOffice(seed=6, modality="csi", n_antennas=4, physics={"impairments": True}, **SMALL).load("test")
    clean = SyntheticOffice(seed=6, modality="csi", n_antennas=4, physics={"impairments": False}, **SMALL).load("test")
    # impairments multiply all antennas of one AP by the same per-packet phase ramp
    rel = t.X[:, 1:4, 0, :] / t.X[:, 0:1, 0, :]
    rel_clean = clean.X[:, 1:4, 0, :] / clean.X[:, 0:1, 0, :]
    np.testing.assert_allclose(rel, rel_clean, rtol=2e-3, atol=1e-5)
    assert t.meta["carrier_hz"] == 5.18e9 and t.meta["subcarrier_offsets_hz"].shape == (56,)


def test_csi_noise_equals_the_thermal_noise_floor():
    # two scans of one reference point differ only by receiver noise: E|H1 - H2|^2 / 2 = N / P_sc with
    # N = k T0 B_sc NF (B_sc = 312.5 kHz, NF = 7 dB) and P_sc = 20 dBm spread over 56 subcarriers
    t = SyntheticOffice(seed=6, modality="csi", physics={"impairments": False}, **SMALL).load("train")
    H = t.X[:, :, 0, :].astype(np.complex128)
    var = np.mean(np.abs(H[0::2] - H[1::2]) ** 2) / 2
    noise_dbm = 10 * np.log10(1.380649e-23 * 290.0 * 312.5e3 / 1e-3) + 7.0
    assert var == pytest.approx(10 ** ((noise_dbm - (20.0 - 10 * np.log10(56))) / 10), rel=0.03)


def test_option_validation():
    with pytest.raises(ValueError, match="modality"):
        SyntheticOffice(modality="lidar")
    with pytest.raises(ValueError, match="physics"):
        SyntheticOffice(physics={"tx_powr_dbm": 3})
    with pytest.raises(ValueError, match="path_loss"):
        SyntheticOffice(path_loss="okumura")
    with pytest.raises(ValueError, match="splits"):
        SyntheticOffice().load("holdout")
    for bad in ({"n_test": 0}, {"n_trajectories": 0}, {"samples_per_point": 0}, {"n_antennas": 0},
                {"n_floors": 0}, {"imu_rate_hz": 0.0}, {"grid_spacing": -1.0}, {"n_scatterers": -1}):
        with pytest.raises(ValueError, match=next(iter(bad))):
            SyntheticOffice(**bad)
    with pytest.raises(ValueError, match="at least one sample"):
        SyntheticOffice(trajectory_duration=0.5, scan_rate_hz=1.0)
    with pytest.raises(ValueError, match="reference point"):
        SyntheticOffice(size=(3.0, 3.0)).load("train")
    with pytest.raises(ValueError, match="n_aps"):
        SyntheticOffice(modality="tdoa", n_aps=1)


def test_config_digest_is_stable_for_numpy_physics_values():
    as_tuple = SyntheticOffice(seed=1, physics={"wall_loss_db": (3.4, 6.9), "tx_power_dbm": 15.0}, **SMALL)
    as_numpy = SyntheticOffice(seed=1, physics={"wall_loss_db": np.array([3.4, 6.9]),
                                                "tx_power_dbm": np.float64(15.0)}, **SMALL)
    a, b = as_tuple.load("test"), as_numpy.load("test")
    assert a.meta["config_sha256"] == b.meta["config_sha256"]
    np.testing.assert_array_equal(a.X, b.X)


# ------------------------------------------------------------------- visible light and magnetic field
def test_optical_copy_agrees_with_signals_vlc():
    from indoorloc.datasets.simulated import optical
    from indoorloc.signals import vlc

    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(0, 8, 50), rng.uniform(0, 6, 50), rng.uniform(0.5, 1.5, 50)])
    leds = np.column_stack([rng.uniform(0, 8, 7), rng.uniform(0, 6, 7), np.full(7, 3.0)])
    for kw in (dict(), dict(order=3.0, area=2e-4, fov=np.deg2rad(50), filter_gain=0.8, concentrator_gain=2.2)):
        np.testing.assert_allclose(optical.lambertian_gain(pts, leds, **kw), vlc.channel_gain(pts, leds, **kw),
                                   rtol=1e-14, atol=0)
    P = np.array([0.0, 1e-6, 1e-4])
    phys = dict(responsivity=0.54, bandwidth=100e6, background_current=5100e-6, area=1e-4, temperature=295.0,
                open_loop_gain=10.0, capacitance_per_area=1.12e-6, channel_noise_factor=1.5, transconductance=30e-3)
    np.testing.assert_allclose(optical.noise_variance(P, **phys, i2=0.562, i3=0.0868), vlc.noise_variance(P),
                               rtol=1e-14)
    assert optical.lambertian_order(np.deg2rad(60.0)) == pytest.approx(vlc.lambertian_order(np.deg2rad(60.0)))


def test_ceiling_led_grid_fills_every_room():
    from indoorloc.datasets.simulated import optical

    plan = bld.office_floor_plan(size=(16.0, 10.0), n_floors=2, random_state=0)
    xyz, fl = optical.room_grid_leds(plan, 2.5, 3.0)
    np.testing.assert_allclose(xyz[:, 2], plan.height_of(fl, 3.0))
    per_storey = 0
    for (x0, y0, x1, y1), f in zip(plan.rooms, plan.room_floor):
        inside = (fl == f) & (xyz[:, 0] > x0) & (xyz[:, 0] < x1) & (xyz[:, 1] > y0) & (xyz[:, 1] < y1)
        assert inside.sum() == max(1, round((x1 - x0) / 2.5)) * max(1, round((y1 - y0) / 2.5))
        per_storey += inside.sum() if f == 0 else 0
    assert len(xyz) == 2 * per_storey and np.array_equal(np.unique(fl), [0, 1])


def test_dipole_field_closed_form_and_maxwell():
    from indoorloc.datasets.simulated import magnetic as mag

    m = np.array([[0.0, 0.0, 100.0]])                                   # A m^2 at the origin
    on_axis = mag.dipole_field([[0.0, 0.0, 2.0]], [[0.0, 0.0, 0.0]], m)
    np.testing.assert_allclose(on_axis, [[0.0, 0.0, 0.1 * 2 * 100 / 8]], atol=1e-15)   # 2 k m / r^3
    equator = mag.dipole_field([[2.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], m)
    np.testing.assert_allclose(equator, [[0.0, 0.0, -0.1 * 100 / 8]], atol=1e-15)      # -k m / r^3
    np.testing.assert_allclose(mag.earth_field(50.0, 60.0, 10.0),
                               50 * np.array([np.cos(np.pi / 3) * np.sin(np.pi / 18),
                                              np.cos(np.pi / 3) * np.cos(np.pi / 18),
                                              -np.sin(np.pi / 3)]))
    # outside the sources the field is divergence- and curl-free (central differences, exact for dipoles)
    rng = np.random.default_rng(1)
    src, mom = rng.uniform(-3, 3, (6, 3)) * [1, 1, 0] - [0, 0, 0.1], rng.normal(0, 80, (6, 3))
    p, h = rng.uniform(-2, 2, (20, 3)) * [1, 1, 0] + [0, 0, 1.2], 1e-4
    J = np.stack([(mag.dipole_field(p + h * e, src, mom) - mag.dipole_field(p - h * e, src, mom)) / (2 * h)
                  for e in np.eye(3)], axis=2)                          # J[n, i, j] = d b_i / d x_j
    scale = np.abs(J).max()
    assert np.abs(np.trace(J, axis1=1, axis2=2)).max() < 1e-6 * scale
    assert np.abs(J - J.transpose(0, 2, 1)).max() < 1e-6 * scale        # curl-free: symmetric gradient


@pytest.mark.parametrize("modality", ["vlc", "magnetic"])
def test_vlc_and_magnetic_contract_and_determinism(modality):
    ds = SyntheticOffice(seed=11, modality=modality, **SMALL)
    for split in ("train", "test", "trajectory"):
        t = ds.load(split)
        assert t.X.dtype == np.float64 and t.X.ndim == 2 and t.pos.shape[1] == 2 and t.floor is not None
        assert len(t.meta["feature_names"]) == t.X.shape[1] and t.meta["modality"] == modality
        assert (t.groups["source"] == "simulated").all() and len(np.unique(t.ids)) == len(t)
        again = SyntheticOffice(seed=11, modality=modality, **SMALL).load(split)
        np.testing.assert_array_equal(t.X, again.X)
        np.testing.assert_array_equal(t.pos, again.pos)
        if modality == "vlc":
            A = len(t.meta["led_positions"])
            assert t.meta["units"] == "W" and t.X.shape[1] == A and t.meta["anchors"].shape == (A, 2)
            assert t.meta["anchor_normals"].shape == (A, 3) and t.meta["lambertian_order"] == pytest.approx(1.0)
            assert t.meta["concentrator_gain"] == pytest.approx(1.5 ** 2 / np.sin(np.deg2rad(70.0)) ** 2)
            seen = np.isfinite(t.X)
            assert seen.any() and (~seen).any() and np.all(t.X[seen] > 0)      # NaN = not seen, never 0
        else:
            assert t.meta["units"] == "uT" and t.meta["feature_names"] == ("B", "B_h", "B_v")
            assert t.meta["dipole_positions"].shape == t.meta["dipole_moments"].shape
            np.testing.assert_allclose(t.X[:, 0], np.hypot(t.X[:, 1], t.X[:, 2]))  # |b|^2 = B_h^2 + B_v^2
    assert t.meta["rate_hz"] == (5.0 if modality == "vlc" else 10.0)
    other = SyntheticOffice(seed=12, modality=modality, **SMALL).load("test")
    assert not np.array_equal(other.X, ds.load("test").X, equal_nan=True)
    with pytest.raises(ValueError, match="n_aps"):
        SyntheticOffice(modality=modality, n_aps=4)


def test_noiseless_vlc_equals_the_lambertian_model_blocked_by_walls_and_floors():
    from indoorloc.signals import vlc

    t = SyntheticOffice(seed=2, modality="vlc", dim=3, n_floors=2, noise_std=0.0, **SMALL).load("test")
    leds, m = t.meta["led_positions"], t.meta["lambertian_order"]
    plan = bld.FloorPlan.from_meta(t.meta["floor_plan"])
    P = 5.0 * vlc.channel_gain(t.pos, leds, order=m, area=1e-4, fov=np.deg2rad(70.0),
                               concentrator_gain=t.meta["concentrator_gain"])
    dev, led = np.repeat(t.pos, len(leds), axis=0), np.tile(leds, (len(t), 1))
    blocked = plan.nlos_mask(dev, led).reshape(len(t), -1)
    visible = (P > 0) & ~blocked
    assert blocked[P > 0].any() and visible.any()
    np.testing.assert_allclose(t.X[visible], P[visible], rtol=1e-12)
    assert np.isnan(t.X[~visible]).all()
    assert np.all(t.meta["anchor_floor"][np.nonzero(visible)[1]] == t.floor[np.nonzero(visible)[0]])


def test_vlc_noise_is_the_shot_and_thermal_noise_of_the_receiver():
    from indoorloc.signals import vlc

    ds = SyntheticOffice(seed=3, modality="vlc", **SMALL)                # two scans per reference point
    t = ds.load("train")
    clean = SyntheticOffice(seed=3, modality="vlc", noise_std=0.0, **SMALL).load("train")
    both = np.isfinite(t.X[0::2]) & np.isfinite(t.X[1::2])
    diff = (t.X[0::2] - t.X[1::2])[both]
    expected = vlc.noise_variance(clean.X[0::2][both]) / 0.54 ** 2       # two readings of the same power
    assert np.mean(diff ** 2) / 2 == pytest.approx(np.mean(expected), rel=0.1)
    threshold = 5.0 * np.sqrt(vlc.noise_variance(0.0)) / 0.54            # detection at SNR 5
    assert np.nanmin(t.X) > threshold and np.isnan(t.X).mean() > 0.5


def test_noiseless_magnetic_features_are_the_field_components():
    from indoorloc.datasets.simulated import magnetic as mag
    from indoorloc.signals import magnetic as smag

    t = SyntheticOffice(seed=4, modality="magnetic", dim=3, n_floors=2, noise_std=0.0, **SMALL).load("test")
    b = t.meta["earth_field"] + mag.dipole_field(t.pos, t.meta["dipole_positions"], t.meta["dipole_moments"])
    np.testing.assert_allclose(t.X, np.column_stack([np.linalg.norm(b, axis=1), np.hypot(b[:, 0], b[:, 1]), b[:, 2]]),
                               rtol=1e-12)
    readings = mag.device_readings(b, np.linspace(-3, 3, len(b)))        # any heading of a flat device
    np.testing.assert_allclose(smag.magnetic_features(readings), t.X, rtol=1e-12)
    assert 0.3 < t.X[:, 0].std() < 10.0                                   # an anomaly of a few uT (documented)
    assert np.all(np.abs(t.X[:, 0] - 50.0) < 25.0)


def test_simulated_readings_share_the_frame_conventions_of_signals_magnetic():
    """L1 (ENU, declination positive east) and L2 (heading from true east, B_v positive up) agree."""
    from indoorloc.datasets.simulated import magnetic as mag
    from indoorloc.signals import magnetic as smag

    walk_heading = np.linspace(-3.0, 3.0, 25)
    earth = mag.earth_field(48.0, 65.0, 7.0)                             # 7 degrees east, dipping 65 degrees
    readings = mag.device_readings(np.tile(earth, (25, 1)), walk_heading)  # flat device, x forward
    true = smag.heading(readings, forward=(1.0, 0.0, 0.0), declination=np.deg2rad(7.0))
    np.testing.assert_allclose(smag.wrap_angle(true - walk_heading), 0.0, atol=1e-12)
    np.testing.assert_allclose(smag.inclination(readings), np.deg2rad(65.0), atol=1e-12)
    np.testing.assert_allclose(smag.magnetic_features(readings),
                               np.tile([48.0, 48.0 * np.cos(np.deg2rad(65.0)), -48.0 * np.sin(np.deg2rad(65.0))],
                                       (25, 1)), atol=1e-12)


def test_magnetic_walks_carry_noisy_sequences_and_a_device_bias():
    kw = dict(seed=5, modality="magnetic", **SMALL)
    walks, clean = SyntheticOffice(**kw).load("trajectory"), SyntheticOffice(noise_std=0.0, **kw).load("trajectory")
    assert walks.meta["rate_hz"] == 10.0 and len(walks) == SMALL["n_trajectories"] * 200
    np.testing.assert_allclose(np.diff(walks.groups["time"][walks.groups["trajectory"] == 0]), 0.1)
    residual = walks.X[:, 0] - clean.X[:, 0]                              # isotropic 0.5 uT per axis
    assert abs(residual.mean()) < 0.05 and 0.45 < residual.std() < 0.55
    biased = SyntheticOffice(noise_std=0.0, physics={"bias_std": 2.0}, **kw).load("trajectory")
    offset = biased.X - clean.X
    walk = biased.groups["trajectory"]
    per_walk = [offset[walk == w] for w in np.unique(walk)]
    assert all(np.ptp(o[:, 2]) < 1e-12 for o in per_walk)                 # B_v: one constant offset per walk
    assert abs(per_walk[0][0, 2] - per_walk[1][0, 2]) > 1e-3              # a new device per walk


def test_vlc_and_magnetic_share_points_and_walks_with_the_other_modalities():
    wifi = SyntheticOffice(seed=6, modality="wifi_rssi", **SMALL)
    a = wifi.load("trajectory")                                          # 1 Hz scans
    for modality in ("vlc", "magnetic"):
        other = SyntheticOffice(seed=6, modality=modality, **SMALL)
        for split in ("train", "test"):
            np.testing.assert_array_equal(wifi.load(split).pos, other.load(split).pos)
            np.testing.assert_array_equal(wifi.load(split).floor, other.load(split).floor)
        b = other.load("trajectory")                                     # 5 Hz (VLC), 10 Hz (magnetic)
        on = np.isclose(b.groups["time"], np.round(b.groups["time"]))
        np.testing.assert_allclose(b.pos[on], a.pos, atol=1e-9)          # join on (trajectory, time)
        np.testing.assert_array_equal(b.groups["trajectory"][on], a.groups["trajectory"])


# ------------------------------------------------------------------------------------ DeepMIMO
def _links():
    rx = np.array([[0.0, 0.0, 1.5], [1.0, 0.0, 1.5], [2.0, 0.0, 1.5]])
    nan = np.nan
    a = dict(tx_id="tx1-0", rx_pos=rx, tx_pos=[[10.0, 0.0, 6.0]],
             power=np.array([[-60.0, -60.0], [-70.0, nan], [nan, nan]]),
             phase=np.array([[0.0, 0.0], [90.0, nan], [nan, nan]]),
             delay=np.array([[20e-9, 10e-9], [30e-9, nan], [nan, nan]]),
             aod_az=np.array([[10.0, 90.0], [45.0, nan], [nan, nan]]),
             aod_el=np.array([[90.0, 90.0], [90.0, nan], [nan, nan]]))
    b = dict(tx_id="tx1-1", rx_pos=rx, tx_pos=[[0.0, 10.0, 6.0]],
             power=np.array([[-80.0, nan], [-75.0, -65.0], [nan, nan]]), phase=np.zeros((3, 2)),
             delay=np.array([[50e-9, nan], [40e-9, 60e-9], [nan, nan]]),
             aod_az=np.array([[-90.0, nan], [0.0, 180.0], [nan, nan]]), aod_el=np.full((3, 2), 90.0))
    return [a, b]


def test_deepmimo_unit_conversions():
    p = np.array([[-60.0, -60.0], [np.nan, np.nan]])
    out = dmx.received_power_dbm(p)
    assert out[0] == pytest.approx(-60.0 + 30.0 + 10 * np.log10(2)) and np.isnan(out[1])
    assert dmx.received_power_dbm(p[:1], [[0.0, 0.0]], coherent=True)[0] == pytest.approx(-30.0 + 20 * np.log10(2))
    assert dmx.received_power_dbm(p[:1], [[0.0, 180.0]], coherent=True)[0] < -200.0        # (numerical) cancellation
    r = dmx.first_arrival_range([[np.nan, 20e-9, 10e-9], [np.nan] * 3])
    assert r[0] == pytest.approx(C * 10e-9) and np.isnan(r[1])
    az = dmx.strongest_path_azimuth([[-70.0, -60.0, np.nan]], [[10.0, 90.0, 0.0]])
    assert az[0] == pytest.approx(np.pi / 2)
    # zenith 90 deg is the horizontal plane: the ULA phase step is pi sin(theta)
    H = dmx.paths_to_csi([[-60.0]], [[0.0]], [[0.0]], [[30.0]], [[90.0]], np.array([0.0]), n_antennas=3)
    np.testing.assert_allclose(np.angle(H[0, 1:, 0] / H[0, :-1, 0]), np.pi * 0.5)
    assert np.abs(H[0, 0, 0]) == pytest.approx(1e-3)                                       # sqrt(10^-6 W)
    # DeepMIMO's "elevation" is a zenith angle (its array response uses cos(theta) for z): zenith 0 is
    # broadside to a horizontal ULA (no phase step) and zenith 60 deg scales the step by sin(60 deg)
    for zenith, step in ((0.0, 0.0), (60.0, np.pi * 0.5 * np.sin(np.deg2rad(60.0)))):
        Hz = dmx.paths_to_csi([[-60.0]], [[0.0]], [[0.0]], [[30.0]], [[zenith]], np.array([0.0]), n_antennas=3)
        np.testing.assert_allclose(np.angle(Hz[0, 1:, 0] / Hz[0, :-1, 0]), step, atol=1e-12)


def test_deepmimo_scenario_to_table():
    t = dmx.scenario_to_table(_links(), modality="rssi")
    assert t.X.shape == (2, 2) and t.X.dtype == np.float32 and list(t.ids) == ["rx0-000000", "rx0-000001"]
    assert t.X[0, 0] == pytest.approx(-30.0 + 10 * np.log10(2), abs=1e-4)          # a 30 dBm (0 dBW) transmitter
    ap20 = dmx.scenario_to_table(_links(), modality="rssi", power_offset_db=20.0 - 30.0)     # a 20 dBm AP
    np.testing.assert_allclose(ap20.X, t.X - 10.0, atol=1e-5)
    assert t.meta["feature_names"] == ("tx1-0", "tx1-1")
    np.testing.assert_allclose(t.meta["anchors"], [[10.0, 0.0, 6.0], [0.0, 10.0, 6.0]])
    assert (t.groups["source"] == "simulated").all()
    r = dmx.scenario_to_table(_links(), modality="ranges")
    np.testing.assert_allclose(r.X, C * np.array([[10e-9, 50e-9], [30e-9, 40e-9]]))
    d = dmx.scenario_to_table(_links(), modality="tdoa")
    np.testing.assert_allclose(d.X[:, 0], r.X[:, 1] - r.X[:, 0])
    a = dmx.scenario_to_table(_links(), modality="aoa", dim=2)
    np.testing.assert_allclose(a.X, np.deg2rad([[10.0, -90.0], [45.0, -180.0]]))           # ties: lowest path index
    assert a.pos.shape == (2, 2)
    c = dmx.scenario_to_table(_links(), modality="csi", n_antennas=4)
    assert c.X.shape == (2, 8, 1, 56) and c.X.dtype == np.complex64
    kept = dmx.scenario_to_table(_links(), modality="rssi", drop_empty=False, sensitivity_dbm=-45.0)
    assert len(kept) == 3 and np.isnan(kept.X[2]).all() and np.isnan(kept.X[0, 1])        # -80 dBm < -45 dBm
    bad = _links()
    bad[1]["rx_pos"] = bad[1]["rx_pos"] + 1.0
    with pytest.raises(ValueError, match="receiver positions"):
        dmx.scenario_to_table(bad)


def test_deepmimo_dataset_with_a_stand_in_package(tmp_path, monkeypatch):
    """Wire DeepMIMO.load to a stand-in module with deepmimo 4's call signatures."""
    (tmp_path / "toy").mkdir()
    (tmp_path / "toy" / "params.json").write_text("{}")
    calls = {}

    class Child(dict):
        pass

    def load(name, **kwargs):
        calls.update(name=name, folder=fake.config.get("scenarios_folder"), **kwargs)
        kids = []
        for i, link in enumerate(_links()):
            kid = Child({k: np.asarray(v) for k, v in link.items() if k != "tx_id"})
            kid.update(txrx={"tx_set_id": 1, "rx_set_id": 0, "tx_idx": i},
                       rt_params={"frequency": 3.5e9, "raytracer_name": "toy", "max_path_depth": 2})
            kids.append(kid)
        return types.SimpleNamespace(datasets=kids)

    class Config:
        values = {"scenarios_folder": "deepmimo_scenarios"}

        def get(self, key):
            return self.values[key]

        def set(self, key, value):
            self.values[key] = value

    fake = types.ModuleType("deepmimo")
    fake.__version__ = "4.0.5-standin"
    fake.config = Config()
    fake.load = load
    monkeypatch.setitem(sys.modules, "deepmimo", fake)
    t = load_dataset("deepmimo", split="all", root=tmp_path, scenario="toy", modality="rssi", download=False)
    assert calls["name"] == "toy" and calls["folder"] == str(tmp_path) and calls["max_paths"] == 25
    assert fake.config.get("scenarios_folder") == "deepmimo_scenarios"                # restored
    assert t.X.shape == (2, 2) and t.meta["carrier_hz"] == 3.5e9 and t.meta["feature_names"] == ("tx1-0", "tx1-1")
    assert t.meta["deepmimo_version"] == "4.0.5-standin" and t.meta["split"] == "all"


def test_deepmimo_needs_files_and_names_the_extra(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        dmx.DeepMIMO(tmp_path, scenario="missing").load()
    (tmp_path / "toy").mkdir()
    (tmp_path / "toy" / "params.json").write_text("{}")
    monkeypatch.setitem(sys.modules, "deepmimo", None)                                    # not installed
    with pytest.raises(ImportError, match=r"indoorloc\[sim\]"):
        dmx.DeepMIMO(tmp_path, scenario="toy").load()
    with pytest.raises(ValueError):
        dmx.DeepMIMO(tmp_path, modality="lidar")
