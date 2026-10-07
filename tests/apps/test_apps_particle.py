"""L5 particle filter: resampling, Bayes update, wall constraints, PDR motion, determinism."""
from __future__ import annotations

import numpy as np
import pytest

from indoorloc.apps.maps import FloorMap
from indoorloc.apps.particle import ParticleFilter, effective_sample_size, systematic_resample


def test_systematic_resampling_known_draw_and_count_bounds():
    assert systematic_resample([0.1, 0.2, 0.3, 0.4], u=0.5).tolist() == [1, 2, 3, 3]  # points 1/8, 3/8, 5/8, 7/8
    rng = np.random.default_rng(0)
    for _ in range(20):
        w = rng.random(50) * (rng.random(50) > 0.3)
        w /= w.sum()
        counts = np.bincount(systematic_resample(w, rng), minlength=50)
        assert counts.sum() == 50
        assert np.all(counts >= np.floor(50 * w) - 1e-9) and np.all(counts <= np.ceil(50 * w) + 1e-9)
        assert np.all(counts[w == 0] == 0)  # a dead particle is never copied
    assert effective_sample_size(np.ones(8)) == pytest.approx(8.0)
    assert effective_sample_size([0.5, 0.5, 0.0, 0.0]) == pytest.approx(2.0)
    assert effective_sample_size([1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_one_fix_gives_the_gaussian_posterior_mean():
    # prior N(m0, s0^2), fix z with std r  ->  posterior mean (m0/s0^2 + z/r^2) / (1/s0^2 + 1/r^2)
    m0, s0, z, r = np.array([0.0, 0.0]), 2.0, np.array([3.0, -1.0]), 1.0
    pf = ParticleFilter(40000, meas_std=r, use_spread=False, resample_threshold=0.0, random_state=0)
    pf.initialize(m0, s0)
    est = pf.correct(z)
    post = (m0 / s0 ** 2 + z / r ** 2) / (1 / s0 ** 2 + 1 / r ** 2)
    assert np.allclose(est, post, atol=0.03)
    assert pf.estimate()[1] == pytest.approx(np.sqrt(2 / (1 / s0 ** 2 + 1 / r ** 2)), rel=0.02)


def test_particles_never_cross_a_wall_and_pass_through_a_door():
    solid = FloorMap([[[5.0, 0.0], [5.0, 10.0]]], bounds=(0, 0, 10, 10))
    pf = ParticleFilter(2000, motion_std=0.6, floor_map=solid, random_state=1).initialize([2.5, 5.0], 0.4)
    assert pf.particles_[:, 0].max() < 5.0  # the whole cloud starts left of the wall
    for _ in range(200):
        pf.predict(1.0)
        assert pf.particles_[:, 0].max() < 5.0 and pf.particles_.min() >= 0.0
    assert pf.n_depleted_ == 0
    door = FloorMap([[[5.0, 0.0], [5.0, 4.0]], [[5.0, 6.0], [5.0, 10.0]]], bounds=(0, 0, 10, 10))
    pf = ParticleFilter(2000, motion_std=0.6, floor_map=door, random_state=1).initialize([2.5, 5.0], 1.0)
    for _ in range(200):
        pf.predict(1.0)
    assert (pf.particles_[:, 0] > 5.0).mean() > 0.2


def test_a_cloud_walking_into_a_wall_stops_instead_of_crossing():
    fmap = FloorMap([[[3.0, -5.0], [3.0, 5.0]]], bounds=(-5, -5, 5, 5))
    pf = ParticleFilter(200, step_length_std=0.0, heading_std=0.0, floor_map=fmap, random_state=0)
    pf.initialize([2.0, 0.0], 0.01)
    before = pf.particles_.copy()
    pf.step(length=2.0, heading=0.0)  # every particle would cross x = 3
    assert pf.n_depleted_ == 1 and np.array_equal(pf.particles_, before)


def test_map_constraints_correct_a_heading_error_in_a_corridor():
    # 60 steps of 0.7 m along a 2 m wide corridor; the PDR heading is 10 degrees off
    corridor = FloorMap([[[0.0, 0.0], [60.0, 0.0]], [[0.0, 2.0], [60.0, 2.0]]], bounds=(-1, 0, 60, 2))
    truth = np.array([0.5 + 0.7 * 60, 1.0])
    bias = np.deg2rad(10.0)
    pdr_only = np.array([0.5, 1.0]) + 0.7 * 60 * np.array([np.cos(bias), np.sin(bias)])
    pf = ParticleFilter(1000, step_length_std=0.05, heading_std=0.05, floor_map=corridor, random_state=2)
    pf.initialize([0.5, 1.0], 0.2, heading_bias_std=0.3)
    for _ in range(60):
        pf.step(0.7, bias)
    est = pf.estimate()[0]
    assert np.linalg.norm(pdr_only - truth) > 7.0
    assert np.linalg.norm(est - truth) < 1.5
    assert pf.particles_[:, 1].min() > 0.0 and pf.particles_[:, 1].max() < 2.0
    assert np.median(pf.heading_bias_) == pytest.approx(-bias, abs=0.05)  # the offset was learned


def test_range_likelihood_localises_a_static_target():
    anchors = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    truth = np.array([6.0, 3.0])
    rng = np.random.default_rng(3)
    pf = ParticleFilter(3000, motion_std=0.05, anchors=anchors, range_std=0.2, random_state=3)
    pf.initialize([5.0, 5.0], 3.0)
    for _ in range(20):
        pf.predict(1.0)
        pf.correct_ranges(np.linalg.norm(anchors - truth, axis=1) + rng.normal(0, 0.2, 4))
    assert np.linalg.norm(pf.estimate()[0] - truth) < 0.15


def test_filter_is_seeded_and_tracks_noisy_fixes():
    rng = np.random.default_rng(4)
    t = np.arange(100, dtype=float)
    truth = np.stack([0.5 * t, 5 + 3 * np.sin(t / 15)], axis=1)
    fixes = truth + rng.normal(0, 2.0, truth.shape)
    fixes[40:45] = np.nan
    def run(seed):
        return ParticleFilter(500, motion_std=0.8, meas_std=2.0, random_state=seed).filter(fixes, t)

    a, b, c = run(7), run(7), run(8)
    assert np.array_equal(a.pos, b.pos) and not np.array_equal(a.pos, c.pos)
    err = lambda p: np.sqrt(np.mean(np.sum((p - truth) ** 2, axis=1)))  # noqa: E731
    ok = np.isfinite(fixes).all(axis=1)
    assert err(a.pos) < 0.8 * np.sqrt(np.mean(np.sum((fixes[ok] - truth[ok]) ** 2, axis=1)))
    assert np.isfinite(a.pos).all() and np.all(a.spread > 0)


def test_online_update_initialises_from_the_first_fix_inside_the_map():
    fmap = FloorMap.from_polygons([[(0, 0), (20, 0), (20, 10), (0, 10)]])
    pf = ParticleFilter(300, floor_map=fmap, random_state=0).reset()
    assert np.isnan(pf.update(0.0, None)).all() and not pf.initialized
    pf.update(1.0, [25.0, 5.0], spread=0.5)  # a fix outside the plan starts the cloud at its edge
    assert pf.initialized and fmap.contains(pf.particles_[pf.weights_ > 0]).all()
    with pytest.raises(ValueError, match="backwards"):
        pf.update(0.5, [1.0, 1.0])
    with pytest.raises(ValueError, match="floor_map"):
        ParticleFilter(10).initialize()


def test_augmented_mcl_recovers_a_cloud_started_at_a_wrong_fix():
    # the first fix is 60 m off; later fixes agree on the truth. Plain SIR crawls towards them,
    # sensor resetting (Thrun et al. 2005, Sec. 8.3.5) jumps within a few fixes.
    rng = np.random.default_rng(5)
    truth = np.array([10.0, 10.0])
    fixes = [truth + [60.0, 0.0]] + [truth + rng.normal(0, 2.0, 2) for _ in range(15)]

    def run(recovery):
        pf = ParticleFilter(1000, motion_std=0.5, meas_std=2.0, use_spread=False, recovery=recovery, random_state=0)
        pf.reset()
        errors = [np.linalg.norm(pf.update(float(k), z) - truth) for k, z in enumerate(fixes)]
        return np.array(errors), pf

    plain, _ = run(None)
    robust, pf = run((0.05, 0.5))
    assert plain[0] == robust[0] == pytest.approx(60.0, abs=1.0)
    assert robust[5:].max() < 3.0 and plain[5:].min() > 20.0
    assert pf.n_injected_ > 0
    quiet, pf = run((0.05, 0.5))
    assert np.array_equal(quiet, robust)  # seeded: the injection draws are reproducible


def test_augmented_mcl_rates_follow_the_closed_form():
    # Thrun et al. (2005), Table 8.3: w_avg = sum_i w_i p(z | x_i); w_slow += a_s (w_avg - w_slow),
    # w_fast += a_f (w_avg - w_fast); each particle is redrawn with probability max(0, 1 - w_fast / w_slow).
    # A cloud collapsed at the origin and a fix d = 2 sigma away: p(z | x_i) = exp(-d^2 / 2 sigma^2) = e^-2.
    a_s, a_f, sigma, n = 0.05, 0.5, 1.5, 200000
    pf = ParticleFilter(n, meas_std=sigma, use_spread=False, recovery=(a_s, a_f), random_state=0)
    pf.initialize([0.0, 0.0], 1e-9)
    pf.correct([2 * sigma, 0.0])
    w = np.exp(-2.0)
    slow, fast = 0.5 + a_s * (w - 0.5), 0.5 + a_f * (w - 0.5)  # both averages start at 1/2
    assert pf.w_slow_ == pytest.approx(slow, rel=1e-6) and pf.w_fast_ == pytest.approx(fast, rel=1e-6)
    rate = 1 - fast / slow
    assert abs(pf.n_injected_ / n - rate) < 4 * np.sqrt(rate * (1 - rate) / n)  # binomial draw
    assert np.sum(np.linalg.norm(pf.particles_, axis=1) > 1e-6) == pf.n_injected_  # redrawn around the fix


def test_saved_filter_continues_the_same_random_sequence(tmp_path):
    from indoorloc.core import load_model

    rng = np.random.default_rng(6)
    fixes = np.cumsum(rng.normal(0, 1, (30, 2)), axis=0)
    pf = ParticleFilter(300, motion_std=0.7, meas_std=1.5, recovery=(0.05, 0.5), random_state=11).reset()
    head = [pf.update(float(k), z) for k, z in enumerate(fixes[:15])]
    again = load_model(pf.save(tmp_path / "pf"))  # particles, weights and the PCG64 state, no pickle
    assert np.array_equal(again.particles_, pf.particles_)
    tail = [again.update(float(k), z) for k, z in enumerate(fixes[15:], start=15)]
    full = ParticleFilter(300, motion_std=0.7, meas_std=1.5, recovery=(0.05, 0.5), random_state=11).filter(fixes)
    assert np.array_equal(np.array(head + tail), full.pos)


def test_ranges_to_3d_anchors_with_a_known_tag_height():
    anchors = np.array([[0.0, 0.0, 2.6], [10.0, 0.0, 2.4], [0.0, 10.0, 2.5], [10.0, 10.0, 2.7]])
    truth, h = np.array([3.0, 7.0]), 1.2
    r = np.linalg.norm(anchors - np.r_[truth, h], axis=1)  # slant ranges
    pf = ParticleFilter(4000, motion_std=0.05, anchors=anchors, height=h, range_std=0.1, random_state=0)
    pf.initialize([5.0, 5.0], 3.0)
    for _ in range(10):
        pf.predict(1.0)
        pf.correct_ranges(r)
    assert np.linalg.norm(pf.estimate()[0] - truth) < 0.1
    with pytest.raises(ValueError, match="height="):
        ParticleFilter(10, anchors=anchors, random_state=0).initialize([5.0, 5.0]).correct_ranges(r)
    with pytest.raises(ValueError, match="one time per row"):
        ParticleFilter(10, random_state=0).filter(np.zeros((5, 2)), t=[0.0, 1.0])
