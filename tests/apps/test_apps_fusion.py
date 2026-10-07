"""L5 fusion: PDR steps + noisy fixes in a map-constrained particle filter."""
from __future__ import annotations

import numpy as np
import pytest

from indoorloc.apps.fusion import PDRFusion
from indoorloc.apps.maps import FloorMap
from indoorloc.apps.particle import ParticleFilter
from indoorloc.core import Prediction


def _ring_corridor():
    """A 40 x 20 m floor with a 24 x 4 m inner block: a 3-7 m wide corridor loop around it."""
    outer = [(0, 0), (40, 0), (40, 20), (0, 20)]
    inner = [(8, 8), (32, 8), (32, 12), (8, 12)]
    return FloorMap.from_polygons([outer, inner])


def _loop_walk(seed=0, heading_offset=0.6, length_scale=1.05, fix_every=4, fix_std=3.0):
    corners = np.array([[5.0, 5.0], [35.0, 5.0], [35.0, 15.0], [5.0, 15.0], [5.0, 5.0]])
    rng = np.random.default_rng(seed)
    pts, heads = [corners[0]], []
    for a, b in zip(corners[:-1], corners[1:]):
        d = b - a
        n = int(round(np.linalg.norm(d) / 0.7))
        h = np.arctan2(d[1], d[0])
        for i in range(1, n + 1):
            pts.append(a + d * i / n)
            heads.append(h)
    truth = np.array(pts[1:])
    step_len = np.linalg.norm(np.diff(np.array(pts), axis=0), axis=1)
    heads = np.array(heads)
    t = 0.5 * np.arange(1, len(truth) + 1)
    # PDR as a phone would report it: heading in the gyro frame (unknown offset), a scale error, noise
    pdr_head = heads + heading_offset + rng.normal(0, 0.03, len(heads))
    pdr_len = step_len * length_scale + rng.normal(0, 0.03, len(heads))
    fix_idx = np.arange(fix_every - 1, len(truth), fix_every)
    fixes = truth[fix_idx] + rng.normal(0, fix_std, (len(fix_idx), 2))
    return t, truth, pdr_len, pdr_head, t[fix_idx], Prediction(fixes, spread=np.full(len(fix_idx), fix_std)), fix_idx


def _rmse(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def test_fusion_beats_both_pdr_and_the_fixes_and_learns_the_heading_offset():
    t, truth, L, H, fix_t, fixes, fix_idx = _loop_walk()
    pf = ParticleFilter(1500, step_length_std=0.08, heading_std=0.04, heading_drift_std=0.005,
                        floor_map=_ring_corridor(), random_state=0)
    fusion = PDRFusion(pf, heading_bias_std=None)  # the PDR heading offset is unknown
    times, est = fusion.run((t, L, H), fixes, fix_t)
    assert len(times) == len(t) + len(fix_t) and np.all(np.diff(times) >= 0)
    # the estimate after each step (a step comes before a fix at the same time)
    fused = est.pos[np.searchsorted(times, t, side="left")]
    started = np.isfinite(fused).all(axis=1)
    assert started.sum() >= len(t) - 4  # the cloud starts at the first fix (step 4)
    pdr_only = truth[0] - [0.7, 0.0] + np.cumsum(np.stack([L * np.cos(H), L * np.sin(H)], 1), 0)
    e_fused = _rmse(fused[started][10:], truth[started][10:])
    e_fix = _rmse(fixes.pos, truth[fix_idx])
    e_pdr = _rmse(pdr_only, truth)
    assert e_fused < 0.6 * e_fix and e_fused < 0.3 * e_pdr
    offset = np.median(fusion.filter_.heading_bias_)
    assert abs(np.angle(np.exp(1j * (offset + 0.6)))) < 0.15  # the cloud learned heading - 0.6 rad


def test_fusion_is_deterministic_and_can_start_from_a_known_point():
    t, truth, L, H, fix_t, fixes, _ = _loop_walk(seed=1, heading_offset=0.0)
    pf = ParticleFilter(400, step_length_std=0.08, heading_std=0.04, floor_map=_ring_corridor(), random_state=5)
    run = lambda: PDRFusion(pf, heading_bias_std=0.05).run((t, L, H), fixes, fix_t, start=[5.0, 5.0],  # noqa: E731
                                                           start_std=0.3, t0=0.0)[1].pos
    a, b = run(), run()
    assert np.array_equal(a, b) and np.isfinite(a).all()  # started before the first step
    assert np.linalg.norm(a[0] - (truth[0])) < 0.6


def test_online_updates_and_fixes_from_a_localizer():
    class Echo:  # a stand-in localizer: the "scan" is the position itself, spread 2 m
        def localize(self, X):
            return Prediction(np.asarray(X, dtype=float), spread=np.full(len(X), 2.0))

    t, truth, L, H, fix_t, fixes, _ = _loop_walk(seed=2, heading_offset=0.0)
    pf = ParticleFilter(300, random_state=0)
    by_scans = PDRFusion(pf, localizer=Echo(), heading_bias_std=0.05).run((t, L, H), np.asarray(fixes.pos), fix_t)[1]
    same = Prediction(fixes.pos, spread=np.full(len(fix_t), 2.0))
    by_fixes = PDRFusion(pf, heading_bias_std=0.05).run((t, L, H), same, fix_t)[1]
    assert np.allclose(by_scans.pos, by_fixes.pos, equal_nan=True)
    fusion = PDRFusion(pf).reset()
    assert np.isnan(fusion.update(0.1, step=(0.7, 0.0))).all()  # no fix yet: nothing to move
    fusion.update(0.5, fix=[5.0, 5.0], spread=1.0)
    moved = fusion.update(1.0, step=(0.7, 0.0))
    assert np.isfinite(moved).all()
    with pytest.raises(ValueError, match="one time per fix"):
        PDRFusion(pf).run((t, L, H), fixes, fix_t[:-1])


def test_random_state_seeds_the_default_filter_and_a_saved_fusion_resumes(tmp_path):
    from indoorloc.core import load_model

    t, truth, L, H, fix_t, fixes, _ = _loop_walk(seed=3, heading_offset=0.0)
    a = PDRFusion(random_state=0).run((t, L, H), fixes, fix_t)[1].pos
    b = PDRFusion(random_state=0).run((t, L, H), fixes, fix_t)[1].pos
    c = PDRFusion(random_state=1).run((t, L, H), fixes, fix_t)[1].pos
    assert np.array_equal(a, b, equal_nan=True) and not np.array_equal(a, c, equal_nan=True)
    pf = ParticleFilter(200, random_state=9)
    assert PDRFusion(pf, random_state=4).reset().filter_.random_state == 4 and pf.random_state == 9
    fusion = PDRFusion(ParticleFilter(200, heading_std=0.05), random_state=2).reset(0.0, start=truth[0], start_std=0.3)
    events = [(tk, (lk, hk)) for tk, lk, hk in zip(t, L, H)]
    head = [fusion.update(tk, step=s) for tk, s in events[:20]]
    again = load_model(fusion.save(tmp_path / "fusion"))
    tail = [again.update(tk, step=s) for tk, s in events[20:]]
    twin = PDRFusion(ParticleFilter(200, heading_std=0.05), random_state=2).reset(0.0, start=truth[0], start_std=0.3)
    assert np.array_equal(np.array(head + tail), np.array([twin.update(tk, step=s) for tk, s in events]))
