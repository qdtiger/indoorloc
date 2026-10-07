"""L5 streaming: OnlineLocalizer over (t, scan) streams with missing scans and a tracker."""
from __future__ import annotations

import numpy as np
import pytest

from indoorloc.apps.particle import ParticleFilter
from indoorloc.apps.streaming import OnlineLocalizer, stack_estimates
from indoorloc.apps.tracking import KalmanTracker
from indoorloc.core import Prediction
from indoorloc.evaluation import evaluate
from indoorloc.methods import create_model
from indoorloc.signals import FillMissing


def _rssi(rng, points, aps):
    d = np.linalg.norm(points[:, None] - aps[None], axis=-1) + 1.0
    rssi = -40.0 - 25.0 * np.log10(d) + rng.normal(0.0, 4.0, d.shape)
    return np.where(rssi < -95.0, np.nan, np.round(rssi)).astype(np.float32)


@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(0)
    aps = rng.uniform([0, 0], [60, 30], (40, 2))
    grid = np.stack(np.meshgrid(np.arange(0, 60.1, 1.5), np.arange(0, 30.1, 1.5)), -1).reshape(-1, 2)
    model = create_model("wknn", k=4, preprocess=FillMissing(-100.0)).fit(_rssi(rng, grid, aps), grid)
    t = np.arange(121, dtype=float)
    truth = np.stack([5 + 50 * t / 120, 5 + 20 * np.sin(np.pi * t / 120)], axis=1)
    scans = _rssi(rng, truth, aps)
    stream = [(tk, scan) for tk, scan in zip(t, scans)]
    stream[30] = (30.0, None)  # a lost scan
    stream[31] = (31.0, np.full(40, np.nan, np.float32))  # nothing heard
    one = np.full(40, np.nan, np.float32)
    one[0] = -60.0
    stream[32] = (32.0, one)  # a single reading
    return model, truth, stream


def test_stream_with_a_tracker_equals_the_offline_filter_of_the_same_fixes(scene):
    model, truth, stream = scene
    online = OnlineLocalizer(model, KalmanTracker(process_noise=0.3), min_readings=3)
    out = list(online.run(stream))
    assert len(out) == len(stream) and [e.missing for e in out[29:34]] == [False, True, True, True, False]
    assert np.isnan(out[31].fix).all() and np.isfinite(out[31].pos).all()  # the tracker carried the track
    ok = ~np.array([e.missing for e in out])
    fixes = np.full((len(stream), 2), np.nan)
    spread = np.full(len(stream), np.nan)
    pred = model.localize(np.stack([s for (_, s), m in zip(stream, ok) if m]))
    fixes[ok], spread[ok] = pred.pos, pred.spread
    offline = KalmanTracker(process_noise=0.3).filter(Prediction(fixes, spread=spread), [t for t, _ in stream])
    t, tracked = stack_estimates(out)
    assert np.allclose(tracked.pos, offline.pos) and np.allclose(t, np.arange(121))
    raw_err = evaluate(truth[ok], np.stack([e.fix for e in out])[ok]).mean_error
    assert evaluate(truth, tracked).mean_error < raw_err  # tracking helps on this walk
    stats = online.latency_stats()
    assert stats["n"] == stats["n_scans"] == 121 and stats["n_missing"] == 3
    assert 0 < stats["p50_ms"] <= stats["p95_ms"] <= stats["max_ms"]


def test_without_a_tracker_missing_scans_hold_the_last_fix(scene):
    model, _, stream = scene
    out = list(OnlineLocalizer(model, min_readings=3).run(stream[:35]))
    assert np.allclose(out[30].pos, out[29].pos) and np.allclose(out[32].pos, out[29].pos)
    assert np.allclose(out[33].pos, out[33].fix)
    first = OnlineLocalizer(model).update(0.0, None)
    assert first.missing and np.isnan(first.pos).all()


def test_a_particle_filter_can_be_the_tracker(scene):
    model, truth, stream = scene
    online = OnlineLocalizer(model, ParticleFilter(500, motion_std=0.8, random_state=0), min_readings=3)
    _, pred = stack_estimates(online.run(stream))
    again = stack_estimates(online.run(stream))[1]  # run() resets: the same seed gives the same track
    assert np.array_equal(pred.pos, again.pos) and np.isfinite(pred.pos).all()


def test_floor_is_a_majority_vote_over_the_window():
    class Floors:
        def __init__(self, labels):
            self.labels = iter(labels)

        def localize(self, X):
            return Prediction(np.zeros((1, 2)), floor=[next(self.labels)], spread=[1.0])

    labels = [0, 0, 1, 0, 1, 1, 1]
    out = OnlineLocalizer(Floors(labels), floor_window=3).run((float(i), np.zeros(3)) for i in range(7))
    assert [e.floor for e in out] == [0, 0, 0, 0, 1, 1, 1]
    tie = OnlineLocalizer(Floors([2, 5]), floor_window=2).run((float(i), np.zeros(3)) for i in range(2))
    assert [e.floor for e in tie] == [2, 5]  # a tie goes to the most recent label


def test_the_tracker_parameter_is_a_template_and_a_saved_stream_resumes_exactly(scene, tmp_path):
    from indoorloc.core import load_model

    model, _, stream = scene
    template = KalmanTracker(process_noise=0.3)
    a, b = OnlineLocalizer(model, template, min_readings=3), OnlineLocalizer(model, template, min_readings=3)
    a.reset(), b.reset()
    first = a.update(*stream[0])
    b.update(stream[0][0], np.full(40, -50.0, np.float32))  # a different scan through the same template
    assert np.allclose(a.update(*stream[1]).pos, list(OnlineLocalizer(model, template, min_readings=3)
                                                      .run(stream[:2]))[1].pos)  # b did not disturb a
    assert not hasattr(template, "x_") and np.isfinite(first.pos).all()  # the parameter is never modified
    for tracker in (KalmanTracker(process_noise=0.3), ParticleFilter(300, motion_std=0.8, random_state=0)):
        full = [e.pos for e in OnlineLocalizer(model, tracker, min_readings=3, floor_window=3).run(stream)]
        online = OnlineLocalizer(model, tracker, min_readings=3, floor_window=3)
        head = [e.pos for e in online.run(stream[:60])]
        again = load_model(online.save(tmp_path / type(tracker).__name__))
        tail = [again.update(t, scan).pos for t, scan in stream[60:]]
        assert np.array_equal(np.array(head + tail), np.array(full))
