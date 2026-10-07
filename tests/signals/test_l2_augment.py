"""L2 augmentations: seeded, training-only, NaN-preserving."""
from __future__ import annotations

import numpy as np
import pytest

from indoorloc.core import SampleTable, clone, load_model
from indoorloc.signals import APDropout, Compose, FillMissing, GaussianNoise, WiFiSignal

NAN = np.nan


def _heard(n=2000, f=50, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(-100, -30, size=(n, f)).astype(np.float32)
    X[rng.random(X.shape) < 0.3] = NAN
    return X


def test_gaussian_noise_statistics_and_missing_readings():
    X = _heard()
    out = GaussianNoise(std_db=2.0, random_state=0)(X)
    heard = ~np.isnan(X)
    assert out.dtype == np.float32 and np.array_equal(np.isnan(out), ~heard)  # NaN stays NaN, nothing new
    noise = (out - X)[heard].astype(np.float64)
    assert abs(noise.mean()) < 0.02 and abs(noise.std() - 2.0) < 0.02  # 70k draws: se(mean) = 0.008


def test_ap_dropout_rate_and_only_heard_readings_are_dropped():
    X = _heard()
    out = APDropout(p=0.25, random_state=0)(X)
    heard = ~np.isnan(X)
    assert np.isnan(out[~heard]).all() and np.array_equal(out[~np.isnan(out)], X[~np.isnan(out)])
    rate = np.isnan(out[heard]).mean()
    assert abs(rate - 0.25) < 0.01  # 70k Bernoulli draws: se = 0.0016
    assert np.array_equal(APDropout(0.0)(X), X, equal_nan=True)
    assert np.isnan(APDropout(1.0)(X)).all()
    with pytest.raises(ValueError, match=r"p must be in \[0, 1\]"):
        APDropout(1.5)(X)


def test_seeded_stream_is_reproducible_and_advances():
    X = _heard(10, 5)
    a, b = GaussianNoise(1.0, random_state=7), GaussianNoise(1.0, random_state=7)
    first, second = a(X), a(X)
    assert not np.array_equal(first, second, equal_nan=True)  # a new draw on every call (epochs differ)
    assert np.array_equal(b(X), first, equal_nan=True) and np.array_equal(b(X), second, equal_nan=True)
    assert np.array_equal(clone(a)(X), first, equal_nan=True)  # clone restarts the stream
    gen = np.random.default_rng(3)
    assert np.array_equal(GaussianNoise(1.0, random_state=gen)(X), GaussianNoise(1.0, random_state=3)(X),
                          equal_nan=True)


def test_inference_passes_data_through_and_containers_are_kept():
    X = _heard(4, 3)
    aug = GaussianNoise(3.0, random_state=0).fit(X)
    assert aug.transform(X) is X  # unseen data is never perturbed
    table = SampleTable(X, np.zeros((4, 2)), meta={"feature_names": ("a", "b", "c")})
    out = aug(table)
    assert isinstance(out, SampleTable) and out.meta == table.meta and not np.array_equal(out.X, X, equal_nan=True)
    view = APDropout(0.5, random_state=0)(WiFiSignal(X[0], ("a", "b", "c")))
    assert isinstance(view, WiFiSignal) and view.ap_ids == ("a", "b", "c")
    with pytest.raises(ValueError, match="Complex"):
        aug.augment(np.ones((2, 3), np.complex64))


def test_pipeline_augments_training_data_only():
    from indoorloc.methods import LocalizerPipeline, create_model

    X = np.array([[-40.0, -70.0], [-70.0, -40.0], [-60.0, -60.0]])
    y = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 5.0]])
    pre = Compose([GaussianNoise(0.5, random_state=0), FillMissing()])
    trained = pre.fit_transform(X)
    assert not np.array_equal(trained, X)  # fit_transform returns the augmented training set
    assert np.array_equal(pre.transform(X), X)  # inference chain: no noise
    model = LocalizerPipeline(pre, create_model("knn", k=1)).fit(X, y)
    assert np.array_equal(model.preprocess_.transform(X), X)  # what localize applies to queries: no noise
    assert np.array_equal(model.localize(X).pos, y)


def test_save_and_load_restart_the_stream(tmp_path):
    X = _heard(6, 4)
    aug = APDropout(0.3, random_state=11).fit(X)
    first = aug(X)
    loaded = load_model(aug.save(tmp_path / "a"))
    assert loaded.p == 0.3 and loaded.n_features_in_ == 4
    assert np.array_equal(loaded(X), first, equal_nan=True)
