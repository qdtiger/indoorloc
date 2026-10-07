from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import PROJECT
from indoorloc.core import NotFittedError, Prediction, load_model
from indoorloc.methods import BaseLocalizer, LocalizerPipeline, create_model, list_models
from indoorloc.methods.neighbors import KNNLocalizer, WKNNLocalizer, kneighbors, sq_distances
from indoorloc.signals import Compose, FillMissing, RSSINormalize, Transform


def _dbm_fingerprints(seed, n_fit=600, n_query=60, n_ap=520):
    """Integer-dBm scans drawn from 40 distinct fingerprints (plus all-missing rows): exact ties everywhere."""
    rng = np.random.default_rng(seed)
    distinct = np.where(rng.random((40, n_ap)) < 0.03, rng.integers(-100, -30, (40, n_ap)), -104)
    distinct[0] = -104  # an "empty scan", as in UJIIndoorLoc
    X_fit = distinct[rng.integers(0, 40, n_fit)].astype(np.float32)
    X = distinct[rng.integers(0, 40, n_query)].astype(np.float32)
    X[::7, rng.integers(0, n_ap)] = -60  # perturbed queries: ties between distinct fingerprints
    return X_fit, X


def _scaled(seed):
    """The same data after (x + 104) / 104 in float32: non-dyadic values, order-sensitive sums."""
    X_fit, X = _dbm_fingerprints(seed)
    return (X_fit + 104) / np.float32(104), (X + 104) / np.float32(104)


def _lexsort_oracle(X_fit, X, k, dist_of):
    d = np.stack([dist_of(X_fit, q) for q in X])
    return np.stack([np.lexsort((np.arange(len(X_fit)), row))[:k] for row in d]), d


def test_integer_features_match_the_exact_integer_oracle():
    X_fit, X = _dbm_fingerprints(0)
    exact = lambda A, q: ((A.astype(np.int64) - q.astype(np.int64)) ** 2).sum(1)  # noqa: E731
    expected, d = _lexsort_oracle(X_fit, X, 7, exact)
    dist, idx = kneighbors(X_fit, X, k=7, chunk_size=5)
    assert np.array_equal(idx, expected)
    assert np.array_equal(dist, np.sqrt(np.take_along_axis(d, expected, 1)))


def test_float_features_follow_the_documented_order():
    X_fit, X = _scaled(1)
    expected, _ = _lexsort_oracle(X_fit, X, 5, lambda A, q: sq_distances(A.astype(np.float64), q.astype(np.float64)))
    assert np.array_equal(kneighbors(X_fit, X, 5, chunk_size=3)[1], expected)


@pytest.mark.parametrize("n_features", [1, 2])
def test_few_features_keep_the_order_under_heavy_cancellation(n_features):
    """Adversarial for the GEMM prefilter, whose margin is tightest relative to its rounding bound
    at small F: coordinates near 2**30 (|q|^2 + |t|^2 ~ 2**61 cancels to the distance), exact ties
    from mirrored and swapped offsets, and near-ties a few ulps apart."""
    rng = np.random.default_rng(n_features)
    n, eps = 64, np.finfo(np.float64).eps
    q = rng.integers(-2 ** 30, 2 ** 30, (n, n_features)).astype(np.float64)
    off = rng.integers(1, 2 ** 30, (n, n_features)).astype(np.float64)
    near = q + off * (1 + rng.integers(1, 4, (n, 1)) * eps)
    X_fit = np.concatenate([q + off, q - off, near, q - off[:, ::-1]])
    for k in (1, 2, 3, 4):
        expected, d = _lexsort_oracle(X_fit, q, k, sq_distances)
        assert np.array_equal(kneighbors(X_fit, q, k, chunk_size=16)[1], expected)
    first_two = np.sort(d, axis=1)[:, :2]
    assert np.any(first_two[:, 0] == first_two[:, 1])  # exact ties at the top are exercised


def test_neighbours_independent_of_thread_count():
    """Fresh interpreters with 1 and 8 BLAS/OpenMP threads (no threadpoolctl needed)."""
    code = ("import sys, hashlib, numpy as np; sys.path[:0] = [{p!r}, {t!r}]\n"
            "from test_methods import _scaled\nfrom indoorloc.methods.neighbors import kneighbors\n"
            "X_fit, X = _scaled(2)\nprint(hashlib.sha256(kneighbors(X_fit, X, 5)[1].tobytes()).hexdigest())")
    code = code.format(p=str(PROJECT), t=str(PROJECT / "tests"))
    digests = set()
    for threads in ("1", "8"):
        env = {**os.environ, **{v: threads for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
        run = subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True, check=True)
        digests.add(run.stdout.strip())
    assert len(digests) == 1


def test_distance_weights_label_votes_and_spread():
    X = np.array([[0.0, 0], [1, 0], [0, 1]])
    y = np.array([[0.0, 0], [10, 0], [0, 10]])
    exact = WKNNLocalizer(k=3).fit(X, y, floor=[2, 1, 1]).localize(X[:1])
    assert exact.pos.tolist() == [[0.0, 0.0]] and exact.floor.tolist() == [2] and exact.spread.tolist() == [0.0]
    uniform = KNNLocalizer(k=3).fit(X, y, floor=[2, 1, 1]).localize(X[:1])
    assert uniform.floor.tolist() == [1]
    assert uniform.spread[0] == pytest.approx(np.sqrt(np.square(y - y.mean(0)).sum(1).mean()))


def test_input_rules_are_explicit():
    model = KNNLocalizer(k=1)
    with pytest.raises(NotFittedError):
        model.predict(np.zeros((1, 2)))
    with pytest.raises(ValueError, match="FillMissing"):
        model.fit(np.array([[np.nan, 1.0]]), np.array([[0.0, 0.0]]))
    model.fit(np.zeros((2, 3)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match=r"x\[None, :\]"):
        model.predict(np.zeros(3))  # the first axis is always samples
    with pytest.raises(ValueError, match="expecting 3 features"):
        model.predict(np.zeros((1, 4)))
    with pytest.raises(ValueError, match="Complex data"):
        KNNLocalizer(k=1).fit(np.ones((2, 3), np.complex64), np.zeros((2, 2)))
    csi = np.random.default_rng(0).random((10, 2, 3, 4))  # (N, rx, tx, subcarrier) amplitudes
    assert KNNLocalizer(k=2).fit(csi, np.zeros((10, 2))).predict(csi[:3]).shape == (3, 2)
    assert KNNLocalizer(k=1).fit(np.eye(3), [0.0, 1.0, 2.0]).predict(np.eye(3)).tolist() == [0.0, 1.0, 2.0]


def test_a_failed_fit_leaves_the_model_unfitted():
    model = KNNLocalizer(k=1)
    for X, y in ((np.array([[np.nan, 1.0]]), np.zeros((1, 2))), (np.zeros((1, 2)), None),
                 (np.zeros((1, 2)), np.full((1, 2), np.inf))):
        with pytest.raises(ValueError):
            model.fit(X, y)
        with pytest.raises(NotFittedError):  # not an AttributeError from half-set state
            model.predict(np.zeros((1, 2)))
    compose = Compose([FillMissing(), RSSINormalize(lo=None)])
    with pytest.raises(ValueError):
        compose.fit(np.zeros((0, 2)))
    assert not hasattr(compose, "n_features_in_")


def test_fit_params_reach_the_subclass_hook():
    class DomainAware(BaseLocalizer):
        def _fit(self, X, pos, floor, building, sample_domain=None):
            self.domains_ = np.unique(sample_domain)
            self.mean_ = pos.mean(0)

        def _localize(self, X):
            return Prediction(np.repeat(self.mean_[None], len(X), 0))

    m = DomainAware().fit(np.zeros((4, 2)), np.ones((4, 2)), sample_domain=[1, 1, -2, -2])
    assert m.domains_.tolist() == [-2, 1] and m.predict(np.zeros((2, 2))).tolist() == [[1, 1], [1, 1]]
    with pytest.raises(TypeError):
        KNNLocalizer(k=1).fit(np.zeros((2, 2)), np.zeros((2, 2)), sample_domain=[0, 1])


def test_pipeline_owns_preprocessing_and_keeps_floor(tmp_path):
    X = np.array([[-40.0, np.nan], [np.nan, -40.0], [-70.0, -70.0]])
    y = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 5.0]])
    model = create_model("wknn", k=1, preprocess=FillMissing(-104))
    assert isinstance(model, LocalizerPipeline)
    p = model.fit(X, y, floor=[0, 1, 2]).localize(X)
    assert p.pos.tolist() == y.tolist() and p.floor.tolist() == [0, 1, 2]
    loaded = load_model(model.save(tmp_path / "m", info={"data_sha256": "abc"}))
    assert repr(loaded) == repr(model) and loaded.saved_info_ == {"data_sha256": "abc"}
    assert np.array_equal(loaded.localize(X).pos, p.pos) and loaded.localize(X).floor.tolist() == [0, 1, 2]
    assert not any(f.suffix == ".pkl" for f in (tmp_path / "m").iterdir())


class _Shift(Transform):  # learns from extra data at fit time, like a CORAL(target=) adapter
    def fit(self, X, y=None, *, target=None):
        super().fit(X, y)
        self.shift_ = float(np.mean(target)) - float(np.mean(X))
        return self

    def _transform(self, x):
        return x + self.shift_


class _Recorder(BaseLocalizer):  # stands in for DeepLocalizer, whose early stopping reads eval_set
    def _fit(self, X, pos, floor, building, eval_set=None):
        self.eval_set_ = eval_set
        self.mean_ = pos.mean(0)

    def _localize(self, X):
        return Prediction(np.repeat(self.mean_[None], len(X), 0))


def test_pipeline_preprocesses_eval_set_and_routes_preprocess_params():
    X, y = np.array([[-40.0, np.nan], [np.nan, -60.0]]), np.zeros((2, 2))
    pipe = LocalizerPipeline(FillMissing(-104), _Recorder()).fit(X, y, eval_set=[(np.array([[np.nan, -70.0]]), y[:1])])
    (X_val, y_val), = pipe.localizer_.eval_set_
    assert X_val.tolist() == [[-104.0, -70.0]] and y_val.tolist() == [[0.0, 0.0]]  # filled like X, not raw NaN
    shifted = LocalizerPipeline(_Shift(), _Recorder()).fit(np.array([[1.0], [3.0]]), y, preprocess__target=[[10.0]])
    assert shifted.preprocess_.shift_ == 8.0
    with pytest.raises(TypeError, match="target"):  # a transform that takes no target says so
        LocalizerPipeline(FillMissing(), _Recorder()).fit(X, y, preprocess__target=[[10.0]])
    with pytest.raises(TypeError, match="no preprocess step"):
        LocalizerPipeline(None, _Recorder()).fit(np.ones((2, 2)), y, preprocess__target=[[10.0]])


def test_factory_and_sklearn_contract():
    pytest.importorskip("sklearn")
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV, cross_val_score

    assert {"knn", "wknn"} <= set(list_models())
    model = create_model("WKNN", k=3)
    twin = clone(model)
    assert twin is not model and twin.get_params() == model.get_params() == {
        "chunk_size": 256, "k": 3, "weights": "distance"}
    X_fit, _ = _dbm_fingerprints(3)
    X_fit[X_fit == -104] = np.nan  # raw tables: NaN = not heard
    y = np.random.default_rng(0).normal(size=(len(X_fit), 2))
    pipe = LocalizerPipeline(FillMissing(), model)
    assert np.all(cross_val_score(pipe, X_fit, y, cv=3) < 0)  # score = minus the mean error
    search = GridSearchCV(pipe, {"localizer__k": [1, 4]}, cv=3).fit(X_fit, y)
    assert search.best_estimator_.localizer_.k in (1, 4)


class _Net:  # stands in for a torch module: not an array, cannot go into arrays.npz as is
    def __init__(self, w):
        self.w = w


class _NetLocalizer(BaseLocalizer):
    def _fit(self, X, pos, floor, building):
        self.net_ = _Net(pos.mean(0))

    def _localize(self, X):
        return Prediction(np.repeat(self.net_.w[None], len(X), 0))

    def _get_state(self):  # swap the module for its weights (a deep model saves its state_dict)
        state = super()._get_state()
        state["weights_"] = state.pop("net_").w
        return state

    def _set_state(self, state):
        state = dict(state)
        self.net_ = _Net(state.pop("weights_"))
        super()._set_state(state)


def test_state_hooks_let_non_array_state_be_saved(tmp_path):
    model = _NetLocalizer().fit(np.zeros((3, 2)), np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 3.0]]))
    path = model.save(tmp_path / "net")
    with pytest.raises(ValueError, match="trusted_modules"):
        load_model(path)  # a class from outside indoorloc and the registries needs explicit trust
    again = load_model(path, trusted_modules=[_NetLocalizer.__module__])
    assert np.array_equal(again.predict(np.zeros((1, 2))), [[2.0, 1.0]])


def test_load_model_checks_the_module_before_importing_it(tmp_path, monkeypatch):
    (tmp_path / "side_effect.py").write_text(
        "import pathlib\npathlib.Path(__file__).with_suffix('.ran').touch()\nclass Thing:\n    pass\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "side_effect", raising=False)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "format": "indoorloc-estimator", "format_version": 1, "info": {"__dict__": {}},
        "object": {"__estimator__": "side_effect:Thing", "params": {}, "state": {}}}))
    np.savez(model / "arrays.npz")
    with pytest.raises(ValueError, match="refusing to import 'side_effect'"):
        load_model(model)
    assert not (tmp_path / "side_effect.ran").exists() and "side_effect" not in sys.modules  # nothing ran
    with pytest.raises(TypeError, match="not an indoorloc Estimator"):
        load_model(model, trusted_modules=["side_effect"])  # trusted: imported, then still refused
    assert (tmp_path / "side_effect.ran").exists()
    monkeypatch.delitem(sys.modules, "side_effect", raising=False)


def test_metadata_routing_passes_floor_only_when_requested():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.model_selection import GridSearchCV, GroupKFold

    X_fit, _ = _dbm_fingerprints(4, n_fit=90, n_ap=20)
    rng = np.random.default_rng(4)
    y, floor, groups = rng.normal(size=(90, 2)), rng.integers(0, 3, 90), np.arange(90) % 3
    with sklearn.config_context(enable_metadata_routing=True):
        search = GridSearchCV(create_model("wknn").set_fit_request(floor=True), {"k": [1, 3]},
                              cv=GroupKFold(n_splits=3)).fit(X_fit, y, groups=groups, floor=floor)
        assert search.best_estimator_.localize(X_fit[:2]).floor is not None
        with pytest.raises(Exception, match="floor"):  # not requested: sklearn refuses rather than drop it
            GridSearchCV(create_model("wknn"), {"k": [1, 3]}, cv=3).fit(X_fit, y, floor=floor)


def test_tracker_filters_a_stream_of_single_scans():
    from indoorloc.apps.tracking import ConstantVelocityKF, track

    X = np.array([[-40.0, -80.0], [-80.0, -40.0]])
    model = create_model("wknn", k=2).fit(X, np.array([[0.0, 0.0], [10.0, 0.0]]))
    out = list(track(model, ConstantVelocityKF(), [(0.0, X[0]), (1.0, X[0]), (2.0, X[1])]))
    assert len(out) == 3 and np.array_equal(out[0][1], out[0][2])  # the first fix initialises the state
    assert 0.0 < out[2][2][0] < out[2][1][0]  # later fixes are smoothed towards the track
