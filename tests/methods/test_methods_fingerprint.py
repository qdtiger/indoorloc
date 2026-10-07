"""L3 fingerprinting methods: sklearn bridge, Horus, GP radio map, ensembles, hierarchy.

Known-result tests (closed forms, textbook identities, exact recoveries) on synthetic data;
the UJIIndoorLoc check runs only when the files are present.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys

import numpy as np
import pytest

from conftest import PROJECT, UJI_ROOT
from indoorloc.core import Prediction, load_model
from indoorloc.methods import BaseLocalizer, LocalizerPipeline, create_model
from indoorloc.methods.ensemble import EnsembleLocalizer, StackingLocalizer, simplex_weights, weighted_median
from indoorloc.methods.gaussian_process import JITTER, GPRadioMapLocalizer, _sq_dist, gp_log_marginal
from indoorloc.methods.hierarchical import HierarchicalLocalizer
from indoorloc.methods.neighbors import KNNLocalizer, WKNNLocalizer
from indoorloc.methods.probabilistic import HorusLocalizer, centre_of_mass, fixed_sum
from indoorloc.signals import FillMissing


def _path_loss_site(seed, repeats=20, floors=(0, 1), noise=3.0, drop_below=-95.0, n_ap=8, noise_seed=None):
    """Reference grid x floors, ``n_ap`` APs, log-distance path loss + 12 dB per floor, integer dBm,
    NaN below the detection threshold. Returns X, pos, floor, and the noise-free map."""
    rng = np.random.default_rng(seed)
    grid = np.stack(np.meshgrid(np.arange(0.0, 24.0, 4.0), np.arange(0.0, 16.0, 4.0)), -1).reshape(-1, 2)
    aps = rng.uniform(0, 24, (n_ap, 2))
    ap_floor = rng.integers(0, len(floors), n_ap)
    ref_pos = np.repeat(grid, len(floors), axis=0)
    ref_floor = np.tile(np.asarray(floors), len(grid))
    dist = np.linalg.norm(ref_pos[:, None] - aps[None], axis=-1) + 1.0
    clean = -35.0 - 30.0 * np.log10(dist) - 12.0 * np.abs(ref_floor[:, None] - ap_floor[None])
    idx = np.repeat(np.arange(len(ref_pos)), repeats)
    rng = rng if noise_seed is None else np.random.default_rng(noise_seed)
    X = np.round(clean[idx] + rng.normal(0, noise, (len(idx), n_ap)))
    X[X < drop_below] = np.nan
    return X, ref_pos[idx], ref_floor[idx], clean, ref_pos, ref_floor


class _Fixed(BaseLocalizer):
    """Stub member: the same position, floor and spread for every scan."""

    def __init__(self, pos=(0.0, 0.0), floor=None, spread=None):
        self.pos = pos
        self.floor = floor
        self.spread = spread

    def _fit(self, X, pos, floor, building):
        pass

    def _localize(self, X):
        n = len(X)
        return Prediction(np.tile(np.asarray(self.pos, float), (n, 1)),
                          None if self.floor is None else np.full(n, self.floor),
                          spread=None if self.spread is None else np.full(n, float(self.spread)))


class _TrainMean(BaseLocalizer):
    """Stub member: always predicts the mean training position."""

    def _fit(self, X, pos, floor, building):
        self.mean_ = pos.mean(axis=0)

    def _localize(self, X):
        return Prediction(np.repeat(self.mean_[None], len(X), axis=0))


# --------------------------------------------------------------------------- sklearn bridge
def test_forest_matches_sklearn_and_does_not_depend_on_n_jobs():
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    X, pos, floor, *_ = _path_loss_site(0, repeats=4)
    X = np.nan_to_num(X, nan=-104.0)
    params = dict(n_estimators=15, max_features="sqrt", random_state=3)
    model = create_model("rf", **params).fit(X, pos, floor=floor)
    ref = RandomForestRegressor(**params).fit(X, pos)
    p = model.localize(X[:40])
    assert np.array_equal(p.pos, ref.predict(X[:40]))  # trees averaged in index order = sklearn with n_jobs=1
    assert np.array_equal(p.floor, RandomForestClassifier(**params).fit(X, floor).predict(X[:40]))
    per_tree = np.stack([t.predict(X[:40].astype(np.float32)) for t in ref.estimators_])
    assert np.allclose(p.spread, np.sqrt(per_tree.var(axis=0).sum(axis=1)), rtol=1e-12, atol=1e-12)
    parallel = create_model("rf", n_jobs=2, **params).fit(X, pos, floor=floor).localize(X[:40])
    assert np.array_equal(parallel.pos, p.pos) and np.array_equal(parallel.spread, p.spread)


def test_fully_grown_extra_trees_interpolate_their_training_set():
    """Textbook property: without bootstrap, fully grown trees fit distinct training rows exactly."""
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(1)
    X, pos = rng.normal(size=(60, 5)), rng.uniform(0, 50, (60, 2)) + [4.0e5, 4.8e6]
    model = create_model("extratrees", n_estimators=7, max_features=None).fit(X, pos)
    assert model.bootstrap is False
    assert np.allclose(model.predict(X), pos, rtol=0, atol=1e-6) and np.all(model.localize(X).spread < 1e-6)


def test_svm_recovers_a_linear_map_and_is_translation_equivariant():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, (80, 3))
    pos = np.column_stack([2 * X[:, 0] - X[:, 1] + 5, X[:, 2] - 3])
    linear = create_model("svm", kernel="linear", C=100.0, epsilon=1e-4).fit(X, pos)
    assert np.abs(linear.predict(X) - pos).max() < 1e-2  # SVR with a linear kernel is exact on a linear target
    rbf = create_model("svm").fit(X, pos)
    shifted = create_model("svm").fit(X, pos + [4.0e5, 4.8e6])  # Mercator-sized offsets
    # equal up to libsvm's stopping tolerance (1e-3 in standardized units): targets are z-scored
    assert np.abs(shifted.predict(X) - [4.0e5, 4.8e6] - rbf.predict(X)).max() < 1e-3 * pos.std(axis=0).max()


def test_gradient_boosting_takes_raw_nan_scans_and_matches_sklearn():
    pytest.importorskip("sklearn")
    from sklearn.ensemble import HistGradientBoostingRegressor

    X, pos, floor, *_ = _path_loss_site(3, repeats=5)
    assert np.isnan(X).any()
    model = create_model("gbdt", max_iter=20).fit(X, pos, floor=floor)
    ref = [HistGradientBoostingRegressor(max_iter=20, early_stopping=False, random_state=0).fit(X, pos[:, d])
           for d in range(2)]
    assert np.array_equal(model.predict(X), np.column_stack([r.predict(X) for r in ref]))
    assert model.localize(X).floor.dtype == np.int64 and model.n_iter_ == 20


def test_a_single_label_needs_no_classifier():
    pytest.importorskip("sklearn")
    X, pos, *_ = _path_loss_site(4, repeats=2, floors=(3,))
    model = create_model("rf", n_estimators=3).fit(np.nan_to_num(X, nan=-104), pos, floor=np.full(len(X), 3),
                                                   building=np.full(len(X), -1))
    p = model.localize(np.nan_to_num(X[:5], nan=-104))
    assert model.floor_model_ is None and p.floor.tolist() == [3] * 5 and p.building.tolist() == [-1] * 5


@pytest.mark.parametrize("name", ["svm", "rf", "extratrees", "gbdt"])
def test_sklearn_models_save_and_load_without_pickle(tmp_path, name):
    pytest.importorskip("sklearn")
    X, pos, floor, *_ = _path_loss_site(5, repeats=3)
    X = np.nan_to_num(X, nan=-104.0)
    params = {"svm": {}, "rf": {"n_estimators": 5}, "extratrees": {"n_estimators": 5}, "gbdt": {"max_iter": 10}}
    model = create_model(name, **params[name]).fit(X, pos, floor=floor)
    path = model.save(tmp_path / name)
    assert sorted(p.name for p in path.iterdir()) == ["arrays.npz", "config.json"]
    loaded = load_model(path)
    a, b = model.localize(X), loaded.localize(X)
    assert np.array_equal(a.pos, b.pos) and np.array_equal(a.floor, b.floor)
    assert (a.spread is None) == (b.spread is None) and (a.spread is None or np.array_equal(a.spread, b.spread))
    text = (path / "config.json").read_text()
    target = json.loads(text)["object"]["state"]["regressors_"][0]["__dict__"]["__sklearn__"]
    (path / "config.json").write_text(text.replace(target, "os:system"))
    with pytest.raises(ValueError, match="only scikit-learn classes"):
        load_model(path)  # nothing outside sklearn is ever imported from a model file


def test_modules_import_without_sklearn_and_name_the_extra():
    code = ("import sys\nsys.modules['sklearn'] = None\nimport numpy as np\n"
            "import indoorloc.methods.sklearn_wrap, indoorloc.methods.probabilistic, indoorloc.methods.ensemble\n"
            "import indoorloc.methods.gaussian_process, indoorloc.methods.hierarchical\n"
            "from indoorloc.methods import create_model\n"
            "m = create_model('rf')\n"
            "try:\n    m.fit(np.zeros((4, 2)), np.zeros((4, 2)))\nexcept ImportError as e:\n"
            "    assert \"indoorloc[sklearn]\" in str(e), e\nelse:\n    raise AssertionError('expected ImportError')\n"
            "create_model('ensemble', localizers=['knn', 'horus']).fit(np.eye(6), np.eye(6)[:, :2])\n")
    subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT, check=True)


# --------------------------------------------------------------------------- Horus
def _horus_toy():
    nan = np.nan
    X = np.array([[-50, -80, nan], [-52, nan, nan], [-54, -84, nan], [-56, nan, nan],  # location A = (0, 0)
                  [-70, nan, -60], [-74, nan, -60]])                                    # location B = (10, 0)
    pos = np.array([[0.0, 0.0]] * 4 + [[10.0, 0.0]] * 2)
    return X, pos


def test_horus_log_likelihood_matches_a_hand_computation():
    X, pos = _horus_toy()
    model = HorusLocalizer(min_std=1.0, detection_smoothing=0.5, unseen_std=8.0).fit(X, pos, floor=[1, 1, 1, 1, 2, 2])
    lg = lambda x, m, s: -0.5 * math.log(2 * math.pi) - math.log(s) - 0.5 * ((x - m) / s) ** 2  # noqa: E731
    floor_dbm = -84.0  # weakest heard training reading: the mean for an AP never heard at a location
    # AP0 is heard in every training scan: no detection term. Smoothing 0.5 for AP1 and AP2.
    # A: AP0 mean -53, std sqrt 5; AP1 heard 2/4 (mean -82, std 2), missing; AP2 heard 0/4
    ll_a = lg(-55, -53, math.sqrt(5)) + math.log(1 - 2.5 / 5) + math.log(0.5 / 5) + lg(-61, floor_dbm, 8.0)
    # B: AP0 mean -72, std 2; AP1 heard 0/2, missing; AP2 heard 2/2 (std 0 -> min_std 1)
    ll_b = lg(-55, -72, 2.0) + math.log(1 - 0.5 / 3) + math.log(2.5 / 3) + lg(-61, -60, 1.0)
    q = np.array([[-55.0, np.nan, -61.0]])
    assert model.detection_modelled_.tolist() == [False, True, True]
    assert model.locations_.tolist() == [[0.0, 0.0], [10.0, 0.0]] and model.location_floor_.tolist() == [1, 2]
    assert np.allclose(model.log_likelihood(q), [[ll_a, ll_b]], rtol=0, atol=1e-12)
    # Horus's continuous-space estimate: posterior-weighted centre of mass of the top 2
    w_b = 1.0 / (1.0 + math.exp(ll_a - ll_b))
    two = model.set_params(n_candidates=2).localize(q)
    assert two.pos[0] == pytest.approx([10.0 * w_b, 0.0], abs=1e-12)
    assert two.floor.tolist() == [1 if w_b < 0.5 else 2]
    assert two.spread[0] == pytest.approx(10.0 * math.sqrt(w_b * (1 - w_b)), abs=1e-12)


def test_horus_scores_missing_readings_explicitly():
    X, pos = _horus_toy()
    model = HorusLocalizer(n_candidates=1, min_std=1.0).fit(X, pos)
    heard = np.array([[-62.0, np.nan, -60.0]])   # AP0 halfway; AP2 is only ever heard at B
    silent = np.array([[-62.0, np.nan, np.nan]])  # AP2 absent: evidence for A (never heard there)
    assert model.predict(heard).tolist() == [[10.0, 0.0]] and model.predict(silent).tolist() == [[0.0, 0.0]]
    nothing = np.full((1, 3), np.nan)  # a scan that hears nothing scores sum_a log(1 - pi)
    assert np.allclose(model.log_likelihood(nothing)[0], model.log_miss_.sum(axis=1), atol=1e-12)


def test_horus_on_complete_input_is_the_classic_gaussian_model():
    """No NaN: log P(s | L) = sum_a log N(s_a; mu_La, sigma_La) exactly, and a location's scan count
    does not matter (smoothed detection terms would favour the location with 40 scans over 2)."""
    few = np.array([[-60.0, -70.0], [-64.0, -74.0]])  # location A: mean (-62, -72), population std 2
    many = np.tile(few, (20, 1))                       # location B: same mean and std from 40 scans
    X, pos = np.vstack([few, many]), np.array([[0.0, 0.0]] * 2 + [[5.0, 0.0]] * 40)
    model = HorusLocalizer(min_std=1.0).fit(X, pos)
    assert model.location_counts_.tolist() == [2, 40] and not model.detection_modelled_.any()
    q = np.array([[-62.0, -72.0], [-59.0, -75.5]])
    ll = model.log_likelihood(q)
    assert np.array_equal(ll[:, 0], ll[:, 1])  # identical models score identically
    lg = lambda x, m, s: -0.5 * math.log(2 * math.pi) - math.log(s) - 0.5 * ((x - m) / s) ** 2  # noqa: E731
    assert ll[1, 0] == pytest.approx(lg(-59, -62, 2.0) + lg(-75.5, -72, 2.0), abs=1e-12)
    assert model.set_params(n_candidates=1).predict(q[:1]).tolist() == [[0.0, 0.0]]  # the tie goes to index 0
    site_X, site_pos, *_ = _path_loss_site(21, repeats=3)
    filled = np.nan_to_num(site_X, nan=-104.0)
    fitted = HorusLocalizer().fit(filled, site_pos)
    classic = np.stack([(-0.5 * math.log(2 * math.pi) - np.log(fitted.std_)
                         - 0.5 * ((x - fitted.mean_) / fitted.std_) ** 2).sum(axis=1) for x in filled[:9]])
    assert np.allclose(fitted.log_likelihood(filled[:9]), classic, rtol=0, atol=1e-9)


def test_horus_is_the_better_classifier_when_its_gaussian_model_holds():
    """Gaussian reading noise around a path-loss map: the per-location maximum-likelihood
    decision recovers the reference point more often than 1-NN on the same scans."""
    X, pos, floor, *_ = _path_loss_site(6, n_ap=16)
    Xq, pos_q, floor_q, *_ = _path_loss_site(6, repeats=3, n_ap=16, noise_seed=99)  # same site, fresh noise
    p = HorusLocalizer(n_candidates=1).fit(X, pos, floor=floor).localize(Xq)
    nn = KNNLocalizer(k=1).fit(np.nan_to_num(X, nan=-104), pos).predict(np.nan_to_num(Xq, nan=-104))
    horus_exact, nn_exact = np.all(p.pos == pos_q, axis=1).mean(), np.all(nn == pos_q, axis=1).mean()
    assert horus_exact > 0.93 and horus_exact > nn_exact + 0.05 and np.array_equal(p.floor, floor_q)


def test_likelihood_scores_do_not_depend_on_batching():
    X, pos, floor, *_ = _path_loss_site(8, repeats=4)
    filled = np.nan_to_num(X, nan=-104)
    one_floor = floor == 0  # the GP maps position only: fit it per floor
    for model, Xq in ((HorusLocalizer().fit(X, pos, floor=floor), X[::7]),
                      (GPRadioMapLocalizer().fit(filled[one_floor], pos[one_floor]), filled[::7])):
        batch = model.log_likelihood(Xq)
        rows = np.concatenate([model.log_likelihood(Xq[i:i + 1]) for i in range(len(Xq))])
        assert np.array_equal(batch, rows)


def test_fixed_sum_and_centre_of_mass_helpers():
    a = np.random.default_rng(9).normal(size=(4, 13))
    assert np.allclose(fixed_sum(a), a.sum(axis=1)) and fixed_sum(np.zeros((3, 0))).tolist() == [0, 0, 0]
    ll = np.array([[0.0, 0.0, -1.0]])  # a tie between candidates 0 and 1: the lower index ranks first
    p = centre_of_mass(ll, np.array([[0.0], [1.0], [2.0]]), floors=np.array([5, 4, 4]), n=1)
    assert p.pos.tolist() == [[0.0]] and p.floor.tolist() == [5]
    p3 = centre_of_mass(ll, np.array([[0.0], [1.0], [2.0]]), floors=np.array([5, 4, 4]), n=3)
    w = np.array([1.0, 1.0, math.exp(-1.0)]) / (2 + math.exp(-1.0))
    assert p3.pos[0, 0] == pytest.approx(w[1] + 2 * w[2]) and p3.floor.tolist() == [4]


# --------------------------------------------------------------------------- Gaussian process
def test_gp_profile_likelihood_equals_the_raw_gaussian_density():
    """Repeated readings enter through their means and scatter: same value as the full model."""
    rng = np.random.default_rng(10)
    refs, m = rng.uniform(0, 10, (6, 2)), np.array([1, 3, 2, 4, 1, 2])
    idx = np.repeat(np.arange(6), m)
    Y = rng.normal(size=(len(idx), 3)) * 4 - 70
    means = np.stack([Y[idx == i].mean(0) for i in range(6)])
    within = sum(((Y[idx == i] - means[i]) ** 2).sum(0) for i in range(6))
    length, ratio = 2.5, 0.3
    ll, s = gp_log_marginal(_sq_dist(refs, refs), m.astype(float), means - Y.mean(0), within, length, ratio)
    R = np.exp(-_sq_dist(refs[idx], refs[idx]) / (2 * length ** 2)) + JITTER * (idx[:, None] == idx[None])
    for a in range(3):
        C = s[a] * (R + ratio * np.eye(len(idx)))
        r = Y[:, a] - Y[:, a].mean()
        raw = -0.5 * (len(idx) * math.log(2 * math.pi) + np.linalg.slogdet(C)[1] + r @ np.linalg.solve(C, r))
        assert ll[a] == pytest.approx(raw, abs=1e-9)
        for t in (0.9, 1.1):  # the profiled signal variance is the maximiser
            Ct = C * t
            assert raw > -0.5 * (len(idx) * math.log(2 * math.pi) + np.linalg.slogdet(Ct)[1]
                                 + r @ np.linalg.solve(Ct, r))


def test_gp_radio_map_equals_textbook_prediction_on_raw_readings():
    """Rasmussen & Williams, Algorithm 2.1, run on every raw reading (duplicated inputs)."""
    rng = np.random.default_rng(11)
    refs, m = rng.uniform(0, 10, (5, 2)), np.array([2, 1, 3, 2, 2])
    idx = np.repeat(np.arange(5), m)
    Y = rng.normal(size=(len(idx), 2)) * 3 - 75
    gp = GPRadioMapLocalizer(length_scales=[3.0], noise_ratios=[0.2]).fit(Y, refs[idx])
    Xs = rng.uniform(0, 10, (4, 2))
    mean, var = gp.radio_map(Xs)
    R = np.exp(-_sq_dist(refs[idx], refs[idx]) / 18.0) + JITTER * (idx[:, None] == idx[None])
    for a in range(2):
        s, c = gp.signal_var_[a], Y[:, a].mean()
        K, k_star = s * (R + 0.2 * np.eye(len(idx))), s * np.exp(-_sq_dist(refs[idx], Xs) / 18.0)
        mu = c + k_star.T @ np.linalg.solve(K, Y[:, a] - c)
        v = s - np.einsum("ij,ij->j", k_star, np.linalg.solve(K, k_star)) + 0.2 * s  # + noise of a new reading
        assert np.allclose(mean[:, a], mu, atol=1e-9) and np.allclose(var[:, a], v, atol=1e-9)


def test_gp_marginal_likelihood_recovers_the_true_hyperparameters():
    rng = np.random.default_rng(3)
    P = rng.uniform(0, 20, (80, 2))
    K = 25.0 * (np.exp(-_sq_dist(P, P) / (2 * 3.0 ** 2)) + 0.01 * np.eye(80))
    Y = -70 + np.linalg.cholesky(K) @ rng.normal(size=(80, 40))  # 40 APs drawn from one GP
    gp = GPRadioMapLocalizer(length_scales=np.geomspace(0.5, 20, 25)).fit(Y, P)
    assert np.median(gp.length_scale_) == pytest.approx(3.0, rel=0.1)
    assert np.median(gp.noise_ratio_) == pytest.approx(0.01)
    assert np.median(gp.signal_var_) == pytest.approx(25, rel=0.25)


def test_gp_localizes_a_noise_free_map_exactly_and_interpolates_it():
    grid = np.stack(np.meshgrid(np.arange(0, 30, 3.0), np.arange(0, 20, 3.0)), -1).reshape(-1, 2)
    aps = np.array([[0, 0], [30, 0], [0, 20], [30, 20], [15, 10]])
    field = lambda P: -40 - 25 * np.log10(np.linalg.norm(P[:, None] - aps[None], axis=-1) + 1)  # noqa: E731
    gp = GPRadioMapLocalizer().fit(field(grid), grid)
    assert np.array_equal(gp.predict(field(grid)), grid)  # maximum likelihood picks the true reference point
    mid = np.array([[13.5, 7.5], [22.5, 10.5]])
    mean, var = gp.radio_map(mid)
    err = np.abs(mean - field(mid))
    assert err.max() < 1.0 and np.all(err < 3 * np.sqrt(var))  # the predictive std covers the interpolation error
    dense = GPRadioMapLocalizer(candidates=np.vstack([grid, mid])).fit(field(grid), grid)
    assert np.allclose(dense.predict(field(mid)), mid)  # off-grid candidates are found from the map


def test_gp_without_informative_features_ties_every_candidate():
    gp = GPRadioMapLocalizer().fit(np.full((3, 2), -104.0), [[5.0, 1.0], [0.0, 2.0], [0.0, 1.0]])
    assert gp.active_.tolist() == [False, False]
    assert gp.predict(np.array([[-60.0, -104.0]])).tolist() == [[0.0, 1.0]]  # first candidate in sorted order


def test_gp_warns_when_positions_mix_floors():
    grid = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
    X = np.array([[-50.0, -70.0], [-60.0, -60.0], [-70.0, -50.0]])
    with pytest.warns(UserWarning, match="HierarchicalLocalizer"):
        GPRadioMapLocalizer().fit(np.vstack([X, X - 10]), np.vstack([grid, grid]), floor=[0, 0, 0, 1, 1, 1])


# --------------------------------------------------------------------------- ensembles
def test_weighted_median_is_numpys_median_for_equal_weights():
    v = np.random.default_rng(12).normal(size=(6, 5, 2))
    assert np.allclose(weighted_median(v, np.ones(6)), np.median(v, axis=0))
    assert np.allclose(weighted_median(v[:5], np.ones(5)), np.median(v[:5], axis=0))
    assert weighted_median(np.array([[0.0], [1.0], [10.0]]), np.array([1.0, 1.0, 3.0])).tolist() == [10.0]


def test_ensemble_combinations_in_closed_form():
    X, y = np.zeros((3, 2)), np.zeros((3, 2))
    members = [_Fixed((0.0, 0.0), floor=1, spread=1.0), _Fixed((3.0, 0.0), floor=2, spread=2.0),
               _Fixed((30.0, 0.0), floor=2, spread=4.0)]
    mean = EnsembleLocalizer(members, weights=[2, 1, 1]).fit(X, y).localize(X[:1])
    assert mean.pos.tolist() == [[33.0 / 4, 0.0]] and mean.floor.tolist() == [1]  # 2 votes each: smallest label
    median = EnsembleLocalizer(members, combine="median").fit(X, y).localize(X[:1])
    assert median.pos.tolist() == [[3.0, 0.0]]  # robust to the outlying member
    inv = EnsembleLocalizer(members, combine="inverse_spread").fit(X, y).localize(X[:1])
    w = np.array([1.0, 1 / 4, 1 / 16])
    assert inv.pos[0, 0] == pytest.approx((w @ [0.0, 3.0, 30.0]) / w.sum())
    tie = EnsembleLocalizer(members[:2]).fit(X, y).localize(X[:1])
    assert tie.floor.tolist() == [1] and tie.spread[0] == pytest.approx(1.5)  # vote tie: smallest label
    exact = EnsembleLocalizer([_Fixed((0.0, 0.0), spread=0.0), members[1]], combine="inverse_spread")
    assert exact.fit(X, y).localize(X[:1]).pos.tolist() == [[0.0, 0.0]]  # zero spread takes all the weight
    with pytest.raises(ValueError, match="spread"):
        EnsembleLocalizer([_Fixed(), _TrainMean()], combine="inverse_spread").fit(X, y).predict(X)
    with pytest.raises(ValueError, match="at least one"):
        EnsembleLocalizer().fit(X, y)


def test_simplex_weights_recover_a_known_convex_mixture():
    rng = np.random.default_rng(13)
    P = rng.normal(size=(3, 50, 2)) * 5 + [4.0e5, 4.8e6]  # large coordinate offsets
    y = 0.7 * P[0] + 0.3 * P[2]
    assert np.allclose(simplex_weights(P, y), [0.7, 0.0, 0.3], atol=1e-6)
    y_out = 1.6 * P[0] - 0.6 * P[1]  # the unconstrained optimum leaves the simplex
    w = simplex_weights(P, y_out)
    grid = [(a, b, 1 - a - b) for a in np.linspace(0, 1, 101) for b in np.linspace(0, 1, 101) if a + b <= 1 + 1e-12]
    rss = lambda v: np.square(y_out - np.einsum("m,mnd->nd", np.asarray(v), P)).sum()  # noqa: E731
    assert w.min() >= 0 and w.sum() == pytest.approx(1) and rss(w) <= min(rss(v) for v in grid) + 1e-6


def test_stacking_uses_out_of_fold_predictions():
    """Positions unrelated to the features: 1-NN is perfect in sample and useless out of fold, so
    an in-sample stack would trust it fully; the out-of-fold stack gives it (almost) no weight."""
    rng = np.random.default_rng(14)
    X = rng.normal(size=(120, 4))
    y = rng.normal(size=(120, 2)) * 5 + [100.0, 200.0]
    stack = StackingLocalizer([KNNLocalizer(k=1), _TrainMean()], cv=4).fit(X, y)
    assert stack.weights_[0] < 0.2 and stack.member_errors_[0] > stack.member_errors_[1]
    assert np.allclose(stack.predict(X[:3]), np.einsum("m,mnd->nd", stack.weights_,
                                                        np.stack([m.predict(X[:3]) for m in stack.localizers_])))
    fold = np.empty(30, dtype=int)
    for f, (train, test) in enumerate(stack._folds(np.repeat(y[:10], 3, axis=0), None)):
        fold[test] = f
        assert len(np.intersect1d(train, test)) == 0
    assert all(len(set(fold[3 * i:3 * i + 3])) == 1 for i in range(10))  # repeated scans of a position share a fold
    grouped = stack._folds(y, groups=np.arange(120) // 30)
    assert sorted(len(t) for _, t in grouped) == [30] * 4


class _Halves:
    """A splitter object: first half / second half."""

    def split(self, X, y, groups=None):
        idx = np.arange(len(X))
        yield idx[len(X) // 2:], idx[:len(X) // 2]
        yield idx[:len(X) // 2], idx[len(X) // 2:]


def test_stacking_accepts_a_splitter_and_its_out_of_fold_errors_are_exact():
    rng = np.random.default_rng(22)
    X, y = rng.normal(size=(10, 3)), rng.uniform(0, 10, (10, 2))
    stack = StackingLocalizer([_TrainMean()], cv=_Halves()).fit(X, y)
    first, second = y[:5].mean(axis=0), y[5:].mean(axis=0)  # each half is predicted by the other half's mean
    oof = np.concatenate([np.linalg.norm(y[:5] - second, axis=1), np.linalg.norm(y[5:] - first, axis=1)])
    assert stack.member_errors_[0] == pytest.approx(oof.mean(), abs=1e-12)
    assert stack.weights_.tolist() == [1.0] and np.allclose(stack.predict(X[:2]), y.mean(axis=0))
    for bad in ("wknn", KNNLocalizer()):  # one model instead of a list: a clear error, not "unknown method 'w'"
        with pytest.raises(TypeError, match="sequence of localizers"):
            StackingLocalizer(bad).fit(X, y)
        with pytest.raises(TypeError, match="sequence of localizers"):
            EnsembleLocalizer(bad).fit(X, y)


def test_numpy_models_save_and_load_without_pickle(tmp_path):
    X, pos, floor, *_ = _path_loss_site(23, repeats=3)
    one = floor == 0
    filled = np.nan_to_num(X, nan=-104.0)
    cand = np.array([[1.0, 1.0], [5.0, 3.0], [9.5, 7.0]])
    for model, Xm, posm, fl in ((HorusLocalizer(), X, pos, floor),  # raw NaN scans
                                (GPRadioMapLocalizer(n_candidates=2, candidates=cand), filled[one], pos[one], None)):
        model.fit(Xm, posm, floor=fl)
        path = model.save(tmp_path / type(model).__name__)
        assert sorted(p.name for p in path.iterdir()) == ["arrays.npz", "config.json"]
        loaded = load_model(path)
        a, b = model.localize(Xm[:25]), loaded.localize(Xm[:25])
        assert np.array_equal(a.pos, b.pos) and np.array_equal(a.spread, b.spread)
        assert (a.floor is None and b.floor is None) or np.array_equal(a.floor, b.floor)
        assert np.array_equal(model.log_likelihood(Xm[:5]), loaded.log_likelihood(Xm[:5]))
    assert np.array_equal(loaded.candidates, cand)
    assert np.array_equal(loaded.radio_map(cand)[0], model.radio_map(cand)[0])


def test_stacking_with_a_custom_meta_learner_and_votes():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import Ridge

    X, pos, floor, *_ = _path_loss_site(15, repeats=3)
    X = np.nan_to_num(X, nan=-104.0)
    stack = StackingLocalizer(["knn", "wknn"], final_estimator=Ridge(alpha=1.0), cv=3).fit(X, pos, floor=floor)
    p = stack.localize(X[:10])
    assert p.pos.shape == (10, 2) and p.floor is not None and p.spread is None
    assert np.allclose(stack.weights_, 0.5)


def test_meta_models_take_missing_readings_only_if_every_member_does():
    X, pos, floor, *_ = _path_loss_site(20, repeats=3)
    assert np.isnan(X).any()
    with pytest.raises(ValueError, match="FillMissing"):
        EnsembleLocalizer(["knn", "horus"]).fit(X, pos)  # 'knn' cannot take NaN: refused up front
    ok = EnsembleLocalizer([LocalizerPipeline(FillMissing(-104), KNNLocalizer()), "horus"]).fit(X, pos, floor=floor)
    assert np.isfinite(ok.predict(X[:5])).all()
    assert HierarchicalLocalizer("horus", "horus", "horus").fit(X, pos, floor=floor).localize(X[:5]).floor is not None
    with pytest.raises(ValueError, match="FillMissing"):
        HierarchicalLocalizer().fit(X, pos, floor=floor)
    assert not StackingLocalizer(["horus", "knn"])._allow_nan and StackingLocalizer(["horus", "gbdt"])._allow_nan


def test_ensembles_save_load_and_clone(tmp_path):
    X, pos, floor, *_ = _path_loss_site(16, repeats=3)
    X = np.nan_to_num(X, nan=-104.0)
    for model in (EnsembleLocalizer(["knn", WKNNLocalizer(k=3), "horus"], combine="median"),
                  StackingLocalizer(["knn", "horus"], cv=3), HierarchicalLocalizer(position_model="horus")):
        model.fit(X, pos, floor=floor)
        loaded = load_model(model.save(tmp_path / type(model).__name__))
        a, b = model.localize(X[:30]), loaded.localize(X[:30])
        assert np.array_equal(a.pos, b.pos) and np.array_equal(a.floor, b.floor)
    sklearn = pytest.importorskip("sklearn")
    twin = sklearn.base.clone(EnsembleLocalizer(["knn", KNNLocalizer(k=2)], weights=[1, 2]))
    assert twin.localizers[0] == "knn" and twin.localizers[1].k == 2


# --------------------------------------------------------------------------- hierarchy
def _two_buildings(seed):
    X0, pos0, f0, *_ = _path_loss_site(seed, repeats=4)
    X1, pos1, f1, *_ = _path_loss_site(seed + 1, repeats=4)
    X = np.nan_to_num(np.vstack([np.hstack([X0, np.full_like(X0, np.nan)]),   # each building hears its own APs
                                 np.hstack([np.full_like(X1, np.nan), X1])]), nan=-104.0)
    pos = np.vstack([pos0, pos1 + [100.0, 0.0]])
    return X, pos, np.concatenate([f0, f1]), np.repeat([0, 1], [len(X0), len(X1)])


def test_hierarchy_equals_per_group_models_when_routing_is_right():
    X, pos, floor, building = _two_buildings(17)
    model = HierarchicalLocalizer(position_model=WKNNLocalizer(k=3)).fit(X, pos, floor=floor, building=building)
    assert model.levels_ == ("building", "floor") and model.position_keys_.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    p = model.localize(X)
    assert np.array_equal(p.building, building) and np.array_equal(p.floor, floor)
    for b in (0, 1):
        for f in (0, 1):
            rows = (building == b) & (floor == f)
            alone = WKNNLocalizer(k=3).fit(X[rows], pos[rows]).localize(X[rows])
            assert np.array_equal(p.pos[rows], alone.pos) and np.array_equal(p.spread[rows], alone.spread)


def test_hierarchy_levels_can_be_removed_or_be_classifiers():
    X, pos, floor, building = _two_buildings(18)
    no_floor = HierarchicalLocalizer(floor_model=None).fit(X, pos, floor=floor, building=building)
    assert no_floor.levels_ == ("building",) and len(no_floor.position_models_) == 2
    assert no_floor.localize(X).floor is not None  # passed through from the per-building position models
    flat = HierarchicalLocalizer().fit(X, pos)  # no labels: a single position model
    assert flat.levels_ == () and flat.localize(X[:2]).floor is None
    one = HierarchicalLocalizer().fit(X[building == 0], pos[building == 0], floor=floor[building == 0],
                                      building=building[building == 0])
    assert one.building_stage_[0] is None and one.localize(X[:3]).building.tolist() == [0, 0, 0]
    with pytest.raises(ValueError, match=r"group \{'building': 0, 'floor': 0\}"):
        HierarchicalLocalizer(position_model=KNNLocalizer(k=10_000)).fit(X, pos, floor=floor, building=building)
    pytest.importorskip("sklearn")
    from sklearn.neighbors import KNeighborsClassifier

    clf = HierarchicalLocalizer(building_model=KNeighborsClassifier(3), floor_model="rf")
    p = clf.fit(X, pos, floor=floor, building=building).localize(X)
    assert (p.building == building).mean() > 0.99 and (p.floor == floor).mean() > 0.9


@pytest.mark.filterwarnings("ignore:some positions carry several floor labels")  # gp_radiomap on two floors
def test_every_registry_name_builds_a_working_model():
    pytest.importorskip("sklearn")
    X, pos, floor, *_ = _path_loss_site(19, repeats=3)
    X = np.nan_to_num(X, nan=-104.0)
    for name, params in (("svm", {}), ("rf", {"n_estimators": 5}), ("random_forest", {"n_estimators": 5}),
                         ("extratrees", {"n_estimators": 5}), ("gbdt", {"max_iter": 5}), ("horus", {}),
                         ("gp_radiomap", {}), ("ensemble", {"localizers": ["knn", "horus"]}),
                         ("stacking", {"localizers": ["knn", "horus"], "cv": 3}), ("hierarchical", {})):
        p = create_model(name, **params).fit(X, pos, floor=floor).localize(X[:4])
        assert p.pos.shape == (4, 2) and np.all(np.isfinite(p.pos)), name


# --------------------------------------------------------------------------- real data
@pytest.mark.skipif(not (UJI_ROOT / "trainingData.csv").is_file(), reason="UJIIndoorLoc files not found")
def test_horus_on_ujiindoorloc_validation():
    from indoorloc.datasets import load_dataset

    train, test = load_dataset("ujiindoorloc", root=UJI_ROOT, download=False)
    result = HorusLocalizer().fit(train).evaluate(test)  # raw NaN scans: missing readings are modelled
    assert result.n == 1111
    # reproduced by this library (deterministic; sha256-verified files): 7.994 m, 1024 and 1110 of 1111 right
    assert result.mean_error == pytest.approx(7.994, abs=5e-4)
    assert result.floor_accuracy == pytest.approx(100 * 1024 / 1111) and result.building_accuracy == pytest.approx(
        100 * 1110 / 1111)
