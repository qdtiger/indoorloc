"""L2 RSSI transforms: APFilter, APSelect, the Torres-Sospedra representations, HampelFilter."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import UJI_ROOT
from indoorloc.core import NotFittedError, SampleTable, clone, load_model
from indoorloc.signals import (APFilter, APSelect, BLESignal, Compose, ExponentialRepresentation, HampelFilter,
                               PositiveRepresentation, PowedRepresentation, WiFiSignal)
from indoorloc.signals import functional as F

NAN = np.nan
X3 = np.array([[-40.0, NAN, -70.0, -80.0],
               [-60.0, -50.0, -70.0, NAN],
               [-50.0, -52.0, -70.0, NAN]], dtype=np.float32)
NAMES = ("AP0", "AP1", "AP2", "AP3")


def _table(X=X3):
    return SampleTable(X, np.zeros((len(X), 2)), meta={"feature_names": NAMES, "modality": "wifi_rssi"})


@pytest.mark.parametrize("make", [lambda: APFilter(-75), lambda: PositiveRepresentation(min_dbm=-105),
                                  lambda: ExponentialRepresentation(min_dbm=-105),
                                  lambda: PowedRepresentation(min_dbm=-105), lambda: HampelFilter(1)])
def test_row_batch_table_and_views_agree(make):
    t = make()
    batch = t(X3)
    assert batch.dtype == np.float32 and batch.shape == X3.shape
    table = t(_table())
    assert isinstance(table, SampleTable) and np.array_equal(table.X, batch, equal_nan=True)
    for i, row in enumerate(X3):
        view = t(WiFiSignal(row, NAMES))
        assert isinstance(view, WiFiSignal) and view.ap_ids == NAMES
        assert np.array_equal(view.rssi, t(row), equal_nan=True)
        assert isinstance(t(BLESignal(row)), BLESignal)
        if not isinstance(t, HampelFilter):  # Hampel's window runs along the row
            assert np.array_equal(t(row), batch[i], equal_nan=True)


def test_apfilter_turns_weak_readings_into_nan_and_keeps_the_threshold():
    out = APFilter(threshold_dbm=-70)(X3)
    assert np.array_equal(out, [[-40, NAN, -70, NAN], [-60, -50, -70, NAN], [-50, -52, -70, NAN]], equal_nan=True)
    assert np.array_equal(APFilter(-70)(np.array([-71, -70, -69])), [NAN, -70, -69], equal_nan=True)  # ints -> float32
    with pytest.raises(ValueError, match="Complex"):
        APFilter().fit(np.ones((2, 3), np.complex64))


def test_apselect_strategies_pick_the_known_columns():
    # fill -104: variances 66.7, 624.9, 0, 128  | coverage 1, 2/3, 1, 1/3 | means -50, -68.7, -70, -96
    assert APSelect(2, "variance").fit(X3).indices_.tolist() == [1, 3]
    assert APSelect(2, "coverage").fit(X3).indices_.tolist() == [0, 2]
    assert APSelect(2, "strongest").fit(X3).indices_.tolist() == [0, 1]
    assert APSelect(1, "coverage").fit(X3).indices_.tolist() == [0]  # tie 0/2: lower index wins
    np.testing.assert_allclose(APSelect(1, "variance").fit(X3).scores_,
                               np.var(np.where(np.isnan(X3), -104.0, X3), axis=0))


def test_apselect_keeps_names_views_and_sklearn_helpers_aligned():
    sel = APSelect(2, "variance").fit(_table())
    out = sel(_table())
    assert out.X.shape == (3, 2) and out.meta["feature_names"] == ("AP1", "AP3") and out.meta["modality"] == "wifi_rssi"
    assert np.array_equal(out.X, X3[:, [1, 3]], equal_nan=True)
    view = sel(WiFiSignal(X3[1], NAMES))
    assert view.ap_ids == ("AP1", "AP3") and np.array_equal(view.rssi, [-50, NAN], equal_nan=True)
    assert sel(X3[0]).shape == (2,)
    assert sel.get_support().tolist() == [False, True, False, True]
    assert sel.get_feature_names_out(NAMES).tolist() == ["AP1", "AP3"]
    with pytest.raises(ValueError, match="expecting 4 features"):
        sel(X3[:, :3])
    with pytest.raises(NotFittedError):
        APSelect(2).transform(X3)
    with pytest.raises(ValueError, match="k must be"):
        APSelect(5).fit(X3)
    with pytest.raises(ValueError, match="strategy"):
        APSelect(1, "entropy").fit(X3)


def test_representations_match_the_closed_form_of_torres_sospedra_2015():
    X = np.array([[-104.0, -60.0, 0.0, NAN]])
    pos = PositiveRepresentation().fit(X)
    assert pos.min_dbm_ == -105.0  # min = lowest training reading - 1: the weakest reading maps to 1
    assert pos(X).tolist() == [[1.0, 45.0, 105.0, 0.0]]
    alpha, beta, mn = 24.0, np.e, -105.0
    positive = np.array([1.0, 45.0, 105.0, 0.0])
    np.testing.assert_allclose(ExponentialRepresentation().fit(X)(X)[0], np.exp(positive / alpha) / np.exp(-mn / alpha))
    np.testing.assert_allclose(PowedRepresentation().fit(X)(X)[0], positive ** beta / (-mn) ** beta)
    # 0 dBm maps to 1 in both normalised representations; not heard -> exp(min/alpha) and 0
    assert ExponentialRepresentation(min_dbm=-105)(np.array([0.0]))[0] == pytest.approx(1.0)
    assert ExponentialRepresentation(min_dbm=-105)(np.array([NAN]))[0] == pytest.approx(np.exp(-105 / 24))
    assert PowedRepresentation(min_dbm=-105)(np.array([0.0, NAN])).tolist() == [1.0, 0.0]
    # a test reading weaker than the training min counts as not heard
    assert pos(np.array([-110.0, -110.0, -110.0, -110.0])).tolist() == [0, 0, 0, 0] and not np.isnan(pos(X)).any()


def test_representation_min_comes_from_training_data_only():
    rep = PowedRepresentation(beta=2.0).fit(np.array([[-90.0, -40.0]]))
    assert rep.min_dbm_ == -91.0 and rep(np.array([-46.0, 0.0]))[0] == pytest.approx((45 / 91) ** 2)
    with pytest.raises(NotFittedError, match="min_dbm"):
        ExponentialRepresentation()(X3)
    with pytest.raises(ValueError, match="negative"):
        PowedRepresentation(min_dbm=5.0)(X3)


def test_hampel_replaces_a_spike_by_the_window_median():
    # window 2: at the spike the window is [2, 3, 100, 5, 6], median 5, MAD 2 -> 95 > 3 * 1.4826 * 2
    out, mask = F.hampel(np.array([1, 2, 3, 100, 5, 6, 7.0]), 2, 3.0, return_mask=True)
    assert out.tolist() == [1, 2, 3, 5, 5, 6, 7] and mask.tolist() == [0, 0, 0, 1, 0, 0, 0]
    assert F.hampel(np.array([1, 2, 3, 100, 5, 6, 7.0]), 2, 50.0).tolist() == [1, 2, 3, 100, 5, 6, 7]  # 95 < 50*2.97
    assert F.hampel(np.arange(10.0), 3).tolist() == list(range(10))  # a ramp has no outliers
    # NaN is ignored inside the window and stays NaN
    assert np.array_equal(F.hampel(np.array([1.0, NAN, 1.0, 9.0, 1.0, 1.0]), 2), [1, NAN, 1, 1, 1, 1], equal_nan=True)


def test_hampel_axis_and_trajectories():
    X = np.zeros((7, 3), np.float32)
    X[3, 1] = 50.0  # a spike in time (axis 0) for AP1
    assert HampelFilter(2, axis=0)(X).max() == 0.0
    assert HampelFilter(2, axis=-1)(X)[3, 1] == 0.0  # across APs [0, 50, 0] the spike is also an outlier
    assert np.array_equal(HampelFilter(2, axis=0)(X[3]), X[3])  # one sample: nothing to compare along time
    column = np.array([[0.0]] * 6 + [[5.0]])
    table = SampleTable(column, np.zeros((7, 2)), groups={"trajectory": np.array([0] * 6 + [1])})
    assert HampelFilter(3, axis=0)(column)[-1, 0] == 0.0  # one long series: 5 looks like an outlier
    assert HampelFilter(3, axis=0)(table).X[-1, 0] == 5.0  # but it is the only sample of trajectory 1


def test_compose_and_persistence(tmp_path):
    pipe = Compose([APFilter(-75), APSelect(2, "coverage"), PositiveRepresentation()]).fit(_table())
    out = pipe(_table())
    assert out.meta["feature_names"] == ("AP0", "AP2") and out.X.tolist() == [[31, 1], [11, 1], [21, 1]]
    loaded = load_model(pipe.save(tmp_path / "p"))
    assert repr(loaded) == repr(pipe) and np.array_equal(loaded(_table()).X, out.X)
    assert loaded.transforms[1].indices_.tolist() == [0, 2] and loaded.transforms[2].min_dbm_ == -71.0
    twin = clone(pipe)
    assert not hasattr(twin.transforms[1], "indices_")


@pytest.mark.skipif(not (UJI_ROOT / "trainingData.csv").is_file(), reason="UJIIndoorLoc files not found")
def test_ujiindoorloc_min_is_minus_105_as_in_the_paper():
    from indoorloc.datasets import load_dataset

    train = load_dataset("ujiindoorloc", split="train", root=UJI_ROOT, download=False)
    pos = PositiveRepresentation().fit(train)
    assert pos.min_dbm_ == -105.0
    sel = APSelect(200, "coverage").fit(train)
    kept = sel(train)
    assert kept.X.shape == (19937, 200) and len(kept.meta["feature_names"]) == 200
    assert np.isnan(train.X).all(axis=0).sum() > 0 and not np.isnan(kept.X).all(axis=0).any()


@pytest.mark.skipif(not (UJI_ROOT / "trainingData.csv").is_file(), reason="UJIIndoorLoc files not found")
def test_ujiindoorloc_nonlinear_representations_beat_positive_with_1nn():
    """The paper's qualitative finding (Torres-Sospedra et al. 2015): with Euclidean 1-NN on
    UJIIndoorLoc, the exponential and powed representations beat the positive one. Measured
    here (validation set, mean error in EPSG:3857 units): 9.850 / 8.578 / 9.093."""
    from indoorloc.datasets import load_dataset
    from indoorloc.methods import create_model
    from indoorloc.signals import FillMissing

    train, test = load_dataset("ujiindoorloc", root=UJI_ROOT, download=False)
    err = {name: create_model("knn", k=1, preprocess=pre).fit(train).evaluate(test).mean_error
           for name, pre in [("dbm", FillMissing(-105.0)), ("positive", PositiveRepresentation()),
                             ("exponential", ExponentialRepresentation()), ("powed", PowedRepresentation())]}
    assert err["positive"] == pytest.approx(err["dbm"], abs=1e-9)  # a shift: Euclidean k-NN is unchanged
    assert err["exponential"] < err["positive"] - 0.5 and err["powed"] < err["positive"] - 0.3
    # this library's numbers, reproduced by an independent CSV parser and brute-force 1-NN
    assert (err["positive"], err["exponential"], err["powed"]) == pytest.approx((9.850, 8.578, 9.093), abs=2e-3)


@pytest.mark.filterwarnings("ignore:Estimator .* does not inherit from:UserWarning")
def test_sklearn_contract():
    pytest.importorskip("sklearn")
    from sklearn.base import clone as sk_clone
    from sklearn.pipeline import make_pipeline
    from sklearn.utils.estimator_checks import check_estimator

    X = np.where(np.isnan(X3), -104.0, X3)
    pipe = make_pipeline(APSelect(2, "variance"), ExponentialRepresentation())
    assert pipe.fit_transform(X).shape == (3, 2) and sk_clone(pipe).get_params()["apselect__k"] == 2
    by_design = {"check_fit1d", "check_fit2d_predict1d"}  # a 1-D array is one scan, not an error
    for est in (APSelect(k=1), PositiveRepresentation(), HampelFilter()):
        results = check_estimator(est, on_fail=None)
        failed = {r["check_name"] for r in results if r["status"] == "failed"}
        assert failed <= by_design, (type(est).__name__, failed)


def test_normalized_and_exponential_identities_of_the_paper():
    """Torres-Sospedra et al. (2015): Normalized = Positive / (-min), which is FillMissing(min) followed by
    min-max scaling on [min, 0]; Exponential of a heard reading r is exp(r / alpha) whatever min is."""
    X = np.array([[-104.0, -60.0, -1.0, NAN], [-90.0, NAN, -30.0, -70.0]])
    mn = -105.0
    normalized = PositiveRepresentation(min_dbm=mn)(X) / -mn
    from indoorloc.signals import FillMissing, RSSINormalize
    np.testing.assert_allclose(Compose([FillMissing(mn), RSSINormalize(lo=mn, hi=0.0, clip=True)])(X), normalized,
                               rtol=0, atol=1e-15)
    heard = ~np.isnan(X)
    for m in (-105.0, -120.0):
        out = ExponentialRepresentation(min_dbm=m)(X)
        np.testing.assert_allclose(out[heard], np.exp(X[heard] / 24.0), rtol=1e-12)
        np.testing.assert_allclose(out[~heard], np.exp(m / 24.0), rtol=1e-12)


def test_degenerate_parameters_are_clear_errors():
    from indoorloc.signals import RSSINormalize

    with pytest.raises(ValueError, match="hi > lo"):
        RSSINormalize(lo=None, hi=None).fit(np.array([[-50.0, -50.0]]))
    with pytest.raises(ValueError, match="hi > lo"):
        RSSINormalize(lo=0.0, hi=-104.0)(np.array([-50.0]))  # not mistaken for a file sentinel
    with pytest.raises(ValueError, match="alpha must be positive"):
        ExponentialRepresentation(alpha=0.0, min_dbm=-105)(X3)
    with pytest.raises(ValueError, match="beta must be positive"):
        PowedRepresentation(beta=-1.0, min_dbm=-105)(X3)
    with pytest.raises(ValueError, match="non-negative integer"):
        HampelFilter(2.5)(X3)


def _brute_force_hampel(x, k, n_sigmas):
    """Textbook Hampel identifier on a 1-D series: truncated window, NaN ignored."""
    out, flags = x.copy(), np.zeros(len(x), bool)
    for i in range(len(x)):
        w = x[max(0, i - k):i + k + 1]
        w = w[~np.isnan(w)]
        if np.isnan(x[i]) or not len(w):
            continue
        m = np.median(w)
        if abs(x[i] - m) > n_sigmas * 1.4826 * np.median(np.abs(w - m)):
            out[i], flags[i] = m, True
    return out, flags


def test_hampel_matches_a_brute_force_reference():
    rng = np.random.default_rng(4)
    X = np.round(rng.normal(-70.0, 3.0, size=(40, 6)))  # integer dBm: ties in medians and MADs
    X[rng.random(X.shape) < 0.15] = NAN
    X[rng.random(X.shape) < 0.08] += 25.0
    for axis, k in [(0, 1), (0, 3), (-1, 2)]:
        got, mask = F.hampel(X, k, 2.5, axis=axis, return_mask=True)
        series = X if axis == 0 else X.T
        want = [_brute_force_hampel(series[:, j], k, 2.5) for j in range(series.shape[1])]
        ref = np.column_stack([w[0] for w in want])
        refmask = np.column_stack([w[1] for w in want])
        if axis != 0:
            ref, refmask = ref.T, refmask.T
        assert np.array_equal(got, ref, equal_nan=True) and np.array_equal(mask, refmask)
        assert mask.any()


def test_hampel_filters_each_trajectory_in_time_order():
    # a ramp with a spike at t = 3; window 2, 3 sigmas: at t = 3 the window is [1, 2, 23, 4, 5], median 4,
    # MAD 2, |23 - 4| > 3 * 1.4826 * 2 -> 4; no other sample is an outlier
    series = np.array([0.0, 1.0, 2.0, 23.0, 4.0, 5.0, 6.0])
    filtered = np.array([0.0, 1.0, 2.0, 4.0, 4.0, 5.0, 6.0])
    assert F.hampel(series, 2).tolist() == filtered.tolist()
    rows = np.random.default_rng(0).permutation(14)  # two trajectories, rows shuffled
    table = SampleTable(np.concatenate([series, 2 * series])[rows][:, None], np.zeros((14, 2)),
                        groups={"trajectory": np.repeat([7, 3], 7)[rows], "time": np.tile(np.arange(7.0), 2)[rows]})
    out = HampelFilter(2, axis=0)(table).X[:, 0]
    assert out.tolist() == np.concatenate([filtered, 2 * filtered])[rows].tolist()  # input row order kept
    # a plain array has no time column: it is one series in row order, so the shuffled ramp filters differently
    assert HampelFilter(2, axis=0)(table.X)[:, 0].tolist() != out.tolist()
