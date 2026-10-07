"""Cross-layer checks of the integration code: downloads, default splits, evaluation of unplaced
samples, the benchmark scorer, Compose fit keywords, the geometric refinement and model files.

Every test here reproduces a defect found in review (each docstring says what used to happen).
"""
from __future__ import annotations

import gzip
import hashlib
import io
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pytest

from indoorloc.core import Prediction, SampleTable, load_model
from indoorloc.datasets import load_dataset
from indoorloc.datasets._base import Dataset
from indoorloc.methods.base import BaseLocalizer


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def zipped(members: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


class _Response(io.BytesIO):
    def __init__(self, body: bytes, headers=None):
        super().__init__(body)
        self.headers = headers or {}


@pytest.fixture
def serve(monkeypatch):
    """``serve({url: bytes | exception | callable})`` patches urlopen; returns the requested urls."""
    asked = []

    def install(table):
        def urlopen(request, timeout=None):
            asked.append(request.full_url)
            item = table[request.full_url]
            item = item() if callable(item) else item
            if isinstance(item, Exception):
                raise item
            return item if isinstance(item, _Response) else _Response(item)

        monkeypatch.setattr(urllib.request, "urlopen", urlopen)
        monkeypatch.setattr(time, "sleep", lambda s: None)
        return asked

    return install


def dataset(root, urls, files, *, verify=True):
    class Fake(Dataset):
        name = "fake"

        def _parse(self, paths, split):
            return SampleTable(np.zeros((1, 1)), np.zeros((1, 2)))

    Fake.urls, Fake.files = urls, files
    return Fake(root, download=True, verify=verify)


def listing(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


A, B = b"x,y\n1,2\n", b"x,y\n3,4\n"


# --------------------------------------------------------------------------------------------- L1 downloads
def test_download_takes_the_member_whose_digest_matches_among_same_named_ones(tmp_path, serve):
    """An archive with old/data.csv before new/data.csv used to extract the first match, leave the
    wrong file in place and fail with a checksum mismatch on every later call."""
    serve({"https://h/a.zip": zipped({"old/data.csv": B, "new/data.csv": A})})
    ds = dataset(tmp_path, ("https://h/a.zip",), {"all": ("data.csv", sha(A))})
    assert ds.check("all")[0].read_bytes() == A and listing(tmp_path) == ["data.csv"]


def test_download_without_a_digest_refuses_an_ambiguous_member(tmp_path, serve):
    serve({"https://h/a.zip": zipped({"old/data.csv": B, "new/data.csv": A})})
    with pytest.raises(ValueError, match="no sha256 tells which"):
        dataset(tmp_path, ("https://h/a.zip",), {"all": ("data.csv", sha(A))}, verify=False).check("all")
    assert listing(tmp_path) == []
    serve({"https://h/a.zip": zipped({"data.csv": A, "backup/data.csv": B})})  # the shallowest wins
    assert dataset(tmp_path, ("https://h/a.zip",), {"all": ("data.csv", None)}).check("all")[0].read_bytes() == A


def test_a_corrupt_archive_member_leaves_no_partial_file(tmp_path, serve):
    """A CRC error while extracting used to leave a truncated file that every later call took as
    present (a permanent checksum mismatch, or silently truncated data with verify=False)."""
    body = bytearray(zipped({"big.csv": A * 500}))
    body[body.find(b"x,y") + 40] ^= 0xFF
    serve({"https://h/a.zip": bytes(body)})
    for verify in (True, False):
        with pytest.raises(zipfile.BadZipFile):
            dataset(tmp_path, ("https://h/a.zip",), {"all": ("big.csv", sha(A * 500))}, verify=verify).check("all")
        assert listing(tmp_path) == []


def test_a_kept_archive_is_verified_before_it_is_placed(tmp_path, serve):
    """The archive kept as a wanted file (named by its url) used to be copied unverified; a
    truncated download then stayed in place for good."""
    body = zipped({"inner.csv": A})
    serve({"https://h/data.zip?download=1": body[:-10]})
    ds = dataset(tmp_path, ("https://h/data.zip?download=1",), {"all": ("data.zip", sha(body))})
    with pytest.raises(ValueError, match="checksum mismatch"):
        ds.check("all")
    assert listing(tmp_path) == []
    serve({"https://h/data.zip?download=1": body})
    assert ds.check("all")[0].read_bytes() == body


def test_a_plain_file_behind_an_opaque_url_is_recognised_by_its_digest(tmp_path, serve):
    """An .npz (itself a zip) behind ``.../content`` used to be searched for members and then
    reported as not found."""
    buf = io.BytesIO()
    np.savez(buf, a=np.arange(3))
    serve({"https://h/records/1/content": buf.getvalue()})
    ds = dataset(tmp_path, ("https://h/records/1/content",), {"all": ("data.npz", sha(buf.getvalue()))})
    assert ds.check("all")[0].read_bytes() == buf.getvalue()


def test_flaky_and_missing_mirrors_and_gzip_transfer(tmp_path, serve):
    """A 404 is not retried (it used to cost three requests and two sleeps per mirror); a mirror
    that fails once is retried; a gzip body with the wrong digest writes nothing."""
    flaky = iter([OSError("connection reset"), _Response(gzip.compress(A), {"Content-Encoding": "gzip"})])
    asked = serve({"https://gone/a.csv": urllib.error.HTTPError("https://gone/a.csv", 404, "Not Found", {}, None),
                   "https://flaky/a.csv": lambda: next(flaky)})
    ds = dataset(tmp_path, {"a.csv": ("https://gone/a.csv", "https://flaky/a.csv")}, {"all": ("a.csv", sha(A))})
    assert ds.check("all")[0].read_bytes() == A
    assert asked == ["https://gone/a.csv", "https://flaky/a.csv", "https://flaky/a.csv"]
    serve({"https://g/b.csv": _Response(gzip.compress(B), {"Content-Encoding": "gzip"})})
    with pytest.raises(ValueError, match="checksum mismatch"):
        dataset(tmp_path, {"b.csv": "https://g/b.csv"}, {"all": ("b.csv", sha(A))}).check("all")
    assert listing(tmp_path) == ["a.csv"]


def test_an_archive_split_with_some_files_present_fetches_the_rest_once(tmp_path, serve):
    asked = serve({"https://h/a.zip": zipped({"d/one.csv": B, "d/two.csv": A})})
    (tmp_path / "one.csv").write_bytes(A)  # present (and different from the archive's copy): kept
    ds = dataset(tmp_path, ("https://h/a.zip",), {"all": (("one.csv", sha(A)), ("two.csv", sha(A)))})
    assert [p.read_bytes() for p in ds.check("all")] == [A, A] and len(asked) == 1


def test_split_none_means_the_default_splits_everywhere():
    """Dataset.load(None) used to insist on "train" (an error for datasets with other names) and
    subclasses with a default argument (SyntheticOffice.load) refused None altogether."""
    from indoorloc.datasets.simulated.office import SyntheticOffice

    class Named(Dataset):
        name = "named"
        files = {"a": (), "b": ()}

        def _parse(self, paths, split):
            return SampleTable(np.zeros((1, 1)), np.zeros((1, 2)), meta={"which": split})

    assert Named("/nonexistent").default_splits == ("a",) and Named("/nonexistent").load().meta["split"] == "a"
    Named.files = {"train": (), "test": (), "all": ()}
    assert Named("/nonexistent").default_splits == ("train", "test")
    Named.files = {"extra": (), "all": ()}
    assert Named("/nonexistent").load().meta["split"] == "all"
    office = SyntheticOffice(grid_spacing=8.0, samples_per_point=1, n_test=5)
    assert office.load(None).meta["split"] == "train" and office.default_splits == ("train", "test")
    train, test = load_dataset("synthetic_office", grid_spacing=8.0, samples_per_point=1, n_test=5)
    assert len(test) == 5 and train.meta["split"] == "train"


# --------------------------------------------------------------------------------------------- L4 evaluate
def test_evaluate_refuses_unknown_ground_truth_and_names_an_all_failed_run():
    """A NaN in y_true used to be counted as a sample the METHOD could not place (n_failed)."""
    from indoorloc.evaluation import evaluate

    with pytest.raises(ValueError, match="ground truth"):
        evaluate([[0.0, 0.0], [np.nan, np.nan]], [[0.0, 0.0], [1.0, 1.0]])
    res = evaluate([[0.0, 0.0], [1.0, 1.0]], [[np.nan, np.nan], [np.nan, np.nan]])
    assert res.n_failed == 2 and np.isnan(res.mean_error) and res.summary().startswith("no sample placed")
    part = evaluate([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [[3.0, 4.0], [np.nan, 0.0], [1.0, 0.0]])
    assert part.n_failed == 1 and part.mean_error == 3.0 and part.cdf([1.0, 5.0]).tolist() == [1 / 3, 2 / 3]


# --------------------------------------------------------------------------------------------- CLI scorer
class HalfPlaced(BaseLocalizer):
    """Test method: the training mean, NaN for every ``every``-th row (0: none)."""

    _allow_nan = True

    def __init__(self, every: int = 3):
        self.every = every

    def _fit(self, X, pos, floor, building):
        self.mean_ = pos.mean(axis=0)

    def _localize(self, X):
        pos = np.tile(self.mean_, (len(X), 1))
        if self.every:
            pos[:: self.every] = np.nan
        return Prediction(pos)


def test_score_of_partly_and_wholly_unplaced_predictions():
    """_score used to raise on ANY unplaced sample (success_rate refuses NaN) and, with none
    placed, in bootstrap_ci: the failed_note path could never run."""
    from indoorloc.cli.benchmark import _score

    y = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 3.0], [0.0, 30.0]])
    p = np.zeros_like(y)
    p[0] = np.nan
    out = _score(y, p, None, None, None, None, 1.0, 0)
    assert out["n_failed"] == 1 and out["mean_error"] == pytest.approx(34 / 3) and "failed_note" in out
    assert out["cdf"]["<=1"] == 25.0 and out["cdf"]["<=5"] == 50.0 and out["cdf"]["<=20"] == 50.0  # of ALL 4
    assert out["ipin_score"] == pytest.approx(np.percentile([1.0, 3.0, 30.0], 75))
    none = _score(y, np.full_like(y, np.nan), None, None, None, None, 1.0, 0)
    assert none["n_failed"] == 4 and none["mean_error_ci95"] is None and none["ipin_score"] is None
    assert set(none["cdf"].values()) == {0.0} and "no sample was placed" in none["scores_note"]


def test_benchmark_runs_a_method_that_cannot_place_every_sample():
    from indoorloc.cli.benchmark import run_benchmark

    options = dict(grid_spacing=8.0, samples_per_point=1, n_test=12, n_trajectories=1, trajectory_duration=2.0)
    doc = run_benchmark("synthetic_office", [f"{__name__}:HalfPlaced", f"{__name__}:HalfPlaced(every=1)"],
                        dataset_options=options)
    some, none = (m["pooled"] for m in doc["methods"])
    assert some["n"] == 12 and some["n_failed"] == 4 and some["mean_error"] is not None
    assert none["n_failed"] == 12 and none["mean_error"] is None and none["mean_error_ci95"] is None


# --------------------------------------------------------------------------------------------- L2 Compose
def test_compose_routes_sample_keywords_through_nested_and_preceding_steps():
    """A nested Compose holding the CORAL step used to be refused ("no step takes target"), and
    DeviceCalibration's ``reference`` skipped the preceding steps, so X was filled and the
    reference was not."""
    from indoorloc.methods.transfer import CORAL
    from indoorloc.signals import Compose, FillMissing
    from indoorloc.signals.calibration import DeviceCalibration

    rng = np.random.default_rng(0)
    Xs, Xt = rng.normal(-70, 8, (40, 3)), rng.normal(-60, 4, (40, 3))
    Xs[0, 0] = Xt[1, 1] = np.nan
    flat = Compose([FillMissing(-104), CORAL()]).fit_transform(Xs, target=Xt)
    nested = Compose([FillMissing(-104), Compose([CORAL()])]).fit_transform(Xs, target=Xt)
    np.testing.assert_allclose(nested, flat)
    ref = Xs + 5.0
    chain = Compose([FillMissing(-104), DeviceCalibration(method="offset")]).fit(Xs, reference=ref)
    alone = DeviceCalibration(method="offset").fit(FillMissing(-104).transform(Xs), reference=FillMissing(-104)
                                                   .transform(ref))
    assert chain.transforms[1].intercept_ == pytest.approx(alone.intercept_)
    with pytest.raises(TypeError, match="foo"):
        Compose([FillMissing(-104), Compose([CORAL()])]).fit(Xs, target=Xt, foo=1)


# --------------------------------------------------------------------------------------------- L3 geometry
def _tdoa(truth, anchors):
    r = np.linalg.norm(truth[:, None] - anchors[None], axis=2)
    return r[:, 1:] - r[:, :1]


def _bearings(truth, anchors):
    return np.arctan2(truth[:, None, 1] - anchors[None, :, 1], truth[:, None, 0] - anchors[None, :, 0])


def test_tdoa_never_fails_the_batch_on_a_singular_closed_form():
    """A target at the centre of a square of anchors (all differences 0), or a noisy row that makes
    Chan's step 1 singular, used to raise LinAlgError for the WHOLE batch."""
    from indoorloc.methods.geometric import TDOALocalizer, chan_tdoa

    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    truth = np.array([[5.0, 5.0], [2.0, 3.0]])
    X = _tdoa(truth, square)
    assert np.isnan(chan_tdoa(X, square)[0]).all() and np.allclose(chan_tdoa(X, square)[1], [2, 3])
    np.testing.assert_allclose(TDOALocalizer(square).localize(X).pos, truth, atol=1e-9)  # the centroid start
    assert np.isnan(TDOALocalizer(square, refine=False).localize(X).pos[0]).all()
    noisy = np.array([[4.932154927629288, 0.6306539087989051, -4.301502614422425]])  # d1 - d2 + d3 = 0
    pos = TDOALocalizer(square * 2.0).localize(noisy).pos
    assert np.all(np.isfinite(pos)) and np.linalg.norm(pos - [6.95, 14.99]) < 3.0


def test_far_field_rule_keeps_targets_the_data_resolve():
    """Every estimate beyond 10 anchor spreads used to become NaN, even an exact fit: a target
    8 m from a 1 m anchor square, or 40 m down a corridor from anchors spanning 6 m."""
    from indoorloc.methods.aoa import AoALocalizer
    from indoorloc.methods.geometric import TDOALocalizer

    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    far = np.array([[8.0, 2.0], [15.0, 4.0], [30.0, -10.0]])
    np.testing.assert_allclose(TDOALocalizer(square).localize(_tdoa(far, square)).pos, far, atol=1e-6)
    corridor = np.array([[0.0, 0.0], [2.0, 1.5], [4.0, 0.0], [6.0, 1.5]])
    down = np.array([[25.0, 0.8], [40.0, 0.7]])
    np.testing.assert_allclose(TDOALocalizer(corridor).localize(_tdoa(down, corridor)).pos, down, atol=1e-6)
    np.testing.assert_allclose(AoALocalizer(corridor).localize(_bearings(down, corridor)).pos, down, atol=1e-6)
    pair = np.array([[0.0, 0.0], [2.0, 0.0]])  # two bearings: no redundancy, so only a known sigma vouches
    target = np.array([[5.0, 25.0]])
    assert np.isnan(AoALocalizer(pair).localize(_bearings(target, pair)).pos).all()
    np.testing.assert_allclose(AoALocalizer(pair, sigma=1e-3).localize(_bearings(target, pair)).pos, target)
    # the runaways the rule exists for are still refused: large noise, noise scale unknown
    rng = np.random.default_rng(8)
    box = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]])
    truth = rng.uniform(0, 20, (3000, 2))
    ranges = np.linalg.norm(truth[:, None] - box[None], axis=2) + rng.normal(0, 6.0, (3000, 4))
    pred = TDOALocalizer(box).localize(ranges[:, 1:] - ranges[:, :1]).pos
    placed = np.isfinite(pred[:, 0])
    assert placed.mean() > 0.9 and np.linalg.norm(pred[placed] - box.mean(0), axis=1).max() <= 10 * np.sqrt(200)


def test_tdoa_refinement_keeps_the_closed_form_root_of_the_minimal_case():
    """Both roots of the minimal case fit exactly; the centroid start used to replace Chan's
    documented choice (the root nearer the centroid) whenever rounding made its cost smaller."""
    from indoorloc.methods.geometric import TDOALocalizer, chan_tdoa

    tri = np.array([[0.0, 0.0], [10.0, 0.0], [3.0, 8.0]])
    X = _tdoa(np.random.default_rng(3).uniform(-15, 25, (2000, 2)), tri)
    np.testing.assert_allclose(TDOALocalizer(tri).localize(X).pos, chan_tdoa(X, tri), atol=1e-6)


def test_tdoa_3d_exact_too_few_and_calibrated():
    from indoorloc.methods.geometric import TDOALocalizer

    rng = np.random.default_rng(4)
    anchors = np.array([[0, 0, 0], [10, 0, 3], [10, 10, 0], [0, 10, 3], [5, 5, 2.5]], dtype=float)
    truth = rng.uniform([0, 0, 0], [10, 10, 3], (300, 3))
    X = _tdoa(truth, anchors)
    np.testing.assert_allclose(TDOALocalizer(anchors).localize(X).pos, truth, atol=1e-8)
    few = X.copy()
    few[:5, :2] = np.nan  # 2 differences for 3 unknowns
    few[5:10] = np.nan
    pos = TDOALocalizer(anchors).localize(few).pos
    assert np.isnan(pos[:10]).all() and np.allclose(pos[10:], truth[10:], atol=1e-8)
    bias = np.array([0.5, -0.3, 0.2, 0.1])
    noisy = X + bias + rng.normal(0, 0.01, X.shape)
    model = TDOALocalizer(anchors, calibrate=True).fit(noisy, truth)
    np.testing.assert_allclose(model.bias_, bias, atol=0.01)
    assert np.median(np.linalg.norm(model.localize(noisy).pos - truth, axis=1)) < 0.05


# --------------------------------------------------------------------------------------------- core files
def test_a_failed_save_never_pairs_one_models_config_with_anothers_arrays(tmp_path, monkeypatch):
    """save() used to write arrays.npz and then config.json in place: a failure in between left
    the new arrays under the old config, which loaded without complaint as a different model."""
    from pathlib import Path

    from indoorloc.methods import create_model

    rng = np.random.default_rng(0)
    X = rng.normal(-70, 5, (30, 4))
    first = create_model("knn", k=1).fit(X, rng.uniform(0, 10, (30, 2)))
    second = create_model("knn", k=1).fit(X, rng.uniform(50, 60, (30, 2)))
    first.save(tmp_path / "m")

    def disk_full(self, *args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(Path, "write_text", disk_full)
    with pytest.raises(OSError, match="No space"):
        second.save(tmp_path / "m")
    monkeypatch.undo()
    np.testing.assert_array_equal(load_model(tmp_path / "m").predict(X), first.predict(X))
    assert sorted(p.name for p in (tmp_path / "m").iterdir()) == ["arrays.npz", "config.json"]
    second.save(tmp_path / "m")
    np.testing.assert_array_equal(load_model(tmp_path / "m").predict(X), second.predict(X))


def test_clone_keeps_a_namedtuple_parameter():
    """clone() used to rebuild every tuple from a generator, which a namedtuple cannot take."""
    from typing import NamedTuple

    from indoorloc.core import Estimator, clone

    class Box(NamedTuple):
        lo: float
        hi: float

    class Bounded(Estimator):
        def __init__(self, box=None, steps=(1, 2)):
            self.box = box
            self.steps = steps

    twin = clone(Bounded(Box(0.0, 1.0), [3, 4]))
    assert twin.box == Box(0.0, 1.0) and type(twin.box) is Box and twin.steps == [3, 4]


def test_an_archive_wrapping_its_namesake_yields_the_inner_file(tmp_path, serve):
    """data.zip whose declared digest is that of the data.zip INSIDE the download: the member is
    extracted instead of the outer archive being refused."""
    inner = zipped({"x.csv": A})
    serve({"https://h/data.zip": zipped({"release/data.zip": inner})})
    ds = dataset(tmp_path, ("https://h/data.zip",), {"all": ("data.zip", sha(inner))})
    assert ds.check("all")[0].read_bytes() == inner and listing(tmp_path) == ["data.zip"]
