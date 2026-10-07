"""Usability fixes from the final review: one meaning per tracker time argument, documented return
types of the L5 batch methods, warnings for raw RSSI sentinels and sign typos, the IMU default split
of the simulator, ``register_dataset``, missing-file hints and the CLI ``evaluate`` default split."""
from __future__ import annotations

import hashlib
import subprocess
import sys
import warnings

import numpy as np
import pytest

from conftest import PROJECT
from indoorloc.apps import PDR, ExtendedKalmanTracker, KalmanTracker, ParticleFilter, PDRFusion, StepTrack
from indoorloc.core import Prediction, SampleTable
from indoorloc.datasets import DATASETS, Dataset, list_datasets, load_dataset, register_dataset
from indoorloc.signals import APFilter, FillMissing

CSV = b"AP1,AP2,X,Y\n-40,-70,0,0\n-70,-40,10,0\n-55,-55,5,0\n-60,-50,7,0\n"
SHA = hashlib.sha256(CSV).hexdigest()


class _Scans(Dataset):
    """Four scans in one file (test fixture)."""

    name = "usability_scans"
    urls = ()
    files = {"all": ("scans.csv", SHA)}
    meta = {"modality": "wifi_rssi", "units": "dBm", "crs": "local"}

    def _parse(self, path, split):
        raw = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        return SampleTable(raw[:, :2].astype(np.float32), raw[:, 2:], ids=[f"{split}-{i}" for i in range(len(raw))])


class _ScansWithTest(_Scans):
    name = "usability_scans_tt"
    files = {"train": ("scans.csv", SHA), "test": ("scans.csv", SHA)}


@pytest.fixture
def scans_root(tmp_path):
    (tmp_path / "scans.csv").write_bytes(CSV)
    return tmp_path


@pytest.fixture
def cleanup_registry():
    names = []
    yield names
    for name in names:
        DATASETS._entries.pop(name.lower(), None)


def _walk(n=20):
    t = np.arange(n, dtype=np.float64)
    truth = np.column_stack([t, 0.5 * t])
    return t, truth + np.random.default_rng(0).normal(0.0, 0.5, truth.shape)


# ------------------------------------------------------------------ 1. return types of the L5 batch methods
def test_batch_methods_return_the_documented_types():
    t, fixes = _walk()
    assert isinstance(KalmanTracker().smooth(fixes, t=t), Prediction)
    assert isinstance(ParticleFilter(50, random_state=0).filter(fixes, t=t), Prediction)
    rate, n = 50.0, 500  # a 10 s walk: 2 Hz steps of +-3 m/s^2, gyro still
    ts = np.arange(n) / rate
    acc = np.column_stack([np.zeros(n), np.zeros(n), 9.81 + 3.0 * np.sin(2 * np.pi * 2.0 * ts)])
    track = PDR(initial_heading=0.0).run(acc, np.zeros((n, 3)), t=ts)
    assert isinstance(track, StepTrack) and not isinstance(track, Prediction) and len(track) > 5
    out = PDRFusion(ParticleFilter(50), random_state=0).run(track, fixes[:5], t[:5])
    assert isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], Prediction)
    assert len(out[0]) == len(out[1].pos) == len(track) + 5  # one row per event, not per fix


def test_docs_state_the_real_return_types():
    for path in ("docs/architecture/CONTRACTS.md", "docs/guide/apps.md"):
        text = (PROJECT / path).read_text(encoding="utf-8")
        assert "StepTrack" in text and ("(t, Prediction)" in text or "(t, core.Prediction)" in text), path
        assert "predict_to" in text, path
    contracts = (PROJECT / "docs/architecture/CONTRACTS.md").read_text(encoding="utf-8")
    assert "Offline L5 methods (smoothers, PDR, fusion) return a `core.Prediction`" not in contracts
    assert "## 10. Parameter names" in contracts and "range_std" in contracts and "random_state" in contracts


# ------------------------------------------------------------------ 2. predict_to: absolute time on every tracker
def test_kalman_predict_to_is_the_absolute_time_predict():
    t, fixes = _walk(5)
    a, b = KalmanTracker().reset(0.0), KalmanTracker().reset(0.0)
    for tk, z in zip(t, fixes):
        a.update(tk, z)
        b.update(tk, z)
    np.testing.assert_array_equal(a.predict_to(7.5), b.predict(7.5))  # predict(t) is absolute too
    assert a.t_ == b.t_ == 7.5
    with pytest.raises(ValueError, match="backwards"):
        a.predict_to(7.0)
    fresh = KalmanTracker()
    assert np.isnan(fresh.predict_to(3.0)).all() and fresh.t_ == 3.0  # before a fix: only the clock moves
    ekf = ExtendedKalmanTracker(np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]), range_std=0.1).reset(0.0)
    ekf.update(0.0, np.linalg.norm(np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]) - [3.0, 4.0], axis=1))
    assert ekf.predict_to(2.0).shape == (2,) and ekf.t_ == 2.0


def test_particle_predict_to_takes_an_absolute_time_and_moves_the_clock():
    a, b = ParticleFilter(200, random_state=1), ParticleFilter(200, random_state=1)
    a.update(10.0, [1.0, 2.0], spread=1.0)
    b.update(10.0, [1.0, 2.0], spread=1.0)
    np.testing.assert_array_equal(a.predict_to(13.0), b.predict(3.0))  # predict(dt) is an increment
    np.testing.assert_array_equal(a.particles_, b.particles_)
    assert a.t_ == 13.0 and b.t_ == 10.0  # only predict_to moves the clock that update reads
    with pytest.raises(ValueError, match="backwards"):
        a.predict_to(12.0)
    fresh = ParticleFilter(10)
    assert np.isnan(fresh.predict_to(4.0)).all() and fresh.t_ == 4.0


def test_particle_update_is_predict_to_then_correct():
    """update(t) runs exactly predict_to(t) + correct: splitting it up gives the same cloud."""
    a, b = ParticleFilter(100, random_state=2), ParticleFilter(100, random_state=2)
    for pf in (a, b):
        pf.update(0.0, [0.0, 0.0], spread=1.0)
    a.update(2.0, [1.0, 1.0], spread=1.0)
    b.predict_to(2.0)
    b.correct([1.0, 1.0], spread=1.0)
    np.testing.assert_array_equal(a.particles_, b.particles_)
    assert a.t_ == b.t_ == 2.0


# ------------------------------------------------------------------ 3. raw sentinels and sign typos warn
def test_fill_missing_warns_about_a_raw_sentinel_left_in_place():
    raw = np.array([[-50.0, 100.0], [100.0, -80.0]])
    with pytest.warns(UserWarning, match=r"missing=100"):
        out = FillMissing()(raw)
    assert out[0, 1] == 100.0  # still not filled: the warning is the only signal
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        np.testing.assert_array_equal(FillMissing(-104, missing=100)(raw), [[-50, -104], [-104, -80]])
        FillMissing()(np.array([[-50.0, np.nan]]))  # the L1 convention: nothing to warn about
        FillMissing()(np.array([[-50, -70]], dtype=np.int16))
        FillMissing(0.0)(np.array([[1e-6, np.nan]]))  # VLC power in W: a non-dBm fill value
        FillMissing()(np.array([[1 + 1j, np.nan]], dtype=np.complex64))  # CSI is not RSSI
        FillMissing()(np.array([[-50.0, 3.0, np.nan]]))  # a few dB above 0 after DeviceCalibration: real
    table = SampleTable(np.array([[-50.0, 127.0]], dtype=np.float32), np.zeros((1, 2)))
    with pytest.warns(UserWarning, match="far above 0 dBm"):
        FillMissing().transform(table)
    assert "missing=" in FillMissing.__doc__.split("\n")[0] and "100" in FillMissing.__doc__.split("\n")[0]


def test_apfilter_warns_about_a_positive_threshold():
    x = np.array([-50.0, -97.0])
    with pytest.warns(UserWarning, match=r"APFilter\(-95\)"):
        assert np.isnan(APFilter(95)(x)).all()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        np.testing.assert_array_equal(APFilter(-95)(x), [-50.0, np.nan])
        APFilter(0)(x)


# ------------------------------------------------------------------ 4. the simulator's IMU modality
def test_synthetic_office_imu_defaults_to_its_only_split():
    from indoorloc.datasets.simulated.office import SyntheticOffice

    small = {"n_trajectories": 1, "trajectory_duration": 4.0}
    walk = load_dataset("synthetic_office", modality="imu", **small)
    assert isinstance(walk, SampleTable) and walk.meta["split"] == "trajectory" and walk.meta["modality"] == "imu"
    office = SyntheticOffice(modality="imu", **small)
    assert office.default_splits == ("trajectory",) and office.load().meta["split"] == "trajectory"
    np.testing.assert_array_equal(office.load().X, walk.X)
    with pytest.raises(ValueError, match="only has the 'trajectory' split"):
        office.load("train")
    wifi = SyntheticOffice(grid_spacing=8.0, samples_per_point=1, n_test=5)
    assert wifi.default_splits == ("train", "test") and wifi.load().meta["split"] == "train"


# ------------------------------------------------------------------ 5. register_dataset and missing files
def test_register_dataset_directly_and_as_a_decorator(scans_root, cleanup_registry):
    cleanup_registry += ["usability_direct", "usability_deco"]
    assert register_dataset("usability_direct", _Scans) is _Scans
    assert "usability_direct" in list_datasets() and len(load_dataset("usability_direct", root=scans_root)) == 4

    @register_dataset("usability_deco")
    class Decorated(_Scans):
        pass

    assert Decorated.__name__ == "Decorated" and DATASETS.get("USABILITY_DECO") is Decorated
    with pytest.raises(KeyError, match="already registered"):
        register_dataset("usability_deco", _Scans)
    register_dataset("usability_deco", _Scans, force=True)
    assert DATASETS.get("usability_deco") is _Scans
    path = f"{_Scans.__module__}:{_Scans.__qualname__}"  # module paths and classes need no registration
    assert load_dataset(path, root=scans_root).meta["name"] == "usability_scans"
    assert len(load_dataset(_Scans, root=scans_root)) == 4


def test_register_dataset_is_on_the_facade_without_numpy():
    code = ("import sys, indoorloc as iloc\n"
            "assert 'register_dataset' in iloc.__all__ and 'register_dataset' in dir(iloc)\n"
            "assert 'numpy' not in sys.modules\n"
            "from indoorloc.datasets import register_dataset\n"
            "assert iloc.register_dataset is register_dataset\n")
    subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT, check=True)


def test_list_datasets_has_a_docstring():
    assert list_datasets.__doc__ and "registry names" in list_datasets.__doc__.lower()


def test_missing_file_without_urls_does_not_suggest_a_download(tmp_path):
    with pytest.raises(FileNotFoundError) as err:
        _Scans(root=tmp_path).load("all")
    assert "download=True" not in str(err.value) and "by hand" in str(err.value)

    class Mirrored(_Scans):
        urls = ("https://example.invalid/scans.zip",)

    with pytest.raises(FileNotFoundError, match="download=True"):  # a download could help: say so
        Mirrored(root=tmp_path).load("all")

    class PerFile(_Scans):
        files = {"all": (("scans.csv", SHA), ("extra.csv", None))}
        urls = {"extra.csv": "https://example.invalid/extra.csv"}

    with pytest.raises(FileNotFoundError) as err:
        PerFile(root=tmp_path).load("all")  # scans.csv has no url of its own
    assert "download=True" not in str(err.value)


# ------------------------------------------------------------------ 6. CLI evaluate: default split
def _saved_model(tmp_path, table):
    from indoorloc.methods import create_model

    model = create_model("knn", k=1).fit(table)
    return str(model.save(tmp_path / "knn"))


def test_cli_evaluate_defaults_to_test_else_all(scans_root, tmp_path, capsys, cleanup_registry):
    from indoorloc.cli import _evaluation_split, main

    cleanup_registry += ["usability_all", "usability_tt"]
    register_dataset("usability_all", _Scans)
    register_dataset("usability_tt", _ScansWithTest)
    saved = _saved_model(tmp_path, _Scans(scans_root).load("all"))
    assert main(["evaluate", "--model", saved, "--dataset", "usability_all", "--root", str(scans_root),
                 "--no-download"]) == 0
    assert "[all], n=4" in capsys.readouterr().out
    path = f"{_Scans.__module__}:{_Scans.__qualname__}"
    assert main(["evaluate", "--model", saved, "--dataset", path, "--root", str(scans_root), "--no-download"]) == 0
    assert "[all], n=4" in capsys.readouterr().out
    assert main(["evaluate", "--model", saved, "--dataset", "usability_tt", "--root", str(scans_root),
                 "--no-download"]) == 0
    assert "[test], n=4" in capsys.readouterr().out
    assert main(["evaluate", "--model", saved, "--dataset", "usability_tt", "--root", str(scans_root),
                 "--split", "train", "--no-download"]) == 0
    assert "[train]" in capsys.readouterr().out
    for name in ("wlanrssi", "csi_fingerprint", "hwild", "ilc2020"):  # only 'all': used to fail with 'test'
        assert _evaluation_split(name, tmp_path) == "all", name
    for name in ("ujiindoorloc", "uji", "ble_indoor", "synthetic_office", "ujindoorloc"):
        assert _evaluation_split(name, tmp_path) == "test", name
