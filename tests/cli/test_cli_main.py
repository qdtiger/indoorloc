from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from conftest import PROJECT, UJI_ROOT
from indoorloc.cli import main
from indoorloc.cli.benchmark import build_preprocess, parse_spec, run_benchmark
from indoorloc.core import SampleTable
from indoorloc.datasets import Dataset

APS = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
GRID = np.array([(x, y) for x in (0.0, 2.5, 5.0, 7.5, 10.0) for y in (0.0, 2.5, 5.0, 7.5, 10.0)])


def _scans(device_offset: float, floor: int) -> np.ndarray:
    """Integer dBm from a log-distance model; floor 1 is 15 dB weaker; below -70 dBm is not heard."""
    d = np.maximum(np.linalg.norm(GRID[:, None, :] - APS[None], axis=2), 1.0)
    rssi = np.round(-30.0 - 25.0 * np.log10(d)) - 15.0 * floor + device_offset
    return np.where(rssi < -70.0, np.nan, rssi)


def _csv(rows) -> bytes:
    lines = ["AP1,AP2,AP3,AP4,X,Y,FLOOR,DEVICE,TIME"]
    lines += [",".join("nan" if np.isnan(v) else f"{v:g}" for v in row) for row in rows]
    return ("\n".join(lines) + "\n").encode()


def _rows(device: int, offset: float, t0: int):
    out = []
    for floor in (0, 1):
        X = _scans(offset, floor)
        for i, (x, y) in enumerate(GRID):
            out.append([*X[i], x, y, floor, device, t0 + len(out)])
    return out


TRAIN_CSV = _csv(_rows(1, 0.0, 0) + _rows(2, -3.0, 100))  # 100 scans, two devices
TEST_CSV = _csv(_rows(1, 0.0, 1000))  # 50 scans, identical to device 1's reference scans


class TinyGrid(Dataset):
    """A 10 m x 10 m two-floor toy survey with four APs (for CLI tests)."""

    name = "tinygrid"
    files = {"train": ("train.csv", hashlib.sha256(TRAIN_CSV).hexdigest()),
             "test": ("test.csv", hashlib.sha256(TEST_CSV).hexdigest())}
    meta = {"modality": "wifi_rssi", "units": "dBm", "crs": "local", "pos_units": "m", "citation": "test fixture"}

    def _parse(self, path, split):
        raw = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        return SampleTable(raw[:, :4].astype(np.float32), raw[:, 4:6], floor=raw[:, 6],
                           groups={"device": raw[:, 7].astype(np.int64), "time": raw[:, 8]},
                           ids=[f"{split}-{i}" for i in range(len(raw))])


DATASET = f"{__name__}:TinyGrid"  # an L1 class by "module:Class", no registration needed


# One survey file with an "all" split that repeats the "train" and "test" rows under the same ids (as
# ibeacon_rssi does), and an "unlabeled" file without positions (as ble_rssi_uci does).
# the last column assigns each survey row to 0 = train, 1 = test or 2 = neither
SURVEY_CSV = _csv([row[:8] + [i % 3] for i, row in enumerate(_rows(1, 0.0, 0))])
EXTRA_CSV = _csv([[-50.0, -60.0, np.nan, -65.0, np.nan, np.nan, 0, 1, 0]] * 7)


class OverlappingSplits(Dataset):
    """Splits "all" = train + test + 16 other rows, plus 7 unlabelled scans (for CLI tests)."""

    name = "overlapping"
    files = {"all": ("survey.csv", hashlib.sha256(SURVEY_CSV).hexdigest()),
             "train": ("survey.csv", hashlib.sha256(SURVEY_CSV).hexdigest()),
             "test": ("survey.csv", hashlib.sha256(SURVEY_CSV).hexdigest()),
             "unlabeled": ("extra.csv", hashlib.sha256(EXTRA_CSV).hexdigest())}
    meta = {"modality": "wifi_rssi", "units": "dBm", "crs": "local", "pos_units": "m"}

    def _parse(self, path, split):
        raw = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        ids = np.array([f"{path.stem}-{i}" for i in range(len(raw))])
        rows = {"train": raw[:, 8] == 0, "test": raw[:, 8] == 1}.get(split, np.ones(len(raw), dtype=bool))
        raw, ids = raw[rows], ids[rows]
        return SampleTable(raw[:, :4].astype(np.float32), raw[:, 4:6], floor=raw[:, 6],
                           groups={"device": raw[:, 7].astype(np.int64)}, ids=ids)


class RoomsOnly(TinyGrid):
    """Scans labelled by room only: no coordinates."""

    def _parse(self, path, split):
        t = super()._parse(path, split)
        return t.replace(pos=np.zeros((len(t), 0)))


@pytest.fixture
def root(tmp_path):
    (tmp_path / "train.csv").write_bytes(TRAIN_CSV)
    (tmp_path / "test.csv").write_bytes(TEST_CSV)
    (tmp_path / "survey.csv").write_bytes(SURVEY_CSV)
    (tmp_path / "extra.csv").write_bytes(EXTRA_CSV)
    return tmp_path


def _bench(root, *extra):
    return ["benchmark", "--dataset", DATASET, "--root", str(root), "--no-download", "-q", *extra]


def test_benchmark_writes_a_self_describing_result_file(root, tmp_path):
    out, report, preds = tmp_path / "r.json", tmp_path / "r.md", tmp_path / "p.npz"
    assert main(_bench(root, "--method", "knn(k=1)", "--method", "wknn", "--out", str(out), "--report",
                       str(report), "--predictions", str(preds))) == 0
    d = json.loads(out.read_text())
    assert d["format"] == "indoorloc-benchmark" and d["format_version"] == 1 and d["seed"] == 0
    assert d["dataset"]["sha256"] == {"train": TinyGrid.files["train"][1], "test": TinyGrid.files["test"][1]}
    assert d["dataset"]["n_samples"] == {"train": 100, "test": 50} and d["dataset"]["class"] == DATASET
    (fold,) = d["protocol"]["folds"]
    assert fold["test_sha256"] == hashlib.sha256(np.arange(100, 150, dtype="<i8").tobytes()).hexdigest()
    knn, wknn = d["methods"]
    # every test scan equals a reference scan of the same device: 1-NN is exact
    assert knn["pooled"]["mean_error"] == 0.0 and knn["pooled"]["floor_accuracy"] == 100.0
    assert knn["pooled"]["ipin_score"] == 0.0 and knn["pooled"]["cdf"]["<=1"] == 100.0
    assert knn["model_params"]["params"]["localizer"]["params"]["k"] == 1
    assert wknn["model_params"]["params"]["localizer"]["params"]["weights"] == "distance"
    assert d["preprocess"] == {"spec": "fill", "default": True, "resolved": {
        "class": "indoorloc.signals.transforms:FillMissing", "params": {"missing": "nan", "value": -104.0}}}
    env = d["environment"]
    assert env["numpy"] == np.__version__ and len(env["source_sha256"]) == 64 and env["python"]
    assert d["command"][:2] == ["indoorloc", "benchmark"] and d["wall_time_s"] > 0 and "literature" not in d
    assert "## Provenance" in report.read_text() and "| knn(k=1) |" in report.read_text()
    with np.load(preds) as z:
        assert z["knn(k=1)/ids"].tolist() == [f"test-{i}" for i in range(50)]
        assert np.array_equal(z["knn(k=1)/pred"], np.tile(GRID, (2, 1)))


def test_saved_models_reload_and_score_the_same(root, tmp_path, capsys):
    out, models = tmp_path / "r.json", tmp_path / "models"
    assert main(_bench(root, "--method", "wknn(k=3)", "--out", str(out), "--save-models", str(models))) == 0
    saved = json.loads(out.read_text())["methods"][0]["saved_to"]
    assert saved == str(models / "wknn_k=3_")
    from indoorloc.core import load_model

    model = load_model(saved)
    assert model.saved_info_["train_splits"] == ["train"] and model.saved_info_["dataset"] == "tinygrid"
    test = TinyGrid(root).load("test")
    expected = json.loads(out.read_text())["methods"][0]["pooled"]["mean_error"]
    assert main(["evaluate", "--model", saved, "--dataset", DATASET, "--root", str(root), "--no-download",
                 "--format", "markdown"]) == 0
    captured = capsys.readouterr()
    assert f"| wknn(k=3) | {expected:.3f} |" in captured.out and "warning" not in captured.err
    assert np.isfinite(model.predict(test)).all()
    assert main(["evaluate", "--model", saved, "--dataset", DATASET, "--root", str(root), "--split", "train",
                 "--no-download"]) == 0
    assert "used to train this model" in capsys.readouterr().err
    assert main(_bench(root, "--method", "knn", "--protocol", "cross-device", "--save-models", str(models))) == 1


def test_grouped_and_random_protocols(root, tmp_path):
    out = tmp_path / "r.json"
    assert main(_bench(root, "--method", "knn(k=1)", "--protocol", "cross-device", "--out", str(out))) == 0
    d = json.loads(out.read_text())
    assert [f["name"] for f in d["protocol"]["folds"]] == ["device=1", "device=2"]
    assert d["methods"][0]["pooled"]["n"] == 150 and len(d["methods"][0]["folds"]) == 2

    def random_test_rows(seed):
        main(_bench(root, "--method", "knn", "--protocol", "random-80-20", "--seed", str(seed), "--out", str(out)))
        return json.loads(out.read_text())["protocol"]["folds"][0]["test_sha256"]

    assert random_test_rows(1) == random_test_rows(1) != random_test_rows(2)


def test_split_listed_twice_and_unlabelled_rows_are_pooled_once(root, tmp_path, capsys):
    out, preds = tmp_path / "r.json", tmp_path / "p.npz"
    dataset = f"{__name__}:OverlappingSplits"
    assert main(["benchmark", "--dataset", dataset, "--root", str(root), "--no-download", "-q", "--method",
                 "knn(k=1)", "--protocol", "random-80-20", "--out", str(out), "--predictions", str(preds)]) == 0
    d = json.loads(out.read_text())["dataset"]
    assert d["splits"] == ["train", "test", "unlabeled", "all"]  # "all" last: shared rows keep train/test
    assert d["n_samples"] == {"train": 17, "test": 17, "unlabeled": 7, "all": 50}
    assert d["n_pooled"] == 50 and d["duplicate_rows_dropped"] == {"all": 34}
    assert d["unlabelled_rows_left_out"] == {"unlabeled": 7}
    with np.load(preds) as z:  # 20 % of the 50 distinct samples, under their own ids (no "all/" copies)
        test_ids = z["knn(k=1)/ids"].tolist()
    assert len(test_ids) == len(set(test_ids)) == 10 and all(i.startswith("survey-") for i in test_ids)
    assert main(["benchmark", "--dataset", dataset, "--root", str(root), "--no-download", "-q", "--method",
                 "knn(k=1)", "--out", str(out)]) == 0
    (fold,) = json.loads(out.read_text())["protocol"]["folds"]  # official: 17 train rows, 17 test rows, 16 unused
    assert (fold["n_train"], fold["n_test"], fold["n_unused"]) == (17, 17, 16)
    assert main(["report", str(out), "--format", "text"]) == 0
    text = capsys.readouterr().out
    assert "duplicate rows dropped    all: 34" in text and "unlabelled rows left out  unlabeled: 7" in text


def test_datasets_without_coordinates_and_ground_unit_comparisons_are_refused(root, tmp_path, capsys):
    assert main(["benchmark", "--dataset", f"{__name__}:RoomsOnly", "--root", str(root), "--no-download", "-q",
                 "--method", "knn"]) == 1
    assert "has no coordinates" in capsys.readouterr().err
    out = tmp_path / "ground.json"
    out.write_text(json.dumps({"units": "m (ground)", "scale": 0.766, "protocol": {"name": "official"},
                               "methods": [{"label": "knn", "pooled": {"mean_error": 6.7}}]}))
    assert main(["literature", "ujiindoorloc", "--results", str(out)]) == 1
    assert "--units native" in capsys.readouterr().err


def test_specs():
    assert parse_spec("knn") == ("knn", {})
    assert parse_spec("wknn(k=3, weights=distance)") == ("wknn", {"k": 3, "weights": "distance"})
    assert parse_spec("pkg.mod:Cls(alpha=0.1, sizes=[1, 2], flag=true, name='a,b')") == (
        "pkg.mod:Cls", {"alpha": 0.1, "sizes": [1, 2], "flag": True, "name": "a,b"})
    assert parse_spec("m(a=True, b=None, c=(1, 2), d=-104)") == ("m", {"a": True, "b": None, "c": (1, 2), "d": -104})
    with pytest.raises(ValueError, match="key=value"):
        parse_spec("knn(3)")
    pre = build_preprocess("fill(value=-110)+normalize")
    assert type(pre).__name__ == "Compose" and pre.transforms[0].value == -110
    assert build_preprocess("none") is None and type(build_preprocess("FillMissing")).__name__ == "FillMissing"
    with pytest.raises(ValueError, match="unknown preprocessing"):
        build_preprocess("sharpen")


def test_errors_exit_with_status_1_and_a_message(root, capsys):
    assert main(_bench(root, "--method", "no_such_method")) == 1
    assert "indoorloc: error: unknown method 'no_such_method'" in capsys.readouterr().err
    assert main(_bench(root, "--method", "knn", "--protocol", "leave-one-building-out")) == 1
    assert "needs 'building' labels" in capsys.readouterr().err
    assert main(_bench(root, "--method", "knn", "--units", "ground")) == 1
    assert "ground_scale" in capsys.readouterr().err
    with pytest.raises(SystemExit) as usage:
        main(["benchmark", "--dataset", DATASET])  # --method is required
    assert usage.value.code == 2


def test_list_info_literature_and_report(root, tmp_path, capsys):
    assert main(["list", "protocols", "-v"]) == 0
    assert "cross-device" in capsys.readouterr().out
    assert main(["list", "methods"]) == 0 and "wknn" in capsys.readouterr().out.split()
    assert main(["list", "literature"]) == 0 and "ujiindoorloc" in capsys.readouterr().out.split()
    assert main(["info", DATASET, "--root", str(root), "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["modality"] == "wifi_rssi" and [f["present"] for f in info["files"]] == [True, True]
    assert main(["literature", "ujiindoorloc", "--format", "markdown"]) == 0
    text = capsys.readouterr().out
    assert "as reported; not re-run" in text and "| corrected |" in text
    out = tmp_path / "r.json"
    main(_bench(root, "--method", "knn", "--out", str(out)))
    assert main(["literature", "ujiindoorloc", "--results", str(out)]) == 0
    text = capsys.readouterr().out
    assert text.index("Reproduced with indoorloc") < text.index("Literature on UJIIndoorLoc")
    assert main(["report", str(out)]) == 0 and "# Benchmark: tinygrid / protocol official" in capsys.readouterr().out


def test_python_m_entry_point_and_a_light_import():
    run = subprocess.run([sys.executable, "-m", "indoorloc.cli", "list", "protocols"], cwd=PROJECT,
                         capture_output=True, text=True, check=True)
    assert "official" in run.stdout.split()
    subprocess.run([sys.executable, "-c", "import sys, indoorloc.cli\n"
                    "assert 'numpy' not in sys.modules and 'indoorloc.datasets' not in sys.modules"],
                   cwd=PROJECT, check=True)


@pytest.mark.skipif(not (UJI_ROOT / "validationData.csv").is_file(), reason="UJIIndoorLoc files not found")
def test_ujiindoorloc_official_knn_and_wknn_reproduce_the_reference_numbers():
    d = run_benchmark("ujiindoorloc", ["knn", "wknn"], root=UJI_ROOT, download=False)
    knn, wknn = (m["pooled"] for m in d["methods"])
    # FillMissing(-104) then k=5 on integer dBm; reference: docs/architecture guide, section 7.1
    assert knn["mean_error"] == pytest.approx(8.808422949721733, rel=1e-12)
    assert wknn["mean_error"] == pytest.approx(8.793663569441243, rel=1e-9)
    assert round(knn["floor_accuracy"] * 1111 / 100) == 1003 and round(wknn["floor_accuracy"] * 1111 / 100) == 1005
    assert knn["building_accuracy"] == pytest.approx(100 * 1108 / 1111)
    assert d["dataset"]["sha256"]["test"].startswith("5f90c536") and d["literature"]["protocol"] == "official"
