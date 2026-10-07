"""Tests of the benchmark harness: protocols, baseline, matrix, label runner, verification and rendering.

Synthetic data, plus checks of the committed ``results/`` against ``matrix.py`` and ``docs/``, and
one real-data recomputation that is skipped when the UJIIndoorLoc files are absent (no network);
run with ``python -m pytest -q -p no:cacheprovider benchmarks/tests``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from benchmarks import crosscheck, matrix, render, run
from benchmarks.baselines import TrainingCentroid
from benchmarks.labels import run_labels
from benchmarks.protocols import LEAVE_ONE_USER_OUT, POINT_KFOLD_5, WITHIN_MONTH, stratified_kfold
from indoorloc.core import SampleTable
from indoorloc.datasets import Dataset

BENCH = Path(__file__).resolve().parents[1]
RESULTS, DOCS = BENCH / "results", BENCH.parent / "docs"
DATA_ROOT = Path(os.environ.get("INDOORLOC_DATA") or Path.home() / ".cache" / "indoorloc" / "datasets")


def _table(n=60, **groups):
    rng = np.random.default_rng(0)
    return SampleTable(rng.normal(size=(n, 4)).astype(np.float32), rng.uniform(0, 10, size=(n, 2)),
                       groups={k: np.asarray(v) for k, v in groups.items()}, ids=np.arange(n).astype(str))


# --------------------------------------------------------------------------- protocols
def test_stratified_kfold_tests_every_row_once_and_balances_classes():
    y = np.repeat([3, 1, 2], [23, 11, 7])  # unsorted class values, sizes not multiples of 5
    folds = stratified_kfold(y, 5, random_state=0)
    tested = np.concatenate([te for _, te in folds])
    assert sorted(tested.tolist()) == list(range(len(y)))
    sizes = [len(te) for _, te in folds]
    assert max(sizes) - min(sizes) <= 1
    for c, count in ((1, 11), (2, 7), (3, 23)):
        per_fold = [int(np.sum(y[te] == c)) for _, te in folds]
        assert max(per_fold) - min(per_fold) <= 1 and sum(per_fold) == count
    for tr, te in folds:
        assert not np.intersect1d(tr, te).size and len(tr) + len(te) == len(y)
        assert np.all(np.diff(tr) > 0) and np.all(np.diff(te) > 0) and tr.dtype == te.dtype == np.int64
    again = stratified_kfold(y, 5, random_state=0)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(folds, again))
    other = stratified_kfold(y, 5, random_state=1)
    assert any(not np.array_equal(a[1], b[1]) for a, b in zip(folds, other))


def test_stratified_kfold_refuses_too_small_classes():
    with pytest.raises(ValueError, match="fewer than n_splits"):
        stratified_kfold([0, 0, 0, 0, 0, 1, 1], 5)


def test_leave_one_user_out_has_one_fold_per_user():
    users = np.repeat([4, 1, 7], 20)
    folds = LEAVE_ONE_USER_OUT.folds(_table(user=users))
    assert [f.name for f in folds] == ["user=1", "user=4", "user=7"]
    for f in folds:
        value = int(f.name.split("=")[1])
        assert np.all(users[f.test] == value) and not np.any(users[f.train] == value)


def test_point_kfold_never_splits_a_point():
    points = np.repeat(np.arange(12), 5).astype(str)
    folds = POINT_KFOLD_5.folds(_table(point=points), random_state=0)
    assert len(folds) == 5
    assert sorted(np.concatenate([f.test for f in folds]).tolist()) == list(range(60))
    for f in folds:
        assert not set(points[f.train]) & set(points[f.test])


def test_within_month_trains_and_tests_inside_each_month():
    month = np.repeat([1, 2, 3], 20)
    split = np.tile(np.repeat(["train", "test"], 10), 3)
    folds = WITHIN_MONTH.folds(_table(month=month, split=split))
    assert [f.name for f in folds] == ["month=1", "month=2", "month=3"]
    for m, f in zip((1, 2, 3), folds):
        assert np.all(month[f.train] == m) and np.all(month[f.test] == m)
        assert np.all(split[f.train] == "train") and np.all(split[f.test] == "test")


def test_protocols_name_the_missing_group():
    with pytest.raises(ValueError, match="groups\\['user'\\]"):
        LEAVE_ONE_USER_OUT.folds(_table(point=np.zeros(60)))


# --------------------------------------------------------------------------- baseline
def test_training_centroid_known_result():
    pos = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]])
    model = TrainingCentroid().fit(np.zeros((4, 3)), pos, floor=[2, 1, 1, 2], building=[5, 5, 5, 6])
    pred = model.localize(np.full((3, 3), np.nan))  # the scan is never read: NaN is accepted
    np.testing.assert_array_equal(pred.pos, [[2.0, 1.0]] * 3)
    assert pred.floor.tolist() == [1, 1, 1]  # tie between floors 1 and 2: the smallest label
    assert pred.building.tolist() == [5, 5, 5]
    np.testing.assert_allclose(pred.spread, np.sqrt(5.0))  # every corner is sqrt(4 + 1) from (2, 1)


def test_training_centroid_runs_through_the_command_line(tmp_path):
    from indoorloc.cli.benchmark import run_benchmark

    payload = run_benchmark("synthetic_office", [matrix.BASELINE, "wknn"], protocol="official", seed=0,
                            dataset_options={"seed": 3, "samples_per_point": 2, "n_test": 40})
    base, wknn = payload["methods"]
    assert base["pooled"]["mean_error"] > wknn["pooled"]["mean_error"]  # a real method beats the constant answer
    assert base["pooled"]["n"] == wknn["pooled"]["n"] == 40


# --------------------------------------------------------------------------- matrix
def test_matrix_cells_resolve():
    from indoorloc.cli.benchmark import build_preprocess, parse_spec
    from indoorloc.datasets import DATASETS
    from indoorloc.evaluation import get_protocol
    from indoorloc.methods import METHODS

    seen = set()
    for table in matrix.tables():
        assert table.key not in seen, f"duplicate table {table.key}"
        seen.add(table.key)
        DATASETS.get(table.dataset)
        if table.runner == "cli":
            get_protocol(table.protocol)
        else:
            assert table.runner == "labels" and table.label
        cells = list(table.cells())
        assert len(cells) == len(set(cells)), f"duplicate cell in {table.key}"
        for pre, spec in cells:
            build_preprocess(pre)
            assert pre in matrix.PREPROCESS, f"no display name for preprocessing {pre!r}"
            assert spec in matrix.DISPLAY, f"no display name for method {spec!r}"
            name, _ = parse_spec(spec.replace("{anchors}", "[[0,0],[1,0],[0,1]]"))
            METHODS.get(name)
        for s in table.skips:
            assert s.en and s.zh
        for en, zh in table.notes:
            assert en and zh
    assert set(matrix.DATASET_ORDER) == {t.dataset for t in matrix.TABLES}


def test_horus_also_runs_on_the_raw_readings_of_every_rssi_table():
    """Horus models "not heard" itself: wherever it runs after the -104 dBm fill, it also runs on the
    raw readings (``--preprocess none``, NaN = not heard), the input it is designed for."""
    with_fill = [t for t in matrix.tables() if ("fill", "horus") in set(t.cells())]
    assert len(with_fill) == 16  # every RSSI table (the CSI tables fill -15 dB amplitudes, not RSSI)
    for table in with_fill:
        assert ("none", "horus") in set(table.cells()), table.key
        assert next(table.cells())[0] == "fill", table.key  # the table's main preprocessing stays first
    assert "NaN = not heard" in matrix.display_preprocess("none")
    bbil = next(t for t in matrix.tables(["ble_indoor"]) if t.id == "official-office")
    assert [m for pre, m in bbil.cells() if pre == "none"] == ["horus", matrix.PATHLOSS_KNOWN, matrix.PATHLOSS_FREE,
                                                               matrix.CENTROID_KNOWN]


def test_command_is_the_public_command_line(tmp_path):
    table = next(t for t in matrix.tables(["tuji1"]))
    argv = run.command(table, "positive", "knn(k=1)", tmp_path / "cell.json")
    assert argv[:3] == ["-m", "indoorloc", "benchmark"] and "--no-download" in argv
    shown = run._shown(argv)
    assert shown.startswith("indoorloc benchmark --dataset tuji1 --protocol official --preprocess positive")
    assert "--out" not in shown and "'knn(k=1)'" in shown


def test_compare_cells_reports_the_largest_difference():
    a = {"pooled": {"mean_error": 1.0, "floor_accuracy": None}, "folds": [{"metrics": {"mean_error": 1.0}}]}
    b = {"pooled": {"mean_error": 1.25, "floor_accuracy": None}, "folds": [{"metrics": {"mean_error": 1.0}}]}
    assert run.compare_cells(a, a) == (2, 0.0, None)
    count, worst, where = run.compare_cells(a, b)
    assert worst == 0.25 and where == "pooled.mean_error"


def test_rerun_check_records_the_comparison():
    a = {"created": "t0", "source_sha256": "s", "pooled": {"mean_error": 2.0}, "folds": []}
    same = run.rerun_check(a, {"pooled": {"mean_error": 2.0}, "folds": []})
    assert same == {"previous_created": "t0", "previous_source_sha256": "s", "numbers_compared": 1,
                    "largest_difference": 0.0, "where": None}
    assert run.rerun_check(a, {"pooled": {"mean_error": 2.5}, "folds": []})["largest_difference"] == 0.5
    missing = run.rerun_check(a, {"pooled": {}, "folds": []})  # a metric disappeared: not comparable
    assert missing["largest_difference"] is None and missing["where"] == "pooled.mean_error"
    json.dumps(missing, allow_nan=False)


# --------------------------------------------------------------------------- machine facts
def test_memory_stall_and_cgroup_limits_are_read_from_the_cgroup(tmp_path, monkeypatch):
    (tmp_path / "memory.pressure").write_text("some avg10=1.00 avg60=0.50 avg300=0.10 total=2500000\n"
                                              "full avg10=0.50 avg60=0.20 avg300=0.05 total=1000000\n")
    (tmp_path / "memory.high").write_text("4294967296\n")
    (tmp_path / "memory.max").write_text("max\n")
    (tmp_path / "memory.swap.max").write_text("0\n")
    monkeypatch.setattr(run, "_cgroup_dir", lambda: tmp_path)
    assert run.memory_stall_us() == (2500000, 1000000)
    assert run.cgroup_limits() == {"memory_high_mb": 4096.0, "memory_max_mb": None, "swap_max_mb": 0.0}
    monkeypatch.setattr(run, "_cgroup_dir", lambda: None)  # not Linux, or no cgroup v2
    assert run.memory_stall_us() is None and run.cgroup_limits() is None


def test_run_cell_records_outcome_times_and_limits(tmp_path):
    ok = run.run_cell(["-c", "sum(range(10**6))"], cwd=tmp_path, log=tmp_path / "ok.log", timeout=60,
                      max_rss_mb=10_000)
    assert ok["status"] == "ok" and ok["reason"] is None and ok["wall_time_s"] > 0
    assert ok["cpu_time_s"] is not None and ok["peak_rss_mb"] > 0 and len(ok["load_avg_1min"]) == 2
    slow = run.run_cell(["-c", "import time; time.sleep(30)"], cwd=tmp_path, log=tmp_path / "slow.log",
                        timeout=0.5, max_rss_mb=10_000)
    assert slow["status"] == "timeout" and "limit" in slow["reason"] and slow["wall_time_s"] < 20
    bad = run.run_cell(["-c", "raise SystemExit('boom')"], cwd=tmp_path, log=tmp_path / "bad.log", timeout=60,
                       max_rss_mb=10_000)
    assert bad["status"] == "failed" and bad["reason"] == "boom"


def test_resume_skips_cells_that_have_a_result(tmp_path, capsys):
    table = matrix.tables(["tuji1"])[0]
    cells = list(table.cells())
    done = {"preprocess": cells[0][0], "method": cells[0][1], "status": "ok"}
    timed_out = {"preprocess": cells[1][0], "method": cells[1][1], "status": "timeout"}
    (tmp_path / "tuji1.json").write_text(json.dumps({"tables": [{"id": table.id, "cells": [done, timed_out]}]}))
    run.main(["--dataset", "tuji1", "--resume", "--list", "--results-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "1 cells already have a result" in out and f"{len(cells) - 1} cells" in out  # the timeout runs again


def test_rerun_legacy_selects_finished_cells_of_an_older_harness(tmp_path, capsys):
    table = matrix.tables(["tuji1"])[0]
    (pre0, m0), (pre1, m1), (pre2, m2) = list(table.cells())[:3]
    stored = [{"preprocess": pre0, "method": m0, "status": "ok"},  # older harness: no cpu_time_s
              {"preprocess": pre1, "method": m1, "status": "ok", "cpu_time_s": 1.0},
              {"preprocess": pre2, "method": m2, "status": "timeout"}]
    (tmp_path / "tuji1.json").write_text(json.dumps({"tables": [{"id": table.id, "cells": stored}]}))
    run.main(["--dataset", "tuji1", "--rerun-legacy", "--list", "--results-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "1 cells recorded by an older harness" in out and "\n1 cells" in out
    assert run._shown(run.command(table, pre0, m0, tmp_path / "x.json")) in out


# --------------------------------------------------------------------------- label runner
class _Rooms(Dataset):
    """Four rooms, each with its own mean RSSI pattern (synthetic)."""

    name = "rooms_test"
    files = {"all": ()}
    meta = {"modality": "wifi_rssi", "units": "dBm", "crs": None, "pos_names": (), "task": "room_classification"}

    def _parse(self, paths, split):
        rng = np.random.default_rng(1)
        room = np.repeat([1, 2, 3, 4], 25)
        centres = rng.uniform(-90, -40, size=(4, 6))
        X = (centres[room - 1] + rng.normal(0, 2.0, size=(100, 6))).astype(np.float32)
        return SampleTable(X, np.zeros((100, 0)), groups={"room": room}, ids=np.arange(100).astype(str))


def test_label_runner_scores_rooms(tmp_path):
    doc = run_labels(f"{__name__}:_Rooms", "wknn", label="room", seed=0)
    m = doc["methods"][0]
    assert m["pooled"]["n"] == 100 and m["pooled"]["accuracy"] == 100.0  # well separated rooms
    assert len(m["folds"]) == 5 and sum(f["metrics"]["n"] for f in m["folds"]) == 100
    assert doc["protocol"]["name"] == "stratified-kfold-5" and doc["preprocess"]["spec"] == "fill"
    base = run_labels(f"{__name__}:_Rooms", matrix.BASELINE, label="room")["methods"][0]
    assert base["pooled"]["accuracy"] == 25.0  # training majority, one room per test row out of four
    json.dumps(doc, allow_nan=False)


# --------------------------------------------------------------------------- rendering
def _fake_doc():
    cell = lambda method, mean, floor: {  # noqa: E731
        "preprocess": "fill", "method": method, "display": [matrix.display(method, 0), matrix.display(method, 1)],
        "status": "ok", "command": f"indoorloc benchmark --method {method}", "wall_time_s": 1.0, "peak_rss_mb": 100.0,
        "fit_s": 0.5, "predict_s": 0.25, "source_sha256": "ab" * 32,
        "pooled": {"mean_error": mean, "median_error": mean, "p75_error": mean, "p90_error": mean,
                   "floor_accuracy": floor, "building_accuracy": 100.0, "ipin_score": mean + 1},
        "folds": [{"fold": "official", "metrics": {"mean_error": mean}}]}
    return {"format": run.FORMAT, "dataset": "ujiindoorloc", "source_sha256": ["ab" * 32],
            "harness": {"hardware": {"cpu": "cpu", "logical_cpus": 4, "memory_gb": 8, "platform": "linux"},
                        "thread_env": run.THREAD_ENV, "timeout_s": 60, "max_rss_mb": 100,
                        "code": {"git": {"commit": "0" * 40, "branch": "b", "dirty": False}}},
            "environment": {"python": "3", "numpy": "2", "packages": {"sklearn": "1"}},
            "tables": [{"id": "official", "title": {"en": "UJI table", "zh": "UJI 表"}, "runner": "cli",
                        "protocol_spec": "official", "options": {},
                        "dataset": {"name": "ujiindoorloc", "n_samples": {"train": 10, "test": 5},
                                    "sha256": {"train": "c" * 64, "test": "d" * 64}},
                        "protocol": {"name": "official", "summary": "own split",
                                     "folds": [{"n_train": 10, "n_test": 5, "n_unused": 0}]},
                        "units": "m", "skips": [{"method": "svm", "en": "too slow", "zh": "太慢"}],
                        "notes": [{"en": "a note", "zh": "说明"}],
                        "cells": [cell(matrix.BASELINE, 50.0, 40.0), cell("wknn", 8.7937, 90.5),
                                  cell("knn", 8.8084, 90.3), {"preprocess": "fill", "method": "svm",
                                                              "status": "timeout", "reason": "30 minutes"}]}],
            "literature": {"available": True, "display_name": "UJIIndoorLoc", "hidden": 1,
                           "checks_all": {"literature/verified": 1, "literature/unidentified": 1},
                           "entries": [{"method": "k-NN", "values": {"mean_error": 7.9}, "protocol": "official",
                                        "check": "verified", "location": "Table 1",
                                        "source": {"authors": "A. B, C. D, E. F", "year": 2014, "venue": "IPIN",
                                                   "doi": "10/x"}}]}}


def test_render_keeps_measured_and_published_numbers_apart():
    docs = {"ujiindoorloc": _fake_doc()}
    en = render.document(docs, None, render.EN)
    zh = render.document(docs, None, render.ZH)
    assert "**8.794**" in en and "8.808" in en  # the best non-baseline mean is bold
    assert "| WKNN (k=5) | fill -104 dBm | **8.794** |" in en
    assert "Floor %" in en and "IPIN score" in en and "Building %" not in en  # one building in the test rows
    assert "not finished" in en and "svm: too slow" in en and "a note" in en
    measured, published = en.split("#### Published numbers")
    assert "7.9" not in measured.split("UJI table")[1] and "mean 7.9 m" in published and "`verified`" in published
    assert "1 further entries are not shown" in published
    assert "UJI 表" in zh and "已发表结果" in zh and "太慢" in zh and "未完成" in zh
    assert "yes" in en.split("Cross-checks")[1].split("##")[0]  # WKNN 8.7937 agrees with the recorded value


def test_render_marks_times_stretched_by_memory_throttling():
    doc = _fake_doc()
    doc["harness"]["cgroup"] = {"memory_high_mb": 4096.0, "memory_max_mb": 6144.0, "swap_max_mb": 4096.0}
    cells = doc["tables"][0]["cells"]
    cells[1]["memory_stall_s"] = {"some": 0.9, "full": 0.5}  # half of its 1 s wall time stalled: marked
    cells[2]["memory_stall_s"] = {"some": 0.05, "full": 0.0}
    cells[1]["rerun_check"] = {"largest_difference": 0.0}
    en = render.document({"ujiindoorloc": doc}, None, render.EN)
    assert "| WKNN (k=5) | fill -104 dBm | **8.794** |" in en and "0.50 † | 0.25 † |" in en
    assert "| k-NN (k=5) | fill -104 dBm | 8.808 | 8.808 | 8.808 | 8.808 | 90.30 | 9.808 | 0.50 | 0.25 |" in en
    assert "memory.high 4,096 MB" in en and "PSI `some`, the stricter of the two totals; 1 of 2 finished cells)" in en
    assert "1 cells were run twice" in en and "(identical)" in en
    zh = render.document({"ujiindoorloc": doc}, None, render.ZH)
    assert "内存限流" in zh and "0.50 †" in zh


def test_summary_and_observations_use_each_tables_main_preprocessing():
    doc = _fake_doc()
    tb = doc["tables"][0]
    for c in tb["cells"]:
        c["preprocess"] = "CSIAmplitude"  # a CSI table: no 'fill' cells at all
    tb["cells"][2]["pooled"] = {**tb["cells"][2]["pooled"], "mean_error": 60.0}  # k-NN worse than the baseline (50)
    en = render.document({"ujiindoorloc": doc}, None, render.EN)
    summary = en.split("## Summary")[1].split("##")[0]
    assert "| 50.000 | 8.794 | WKNN (k=5) (\\|CSI\\| (linear)): 8.794 |" in summary  # WKNN column found
    observations = en.split("## Observations")[1].split("## Setup")[0]
    assert "  - [ujiindoorloc / official](#ujiindoorloc-official): k-NN (k=5)\n" in observations
    assert "positive representation" not in observations  # no representation rows in a CSI table


def test_observations_compare_raw_and_filled_horus():
    doc = _fake_doc()
    cells = doc["tables"][0]["cells"]
    horus = {**cells[1], "method": "horus", "display": ["Horus", "Horus"]}
    cells += [{**horus, "pooled": {**horus["pooled"], "mean_error": 9.7}},  # fill: worse than WKNN (8.7937)
              {**horus, "preprocess": "none", "pooled": {**horus["pooled"], "mean_error": 7.99}}]
    en = render.document({"ujiindoorloc": doc}, None, render.EN)
    assert ("- Horus on the raw readings (NaN = not heard, the input its detection model is made for) has a lower "
            "mean error than Horus after the -104 dBm fill on 1 of 1 tables, and than WKNN on 1 of 1.") in en
    assert "| Horus | raw RSSI, NaN = not heard | **7.990** |" in en
    assert "Horus (raw RSSI, NaN = not heard): 7.990 |" in en.split("## Summary")[1].split("##")[0]
    zh = render.document({"ujiindoorloc": doc}, None, render.ZH)
    assert "在原始读数上" in zh and "1 / 1 个表；低于 WKNN：1 / 1 个表" in zh


# --------------------------------------------------------------------------- provenance of partial reruns
def test_merge_harness_keeps_the_code_of_cells_that_were_not_rerun():
    old = {"timeout_s": 60, "code": {"source_sha256": "A", "taken": "t0", "frozen": True}}
    new = {"timeout_s": 60, "code": {"source_sha256": "B", "taken": "t1", "frozen": True}}
    merged = run.merge_harness(old, new, {"A", "B"})
    assert merged["code"]["source_sha256"] == "B" and [c["source_sha256"] for c in merged["code_history"]] == ["A"]
    again = run.merge_harness(merged, {**new, "code": {"source_sha256": "C"}}, {"A", "C"})  # B no longer used
    assert [c["source_sha256"] for c in again["code_history"]] == ["A"]
    assert "code_history" not in run.merge_harness(merged, new, {"B"})  # every cell rerun: history dropped
    assert run.merge_harness(merged, None, {"A", "B"}) == merged  # --remerge keeps the stored block


def test_merge_records_both_code_versions_after_a_partial_rerun(tmp_path):
    table = matrix.tables(["tuji1"])[0]
    (pre0, m0), (pre1, m1) = list(table.cells())[:2]
    cell = lambda pre, m, d: {"preprocess": pre, "method": m, "status": "ok", "source_sha256": d,  # noqa: E731
                              "pooled": {"mean_error": 1.0}, "folds": [], "folds_sha256": "f"}
    run.merge("tuji1", {(table.id, pre0, m0): cell(pre0, m0, "A")}, {}, {}, tmp_path,
              {"code": {"source_sha256": "A", "frozen": True, "taken": "2026-01-01T00:00:00"}})
    doc = run.merge("tuji1", {(table.id, pre1, m1): cell(pre1, m1, "B")}, {}, {}, tmp_path,
                    {"code": {"source_sha256": "B", "frozen": False, "taken": "2026-02-01T00:00:00"}})
    assert doc["source_sha256"] == ["A", "B"] and doc["harness"]["code_history"][0]["source_sha256"] == "A"
    versions = render.code_versions({"tuji1": doc})
    assert [(d, n) for d, n, _ in versions] == [("A", 1), ("B", 1)] and all(code for _, _, code in versions)
    harness = {**doc["harness"], "hardware": {"cpu": "c", "logical_cpus": 1, "memory_gb": 1, "platform": "p"},
               "thread_env": run.THREAD_ENV, "timeout_s": 60, "max_rss_mb": 100}
    code = next(ln for ln in render.setup_section({"tuji1": {**doc, "harness": harness}}, None, render.EN)
                if ln.startswith("- Code:"))
    assert "ran 2 versions" in code and "`A` (1 cell): a copy frozen" in code
    assert "`B` (1 cell): the working tree" in code


def test_verification_fails_when_a_rerun_does_not_finish_or_differs():
    ok = {"status": "ok", "source_sha256": "A", "pooled": {"mean_error": 2.0}, "folds": []}
    fresh = {"tuji1": {("official", "fill", "wknn"): {**ok, "source_sha256": "B"},
                       ("official", "fill", "knn"): {"status": "failed", "reason": "boom"},
                       ("official", "fill", "svm"): {**ok, "pooled": {"mean_error": 2.5}},
                       ("official", "fill", "mlp"): ok}}
    stored = {("tuji1", "official", "fill", m): ok for m in ("wknn", "knn", "svm")}
    report, mismatches = run.verification_report(fresh, stored)
    by = {r["method"]: r for r in report}
    assert mismatches == 2  # the crashed rerun and the changed number; the never-stored cell is no mismatch
    assert by["wknn"]["largest_difference"] == 0.0 and by["wknn"]["rerun_source_sha256"] == "B"
    assert by["knn"]["largest_difference"] is None and "did not finish" in by["knn"]["where"]
    assert by["svm"]["largest_difference"] == 0.5 and by["mlp"]["comparable"] is False
    json.dumps(report, allow_nan=False)
    line = render.verification_line({"created": "2026-09-28", "cells": report}, render.EN)
    assert "4 cells" in line and "0.5" in line and "2 could not be compared" in line and "`B`" in line


# --------------------------------------------------------------------------- independent recomputation
def test_numpy_knn_orders_ties_by_training_index_and_matches_the_library():
    from indoorloc.methods import create_model

    A = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 5.0], [5.0, 0.0], [4.0, 3.0], [6.0, 8.0]])
    P = np.arange(12, dtype=float).reshape(6, 2)
    Q = np.array([[0.0, 0.0], [1.0, 1.0]])
    est, info = crosscheck.knn_positions(A, P, Q, 2, "uniform")
    # query 0: row 0 (d=0), then rows 1-4 all at d=5: training index order picks row 1
    np.testing.assert_array_equal(est[0], P[[0, 1]].mean(0))
    # query 1 = (1, 1): rows 1 and 4 tie at d^2 = 13 for the second place, so both queries tie at k
    assert info["mode"] == "exact-gemm" and info["grid_bits"] == 0 and info["ties_at_k"] == 2
    est, _ = crosscheck.knn_positions(A, P, Q, 2, "distance")
    np.testing.assert_array_equal(est[0], P[0])  # an exact match takes all the weight
    rng = np.random.default_rng(0)
    A = rng.integers(-100, -40, size=(200, 7)).astype(float)
    P, Q = rng.uniform(0, 30, size=(200, 2)), rng.integers(-100, -40, size=(25, 7)).astype(float)
    for spec, weights in (("knn", "uniform"), ("wknn", "distance")):
        lib = create_model(spec).fit(A, P).predict(Q)
        np.testing.assert_allclose(crosscheck.knn_positions(A, P, Q, 5, weights)[0], lib, rtol=0, atol=1e-12)


def test_numpy_knn_distance_modes():
    # BBIL-like: 9 receivers, readings in [-110, -35] dBm averaged to thirds and stored as float32 (common
    # grid 2**-18); the scaled magnitudes are too large for an exact matrix product, the differences are not
    rng = np.random.default_rng(1)
    thirds = np.float32(rng.integers(-330, -105, size=(40, 9)) / 3.0)
    _, info = crosscheck.knn_positions(thirds, np.zeros((40, 2)), thirds[:3], 1, "uniform")
    assert info["mode"] == "exact-diff" and info["grid_bits"] == 18
    _, info = crosscheck.knn_positions(rng.random((50, 52)), np.zeros((50, 3)), rng.random((5, 52)), 5, "distance")
    assert info["mode"] == "float" and info["near_ties"] == 0 and info["grid_bits"] is None
    with pytest.raises(ValueError, match="weights"):
        crosscheck.knn_positions(np.zeros((3, 1)), np.zeros((3, 1)), np.zeros((1, 1)), 1, "gaussian")


def test_positive_representation_is_the_papers_definition():
    train = np.array([[-80.0, np.nan], [-60.0, -95.0]])
    # min = lowest training reading - 1 = -96; heard readings -> x - min, missing -> 0
    np.testing.assert_array_equal(crosscheck.positive(train, np.array([[-70.0, np.nan], [-96.0, -100.0]])),
                                  [[26.0, 0.0], [0.0, 0.0]])


@pytest.mark.skipif(not (DATA_ROOT / "ujiindoorloc" / "validationData.csv").is_file(),
                    reason="UJIIndoorLoc files not found")
def test_uji_knn_cells_are_reproduced_outside_the_library():
    stored = render.load_docs(RESULTS)["ujiindoorloc"]
    for check in crosscheck.CHECKS[:2]:
        row = crosscheck.recompute(check, DATA_ROOT / "ujiindoorloc")
        cell = render._cell(stored["tables"], check.table, check.preprocess, check.method)
        assert row["mode"] == "exact-gemm" and row["n_test"] == 1111
        assert row["numpy_mean_error"] == pytest.approx(cell["pooled"]["mean_error"], rel=1e-12)


# --------------------------------------------------------------------------- the committed results and docs
def _stored():
    return render.load_docs(RESULTS)


def test_results_cover_every_cell_of_the_matrix():
    docs = _stored()
    assert set(docs) == set(matrix.DATASET_ORDER)
    for table in matrix.tables():
        doc = docs[table.dataset]
        tb = next(t for t in doc["tables"] if t["id"] == table.id)
        assert [(c["preprocess"], c["method"]) for c in tb["cells"]] == list(table.cells()), table.key
        assert [s["method"] for s in tb["skips"]] == [s.method for s in table.skips], table.key
        assert tb["same_folds_in_every_cell"], table.key
        known = {c["source_sha256"] for c in [doc["harness"]["code"], *doc["harness"].get("code_history", [])]}
        for c in tb["cells"]:
            assert c["status"] in ("ok", "timeout", "memory", "failed"), f"{table.key} {c['method']}: {c['status']}"
            if c["status"] == "ok":
                assert c["source_sha256"] in doc["source_sha256"] and c["source_sha256"] in known
                assert c["folds_sha256"] and ("accuracy" if table.runner == "labels" else "mean_error") in c["pooled"]
            else:
                assert c["reason"] and c["cpu_time_s"] is not None


def test_cross_checks_agree():
    cc = json.loads((RESULTS / "crosscheck.json").read_text(encoding="utf-8"))
    lines = "\n".join(render.checks_section(_stored(), render.EN, cc)).splitlines()
    table = [ln for ln in lines if ln.startswith("| ")][2:]  # after the header and the rule
    assert len(table) == len(render.CHECKS) and all(ln.endswith("| yes |") for ln in table)
    assert {(r["dataset"], r["table"], r["preprocess"], r["method"]) for r in cc["cells"]} == \
        {c[:4] for c in render.CHECKS}  # every cross-checked cell is recomputed outside the library


def test_committed_docs_are_generated_from_the_results():
    docs = _stored()
    verification, cc = (render._load_optional(RESULTS / f"{n}.json") for n in ("verification", "crosscheck"))
    for lang, name in ((render.EN, "benchmarks.md"), (render.ZH, "benchmarks_zh.md")):
        assert (DOCS / name).read_text(encoding="utf-8") == render.document(docs, verification, lang, cc), \
            f"docs/{name} is stale: run python benchmarks/render.py"


def test_changelog_quotes_the_matrix_counts():
    stats = render._run_stats(_stored())
    verified = json.loads((RESULTS / "verification.json").read_text(encoding="utf-8"))["cells"]
    text = " ".join((BENCH.parent / "CHANGELOG.md").read_text(encoding="utf-8").split())
    assert (f"Of the {stats['run']} cells, {stats['ok']} finished and {stats['limit']} stopped at the 2,500 MB "
            f"memory limit. {stats['reruns']} cells were run twice in separate processes, and {len(verified)} were "
            "run again") in text
