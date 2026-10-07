from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from indoorloc.evaluation import evaluate, literature
from indoorloc.evaluation.literature import CHECKS, compare, list_tables, load, table

LIT_DIR = Path(literature.__file__).parent
# entries per indoorloc 0.1 evaluation/benchmark_data/<file>.py (commented-out entries excluded)
PORTED_FROM_01 = {"ujiindoorloc": 14, "tampere": 8, "sodindoorloc": 7, "tuji1": 4, "longtermwifi": 5,
                  "ble_rssi_uci": 6, "wlanrssi": 4, "ibeacon_rssi": 2, "ble_indoor": 2, "csi_fingerprint": 2,
                  "csiindoor": 2, "magneticindoor": 2, "csi2taoa": 2, "wildv2": 2, "wificsid2d": 2}
KINDS = ("literature", "indoorloc-0.1", "indoorloc-0.1-demo")


@pytest.mark.parametrize("path", sorted(LIT_DIR.glob("*.json")), ids=lambda p: p.stem)
def test_every_file_follows_the_schema(path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["format"] == "indoorloc-literature" and doc["format_version"] == 1 and doc["dataset"] == path.stem
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", doc["checked_on"]) and doc["ported_from"].startswith("indoorloc 0.1")
    for e in doc["entries"]:
        assert e["check"] in CHECKS and e["kind"] in KINDS, e["method"]
        assert set(e["values"]) <= set(doc["metrics"]), (e["method"], set(e["values"]) - set(doc["metrics"]))
        for metric, value in e["values"].items():
            assert value is None or isinstance(value, (int, float))
            if value is not None and doc["metrics"][metric]["unit"] == "%":
                assert 1.0 < value <= 100.0, (e["method"], metric, value)  # 0.1 fractions became percent
        if e["check"] in ("verified", "corrected"):
            assert e["location"] and e["source"]["doi"], e["method"]  # a checked number says where it is
        if e["check"] in ("verified", "corrected", "unchecked", "missing"):
            assert e["source"] and (e["source"]["doi"] or e["source"]["url"]) and e["source"]["year"]
        if e["kind"] != "literature":
            assert e["check"] == "not-literature"
        if e["check"] == "missing":
            assert all(v is None for v in e["values"].values())


def test_every_01_entry_is_ported_with_its_original_values():
    assert set(PORTED_FROM_01) == set(list_tables())
    for name, count in PORTED_FROM_01.items():
        ported = [e["ported"] for e in load(name)["entries"] if e["ported"] is not None]
        assert len(ported) == count, name
        assert all({"citation", "values", "notes", "is_sota"} <= set(p) for p in ported)


def test_corrections_found_while_porting_are_recorded():
    uji = {e["method"]: e for e in load("ujiindoorloc")["entries"]}
    kim = uji["Scalable DNN (hierarchical)"]
    assert kim["check"] == "corrected" and kim["values"]["floor_accuracy"] == 91.27
    assert kim["ported"]["values"]["floor_accuracy"] == 0.924
    assert uji["GBDT + sample differences"]["source"]["authors"].startswith("X. Cao")  # 0.1 said "Li et al."
    assert uji["CCpos (CDAE-CNN)"]["check"] == "verified"
    tuji = {e["method"].split(" (")[0]: e for e in load("tuji1")["entries"]}
    assert tuji["k-NN, tuned"]["values"] == {"mean_error_3d": 2.27}  # 0.1 had 1.8 m
    assert tuji["1-NN"]["values"] == {"mean_error_3d": 3.34}  # 0.1 had 2.5 m


def test_aliases_and_default_selection():
    assert table("uji").dataset == table("UJIndoorLoc").dataset == "ujiindoorloc"
    default = table("ujiindoorloc")
    assert all(e["check"] in ("verified", "corrected", "unchecked") and e["kind"] == "literature"
               for e in default.entries)
    assert default.excluded == 2 and len(table("ujiindoorloc", include="all")) == 20
    means = [e["values"]["mean_error"] for e in default.entries if e["values"].get("mean_error") is not None]
    assert means == sorted(means)  # best first by the primary metric
    assert len(table("tuji1")) == 2 and len(table("tuji1", kinds=("literature", "indoorloc-0.1"))) == 4
    assert len(table("csiindoor", include="all")) == 0 and len(table("csiindoor", kinds="all")) == 2  # demo data
    with pytest.raises(KeyError, match="no literature table"):
        table("nope")
    with pytest.raises(ValueError, match="unknown check"):
        table("uji", include=("great",))


def test_compare_keeps_reproduced_and_published_numbers_apart():
    res = evaluate([[0, 0], [3, 4]], [[0, 1], [0, 0]], floor_true=[0, 1], floor_pred=[0, 1],
                   building_true=[0, 0], building_pred=[0, 0])
    comparison = compare({"knn": res}, "ujiindoorloc")
    d = comparison.to_dict()
    assert d["reproduced"] == [{"method": "knn", "mean_error": 3.0, "floor_accuracy": 100.0,
                                "building_accuracy": 100.0}]
    assert d["metrics"] == ["mean_error", "floor_accuracy", "building_accuracy"]
    same = {e["method"]: e["same_protocol"] for e in d["literature"]}
    assert same["k-NN (k=1), dataset baseline"] is True and same["P-MIMO LSTM"] is False
    assert same["GBDT + sample differences"] is None  # the entry does not record the paper's protocol
    assert not any(k in d for k in ("rank", "ranking", "sota", "beats"))  # no verdicts across protocols
    text = comparison.to_markdown()
    assert text.index("Reproduced with indoorloc") < text.index("Literature on UJIIndoorLoc")
    assert "EPSG:3857" in text and "not directly comparable" in text
    single = compare(res, "uji", label="mine").to_dict()["reproduced"][0]
    assert single["method"] == "mine" and single["mean_error"] == 3.0
    assert compare(res.to_dict(), "uji").to_dict()["reproduced"][0]["mean_error"] == 3.0


def test_rendered_tables_name_the_check_and_the_source():
    text = table("ujiindoorloc").to_text()
    assert "as reported; not re-run" in text and "doi:10.1186/s41044-018-0031-2" in text and "corrected" in text
    md = table("ble_rssi_uci").to_markdown()
    assert md.count("| verified |") == 11 and "Sun et al. (2021)" in md


def test_compare_shows_a_published_number_reported_under_another_metric():
    # TUJI1's paper reports mean 3-D errors only: they get their own column instead of vanishing
    d = compare({"knn": {"mean_error": 2.5}}, "tuji1").to_dict()
    assert d["metrics"] == ["mean_error_3d", "mean_error"]
    assert d["reproduced"] == [{"method": "knn", "mean_error_3d": None, "mean_error": 2.5}]
    assert sorted(e["values"]["mean_error_3d"] for e in d["literature"]) == [2.27, 3.34]
    assert all(e["same_protocol"] is True for e in d["literature"])
    assert compare({"knn": {"mean_error": 2.5}}, "tuji1", metrics=["mean_error"]).metrics == ("mean_error",)
