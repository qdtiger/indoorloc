"""L1 WiFi RSSI loaders: SODIndoorLoc, Tampere, TUJI1, LongTermWiFi, WLANRSSI.

Synthetic files that copy each official format exercise the parsers (no network); the
real-data checks at the end run only when the files are in ``$INDOORLOC_DATA``.
"""
from __future__ import annotations

import hashlib
import re
import zipfile
from datetime import datetime, timezone

import numpy as np
import pytest

from conftest import DATA_ROOT
from indoorloc.datasets import DATASETS, dataset_info, load_dataset
from indoorloc.datasets.longtermwifi import LongTermWiFi
from indoorloc.datasets.sodindoorloc import BUILDINGS, SODIndoorLoc
from indoorloc.datasets.tampere import Tampere
from indoorloc.datasets.tuji1 import TUJI1
from indoorloc.datasets.wlanrssi import WLANRSSI

WIFI = {"sodindoorloc": SODIndoorLoc, "tampere": Tampere, "tuji1": TUJI1, "longtermwifi": LongTermWiFi,
        "wlanrssi": WLANRSSI}
LABELS = "ECoord,NCoord,FloorID,BuildingID,SceneID,UserID,PhoneID,SampleTimes"


def utc(*args) -> int:
    return int(datetime(*args, tzinfo=timezone.utc).timestamp())


# ---------------------------------------------------------------- declarations
@pytest.mark.parametrize("name", sorted(WIFI))
def test_class_declarations_follow_the_dataset_contract(name):
    cls = DATASETS.get(name)
    assert cls is WIFI[name] and cls.name == name
    info = dataset_info(name)
    for key in ("modality", "units", "crs", "pos_units", "license", "doi", "citation", "url"):
        assert key in info, key
    assert info["modality"] == "wifi_rssi" and info["units"] == "dBm"
    dataset = cls()
    for split in dataset.splits:
        for rel, sha in dataset._entries(split):
            assert re.fullmatch(r"[0-9a-f]{64}", sha), (rel, sha)  # every file has a real checksum
            if isinstance(cls.urls, dict):
                assert rel in cls.urls, rel  # every declared file can be fetched
    assert cls.urls


def test_sod_declares_every_sheet_with_a_pinned_url():
    assert len(SODIndoorLoc.urls) == 14  # 9 training and 5 testing sheets
    for rel, mirrors in SODIndoorLoc.urls.items():
        assert all(re.search(r"/[0-9a-f]{40}/", url) and url.endswith(rel) for url in mirrors)


# ---------------------------------------------------------------- SODIndoorLoc
def _sod_sheet(path, macs, rows):
    """rows: (rssi per MAC in ``macs`` order, (E, N, floor, building, scene, user, phone, times))."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(list(macs) + LABELS.split(","))]
    lines += [",".join(str(v) for v in list(rssi) + list(labels)) for rssi, labels in rows]
    path.write_text("\n".join(lines) + "\n")


def _sod_root(tmp_path):
    root = tmp_path / "sod"
    # CETC331: columns deliberately out of MAC order, parsed by name
    _sod_sheet(root / "CETC331/Training_CETC331.csv", ["MAC2", "MAC1"],
               [((-40, 100), (45.0, 17.5, 1, 1, 1, 4, 3, 1)), ((100, -71), (46.5, 18.25, 3, 1, 2, 4, 3, 1))])
    _sod_sheet(root / "CETC331/Testing_CETC331.csv", ["MAC2", "MAC1"], [((-44, -60), (45.9, 18.0, 2, 1, 3, 4, 3, 7))])
    _sod_sheet(root / "HCXY/Training_HCXY_All_30.csv", ["MAC1", "MAC2", "MAC10"],
               [((-50, 100, -90), (858.542, 917.094, 4, 2, 1, 5, 4, 1)),
                ((-51, -77, 100), (858.542, 917.094, 4, 2, 1, 6, 9, 30))])
    _sod_sheet(root / "HCXY/Testing_HCXY_All.csv", ["MAC1", "MAC2", "MAC10"],
               [((100, -60, -80), (860.906, 916.251, 4, 2, 1, 10, 9, 2))])
    _sod_sheet(root / "SYL/Training_SYL_All_30.csv", ["MAC1", "MAC2"], [((-30, -45), (19.03, 20.05, 4, 3, 1, 1, 1, 1))])
    _sod_sheet(root / "SYL/Testing_SYL_All.csv", ["MAC1", "MAC2"], [((-31, 100), (19.63, 19.45, 4, 3, 2, 2, 2, 1))])
    return root


def test_sod_stacks_buildings_block_diagonally_with_names_units_and_groups(tmp_path):
    train, test = load_dataset("sodindoorloc", root=_sod_root(tmp_path), download=False, verify=False)
    names = ("CETC331/MAC1", "CETC331/MAC2", "HCXY/MAC1", "HCXY/MAC2", "HCXY/MAC10", "SYL/MAC1", "SYL/MAC2")
    assert train.meta["feature_names"] == names and test.meta["feature_names"] == names
    nan = np.nan
    expected = np.array([[100, -40, nan, nan, nan, nan, nan], [-71, 100, nan, nan, nan, nan, nan],
                         [nan, nan, -50, 100, -90, nan, nan], [nan, nan, -51, -77, 100, nan, nan],
                         [nan, nan, nan, nan, nan, -30, -45]], dtype=np.float32)
    expected[expected == 100] = nan  # the file's "not detected" marker
    np.testing.assert_array_equal(train.X, expected)
    assert train.X.dtype == np.float32 and train.pos.dtype == np.float64
    np.testing.assert_array_equal(train.pos, [[45.0, 17.5], [46.5, 18.25], [858.542, 917.094], [858.542, 917.094],
                                              [19.03, 20.05]])
    np.testing.assert_array_equal(train.floor, [1, 3, 4, 4, 4])
    np.testing.assert_array_equal(train.building, [1, 1, 2, 2, 3])
    np.testing.assert_array_equal(train.groups["device"], [3, 3, 4, 9, 1])
    np.testing.assert_array_equal(train.groups["user"], [4, 4, 5, 6, 1])
    np.testing.assert_array_equal(train.groups["scene"], [1, 2, 1, 1, 1])
    np.testing.assert_array_equal(train.groups["scan_index"], [1, 1, 1, 30, 1])
    assert train.ids.tolist() == ["Training_CETC331-00000", "Training_CETC331-00001", "Training_HCXY_All_30-00000",
                                  "Training_HCXY_All_30-00001", "Training_SYL_All_30-00000"]
    assert test.ids[-1] == "Testing_SYL_All-00000" and len(set(test.ids)) == len(test)
    assert train.meta["crs"] == "local-per-building" and train.meta["pos_units"] == "m"
    assert set(train.meta["sha256"]) == {"CETC331/Training_CETC331.csv", "HCXY/Training_HCXY_All_30.csv",
                                         "SYL/Training_SYL_All_30.csv"}


def test_sod_building_option_selects_sheets_in_canonical_order(tmp_path):
    root = _sod_root(tmp_path)
    hcxy = load_dataset("sodindoorloc", split="test", root=root, download=False, verify=False, building="hcxy")
    assert hcxy.meta["feature_names"] == ("HCXY/MAC1", "HCXY/MAC2", "HCXY/MAC10")
    np.testing.assert_array_equal(hcxy.X, [[np.nan, -60, -80]])
    pair = SODIndoorLoc(root, verify=False, building=["SYL", 1])  # names in any case, or BuildingID
    assert pair.selected == ("CETC331", "SYL")
    assert [len(t) for t in (pair.load("train"), pair.load("test"))] == [3, 2]
    assert SODIndoorLoc(root, building=np.int64(2)).selected == ("HCXY",)
    for bad in ("UJI", [], ["HCXY", 7], 2.0, True, [np.True_]):
        with pytest.raises(ValueError, match="building"):
            SODIndoorLoc(root, building=bad)
    with pytest.raises(ValueError, match="macs"):
        SODIndoorLoc(root, macs="AP")


def test_sod_variant_options_map_to_the_official_sheets():
    files = lambda **kw: [rel for rel, _ in SODIndoorLoc(**kw).files["train"]]  # noqa: E731
    assert files() == ["CETC331/Training_CETC331.csv", "HCXY/Training_HCXY_All_30.csv", "SYL/Training_SYL_All_30.csv"]
    assert files(macs="preinstalled", averaged=True) == [
        "CETC331/Training_CETC331.csv", "HCXY/Training_HCXY_AP_Avg.csv", "SYL/Training_SYL_AP_Avg.csv"]
    syl = SODIndoorLoc(building="SYL", macs="preinstalled")
    assert [rel for rel, _ in syl.files["test"]] == ["SYL/Testing_SYL_AP.csv"]


def test_sod_averaged_sheets_decode_their_own_missing_marker_and_building_erratum(tmp_path):
    root = tmp_path / "sod"
    # SYL averaged sheets substitute -105 for undetected scans; its BuildingID column counts 3, 4, ...
    _sod_sheet(root / "SYL/Training_SYL_All_Avg.csv", ["MAC1", "MAC2"],
               [((-105, -60), (19.03, 18.85, 4, 3, 1, 1, 1, 1)), ((-104, -105), (20.23, 20.05, 4, 4, 1, 1, 1, 1))])
    # HCXY "all" averaged sheet averages the detected scans only and keeps +100
    _sod_sheet(root / "HCXY/Training_HCXY_All_Avg.csv", ["MAC1"], [((100,), (858.5, 917.1, 4, 2, 1, 5, 4, 1)),
                                                                   ((-98,), (859.5, 917.1, 4, 2, 1, 5, 4, 1))])
    syl = SODIndoorLoc(root, verify=False, building="SYL", averaged=True).load("train")
    np.testing.assert_array_equal(syl.X, [[np.nan, -60], [-104, np.nan]])
    np.testing.assert_array_equal(syl.building, [3, 3])
    assert syl.meta["raw_missing_value"] == (-105, 100)  # the markers of the sheets actually loaded
    hcxy = SODIndoorLoc(root, verify=False, building="HCXY", averaged=True).load("train")
    np.testing.assert_array_equal(hcxy.X, [[np.nan], [-98]])
    assert hcxy.meta["raw_missing_value"] == 100


def test_sod_rejects_a_sheet_of_another_building_and_missing_columns(tmp_path):
    root = tmp_path / "sod"
    _sod_sheet(root / "HCXY/Testing_HCXY_All.csv", ["MAC1"], [((-50,), (1.0, 2.0, 4, 3, 1, 5, 4, 1))])
    with pytest.raises(ValueError, match="BuildingID"):
        SODIndoorLoc(root, verify=False, building="HCXY").load("test")
    (root / "HCXY/Testing_HCXY_All.csv").write_text("MAC1,ECoord,NCoord\n-50,1,2\n")
    with pytest.raises(ValueError, match="not found"):
        SODIndoorLoc(root, verify=False, building="HCXY").load("test")


def test_sod_downloads_each_sheet_from_its_mirrors_and_verifies_it(tmp_path):
    source = _sod_root(tmp_path)
    sha = lambda rel: hashlib.sha256((source / rel).read_bytes()).hexdigest()  # noqa: E731
    sheets = ("HCXY/Training_HCXY_All_30.csv", "HCXY/Testing_HCXY_All.csv")

    class Mirrored(SODIndoorLoc):
        urls = {rel: ((tmp_path / "gone" / rel).as_uri(), (source / rel).as_uri()) for rel in sheets}

    data = Mirrored(tmp_path / "data", download=True, building="HCXY")
    data.files = {"train": ((sheets[0], sha(sheets[0])),), "test": ((sheets[1], sha(sheets[1])),)}
    assert len(data.load("train")) == 2 and len(data.load("test")) == 1
    assert sorted(p.name for p in (tmp_path / "data" / "HCXY").iterdir()) == ["Testing_HCXY_All.csv",
                                                                                "Training_HCXY_All_30.csv"]


# ---------------------------------------------------------------- Tampere
class TinyTampere(Tampere):
    n_aps = 3


def _tampere_root(tmp_path, z=(0.0, 3.7, 14.8)):
    root = tmp_path / "tampere"
    root.mkdir(parents=True)
    for prefix in ("Training", "Test"):
        (root / f"{prefix}_rss_21Aug17.csv").write_text("100,-52,-84\n-99,100,100\n100,100,-14\n")
        (root / f"{prefix}_coordinates_21Aug17.csv").write_text(
            "".join(f"{x},{y},{h}\n" for (x, y), h in zip([(137.24, 19.731), (65.521, -5.1727), (201.51, 74.318)], z)))
        (root / f"{prefix}_device_21Aug17.csv").write_text("HUAWEI T1 7.0\nsamsung SM-A310F\nLetv x600\n")
        (root / f"{prefix}_date_21Aug17.csv").write_text(
            "2017-08-18 11:59:23\n2017-03-21 16:10:05\n2017-02-10 10:24:32\n")
    return root


def test_tampere_keeps_xyz_and_derives_floors_from_height(tmp_path):
    table = TinyTampere(_tampere_root(tmp_path), verify=False).load("test")
    np.testing.assert_array_equal(table.X, [[np.nan, -52, -84], [-99, np.nan, np.nan], [np.nan, np.nan, -14]])
    np.testing.assert_array_equal(table.pos, [[137.24, 19.731, 0.0], [65.521, -5.1727, 3.7], [201.51, 74.318, 14.8]])
    np.testing.assert_array_equal(table.floor, [0, 1, 4])  # round(z / 3.7), the benchmark software's rule
    assert table.building is None
    assert table.groups["device"].tolist() == ["HUAWEI T1 7.0", "samsung SM-A310F", "Letv x600"]
    np.testing.assert_array_equal(table.groups["time"], [utc(2017, 8, 18, 11, 59, 23), utc(2017, 3, 21, 16, 10, 5),
                                                         utc(2017, 2, 10, 10, 24, 32)])
    assert table.groups["time"][0] == 1503057563
    assert table.meta["pos_names"] == ("x", "y", "z") and table.meta["feature_names"] == ("WAP001", "WAP002", "WAP003")
    assert table.ids.tolist() == ["test-00000", "test-00001", "test-00002"]


def test_tampere_rejects_heights_between_floors_and_ragged_files(tmp_path):
    with pytest.raises(ValueError, match="whole number"):
        TinyTampere(_tampere_root(tmp_path, z=(0.0, 5.0, 14.8)), verify=False).load("train")
    root = _tampere_root(tmp_path / "b")
    (root / "Training_device_21Aug17.csv").write_text("only one\n")
    with pytest.raises(ValueError, match="row"):
        TinyTampere(root, verify=False).load("train")


def test_tampere_download_flattens_the_zenodo_archive_and_verifies_members(tmp_path):
    source = _tampere_root(tmp_path)
    archive = tmp_path / "DISTRIBUTED_OPENSOURCE_version2.zip"
    with zipfile.ZipFile(archive, "w") as zf:  # the Zenodo layout: FINGERPRINTING_DB/<file>
        zf.writestr("README.pdf", "manual")
        for path in sorted(source.iterdir()):
            zf.write(path, f"FINGERPRINTING_DB/{path.name}")

    class Mirrored(TinyTampere):
        urls = (archive.as_uri(),)
        files = {split: tuple((rel, hashlib.sha256((source / rel).read_bytes()).hexdigest()) for rel, _ in entries)
                 for split, entries in Tampere.files.items()}

    table = Mirrored(tmp_path / "data", download=True).load("train")
    assert len(table) == 3 and sorted(table.meta["sha256"]) == sorted(rel for rel, _ in Tampere.files["train"])
    wanted = sorted(rel for rel, _ in Tampere.files["train"])  # only the split's own members are extracted
    assert sorted(p.name for p in (tmp_path / "data").iterdir()) == wanted


# ---------------------------------------------------------------- TUJI1
class TinyTUJI1(TUJI1):
    n_aps = 3


def _tuji1_root(tmp_path, placeholder=0):
    root = tmp_path / "tuji1"
    root.mkdir(parents=True)
    (root / "Device_labels.csv").write_text("Label,Device\n1,S20\n2,S7\n5,A12\n")  # columns by name, any order
    for suffix in ("training", "testing"):
        (root / f"RSS_{suffix}.csv").write_text("-54,100,-88\n100,-101,-29\n")
        (root / f"Coordinates_{suffix}.csv").write_text(f"22.64,14.903,0,0,{placeholder},1\n0.2985,32.792,0,0,0,5\n")
    return root


def test_tuji1_reads_device_names_and_drops_the_constant_placeholder_columns(tmp_path):
    table = TinyTUJI1(_tuji1_root(tmp_path), verify=False).load("train")
    np.testing.assert_array_equal(table.X, [[-54, np.nan, -88], [np.nan, -101, -29]])
    np.testing.assert_array_equal(table.pos, [[22.64, 14.903], [0.2985, 32.792]])
    assert table.floor is None and table.building is None
    assert table.groups["device"].tolist() == ["S20", "A12"]
    assert table.meta["device_names"] == ("S20", "S7", "A12")


def test_tuji1_refuses_to_drop_placeholder_columns_that_hold_values(tmp_path):
    with pytest.raises(ValueError, match="z/floor/building"):
        TinyTUJI1(_tuji1_root(tmp_path, placeholder=1), verify=False).load("test")


def test_tuji1_names_the_expected_columns_of_a_malformed_label_file(tmp_path):
    root = _tuji1_root(tmp_path)
    (root / "Device_labels.csv").write_text("id,name\n1,S20\n")
    with pytest.raises(ValueError, match="'Device' and 'Label'"):
        TinyTUJI1(root, verify=False).load("train")


# ---------------------------------------------------------------- LongTermWiFi
class TinyLongTermWiFi(LongTermWiFi):
    n_aps = 3


def _ltw_archive(path, ids_width=10):
    """A v2.2-shaped archive: month 1 (trn01, tst01) and month 25 (trn01, trn02, tst01, tst06)."""
    sets = {(1, "trn", 1): 2, (1, "tst", 1): 1, (25, "trn", 1): 1, (25, "trn", 2): 1, (25, "tst", 1): 1,
            (25, "tst", 6): 2}
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("Readme.txt", "synthetic")
        for (month, kind, campaign), n in sets.items():
            base = f"db/{month:02d}/{kind}{campaign:02d}"
            code = month * 10**8 + campaign * 10**6 + (1 if kind == "trn" else 2) * 10**5 + 100  # point 001
            ids = [str(code + s + 1).zfill(ids_width) for s in range(n)]
            archive.writestr(f"{base}rss.csv", "".join(f"-{50 + s},100,-{90 + month}\n" for s in range(n)))
            archive.writestr(f"{base}crd.csv", "".join(f"12.91385188,29.21654402,{3 if s % 2 == 0 else 5}\n"
                                                       for s in range(n)))
            archive.writestr(f"{base}ids.csv", "\n".join(ids) + "\n")
            archive.writestr(f"{base}tms.csv", "".join(("20160727123260000" if s == 0 else "20160602155217232") + "\n"
                                                       for s in range(n)))
    return path


def test_longtermwifi_reads_the_archive_in_place_with_month_groups_and_devices(tmp_path):
    root = tmp_path / "ltw"
    _ltw_archive(root / "UJI_LIB_DB_v2.2.zip", ids_width=9)  # v2.1 wrote 9-digit ids for months 1-9
    train = TinyLongTermWiFi(root, verify=False).load("train")
    assert len(train) == 4
    np.testing.assert_array_equal(train.X, [[-50, np.nan, -91], [-51, np.nan, -91], [-50, np.nan, -115],
                                            [-50, np.nan, -115]])
    assert train.ids.tolist() == ["0101100101", "0101100102", "2501100101", "2502100101"]
    np.testing.assert_array_equal(train.groups["month"], [1, 1, 25, 25])
    np.testing.assert_array_equal(train.groups["campaign"], [1, 1, 1, 2])
    assert train.groups["device"].tolist() == ["Samsung Galaxy S3"] * 3 + ["Samsung Galaxy A5 (2017)"]
    np.testing.assert_array_equal(train.floor, [3, 5, 3, 3])
    # "...123260000" writes second 60: read as 12:33:00.000; the other stamp keeps its milliseconds
    np.testing.assert_allclose(train.groups["time"][:2],
                               [utc(2016, 7, 27, 12, 33, 0), utc(2016, 6, 2, 15, 52, 17) + 0.232], rtol=0, atol=1e-6)
    test = TinyLongTermWiFi(root, verify=False, month=[25]).load("test")
    assert test.groups["device"].tolist() == ["Samsung Galaxy S3"] + ["Samsung Galaxy A5 (2017)"] * 2
    assert test.meta["selected_months"] == (25,) and test.meta["source_files"] == ("UJI_LIB_DB_v2.2.zip",)


def test_longtermwifi_month_option_is_validated(tmp_path):
    for bad in (0, 26, True, [], [1, 30], "3", ["3"], 2.5, [np.True_], object()):
        with pytest.raises(ValueError, match="month"):
            LongTermWiFi(tmp_path, month=bad)
    assert LongTermWiFi(tmp_path, month=np.int64(3)).selected == (3,)
    assert LongTermWiFi(tmp_path, month=np.array(3)).selected == (3,)
    assert LongTermWiFi(tmp_path, month=[5, 2, 5]).selected == (2, 5)
    assert LongTermWiFi(tmp_path, month=range(24, 26)).selected == (24, 25)
    _ltw_archive(tmp_path / "UJI_LIB_DB_v2.2.zip")
    with pytest.raises(ValueError, match="no tst sets"):
        TinyLongTermWiFi(tmp_path, verify=False, month=7).load("test")


def test_longtermwifi_download_keeps_the_archive_and_verifies_it(tmp_path):
    archive = _ltw_archive(tmp_path / "mirror" / "UJI_LIB_DB_v2.2.zip")
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()

    class Mirrored(TinyLongTermWiFi):
        urls = {"UJI_LIB_DB_v2.2.zip": ((tmp_path / "gone.zip").as_uri(), archive.as_uri())}  # first mirror down
        files = {"train": ("UJI_LIB_DB_v2.2.zip", sha), "test": ("UJI_LIB_DB_v2.2.zip", sha)}

    table = Mirrored(tmp_path / "data", download=True).load("test")
    assert len(table) == 4 and table.meta["sha256"] == sha
    assert sorted(p.name for p in (tmp_path / "data").iterdir()) == ["UJI_LIB_DB_v2.2.zip"]  # no .part left

    class Broken(Mirrored):
        urls = {"UJI_LIB_DB_v2.2.zip": ((tmp_path / "gone.zip").as_uri(),)}

    with pytest.raises(RuntimeError, match="could not download"):
        Broken(tmp_path / "other", download=True).load("train")


# ---------------------------------------------------------------- WLANRSSI
def _wlan_root(tmp_path, rows):
    root = tmp_path / "wlan"
    root.mkdir(parents=True)
    (root / "wifi_localization.txt").write_bytes(
        "".join("\t".join(str(v) for v in row) + "\r\n" for row in rows).encode())  # tabs + CRLF, as UCI ships it
    return root


# two well-separated rooms: room 1 hears AP1 strongly, room 3 hears AP7 strongly
TOY = [[-40, -60, -60, -70, -70, -80, -90, 1], [-42, -61, -60, -70, -71, -80, -90, 1],
       [-44, -60, -62, -70, -70, -81, -91, 1], [-90, -80, -70, -70, -60, -60, -40, 3],
       [-91, -80, -71, -70, -60, -61, -42, 3], [-90, -81, -70, -72, -61, -60, -44, 3]]


def test_wlanrssi_keeps_room_labels_and_an_empty_position_array(tmp_path):
    table = load_dataset("wlanrssi", split="all", root=_wlan_root(tmp_path, TOY), download=False, verify=False)
    assert table.X.shape == (6, 7) and table.X.dtype == np.float32
    assert table.pos.shape == (6, 0) and table.pos.dtype == np.float64
    assert table.floor is None and table.building is None
    np.testing.assert_array_equal(table.groups["room"], [1, 1, 1, 3, 3, 3])
    assert table.meta["task"] == "room_classification" and table.meta["pos_names"] == ()
    assert table.meta["feature_names"] == tuple(f"AP{j}" for j in range(1, 8))
    default = load_dataset("wlanrssi", root=tmp_path / "wlan", download=False, verify=False)  # no train/test: "all"
    assert isinstance(default, type(table)) and default.meta["split"] == "all" and len(default) == 6


def test_wlanrssi_tables_pass_through_l3_and_l4_and_rooms_are_classified(tmp_path):
    from indoorloc import create_model
    from indoorloc.evaluation import label_accuracy
    from indoorloc.signals.transforms import FillMissing

    table = load_dataset("wlanrssi", split="all", root=_wlan_root(tmp_path, TOY), download=False, verify=False)
    train, test = table[[0, 1, 3, 4]], table[[2, 5]]
    model = create_model("wknn", k=1, preprocess=FillMissing(-100)).fit(train)  # the (N, 0) target is accepted
    assert model.localize(test).pos.shape == (2, 0)
    assert model.evaluate(test).n == 2
    # room classification: the k-NN vote over the room passed as the label, a known 1-NN answer
    knn = create_model("wknn", k=1).fit(train.X, train.pos, floor=train.groups["room"])
    np.testing.assert_array_equal(knn.localize(test.X).floor, [1, 3])
    assert label_accuracy(test.groups["room"], knn.localize(test.X).floor) == 100.0


def test_wlanrssi_docstring_recipe_runs_as_written(tmp_path):
    from indoorloc import create_model
    from indoorloc.evaluation import label_accuracy, random_split

    table = load_dataset("wlanrssi", split="all", root=_wlan_root(tmp_path, TOY), download=False, verify=False)
    train_idx, test_idx = random_split(len(table), 0.2, stratify=table.groups["room"], random_state=0)
    train, test = table[train_idx], table[test_idx]
    np.testing.assert_array_equal(test.groups["room"], [1, 3])  # ceil(0.2 * 6) = 2 rows, one per room
    knn = create_model("wknn", k=1).fit(train.X, train.pos, floor=train.groups["room"])
    assert label_accuracy(test.groups["room"], knn.localize(test.X).floor) == 100.0  # well-separated rooms


def test_wlanrssi_rejects_unknown_rooms_and_positive_readings(tmp_path):
    with pytest.raises(ValueError, match="room"):
        WLANRSSI(_wlan_root(tmp_path, [TOY[0][:7] + [5]]), verify=False).load("all")
    with pytest.raises(ValueError, match="non-negative"):
        WLANRSSI(_wlan_root(tmp_path / "b", [[0] + TOY[0][1:]]), verify=False).load("all")


# ---------------------------------------------------------------- real data (skipped when absent)
def _have(name, *rels):
    return pytest.mark.skipif(not all((DATA_ROOT / name / rel).is_file() for rel in rels),
                              reason=f"{name} files not found under {DATA_ROOT}")


_SOD_ALL_SHEETS = tuple(SODIndoorLoc.urls)  # the 9 training and 5 testing sheets


@_have("sodindoorloc", *(rel for rel, _ in SODIndoorLoc.files["train"] + SODIndoorLoc.files["test"]))
def test_real_sodindoorloc_counts_and_groups():
    train, test = load_dataset("sodindoorloc", root=DATA_ROOT / "sodindoorloc", download=False)
    assert train.X.shape == (21205, 52 + 347 + 363) and test.X.shape == (2720, 762)  # README: 21205 / 2720
    assert np.unique(train.building).tolist() == [1, 2, 3] and np.unique(train.floor).tolist() == [1, 2, 3, 4]
    hcxy = train[train.building == 2]
    assert np.unique(hcxy.groups["device"]).tolist() == [4, 5, 6, 7, 8, 9]  # six phones
    assert np.isnan(hcxy.X[:, :52]).all() and np.isnan(hcxy.X[:, 399:]).all()  # other buildings' MACs
    assert np.nanmin(train.X) >= -104 and np.nanmax(train.X) < 0
    assert len(np.unique(train.ids)) == len(train)


def _round_half_away(values):
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


@_have("sodindoorloc", *_SOD_ALL_SHEETS)
@pytest.mark.parametrize("building, macs", [("HCXY", "all"), ("HCXY", "preinstalled"), ("SYL", "all"),
                                            ("SYL", "preinstalled")])
def test_real_sodindoorloc_averaged_sheets_are_rebuilt_exactly_from_the_30_scan_sheets(building, macs):
    """Known result: every cell of the four "Avg" sheets follows from its 30-scan sheet."""
    root = DATA_ROOT / "sodindoorloc"
    scans = SODIndoorLoc(root, building=building, macs=macs).load("train")
    avg = SODIndoorLoc(root, building=building, macs=macs, averaged=True).load("train")
    n = scans.X.shape[1]
    assert avg.meta["feature_names"] == scans.meta["feature_names"] and len(avg) * 30 == len(scans)
    np.testing.assert_array_equal(scans.groups["scan_index"].reshape(-1, 30), np.tile(np.arange(1, 31), (len(avg), 1)))
    np.testing.assert_array_equal(avg.pos, scans.pos[::30])  # one row per point, same order
    X = scans.X.astype(np.float64).reshape(-1, 30, n)
    heard = (~np.isnan(X)).sum(axis=1)
    if (building, macs) == ("HCXY", "all"):  # mean of the detected scans; never detected -> +100 -> NaN
        expected = _round_half_away(np.nansum(X, axis=1) / np.maximum(heard, 1))
        expected[heard == 0] = np.nan
    else:  # -105 dBm substituted for every undetected scan; a -105 result -> NaN
        expected = _round_half_away(np.where(np.isnan(X), -105.0, X).mean(axis=1))
        expected[expected == -105] = np.nan
        assert np.all(heard[np.isnan(expected)] <= 1)  # a -105 average: never heard, or heard once
    np.testing.assert_array_equal(avg.X, expected.astype(np.float32))


@_have("sodindoorloc", *_SOD_ALL_SHEETS)
def test_real_sodindoorloc_preinstalled_columns_are_the_same_macs_as_in_the_all_sheets():
    root = DATA_ROOT / "sodindoorloc"
    every = SODIndoorLoc(root, building=["HCXY", "SYL"]).load("test")
    ap = SODIndoorLoc(root, building=["HCXY", "SYL"], macs="preinstalled").load("test")
    assert ap.X.shape == (1880, 56 + 46)  # 56 single-band HCXY APs, 23 dual-band SYL APs (README)
    column = {name: j for j, name in enumerate(every.meta["feature_names"])}
    np.testing.assert_array_equal(ap.X, every.X[:, [column[name] for name in ap.meta["feature_names"]]])
    np.testing.assert_array_equal(ap.pos, every.pos)


@_have("tampere", *(rel for rel, _ in Tampere.files["train"] + Tampere.files["test"]))
def test_real_tampere_counts():
    train, test = load_dataset("tampere", root=DATA_ROOT / "tampere", download=False)
    assert train.X.shape == (697, 992) and test.X.shape == (3951, 992)  # 4648 fingerprints (paper)
    assert len(set(train.groups["device"]) | set(test.groups["device"])) == 21  # 21 devices (paper)
    assert np.unique(test.floor).tolist() == [0, 1, 2, 3, 4]


@_have("tampere", *(rel for rel, _ in Tampere.files["train"] + Tampere.files["test"]))
def test_real_tampere_matches_the_worked_examples_of_its_readme():
    """FINGERPRINTING_DB/README.txt quotes rows of the files; the loaded tables must agree."""
    train, test = load_dataset("tampere", root=DATA_ROOT / "tampere", download=False)
    np.testing.assert_array_equal(test.pos[0], [137.24, 19.731, 0.0])  # "(x,y,z)=(137.24,19.731,0) ... first"
    first = train.X[0]  # "access point 2 was not heard, 420 was heard at -84 dB, and 489 with -52 dB"
    assert np.isnan(first[1]) and first[419] == -84 and first[488] == -52
    assert test.groups["time"][0] == utc(2017, 8, 18, 11, 59, 23)  # "measurement indexed 1 ... 18.8.2017 11:59:23"
    assert train.groups["device"][2] == "samsung SM-A310F"  # "3rd measurement ... Samsung SM-A10F" (sic)
    assert test.groups["device"][14] == "Xiaomi MI MAX 2"  # "15th measurement in the test data"


@_have("tuji1", *(rel for rel, _ in TUJI1.files["train"] + TUJI1.files["test"]))
def test_real_tuji1_reproduces_the_papers_1nn_benchmark():
    train, test = load_dataset("tuji1", root=DATA_ROOT / "tuji1", download=False)
    assert train.X.shape == (6752, 310) and test.X.shape == (2147, 310)  # 8899 scans (paper)
    # Klus et al. (2024), Table 3, "1NN": positive data representation (missing = overall
    # minimum - 1 = -102 dBm), Euclidean distance, k = 1, every candidate tied with the nearest
    # averaged (knn_ips.m). Integer dBm make every squared distance an exact integer.
    fill = min(np.nanmin(train.X), np.nanmin(test.X)) - 1
    A = np.nan_to_num(train.X.astype(np.float64), nan=fill)
    Q = np.nan_to_num(test.X.astype(np.float64), nan=fill)
    d2 = np.einsum("ij,ij->i", Q, Q)[:, None] + np.einsum("ij,ij->i", A, A)[None] - 2.0 * Q @ A.T
    nearest = d2 == d2.min(axis=1, keepdims=True)
    estimate = nearest @ train.pos / nearest.sum(axis=1, keepdims=True)
    error = np.linalg.norm(estimate - test.pos, axis=1).mean()
    assert fill == -102 and round(error, 2) == 3.34


@_have("longtermwifi", "UJI_LIB_DB_v2.2.zip")
def test_real_longtermwifi_months():
    data = LongTermWiFi(DATA_ROOT / "longtermwifi", month=[1, 25])
    train, test = data.load("train"), data.load("test")
    assert (train.groups["month"] == 1).sum() == 15 * 576 and (train.groups["month"] == 25).sum() == 2 * 576
    assert set(train.groups["device"][train.groups["month"] == 25]) == {"Samsung Galaxy S3", "Samsung Galaxy A5 (2017)"}
    assert train.X.shape[1] == test.X.shape[1] == 620 and np.unique(test.floor).tolist() == [3, 5]
    assert len(np.unique(np.concatenate([train.ids, test.ids]))) == len(train) + len(test)
    assert np.all(np.diff(train.groups["time"][train.groups["campaign"] == 1][:576]) > 0)  # stamps in order


@_have("longtermwifi", "UJI_LIB_DB_v2.2.zip")
def test_real_longtermwifi_ids_stamps_and_devices_follow_the_release_documentation():
    test = LongTermWiFi(DATA_ROOT / "longtermwifi", month=[2, 25]).load("test")
    ids = test.ids.astype("U10")
    # ids are MM CC T PPP SS (findMonth.m, findCampNumber.m, findTrainOrTest.m): they agree with the groups
    np.testing.assert_array_equal(np.array([int(i[:2]) for i in ids]), test.groups["month"])
    np.testing.assert_array_equal(np.array([int(i[2:4]) for i in ids]), test.groups["campaign"])
    assert {i[4] for i in ids} == {"2"}  # 2 = test set
    # db/02/tst03tms.csv writes "20160727123260000" (second 60): read as 12:33:00.000
    assert np.sum(test.groups["time"] == utc(2016, 7, 27, 12, 33, 0)) == 1
    # db/Readme.txt: only training 2 and tests 6-10 of month 25 are Galaxy A5 (2017), at the positions of 1-5
    a5 = test.groups["device"] == "Samsung Galaxy A5 (2017)"
    assert set(test.groups["campaign"][a5]) == {6, 7, 8, 9, 10} and set(test.groups["month"][a5]) == {25}
    m25 = test.groups["month"] == 25
    np.testing.assert_array_equal(test.pos[m25 & a5], test.pos[m25 & ~a5])


@_have("wlanrssi", "wifi_localization.txt")
def test_real_wlanrssi():
    table = load_dataset("wlanrssi", split="all", root=DATA_ROOT / "wlanrssi", download=False)
    assert table.X.shape == (2000, 7) and not np.isnan(table.X).any()
    assert np.bincount(table.groups["room"]).tolist() == [0, 500, 500, 500, 500]


def test_building_names_constant_matches_meta():
    assert tuple(SODIndoorLoc.meta["building_names"].values()) == BUILDINGS
