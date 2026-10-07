"""L1 BLE and CSI loaders: BLEIndoor (BBIL), BLERSSIUCI, IBeaconRSSI, CSIFingerprint, HWILD, HALOC.

Synthetic files that copy each official format exercise the parsers with known answers (no
network); the real-data checks at the end run only when the files are in ``$INDOORLOC_DATA``
(default ``~/.cache/indoorloc/datasets/<name>``) and pin counts measured on the official files.
"""
from __future__ import annotations

import calendar
import csv
import hashlib
import io
import zipfile

import numpy as np
import pytest

from conftest import DATA_ROOT
from indoorloc.datasets import DATASETS, dataset_info, load_dataset
from indoorloc.datasets.ble_indoor import BLEIndoor
from indoorloc.datasets.ble_rssi_uci import BLERSSIUCI, cell_to_grid
from indoorloc.datasets.csi_fingerprint import CSIFingerprint, grid_position
from indoorloc.datasets.haloc import HALOC, SUBCARRIERS, parse_esp32_csi
from indoorloc.datasets.hwild import HWILD, aoa_geometry
from indoorloc.datasets.ibeacon_rssi import PROTOCOLS, IBeaconRSSI

CLASSES = {"ble_indoor": BLEIndoor, "ble_rssi_uci": BLERSSIUCI, "ibeacon_rssi": IBeaconRSSI,
           "csi_fingerprint": CSIFingerprint, "hwild": HWILD, "haloc": HALOC}


def sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------- declarations
@pytest.mark.parametrize("name", sorted(CLASSES))
def test_class_declarations_follow_the_dataset_contract(name):
    cls = DATASETS.get(name)
    assert cls is CLASSES[name] and cls.name == name
    info = dataset_info(name)
    for key in ("modality", "units", "crs", "pos_units", "license", "doi", "citation", "url"):
        assert key in info, key
    assert info["modality"] in ("ble_rssi", "csi", "csi_amp")
    for split in cls.files:  # every declared file has a full sha256
        for rel, digest in cls(root="/nonexistent")._entries(split):
            assert isinstance(rel, str) and len(digest) == 64 and int(digest, 16) >= 0


def test_grid_units_are_never_called_metres():
    assert "grid" in dataset_info("ble_rssi_uci")["pos_units"]
    assert "grid" in dataset_info("csi_fingerprint")["pos_units"]
    for name in ("ble_indoor", "ibeacon_rssi", "hwild", "haloc"):
        assert dataset_info(name)["pos_units"] == "m"


# ---------------------------------------------------------------- BLE RSSI UCI
UCI_HEADER = ["location", "date", *[f"b{3001 + i}" for i in range(13)]]


def uci_row(cell, date, heard):
    values = [-200] * 13
    for j, v in heard.items():
        values[j] = v
    return [cell, date, *values]


def test_uci_cells_sentinels_and_month_first_dates(tmp_path):
    rows = [uci_row("K04", "10-18-2016 11:15:21", {5: -78}),
            uci_row("A01", "4-19-2016 9:37:23", {0: -60, 11: -199}),  # -199: the sentinel off by one
            uci_row("W15", "10-3-2016 13:00:00", {12: -61, 3: -198})]
    write_csv(tmp_path / "iBeacon_RSSI_Labeled.csv", UCI_HEADER, rows)
    write_csv(tmp_path / "iBeacon_RSSI_Unlabeled.csv", UCI_HEADER, [uci_row("?", "11-7-2016 12:29:01", {2: -80})])
    t = BLERSSIUCI(tmp_path, verify=False).load("labeled")  # alias of "all"
    assert t.X.shape == (3, 13) and t.X.dtype == np.float32 and t.meta["split"] == "all"
    assert np.array_equal(np.isnan(t.X).sum(axis=1), [12, 12, 12])  # -200, -199, -198 are all "not heard"
    assert t.X[0, 5] == -78 and t.X[1, 0] == -60 and t.X[2, 12] == -61
    assert t.pos.tolist() == [[11, 4], [1, 1], [23, 15]]  # column letter (A = 1), row number
    assert t.groups["point"].tolist() == ["K04", "A01", "W15"] and t.floor is None and t.building is None
    assert t.groups["time"][0] == calendar.timegm((2016, 10, 18, 11, 15, 21))  # month-day-year
    assert t.groups["time"][2] == calendar.timegm((2016, 10, 3, 13, 0, 0))
    assert t.meta["feature_names"][0] == "b3001" and "grid" in t.meta["pos_units"]

    u = BLERSSIUCI(tmp_path, verify=False).load("unlabeled")
    assert np.isnan(u.pos).all() and u.meta["unknown_groups"] == ("point",) and sorted(u.groups) == ["time"]


def test_uci_cell_parser_rejects_non_cells():
    assert cell_to_grid("u15") == (21.0, 15.0)
    for bad in ("?", "15", "K"):
        with pytest.raises(ValueError, match="grid cell"):
            cell_to_grid(bad)


# ---------------------------------------------------------------- BBIL (ble_indoor)
EDGES_OFFICE = {1: (19.82, 6.55), 2: (24.71, 10.31), 8: (0.0, 9.79)}
EDGES_LAB = {50: (0.579, 7.88), 51: (0.0, 3.431)}


def bbil_archive(path):
    """A miniature experiment.zip: two rooms, three splits, receivers named edge_<id>."""
    def edges_csv(edges):
        lines = ["beaconid,edgenodeid,gamma,bias,edge_x,edge_y,edge_z"]
        for beacon in (1, 9):  # one row per beacon and receiver, as in the release
            lines += [f"{beacon},{e},2.0,-50.0,{x},{y},1.6" for e, (x, y) in edges.items()]
        return "\n".join(lines) + "\n"

    with zipfile.ZipFile(path, "w") as z:
        for split in ("train", "valid", "test"):
            z.writestr(f"experiment1/{split}/edges.csv", edges_csv(EDGES_OFFICE))
            z.writestr(f"experiment2/{split}/edges.csv", edges_csv(EDGES_LAB))
            # columns in a shuffled order and one receiver missing from the recording
            z.writestr(f"experiment1/{split}/2018-09-18T20-03-15-500000_1_data_wide.csv",
                       "Datetime,beaconid,edge_8,realx,realy,edge_1\n"
                       "2018-09-18T20:03:15.500000Z,1,-79.0,18.5,8.0,-68.0\n"
                       "2018-09-18T20:03:16.000000Z,1,-84.0,18.25,7.5,-74.5\n")
            z.writestr(f"experiment1/{split}/2018-09-18T20-03-15-500000_1_data.csv", "ignored\n")
            z.writestr(f"experiment2/{split}/2019-03-20T23-38-02-000000_8_data_wide.csv",
                       "Datetime,beaconid,realx,realy,edge_50,edge_51\n"
                       "2019-03-20T23:38:02.000000Z,8,1.0,2.0,-70.0,-66.0\n")
    return path


def test_bbil_rooms_are_buildings_with_their_own_receivers(tmp_path):
    archive = bbil_archive(tmp_path / "experiment.zip")
    t = BLEIndoor(tmp_path, verify=False).load("test")
    assert t.meta["feature_names"] == ("edge_1", "edge_2", "edge_8", "edge_50", "edge_51")
    nan = np.nan
    assert np.array_equal(t.X, np.array([[-68.0, nan, -79.0, nan, nan], [-74.5, nan, -84.0, nan, nan],
                                         [nan, nan, nan, -70.0, -66.0]], dtype=np.float32), equal_nan=True)
    assert t.pos.tolist() == [[18.5, 8.0], [18.25, 7.5], [1.0, 2.0]]
    assert t.building.tolist() == [0, 0, 1] and t.floor is None
    assert t.meta["building_names"] == {0: "office", 1: "lab"}
    assert np.array_equal(t.meta["anchors"], [[19.82, 6.55], [24.71, 10.31], [0.0, 9.79], [0.579, 7.88], [0.0, 3.431]])
    assert t.groups["device"].tolist() == [1, 1, 8]
    assert t.groups["time"][0] == calendar.timegm((2018, 9, 18, 20, 3, 15)) + 0.5
    assert t.groups["trajectory"][0] == "office/2018-09-18T20-03-15-500000_1"
    assert t.ids.tolist() == ["office-test-00000", "office-test-00001", "lab-test-00000"]
    assert t.meta["source_files"] == ("experiment.zip",)
    assert t.meta["sha256"] == BLEIndoor._archive[1] and sha(archive) != BLEIndoor._archive[1]  # declared digest

    lab = BLEIndoor(tmp_path, verify=False, room="experiment2").load("val")  # folder name and split alias
    assert lab.meta["feature_names"] == ("edge_50", "edge_51") and lab.building.tolist() == [1]
    with pytest.raises(ValueError, match="unknown room"):
        BLEIndoor(tmp_path, room="floor1")


def test_bbil_download_keeps_the_release_archive(tmp_path):
    """The base downloader stores a zip whose url names a declared file instead of unpacking it."""
    (tmp_path / "mirror").mkdir()
    archive = bbil_archive(tmp_path / "mirror" / "experiment.zip")  # named like the release asset

    class Mirrored(BLEIndoor):
        urls = ((tmp_path / "gone" / "experiment.zip").as_uri(), archive.as_uri())  # first mirror is down
        files = dict.fromkeys(("train", "valid", "test"), ("experiment.zip", sha(archive)))

    t = Mirrored(tmp_path / "data", download=True, room="office").load("train")
    assert len(t) == 2 and sorted(p.name for p in (tmp_path / "data").iterdir()) == ["experiment.zip"]
    assert sha(tmp_path / "data" / "experiment.zip") == sha(archive)


# ---------------------------------------------------------------- UJI BLE DB (ibeacon_rssi)
def uji_ble_files(root):
    """lib: 2 beacons; point 1 of campaign 2 (train) and points 1-2 of campaign 3 (test); geo: 1 beacon."""
    (root / "data" / "rss").mkdir(parents=True)
    (root / "data" / "dep").mkdir()
    lib_ids = [12200101, 22200102, 12300101, 32300201, 32300201]  # phone, power, campaign, point, sample
    np.savetxt(root / "data/rss/lib_ids.csv", lib_ids, fmt="%d")
    np.savetxt(root / "data/rss/lib_rss.csv", [[-57, 100], [100, 100], [-63, -52], [-70, -71], [-72, -73]],
               fmt="%d", delimiter=",")
    np.savetxt(root / "data/rss/lib_crd.csv", [[29.22, 12.91], [29.22, 12.91], [27.43, 8.52], [25.0, 4.13],
                                                [25.0, 4.13]], fmt="%.2f", delimiter=",")
    (root / "data/dep/lib.csv").write_text("id,x,y\n23,30.50,14.25\n24,30.50,09.90")
    np.savetxt(root / "data/rss/geo_ids.csv", [11100101, 13100201], fmt="%d")
    np.savetxt(root / "data/rss/geo_rss.csv", [[-60], [100]], fmt="%d", delimiter=",")
    np.savetxt(root / "data/rss/geo_crd.csv", [[16.091, 2.795], [16.091, 5.174]], fmt="%.3f", delimiter=",")
    (root / "data/dep/geo.csv").write_text("id,x,y\nB1,15.993,10.168\n")


def test_uji_ble_ids_decode_into_groups_and_zones_into_buildings(tmp_path):
    uji_ble_files(tmp_path)
    t = IBeaconRSSI(tmp_path, verify=False).load("all")
    assert t.meta["feature_names"] == ("B1", "23", "24") and t.X.shape == (7, 3)
    assert np.isnan(t.X[1]).all() and t.X[2, 1] == -57 and np.isnan(t.X[2, 2])  # 100 and the other zone -> NaN
    assert t.building.tolist() == [1, 1, 2, 2, 2, 2, 2] and t.meta["building_names"] == {1: "geotec", 2: "library"}
    assert t.groups["device"].tolist() == ["A5", "A5", "A5", "BQ", "A5", "S6", "S6"]
    assert t.groups["power"].tolist() == [-4, -20, -12, -12, -12, -12, -12]
    assert t.groups["campaign"].tolist() == [1, 1, 2, 2, 3, 3, 3]
    assert t.groups["point_number"].tolist() == [1, 2, 1, 1, 1, 2, 2]
    # the position id comes from the coordinates as written in the file, not from the point number
    assert t.groups["point"].tolist() == ["geo/16.091,2.795", "geo/16.091,5.174", "lib/29.22,12.91",
                                          "lib/29.22,12.91", "lib/27.43,8.52", "lib/25.00,4.13", "lib/25.00,4.13"]
    assert len(set(t.ids)) == 7  # the file repeats id 32300201; sample ids are file rows
    assert np.array_equal(t.meta["anchors"], [[15.993, 10.168], [30.5, 14.25], [30.5, 9.9]])


def test_uji_ble_protocols_follow_the_authors_point_lists(tmp_path):
    uji_ble_files(tmp_path)
    lib = IBeaconRSSI(tmp_path, verify=False, zone="lib")
    assert lib.files["all"] == IBeaconRSSI._zone_files["lib"]  # only the selected zone is checked
    train, test = lib.load("train"), lib.load("test")
    assert train.meta["protocol"] == {"lib": "full_train_no_outer_test"}  # the authors' f7_basicPos.m
    assert train.groups["campaign"].tolist() == [2, 2] and train.groups["point_number"].tolist() == [1, 1]
    assert test.groups["campaign"].tolist() == [3, 3] and test.groups["point_number"].tolist() == [2, 2]  # 1: outer
    full = IBeaconRSSI(tmp_path, verify=False, zone="lib", protocol={"lib": "full_train_full_test"}).load("test")
    assert full.groups["point_number"].tolist() == [1, 2, 2]
    # the authors' lists, independently restated
    _, lib_train, _, lib_test = PROTOCOLS["lib"]["full_train_no_outer_test"]
    assert lib_train == tuple(range(1, 49)) and len(lib_test) == 60
    assert {1, 17, 18, 34, 35, 51, 52, 68}.isdisjoint(lib_test)
    for name, (_, train_p, _, test_p) in PROTOCOLS["geo"].items():
        assert set(train_p).isdisjoint(test_p) and set(train_p) | set(test_p) == set(range(1, 69)), name
    # point numbers k and n + 1 - k share a position: every configuration keeps such pairs together
    for zone, n_train, n_test in (("geo", 68, 68), ("lib", 48, 68)):
        for name, (_, train_p, _, test_p) in PROTOCOLS[zone].items():
            assert {n_train + 1 - k for k in train_p} == set(train_p), (zone, name)
            assert {n_test + 1 - k for k in test_p} == set(test_p), (zone, name)
    assert len(PROTOCOLS["geo"]["full_limits"][1]) == 48 and len(PROTOCOLS["geo"]["reduced_limits"][1]) == 22
    with pytest.raises(ValueError, match="unknown protocol"):
        IBeaconRSSI(tmp_path, protocol={"lib": "triangles"})
    with pytest.raises(ValueError, match="unknown zone"):
        IBeaconRSSI(tmp_path, zone="library")


# ---------------------------------------------------------------- HALOC
def esp32_buffer(h: np.ndarray, pairs: np.ndarray) -> str:
    """128 (imag, real) int8 pairs with ``h`` at ``pairs`` and zeros elsewhere, as ESP-IDF prints it."""
    raw = np.zeros((128, 2), dtype=int)
    raw[pairs, 0], raw[pairs, 1] = h.imag, h.real
    return "[" + ",".join(map(str, raw.ravel())) + "]"


def test_esp32_buffer_holds_imaginary_then_real():
    pairs, indices = SUBCARRIERS["lltf"]
    h = np.arange(52) - 20 + 1j * (np.arange(52) % 7 - 3)
    got = parse_esp32_csi(esp32_buffer(h, pairs), pairs)
    assert got.dtype == np.complex64 and np.array_equal(got, h.astype(np.complex64))
    assert indices.tolist() == [*range(-26, 0), *range(1, 27)]
    assert SUBCARRIERS["htltf"][1].tolist() == [*range(-28, 0), *range(1, 29)]


# the official header of the six sequence files
HALOC_HEADER = ["type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing", "not_sounding",
                "aggregation", "stbc", "fec_coding", "sgi", "noise_floor", "ampdu_cnt", "channel",
                "secondary_channel", "local_timestamp", "ant", "sig_len", "rx_state", "len", "first_word", "data",
                "x", "y", "z"]


def haloc_archive(root, n=3, **override):
    """HALOC.zip as on Zenodo (sequences under ``HALOC/``) with random I/Q; returns the true CSI per sequence."""
    rng = np.random.default_rng(0)
    pairs = SUBCARRIERS["lltf"][0]
    truth = {}
    clock = [2**31 - 5, 2**31 + 5, 2**32 + 15]  # microseconds; ESP-IDF prints int32, so it wraps to negative
    clock = [(c + 2**31) % 2**32 - 2**31 for c in clock]
    with zipfile.ZipFile(root / "HALOC.zip", "w") as z:
        for seq in range(6):
            h = rng.integers(-60, 60, (n, 52)) + 1j * rng.integers(-60, 60, (n, 52))
            rows = []
            for k in range(n):
                field = {"type": "CSI_DATA", "id": k, "mac": "1a:00:00:00:00:00", "rssi": -43, "rate": 11,
                         "sig_mode": 1, "mcs": 0, "bandwidth": 0, "smoothing": 1, "not_sounding": 1,
                         "aggregation": 0, "stbc": 0, "fec_coding": 0, "sgi": 1, "noise_floor": -96, "ampdu_cnt": 0,
                         "channel": 11, "secondary_channel": 2, "local_timestamp": clock[k], "ant": 0,
                         "sig_len": 44, "rx_state": 0, "len": 256, "first_word": 0,
                         "data": esp32_buffer(h[k], pairs), "x": seq + 0.5 * k, "y": 0.1, "z": 1.25}
                field.update(override if seq == 5 and k == 1 else {})
                rows.append([field[c] for c in HALOC_HEADER])
            buf = io.StringIO()
            csv.writer(buf).writerows([HALOC_HEADER, *rows])
            z.writestr(f"HALOC/{seq}.csv", buf.getvalue())
            truth[seq] = h
        z.writestr("HALOC/readme.txt", "training: 0.csv, 1.csv, 2.csv and 3.csv\nvalidation: 4.csv\ntest: 5.csv")
    return truth


def test_haloc_reads_the_archive_by_the_authors_split(tmp_path):
    truth = haloc_archive(tmp_path)
    tr, va, te = (HALOC(tmp_path, verify=False).load(s) for s in ("train", "validation", "test"))
    assert tr.X.shape == (12, 1, 1, 52) and tr.X.dtype == np.complex64 and len(va) == len(te) == 3
    assert np.array_equal(tr.X[3:6, 0, 0], truth[1].astype(np.complex64))
    assert np.array_equal(te.X[:, 0, 0], truth[5].astype(np.complex64))
    assert tr.groups["trajectory"].tolist() == [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3
    assert np.allclose(tr.groups["time"][:3], [0.0, 10e-6, (2**31 + 20) * 1e-6])  # unwrapped 32-bit microseconds
    assert np.allclose(te.pos, [[5.0, 0.1, 1.25], [5.5, 0.1, 1.25], [6.0, 0.1, 1.25]])
    assert te.meta["subcarriers"].tolist() == SUBCARRIERS["lltf"][1].tolist() and te.meta["modality"] == "csi"
    # 802.11n at 20 MHz: 312.5 kHz spacing, subcarrier -26 sits 8.125 MHz below the channel 11 carrier
    assert te.meta["subcarrier_offsets_hz"][0] == -8.125e6 and te.meta["carrier_hz"] == 2.462e9
    assert te.ids[0] == "seq5-00000" and tr.meta["sequences"] == (0, 1, 2, 3)
    assert te.meta["source_files"] == ("HALOC.zip",) and sorted(p.name for p in tmp_path.iterdir()) == ["HALOC.zip"]

    ht = HALOC(tmp_path, verify=False, subcarriers="htltf").load("test")
    assert ht.X.shape == (3, 1, 1, 56) and not np.any(ht.X)  # the synthetic HT-LTF field is all zero

    one = HALOC(tmp_path, verify=False, sequences=[2])
    assert one.load("train").groups["trajectory"].tolist() == [2, 2, 2] and one.splits == ("train", "all")
    assert one.load("all").meta["sequences"] == (2,)
    with pytest.raises(ValueError, match="not 'test'"):
        one.load("test")
    with pytest.raises(ValueError, match="sequences"):
        HALOC(tmp_path, sequences=[6])
    with pytest.raises(ValueError, match="subcarriers"):
        HALOC(tmp_path, subcarriers="all")


@pytest.mark.parametrize("field, value", [("secondary_channel", 1), ("bandwidth", 1), ("sig_mode", 0)])
def test_haloc_refuses_packets_of_another_format(tmp_path, field, value):
    """The subcarrier positions hold only for HT20 packets with the secondary channel below."""
    haloc_archive(tmp_path, **{field: value})
    assert len(HALOC(tmp_path, verify=False).load("train")) == 12  # sequences 0-3 are unaffected
    with pytest.raises(ValueError, match=field):
        HALOC(tmp_path, verify=False).load("test")


# ---------------------------------------------------------------- CSI-dataset (csi_fingerprint)
def test_grid_position_follows_the_readme():
    assert grid_position("coordinate715.mat") == (7, 15)  # "coordinate 715 means [7, 15]"
    assert grid_position("coordinate1006.mat") == (10, 6) and grid_position("1101.mat") == (11, 1)
    with pytest.raises(ValueError):
        grid_position("imaginary715.mat")


def test_csi_fingerprint_keeps_db_amplitude_per_packet(tmp_path):
    sio = pytest.importorskip("scipy.io")
    rng = np.random.default_rng(1)
    data = {"Lab Dataset/coordinate 1-100/coordinate715.mat": rng.normal(20, 4, (3, 30, 4)),
            "Lab Dataset/coordinate 1-100/coordinate1006.mat": rng.normal(20, 4, (3, 30, 4)),
            "miniLab/coordinate 1-35/203.mat": rng.normal(15, 4, (3, 30, 2))}
    data["Lab Dataset/coordinate 1-100/coordinate715.mat"][1, 4, 3] = -np.inf  # |h| = 0 stored as dB
    for rel, arr in data.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        sio.savemat(tmp_path / rel, {"myData": arr})

    class Tiny(CSIFingerprint):
        files = {"all": tuple((rel, None) for rel in data)}

    t = Tiny(tmp_path).load("all")
    assert t.X.shape == (10, 3, 1, 30) and t.X.dtype == np.float32 and t.meta["modality"] == "csi_amp"
    first = data["Lab Dataset/coordinate 1-100/coordinate715.mat"]
    assert np.allclose(t.X[2, :, 0, :], first[:, :, 2])  # packet k is myData[:, :, k], antennas x subcarriers
    assert np.isnan(t.X[3, 1, 0, 4]) and np.isnan(t.X).sum() == 1  # -inf dB (zero amplitude) -> NaN
    assert t.pos[:4].tolist() == [[7, 15]] * 4 and t.pos[4].tolist() == [10, 6] and t.pos[-1].tolist() == [2, 3]
    assert t.building.tolist() == [0] * 8 + [3] * 2 and t.meta["building_names"][3] == "minilab"
    assert t.groups["point"][[0, 4, 8]].tolist() == ["lab/715", "lab/1006", "minilab/203"]
    assert t.groups["packet"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1] and t.ids[-1] == "minilab-203-0001"

    few = Tiny(tmp_path, area=["minilab", "lab"], packets=1).load("all")
    assert len(few) == 3 and few.groups["packet"].tolist() == [0, 0, 0]
    assert few.meta["building_names"] == {0: "lab", 3: "minilab"} and few.meta["packets_per_point"] == 1
    assert len(Tiny(tmp_path, area="minilab").files["all"]) == 1  # other areas are neither checked nor fetched
    with pytest.raises(ValueError, match="packets"):
        Tiny(tmp_path, packets=0)


def test_csi_fingerprint_manifest_covers_the_repository():
    rels = [rel for rel, _ in CSIFingerprint.files["all"]]
    folders = {area: sum(r.startswith(folder) for r in rels) for area, (_, folder) in CSIFingerprint.areas.items()}
    assert folders == {"lab": 317, "meeting": 176, "conference": 160, "minilab": 35}
    assert len(set(rels)) == len(rels) == len(CSIFingerprint.urls) == 688
    assert all(url.startswith("https://raw.githubusercontent.com/") and " " not in url
               for url in CSIFingerprint.urls.values())
    positions = {(r.split("/")[0], grid_position(r.rsplit("/", 1)[1])) for r in rels}
    assert len(positions) == 688  # one file per reference point


# ---------------------------------------------------------------- H-WILD
def orientation_xy(dx, dy, toward):
    """The authors' orientation_xy.m, line by line (degrees)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if toward == 0:
            return np.where(dy >= 0, -1, 1) * np.abs(np.degrees(np.arctan(dy / dx)))
        if toward == 90:
            return np.where(dx < 0, -1, 1) * np.abs(np.degrees(np.arctan(dx / dy)))
        if toward == -90:
            return np.where(dx < 0, 1, -1) * np.abs(np.degrees(np.arctan(dx / dy)))
        return np.where(dy < 0, -1, 1) * np.abs(np.degrees(np.arctan(dy / dx)))  # 180


def hwild_walk(root, room_folder, prefix, aps, user, state, n, rng, shift_uwb=0.0, drop=0):
    h5py = pytest.importorskip("h5py")
    x, y = rng.uniform(0, 5, n), rng.uniform(0, 5, n)
    truth = {}
    for k, ap in enumerate(aps):
        csi = rng.normal(size=(n, 90)) + 1j * rng.normal(size=(n, 90))
        truth[ap] = csi
        path = root / room_folder / f"{prefix}_{ap}_user{user}_{state}.mat"
        path.parent.mkdir(parents=True, exist_ok=True)
        compound = np.zeros((90, n), dtype=[("real", "<f8"), ("imag", "<f8")])  # MATLAB (n, 90) as h5py sees it
        compound["real"], compound["imag"] = csi.real.T, csi.imag.T
        with h5py.File(path, "w") as f:
            f["features_csi"] = compound
            f["estimations_aoa"] = np.full((1, n), 10.0 * (k + 1))
            last = n - (drop if k == 3 else 0)
            f["uwb_coordinate_x"] = (x + (shift_uwb if k >= 2 else 0.0))[None, :last]
            f["uwb_coordinate_y"] = y[None, :last]
    return truth, np.column_stack([x, y])


def test_hwild_stacks_four_synchronous_aps_per_packet(tmp_path):
    rng = np.random.default_rng(3)
    aps = ("sRE22", "sRE5", "sRE6", "sRE7")
    t1, p1 = hwild_walk(tmp_path, "Conference", "Con", aps, 1, "w", 4, rng)
    t2, p2 = hwild_walk(tmp_path, "Conference", "Con", aps, 2, "wo", 3, rng)
    _, p3 = hwild_walk(tmp_path, "Lounge", "Lounge", ("sRE4", "sRE5", "sRE6", "sRE7"), 6, "w", 2, rng)
    rels = sorted(str(q.relative_to(tmp_path)) for q in tmp_path.rglob("*.mat"))

    class Tiny(HWILD):
        files = {"all": tuple((rel, None) for rel in rels)}

    t = Tiny(tmp_path).load("all")
    assert t.X.shape == (9, 12, 1, 30) and t.X.dtype == np.complex64 and t.meta["modality"] == "csi"
    # receive chain = AP * 3 + antenna; the 90 values of a packet are antenna-major (MATLAB reshape(csi, 30, 3))
    assert np.allclose(t.X[1, 3 * 2 + 1, 0], t1["sRE6"][1, 30:60].astype(np.complex64))
    assert np.allclose(t.X[4, 11, 0], t2["sRE7"][0, 60:90].astype(np.complex64))
    assert t.meta["rx_names"][7] == "AP3/ant2" and t.meta["ap_names"][3] == ("sRE4", "sRE5", "sRE6", "sRE7")
    assert t.meta["antenna_anchor"].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]  # row -> AP (contract, csi)
    assert "anchors" not in t.meta  # two rooms: two AP layouts, no single anchor array
    assert np.allclose(t.pos, np.concatenate([p1, p2, p3]))
    assert t.building.tolist() == [0] * 7 + [3] * 2 and t.groups["user"].tolist() == [1] * 4 + [2] * 3 + [6] * 2
    assert t.groups["interference"].tolist() == [True] * 4 + [False] * 3 + [True] * 2
    assert t.groups["time"].tolist() == [0, 1, 2, 3, 0, 1, 2, 0, 1]
    assert t.groups["trajectory"][4] == "conference/user2_wo" and t.ids[0] == "conference-user1_w-00000"

    sub = Tiny(tmp_path, environment="conference", users=[2], interference=False)
    assert len(sub.files["all"]) == 4 and len(sub.load("all")) == 3  # only that walk's four AP files
    assert np.array_equal(sub.load("all").meta["anchors"], [[-1.7, 3.0], [2.0, -0.6], [4.6, 3.4], [2.0, 6.6]])
    with pytest.raises(ValueError, match="no walk"):
        Tiny(tmp_path, environment="lounge", interference=False)
    with pytest.raises(ValueError, match="single environment"):
        Tiny(tmp_path, features="aoa")

    aoa = Tiny(tmp_path, environment="conference", features="aoa").load("all")
    anchors, boresight, sign = aoa_geometry("conference")
    assert aoa.X.shape == (7, 4) and aoa.meta["modality"] == "aoa" and aoa.meta["units"] == "rad"
    assert np.allclose(aoa.X[0], np.radians([10.0, 20.0, 30.0, 40.0]) * sign)
    assert np.array_equal(aoa.meta["anchors"], anchors) and np.array_equal(aoa.meta["anchor_orientations"], boresight)


@pytest.mark.parametrize("shift, drop, error",
                         [(0.004, 0, None), (0.5, 0, "UWB positions"), (0.0, 1, "number of packets")])
def test_hwild_checks_that_the_four_ap_files_are_one_walk(tmp_path, shift, drop, error):
    _, pos = hwild_walk(tmp_path, "Lounge", "Lounge", ("sRE4", "sRE5", "sRE6", "sRE7"), 1, "w", 3,
                        np.random.default_rng(4), shift_uwb=shift, drop=drop)
    rels = sorted(str(q.relative_to(tmp_path)) for q in tmp_path.rglob("*.mat"))

    class Tiny(HWILD):
        files = {"all": tuple((rel, None) for rel in rels)}

    if error:
        with pytest.raises(ValueError, match=error):
            Tiny(tmp_path).load("all")
    else:  # lounge-like label jitter: two files at x, two at x + 4 mm -> the median is x + 2 mm
        assert np.allclose(Tiny(tmp_path).load("all").pos, pos + [shift / 2, 0.0])


@pytest.mark.parametrize("room", ["conference", "laboratory", "office", "lounge"])
def test_hwild_aoa_convention_matches_orientation_xy(room):
    """Closed form: for targets in the room, sign * orientation_xy (the authors' label) is the
    counter-clockwise angle from the boresight, and boresight + angle is the true bearing."""
    anchors, boresight, sign = aoa_geometry(room)
    toward = __import__("indoorloc.datasets.hwild", fromlist=["AP_TOWARD"]).AP_TOWARD[room]
    for k in range(4):
        u = np.array([np.cos(boresight[k]), np.sin(boresight[k])])
        v = np.array([-u[1], u[0]])
        angles = np.radians(np.linspace(-80, 80, 17))  # in front of the boresight, 2 m away
        targets = anchors[k] + 2.0 * (np.outer(np.cos(angles), u) + np.outer(np.sin(angles), v))
        dx, dy = (targets - anchors[k]).T
        label = orientation_xy(dx, dy, toward[k])
        assert np.allclose(sign[k] * np.radians(label), angles, atol=1e-9), (room, k)
        bearing = np.arctan2(dy, dx)
        assert np.allclose(np.angle(np.exp(1j * (boresight[k] + angles - bearing))), 0, atol=1e-9)


def test_hwild_manifest_covers_the_repository():
    rels = [rel for rel, _ in HWILD.files["all"]]
    assert len(rels) == len(set(rels)) == 172 and len(HWILD.urls) == 172
    counts = {folder: sum(r.startswith(folder + "/") for r in rels)
              for folder in ("Conference", "Laboratory", "Office", "Lounge")}
    assert counts == {"Conference": 36, "Laboratory": 40, "Office": 40, "Lounge": 56}
    walks = {r.rsplit("_", 3)[0].split("/")[0] + r.split("_user")[1] for r in rels}
    assert len(walks) == 43  # 43 walks x 4 AP files


# ---------------------------------------------------------------- real data
def _real(name, rel):
    return (DATA_ROOT / name / rel).is_file()


@pytest.mark.skipif(not _real("ble_indoor", "experiment.zip"), reason="BBIL release archive not found")
def test_real_bbil_split_sizes():
    tr, va, te = load_dataset("ble_indoor", split=("train", "valid", "test"), download=False)
    assert (len(tr), len(va), len(te)) == (35475, 6284, 8304)
    office = te[te.building == 0]
    assert len(office) == 5110 and not np.isnan(office.X[:, :9]).any() and np.isnan(office.X[:, 9:]).all()
    assert len(np.unique(te.groups["trajectory"])) == 39


@pytest.mark.skipif(not _real("ble_rssi_uci", "iBeacon_RSSI_Labeled.csv"), reason="UCI BLE RSSI files not found")
def test_real_uci_ble_counts():
    t = load_dataset("ble_rssi_uci", split="all", download=False)
    assert t.X.shape == (1420, 13) and len(np.unique(t.groups["point"])) == 105
    assert np.isnan(t.X).sum() == 16417 + 15  # -200, and the 15 readings of -198/-199
    assert np.nanmin(t.X) == -88 and np.nanmax(t.X) == -55
    assert len(load_dataset("ble_rssi_uci", split="unlabeled", download=False)) == 5191


@pytest.mark.skipif(not _real("ibeacon_rssi", "data/rss/lib_rss.csv"), reason="UJI BLE RSS files not found")
def test_real_uji_ble_counts():
    t = load_dataset("ibeacon_rssi", split="all", download=False)
    assert t.X.shape == (4752, 46) and np.isnan(t.X).all(axis=1).sum() == 589
    # 68 geo, 48 + 68 lib point numbers, but k and n + 1 - k are one position: 34 + 24 + 34 positions
    assert len(np.unique(t.groups["point"])) == len(np.unique(np.column_stack([t.building, t.pos]), axis=0)) == 92
    for z in ("geo", "lib"):
        for c in np.unique(t.groups["campaign"][t.building == IBeaconRSSI.zones[z][0]]):
            rows = (t.building == IBeaconRSSI.zones[z][0]) & (t.groups["campaign"] == c)
            k, n = t.groups["point_number"][rows], t.groups["point_number"][rows].max()
            first = {int(a): tuple(p) for a, p in zip(k, t.pos[rows])}
            assert all(first[a] == first[n + 1 - a] for a in first), (z, c)
    train, test = load_dataset("ibeacon_rssi", split=("train", "test"), download=False)
    assert (len(train), len(test)) == (876 + 1872, 1080 + 780)
    assert not set(train.groups["point"]) & set(test.groups["point"])  # never tested at a training position


@pytest.mark.skipif(not _real("haloc", "HALOC.zip"), reason="HALOC.zip not found")
def test_real_haloc_sizes():
    tr, va, te = load_dataset("haloc", split=("train", "valid", "test"), download=False)
    assert (len(tr), len(va), len(te)) == (96491, 28111, 14277)
    assert 0 <= te.pos[:, 0].min() and te.pos[:, 0].max() < 20.01
    # ESP-IDF (imag, real) pairs give a smooth spectrum: mean amplitude step between adjacent subcarriers
    # 0.41 on the first 3,000 packets of sequence 0 (0.60 with the authors' pairing of neighbouring bytes)
    amp = np.abs(tr.X[:3000, 0, 0])
    steps = np.concatenate([np.diff(amp[:, :26]), np.diff(amp[:, 26:])], axis=1)  # skip the DC gap
    assert abs(np.abs(steps).mean() - 0.411) < 0.001


@pytest.mark.skipif(not _real("csi_fingerprint", "miniLab/coordinate 1-35/101.mat"),
                    reason="CSI-dataset files not found")
def test_real_csi_fingerprint_small_areas():
    t = load_dataset("csi_fingerprint", split="all", area=["conference", "minilab"], download=False)
    assert t.X.shape == ((160 + 35) * 50, 3, 1, 30) and np.isfinite(t.X).all()
    assert len(np.unique(t.groups["point"])) == 195 and -15 < t.X.min() and t.X.max() < 40  # dB amplitudes


@pytest.mark.skipif(not _real("hwild", "Conference/Con_sRE7_user5_wo.mat"), reason="H-WILD conference files not found")
def test_real_hwild_conference_aoa_is_in_the_library_convention():
    t = load_dataset("hwild", split="all", environment="conference", features="aoa", download=False)
    assert t.X.shape[1] == 4 and len(np.unique(t.groups["trajectory"])) == 9
    anchors, boresight = t.meta["anchors"], t.meta["anchor_orientations"]
    # the geometric angle of the UWB position seen from each AP, counter-clockwise from its boresight
    geo = np.arctan2(t.pos[:, None, 1] - anchors[:, 1], t.pos[:, None, 0] - anchors[:, 0]) - boresight
    err = np.degrees(np.abs(np.angle(np.exp(1j * (t.X - geo)))))
    assert np.all(np.median(err, axis=0) < 20)  # 2-D FFT estimates vs geometry: same convention, same sign

    # antenna order of the raw CSI: the adjacent-antenna phase step grows with sin(angle ccw from ap_toward)
    c = load_dataset("hwild", split="all", environment="conference", users=[1], interference=False, download=False)
    toward = np.radians(__import__("indoorloc.datasets.hwild", fromlist=["AP_TOWARD"]).AP_TOWARD["conference"])
    psi = np.arctan2(c.pos[:, None, 1] - anchors[:, 1], c.pos[:, None, 0] - anchors[:, 0]) - toward
    H = c.X.reshape(len(c), 4, 3, 30)
    kappa = np.linspace(-6, 6, 241)
    for k in range(4):
        step = np.angle((H[:, k, 1:] * H[:, k, :-1].conj()).sum(-1))
        fit = [np.abs(np.exp(1j * (step - kk * np.sin(psi[:, k, None]))).mean()) for kk in kappa]
        assert 1.5 < kappa[int(np.argmax(fit))] < 3.5, k  # about 0.4 wavelength spacing, positive in every AP
