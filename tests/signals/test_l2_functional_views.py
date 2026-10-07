"""L2 functional RSSI helpers (power units, aggregation) and the one-scan views."""
from __future__ import annotations

import numpy as np
import pytest

from indoorloc import _legacy
from indoorloc.core import SampleTable
from indoorloc.signals import APSelect, BLESignal, FillMissing, WiFiSignal
from indoorloc.signals import functional as F

NAN = np.nan


def test_dbm_mw_round_trip():
    np.testing.assert_allclose(F.dbm_to_mw(np.array([0.0, -30.0, 10.0])), [1.0, 1e-3, 10.0])
    assert F.dbm_to_mw(np.array([-50.0], np.float32)).dtype == np.float32
    out = F.mw_to_dbm(np.array([1.0, 1e-9, 0.0, NAN]))
    assert np.array_equal(out, [0.0, -90.0, NAN, NAN], equal_nan=True)  # no power received -> not heard
    x = np.array([-97.0, -45.5, -60.0])
    np.testing.assert_allclose(F.mw_to_dbm(F.dbm_to_mw(x)), x)


def test_aggregate_rssi_methods():
    scans = np.array([[-50.0, NAN, -90.0],
                      [-60.0, NAN, NAN],
                      [-70.0, -80.0, NAN]], dtype=np.float32)
    assert np.array_equal(F.aggregate_rssi(scans), [-60.0, -80.0, -90.0])
    assert F.aggregate_rssi(scans).dtype == np.float32
    assert np.array_equal(F.aggregate_rssi(scans, "median"), [-60.0, -80.0, -90.0])
    assert np.array_equal(F.aggregate_rssi(scans, "max"), [-50.0, -80.0, -90.0])
    assert np.array_equal(F.aggregate_rssi(scans, min_count=2), [-60.0, NAN, NAN], equal_nan=True)
    # mean power: 10 log10((1e-5 + 1e-6 + 1e-7) / 3) = -54.33 dBm, above the dBm mean -60
    assert F.aggregate_rssi(scans, "power_mean")[0] == pytest.approx(10 * np.log10(1.11e-5 / 3), abs=1e-4)
    assert np.array_equal(F.aggregate_rssi(scans, axis=1), [-70.0, -60.0, -75.0])
    with pytest.raises(ValueError, match="method"):
        F.aggregate_rssi(scans, "sum")


def test_aggregate_rssi_groups_averages_repeated_scans_per_point():
    X = np.array([[-50.0, NAN], [-70.0, -80.0], [-40.0, -40.0], [-60.0, NAN]])
    pos = np.array([[1.0, 2.0], [0.0, 0.0], [1.0, 2.0], [0.0, 0.0]])
    keys, agg, counts = F.aggregate_rssi_groups(X, pos)
    assert keys.tolist() == [[0.0, 0.0], [1.0, 2.0]] and counts.tolist() == [2, 2]
    assert np.array_equal(agg, [[-65.0, -80.0], [-45.0, -40.0]])
    keys, agg, _ = F.aggregate_rssi_groups(X, np.array(["b", "a", "b", "a"]), "max", min_count=2)
    assert keys.tolist() == ["a", "b"] and np.array_equal(agg, [[-60.0, NAN], [-40.0, NAN]], equal_nan=True)


def test_wifi_view_from_readings_and_legacy_subclass():
    ids = ("AP1", "AP2", "AP3")
    s = WiFiSignal.from_readings({"AP3": -70, "AP1": -40, "ROGUE": -30}, ids)
    assert np.array_equal(s.rssi, [-40.0, NAN, -70.0], equal_nan=True) and s.detected == {"AP1": -40.0, "AP3": -70.0}
    assert s.take([2, 0]).ap_ids == ("AP3", "AP1") and len(s) == 3 and s.ids == ids
    with pytest.raises(ValueError, match="unique"):
        WiFiSignal.from_readings({}, ("a", "a"))
    with pytest.raises(ValueError, match="3 transmitter ids"):
        WiFiSignal(np.zeros(2), ids)
    with pytest.warns(FutureWarning):
        old = _legacy.WiFiSignal(rssi_values=[100, -50, -60])
    filled = FillMissing()(old)  # a 0.1 signal goes through 0.2 transforms and stays a 0.1 signal
    assert type(filled) is _legacy.WiFiSignal and filled.rssi.tolist() == [-104, -50, -60]


def test_ble_view_needs_an_explicit_sentinel():
    ids = ("b1", "b2", "b3")
    s = BLESignal.from_raw([-100, -200, -75], missing=-200, beacon_ids=ids)
    assert np.array_equal(s.rssi, [-100.0, NAN, -75.0], equal_nan=True)  # -100 dBm is a real reading here
    with pytest.raises(TypeError):
        BLESignal.from_raw([-100, -75])  # no default sentinel
    assert BLESignal.beacon_id("FDA50693-A4E2-4FB1-AFCF-C6EB07647825", 10001, 19641) == \
        "fda50693-a4e2-4fb1-afcf-c6eb07647825:10001:19641"
    table = SampleTable(np.array([[-60.0, NAN, -80.0]]), np.zeros((1, 2)),
                        meta={"feature_names": ids, "modality": "ble_rssi"})
    view = BLESignal.from_table(table, 0)
    kept = APSelect(2, "coverage").fit(table)(view)
    assert isinstance(kept, BLESignal) and kept.beacon_ids == ("b1", "b3")
    assert BLESignal.from_readings({"b2": -66}, ids).detected == {"b2": -66.0}
