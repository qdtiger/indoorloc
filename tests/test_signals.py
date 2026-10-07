from __future__ import annotations

import numpy as np
import pytest

from indoorloc.core import SampleTable
from indoorloc.signals import Compose, FillMissing, RSSINormalize, WiFiSignal


def test_row_batch_table_and_view_give_identical_values():
    raw = np.array([[-50, 100, -104], [100, -80, -26]], dtype=np.float32)  # 100 = file sentinel
    batch = Compose([FillMissing(-104, missing=100), RSSINormalize()])(raw)
    assert batch.dtype == np.float32
    assert np.allclose(batch, [[54 / 104, 0, 0], [0, 24 / 104, 78 / 104]])
    rows = [Compose([FillMissing(-104, missing=100), RSSINormalize()])(r) for r in raw]
    assert all(np.array_equal(r, b) for r, b in zip(rows, batch))

    table = SampleTable(np.where(raw == 100, np.nan, raw), np.zeros((2, 2)))
    nan_pipeline = Compose([FillMissing(), RSSINormalize()])
    assert np.array_equal(nan_pipeline(table).X, batch)
    view = nan_pipeline(WiFiSignal.from_table(table, 1))
    assert isinstance(view, WiFiSignal) and np.array_equal(view.rssi, batch[1])
    assert np.array_equal(nan_pipeline(WiFiSignal.from_raw(raw[1])).rssi, batch[1])


def test_raw_sentinel_is_an_error_not_a_silent_scale():
    with pytest.warns(UserWarning, match="missing=100"), pytest.raises(ValueError, match="missing=100"):
        Compose([FillMissing(), RSSINormalize()])(np.array([-50.0, 100.0]))  # FillMissing warns, RSSINormalize refuses


def test_fitted_range_comes_from_training_data_only():
    norm = RSSINormalize(lo=None, hi=None).fit(np.array([[-90.0, -30.0]]))
    assert (norm.lo_, norm.hi_) == (-90.0, -30.0)
    assert np.allclose(norm(np.array([[-60.0, -45.0]])), [[0.5, 0.75]])


def test_view_reports_heard_aps():
    signal = WiFiSignal(np.array([np.nan, -70.0]), ("AP1", "AP2"))
    assert signal.detected == {"AP2": -70.0} and np.asarray(signal).shape == (2,)
