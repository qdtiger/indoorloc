"""0.1 compatibility (rule 5.7): the 0.1 README snippets through _legacy and the two L3 hooks."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from conftest import PROJECT, UJI_ROOT
from indoorloc import _legacy as legacy
from indoorloc.core import SampleTable
from indoorloc.datasets import load_dataset
from indoorloc.methods import create_model


def test_readme_l3_snippet_fit_signals_and_locations_predict_one_signal():
    X_own = np.array([[-40.0, -70, -90], [-70, -40, -90], [-90, -70, -40]])
    with pytest.warns(FutureWarning, match="rssi_values"):
        signals = [legacy.WiFiSignal(rssi_values=row) for row in X_own]
    locations = [legacy.Location(coordinate=legacy.Coordinate(x, y), floor=f)
                 for (x, y), f in [((0.0, 0.0), 0), ((10.0, 0.0), 0), ((10.0, 10.0), 1)]]
    model = create_model("wknn", k=1).fit(signals, locations)  # hook 1: a list of 0.1 Locations as y
    result = model.predict(signals[2])  # hook 2: _legacy_wrap -> LocalizationResult
    assert isinstance(result, legacy.LocalizationResult) and (result.x, result.y, result.floor) == (10.0, 10.0, 1)
    assert model.predict(X_own[:1]).tolist() == [[0.0, 0.0]]  # arrays keep the 0.2 contract


def test_readme_l4_snippet_from_predictions_is_added_by_the_shim_not_by_l4():
    code = ("import warnings\nfrom indoorloc.evaluation import EvaluationResults\n"
            "assert not hasattr(EvaluationResults, 'from_predictions')  # L4 holds no 0.1 code\n"
            "from indoorloc._legacy import Coordinate, Location\n"
            "truths = [Location(coordinate=Coordinate(x, y), floor=1) for x, y in [(0, 0), (3, 4)]]\n"
            "preds = [Location(coordinate=Coordinate(0, y), floor=f) for y, f in [(1, 1), (0, 2)]]\n"
            "with warnings.catch_warnings(record=True) as caught:\n"
            "    warnings.simplefilter('always')\n"
            "    r = EvaluationResults.from_predictions(preds, truths)\n"
            "assert (r.mean_error, r.floor_accuracy) == (3.0, 50.0) and caught[0].category is FutureWarning")
    subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT, check=True)


@pytest.mark.skipif(not (UJI_ROOT / "validationData.csv").is_file(), reason="UJIIndoorLoc files not found")
def test_old_dataset_id_gives_the_01_object_new_id_the_table():
    with pytest.warns(FutureWarning, match="ujiindoorloc"):
        old = load_dataset("ujindoorloc", split="test", root=UJI_ROOT, download=False)
    new = load_dataset("ujiindoorloc", split="test", root=UJI_ROOT, download=False)
    assert isinstance(old, legacy.LegacyDataset) and type(new) is SampleTable
    assert np.nanmax(new.X) <= 0 and 0 <= old.X.min() and old.X.max() <= 1  # dBm vs 0.1 min-max
    signal, location = old[0]
    assert isinstance(signal, legacy.WiFiSignal) and location.floor == int(new.floor[0])
    assert old.to_tensors()[1].shape == (1111, 4)
    assert create_model("wknn", k=5).fit(old).evaluate(old).n == 1111  # the 0.1 README headline flow
