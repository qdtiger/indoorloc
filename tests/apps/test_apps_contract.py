"""L5 classes follow the Estimator contract (parameters only in __init__, clone, repr)."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from conftest import HEAVY, PROJECT
from indoorloc.apps import (PDR, ConstantVelocityKF, ExtendedKalmanTracker, FloorMap, KalmanTracker, Navigator,
                            OnlineLocalizer, ParticleFilter, PDRFusion, StepDetector)
from indoorloc.core import clone

ANCHORS = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
ESTIMATORS = [
    KalmanTracker(motion="ca", gate=3.0), ExtendedKalmanTracker(ANCHORS, range_std=0.2), ConstantVelocityKF(0.3),
    ParticleFilter(50, floor_map=FloorMap.from_polygons([[(0, 0), (5, 0), (5, 5)]]), recovery=(0.05, 0.5)),
    StepDetector(min_interval=0.25), PDR(StepDetector(), step_model="kim"), PDRFusion(ParticleFilter(10)),
    OnlineLocalizer(tracker=KalmanTracker()), Navigator(resolution=0.5),
]


@pytest.mark.parametrize("est", ESTIMATORS, ids=lambda e: type(e).__name__)
def test_clone_keeps_parameters_and_drops_state(est):
    twin = clone(est)
    assert type(twin) is type(est) and set(twin.get_params()) == set(est.get_params())
    assert repr(twin) == repr(est) and repr(est).startswith(type(est).__name__ + "(")  # same parameters
    assert not [k for k in vars(twin) if k.endswith("_") and not k.startswith("_")]  # no learned state


def test_sklearn_clone_accepts_the_trackers():
    base = pytest.importorskip("sklearn.base")
    for est in ESTIMATORS:
        assert type(base.clone(est)) is type(est)


def test_apps_import_only_numpy():
    code = (f"import sys\nfor m in {HEAVY!r}: sys.modules[m] = None\n"
            "import indoorloc.apps, indoorloc.apps.navigation, indoorloc.apps.fusion\n"
            "assert not [m for m in sys.modules if m.startswith(('indoorloc.datasets', 'indoorloc.methods'))]")
    subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT, check=True)
