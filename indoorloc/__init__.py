"""IndoorLoc: wireless indoor localization in five layers that exchange plain arrays.

    L1 datasets    WiFi / BLE / CSI / IMU datasets and simulators   -> SampleTable
    L2 signals     preprocessing and signal representations         SampleTable -> SampleTable
    L3 methods     fingerprinting, model-based, deep, transfer      fit(X, y) / localize(X) -> Prediction
    L4 evaluation  metrics, protocols, bounds, literature            evaluate(y_true, y_pred)
    L5 apps        tracking, PDR, fusion, streaming, navigation      built on L2-L4

Your own data and code plug in at each layer: ``SampleTable(X, pos, ...)`` or a ``Dataset``
subclass for ``register_dataset`` (L1), a ``Transform`` subclass (L2), a ``BaseLocalizer``
subclass for ``register_model`` (L3), a ``Protocol`` whose split returns ``Fold``s for
``register_protocol`` (L4); see docs/guide/extending.md.

``import indoorloc`` runs only this file. Every name below is imported on first access
(PEP 562), so using one layer never loads the others, and nothing here imports numpy,
torch or scikit-learn.
"""
from __future__ import annotations

import importlib
import sys
import warnings
from typing import TYPE_CHECKING

from ._version import __version__

_SUBMODULES = {
    "core": "core", "datasets": "datasets", "signals": "signals", "methods": "methods",
    "evaluation": "evaluation", "apps": "apps", "transforms": "signals.transforms",
}
_EXPORTS = {
    # core
    "SampleTable": "core", "Prediction": "core", "load_model": "core", "clone": "core",
    # L1
    "load_dataset": "datasets", "list_datasets": "datasets", "dataset_info": "datasets", "Dataset": "datasets",
    "register_dataset": "datasets",
    # L2
    "Transform": "signals", "Compose": "signals", "FillMissing": "signals", "RSSINormalize": "signals",
    "APFilter": "signals", "APSelect": "signals", "APDropout": "signals", "GaussianNoise": "signals",
    "DeviceCalibration": "signals",
    "PositiveRepresentation": "signals", "ExponentialRepresentation": "signals", "PowedRepresentation": "signals",
    "CSIAmplitude": "signals", "CSIPhaseSanitize": "signals", "HampelFilter": "signals", "SubcarrierSelect": "signals",
    "BLESignal": "signals", "MagneticFeatures": "signals", "MagnetometerCalibration": "signals",
    # L3 (the localizers also have a registry name for create_model; RadioMapInterpolator, CORAL, TCA do not)
    "create_model": "methods", "list_models": "methods", "register_model": "methods", "BaseLocalizer": "methods",
    "LocalizerPipeline": "methods", "KNNLocalizer": "methods.neighbors", "WKNNLocalizer": "methods.neighbors",
    "SVMLocalizer": "methods.sklearn_wrap", "RandomForestLocalizer": "methods.sklearn_wrap",
    "ExtraTreesLocalizer": "methods.sklearn_wrap", "GradientBoostingLocalizer": "methods.sklearn_wrap",
    "HorusLocalizer": "methods.probabilistic", "GPRadioMapLocalizer": "methods.gaussian_process",
    "EnsembleLocalizer": "methods.ensemble", "StackingLocalizer": "methods.ensemble",
    "HierarchicalLocalizer": "methods.hierarchical", "RadioMapInterpolator": "methods.interpolation",
    "TrilaterationLocalizer": "methods.geometric", "TDOALocalizer": "methods.geometric",
    "WeightedCentroidLocalizer": "methods.geometric", "PathLossLocalizer": "methods.pathloss",
    "AoALocalizer": "methods.aoa", "LambertianLocalizer": "methods.vlc",
    "MagneticDTWLocalizer": "methods.magnetic",
    "MLPLocalizer": "methods.deep", "CNN1DLocalizer": "methods.deep", "DeepLocalizer": "methods.deep",
    "CORAL": "methods.transfer", "TCA": "methods.transfer",
    # L4
    "evaluate": "evaluation", "EvaluationResults": "evaluation", "ipin_score": "evaluation",
    "get_protocol": "evaluation", "list_protocols": "evaluation", "register_protocol": "evaluation",
    "Protocol": "evaluation", "Fold": "evaluation",
    # L5
    "KalmanTracker": "apps", "ExtendedKalmanTracker": "apps", "ParticleFilter": "apps", "FloorMap": "apps",
    "StepDetector": "apps", "PDR": "apps", "PDRFusion": "apps", "OnlineLocalizer": "apps", "Navigator": "apps",
}
# 0.1 names: still importable in 0.2.x (FutureWarning where the behaviour changed), removed in 0.3.
_LEGACY = {
    "WiFiSignal": ("_legacy", "WiFiSignal", None),  # accepts rssi= (0.2) and rssi_values= (0.1, warns)
    "Location": ("_legacy", "Location", "indoorloc._legacy.Location (0.1 value type)"),
    "Coordinate": ("_legacy", "Coordinate", "indoorloc._legacy.Coordinate (0.1 value type)"),
    "LocalizationResult": ("_legacy", "LocalizationResult", "Prediction"),
    "list_available_datasets": ("datasets", "list_datasets", "list_datasets"),
}
__all__ = sorted([*_SUBMODULES, *_EXPORTS, "WiFiSignal", "__version__"])


def __getattr__(name: str):
    if name in _SUBMODULES:
        value = importlib.import_module(f"{__name__}.{_SUBMODULES[name]}")
    elif name in _EXPORTS:
        value = getattr(importlib.import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    elif name in _LEGACY:
        module, attr, replacement = _LEGACY[name]
        if replacement:
            warnings.warn(f"indoorloc.{name} is deprecated and will be removed in 0.3; use {replacement}",
                          FutureWarning, stacklevel=2)
        return getattr(importlib.import_module(f"{__name__}.{module}"), attr)  # not cached: warn every time
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}{_where(name)}")
    globals()[name] = value  # later lookups skip __getattr__
    return value


def _where(name: str) -> str:
    """Point a public name that is not re-exported here to its layer. Only layers that are
    already imported are searched, so a failed lookup (``hasattr``, IDE probes) imports nothing."""
    import difflib

    if name.startswith("_"):
        return ""
    for layer in ("core", "datasets", "signals", "methods", "evaluation", "apps"):
        module = sys.modules.get(f"{__name__}.{layer}")
        if module is not None and name in getattr(module, "__all__", ()):
            return f"; it is in indoorloc.{layer}: from indoorloc.{layer} import {name}"
    if difflib.get_close_matches(name, __all__, n=1):
        return ""  # a typo of an exported name: Python (3.12+) adds "Did you mean ...?" itself
    return ("; names not re-exported here live in the layer packages indoorloc.core, .datasets, "
            ".signals, .methods, .evaluation and .apps")


def __dir__() -> list[str]:
    return __all__


if TYPE_CHECKING:  # static analysers and IDEs see the real names
    from . import apps, core, datasets, evaluation, methods, signals  # noqa: F401
    from ._legacy import WiFiSignal  # noqa: F401
    from .core import Prediction, SampleTable, clone, load_model  # noqa: F401
    from .datasets import Dataset, dataset_info, list_datasets, load_dataset, register_dataset  # noqa: F401
    from .apps import (PDR, ExtendedKalmanTracker, FloorMap, KalmanTracker, Navigator,  # noqa: F401
                       OnlineLocalizer, ParticleFilter, PDRFusion, StepDetector)
    from .evaluation import (EvaluationResults, Fold, Protocol, evaluate, get_protocol,  # noqa: F401
                             ipin_score, list_protocols, register_protocol)
    from .methods import BaseLocalizer, LocalizerPipeline, create_model, list_models, register_model  # noqa: F401
    from .methods.aoa import AoALocalizer  # noqa: F401
    from .methods.magnetic import MagneticDTWLocalizer  # noqa: F401
    from .methods.vlc import LambertianLocalizer  # noqa: F401
    from .methods.deep import CNN1DLocalizer, DeepLocalizer, MLPLocalizer  # noqa: F401
    from .methods.ensemble import EnsembleLocalizer, StackingLocalizer  # noqa: F401
    from .methods.gaussian_process import GPRadioMapLocalizer  # noqa: F401
    from .methods.geometric import TDOALocalizer, TrilaterationLocalizer, WeightedCentroidLocalizer  # noqa: F401
    from .methods.hierarchical import HierarchicalLocalizer  # noqa: F401
    from .methods.interpolation import RadioMapInterpolator  # noqa: F401
    from .methods.neighbors import KNNLocalizer, WKNNLocalizer  # noqa: F401
    from .methods.pathloss import PathLossLocalizer  # noqa: F401
    from .methods.probabilistic import HorusLocalizer  # noqa: F401
    from .methods.sklearn_wrap import (ExtraTreesLocalizer, GradientBoostingLocalizer,  # noqa: F401
                                       RandomForestLocalizer, SVMLocalizer)
    from .methods.transfer import CORAL, TCA  # noqa: F401
    from .signals import (APDropout, APFilter, APSelect, BLESignal, Compose, CSIAmplitude,  # noqa: F401
                          CSIPhaseSanitize, DeviceCalibration, ExponentialRepresentation, FillMissing,
                          GaussianNoise, HampelFilter, MagneticFeatures, MagnetometerCalibration,
                          PositiveRepresentation, PowedRepresentation, RSSINormalize, SubcarrierSelect,
                          Transform)
    from .signals import transforms  # noqa: F401
