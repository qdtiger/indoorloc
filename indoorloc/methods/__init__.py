"""L3: localization methods behind one sklearn-style contract.

Only numpy is imported here. Method modules load when first created, so a torch-backed
``"mlp"`` entry costs nothing until someone asks for it. Every entry is a
``"module:Class"`` string; ``create_model`` also accepts such a string directly.

Families:
  fingerprinting  knn, wknn, svm, rf, extratrees, gbdt, horus, gp_radiomap, ensemble, stacking, hierarchical
  model-based     trilateration, tdoa, centroid, pathloss, aoa, vlc (visible light)
  sequence        magnetic_dtw (geomagnetic sequence matching)
  deep            mlp, cnn1d, deep (timm backbone)
  (domain adaptation transforms live in ``methods.transfer``: CORAL, TCA)
"""
from __future__ import annotations

from ..core import Registry
from .base import BaseLocalizer
from .pipeline import LocalizerPipeline

_P = "indoorloc.methods"
METHODS = Registry("method", {
    # Fingerprinting
    "knn": f"{_P}.neighbors:KNNLocalizer",
    "wknn": f"{_P}.neighbors:WKNNLocalizer",
    "svm": f"{_P}.sklearn_wrap:SVMLocalizer",
    "rf": f"{_P}.sklearn_wrap:RandomForestLocalizer",
    "extratrees": f"{_P}.sklearn_wrap:ExtraTreesLocalizer",
    "gbdt": f"{_P}.sklearn_wrap:GradientBoostingLocalizer",
    "horus": f"{_P}.probabilistic:HorusLocalizer",
    "gp_radiomap": f"{_P}.gaussian_process:GPRadioMapLocalizer",
    "ensemble": f"{_P}.ensemble:EnsembleLocalizer",
    "stacking": f"{_P}.ensemble:StackingLocalizer",
    "hierarchical": f"{_P}.hierarchical:HierarchicalLocalizer",
    # Model-based (ranging / angles / propagation models)
    "trilateration": f"{_P}.geometric:TrilaterationLocalizer",
    "tdoa": f"{_P}.geometric:TDOALocalizer",
    "centroid": f"{_P}.geometric:WeightedCentroidLocalizer",
    "pathloss": f"{_P}.pathloss:PathLossLocalizer",
    "aoa": f"{_P}.aoa:AoALocalizer",
    "vlc": f"{_P}.vlc:LambertianLocalizer",
    # Sequence matching (magnetic field)
    "magnetic_dtw": f"{_P}.magnetic:MagneticDTWLocalizer",
    # Deep learning (extra [deep])
    "mlp": f"{_P}.deep:MLPLocalizer",
    "cnn1d": f"{_P}.deep:CNN1DLocalizer",
    "deep": f"{_P}.deep:DeepLocalizer",
    # Aliases
    "random_forest": "rf",
    "weighted_knn": "wknn",
    "multilateration": "trilateration",
})


def list_models() -> list[str]:
    """Registry names of the built-in and registered methods (aliases left out), sorted."""
    return METHODS.names()


def create_model(name, *, preprocess=None, **params) -> BaseLocalizer:
    """``create_model("wknn", k=5)``; ``preprocess=`` wraps it in a LocalizerPipeline.

    ``name`` may also be ``"package.module:Class"`` or the localizer class itself (no
    registration needed). ``preprocess`` is one L2 transform or a list of them, applied in
    order (a list becomes a ``Compose``).
    """
    model = METHODS.get(name)(**params)
    if isinstance(preprocess, (list, tuple)):
        from ..signals import Compose  # L3 -> L2 edge, loaded only when a list is given

        preprocess = Compose(list(preprocess)) if preprocess else None
    return model if preprocess is None else LocalizerPipeline(preprocess, model)


def register_model(name: str, target=None, *, force: bool = False):
    """Register a class or ``"module:Class"`` string; usable as a decorator."""
    return METHODS.register(name, target, force=force)


__all__ = ["METHODS", "BaseLocalizer", "LocalizerPipeline", "create_model", "list_models", "register_model"]
