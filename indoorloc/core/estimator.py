"""scikit-learn's estimator contract, implemented without importing scikit-learn.

Any subclass works with ``sklearn.base.clone``, ``Pipeline`` and ``GridSearchCV``
(constructor arguments are the parameters, learned state is set in ``fit``),
while code that never touches sklearn does not pay for importing it.
"""
from __future__ import annotations

import copy
import inspect

from ._optional import requires


class NotFittedError(ValueError, AttributeError):
    """Raised when an estimator is used before ``fit`` (same bases as sklearn's)."""


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True  # NaN defaults (e.g. FillMissing(missing=nan)) survive save/load
    try:
        return a is b or bool(a == b)
    except (TypeError, ValueError):  # arrays and other non-scalar parameters
        return False


def clone(obj):
    """Unfitted copy with the same parameters (sklearn.base.clone without sklearn)."""
    if isinstance(obj, (list, tuple)):
        items = [clone(o) for o in obj]
        return type(obj)(*items) if hasattr(obj, "_fields") else type(obj)(items)  # a namedtuple takes fields
    if not isinstance(obj, Estimator):
        return copy.deepcopy(obj)
    twin = type(obj)(**{k: clone(v) for k, v in obj.get_params(deep=False).items()})
    if "_metadata_request" in vars(obj):  # set_fit_request survives cloning, as in sklearn
        twin._metadata_request = copy.deepcopy(obj._metadata_request)
    return twin


class Estimator:
    """Base for L2 transforms and L3 localizers.

    ``__init__`` only stores its arguments (no validation, no computation), as
    sklearn requires; validation happens in ``fit``. Learned attributes end in ``_``.
    """

    _estimator_type: str | None = None  # "regressor" or "transformer"
    _allow_nan = False
    _requires_fit = True

    @classmethod
    def _get_param_names(cls) -> list[str]:
        if cls.__init__ is object.__init__:
            return []
        params = inspect.signature(cls.__init__).parameters.values()
        return sorted(p.name for p in params
                      if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD))

    def get_params(self, deep: bool = True) -> dict:
        out = {}
        for name in self._get_param_names():
            value = getattr(self, name)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                out.update({f"{name}__{k}": v for k, v in value.get_params().items()})
            out[name] = value
        return out

    def set_params(self, **params):
        valid = self._get_param_names()
        nested: dict[str, dict] = {}
        for key, value in params.items():
            name, sep, sub = key.partition("__")
            if name not in valid:
                raise ValueError(f"invalid parameter {name!r} for {type(self).__name__}; valid: {valid}")
            if sep:
                nested.setdefault(name, {})[sub] = value
            else:
                setattr(self, name, value)
        for name, sub in nested.items():
            getattr(self, name).set_params(**sub)
        return self

    def _check_fitted(self, attr: str = "n_features_in_") -> None:
        if not hasattr(self, attr):
            raise NotFittedError(f"this {type(self).__name__} is not fitted yet; call fit first")

    def _get_state(self) -> dict:
        """Everything ``fit`` learned, for ``save``. Arrays, numbers, str, containers, other
        Estimators and opted-in data objects (``_save_via_dict = True`` with ``to_dict`` /
        ``from_dict``, e.g. a ``FloorMap``; see ``core.persistence``) are stored as they are;
        override to swap any other state (a torch module) for arrays (its state_dict), and
        ``_set_state`` to rebuild it."""
        params = self._get_param_names()
        return {k: v for k, v in vars(self).items() if k not in params}

    def _set_state(self, state: dict) -> None:
        self.__dict__.update(state)

    def save(self, path, *, info: dict | None = None, compress: bool = True):
        """Write ``path/config.json`` + ``path/arrays.npz`` (no pickle); ``core.load_model``
        reads it back. Parameters and state follow the rules of :meth:`_get_state`; see
        ``core.persistence`` for the format."""
        from .persistence import save

        return save(self, path, info=info, compress=compress)

    def __repr__(self) -> str:
        defaults = inspect.signature(type(self).__init__).parameters
        args = [f"{k}={v!r}" for k, v in self.get_params(deep=False).items()
                if not _same(v, defaults[k].default)]
        return f"{type(self).__name__}({', '.join(args)})"

    def set_fit_request(self, **requests):
        """sklearn metadata routing (``sklearn.set_config(enable_metadata_routing=True)``): the
        fit keywords a meta-estimator should pass on, e.g. ``set_fit_request(floor=True)``.
        Values as in sklearn: True, False, None or an alias string."""
        self._metadata_request = {**getattr(self, "_metadata_request", {}), **requests}
        return self

    def get_metadata_routing(self):
        # Only sklearn calls this (with metadata routing enabled), so sklearn is importable.
        routing = requires("sklearn.utils.metadata_routing", "sklearn")
        request = routing.MetadataRequest(owner=type(self).__name__)
        for param, alias in getattr(self, "_metadata_request", {}).items():
            request.fit.add_request(param=param, alias=alias)
        return request

    def __sklearn_tags__(self):
        # Only sklearn calls this, so sklearn is importable whenever it runs.
        su = requires("sklearn.utils", "sklearn")
        tags = su.Tags(estimator_type=None, target_tags=su.TargetTags(required=False))
        tags.input_tags.allow_nan = self._allow_nan
        tags.requires_fit = self._requires_fit
        if self._estimator_type == "regressor":
            tags.estimator_type = "regressor"
            tags.regressor_tags = su.RegressorTags()
            tags.target_tags.required = True
            tags.target_tags.multi_output = True
        elif self._estimator_type == "transformer":
            tags.transformer_tags = su.TransformerTags()
        return tags
