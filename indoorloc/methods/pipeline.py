"""A localizer that owns its preprocessing, so raw tables (NaN = missing) go straight in."""
from __future__ import annotations

from ..core import Prediction, clone
from .base import BaseLocalizer


class LocalizerPipeline(BaseLocalizer):
    """``preprocess`` (any L2 transform or a list of them, fitted on training data only) then ``localizer``.

    Unlike an sklearn Pipeline, ``localize``/``evaluate`` keep floor and building, and
    a deployed model accepts raw scans. Nested parameters work as usual
    (``localizer__k``), so GridSearchCV and clone see one estimator. Rule: preprocessing
    lives here or before the model, never both (it would be applied twice).
    """

    _allow_nan = True

    def __init__(self, preprocess=None, localizer=None):
        self.preprocess = preprocess
        self.localizer = localizer

    def fit(self, X, y=None, *, floor=None, building=None, **fit_params):
        """``preprocess__<name>=`` goes to ``preprocess.fit`` (e.g. a CORAL target);
        ``eval_set=[(X_val, y_val), ...]`` is preprocessed like X; the rest goes to the localizer."""
        if self.localizer is None:
            raise ValueError("LocalizerPipeline needs a localizer")
        if hasattr(X, "toarray"):
            raise TypeError("sparse input is not supported; pass a dense array (X.toarray())")
        pre = {k.removeprefix("preprocess__"): fit_params.pop(k) for k in list(fit_params)
               if k.startswith("preprocess__")}
        step = self.preprocess
        if isinstance(step, (list, tuple)):  # [t1, t2]: applied in order, as a Compose
            from ..signals import Compose  # L3 -> L2 edge, loaded only when a list is given

            step = Compose(list(step)) if step else None
        if pre and step is None:
            raise TypeError(f"got {sorted(pre)} for preprocess, but this pipeline has no preprocess step")
        self.preprocess_ = None if step is None else clone(step)
        self.localizer_ = clone(self.localizer)
        Xt = X if self.preprocess_ is None else self.preprocess_.fit_transform(X, **pre)  # a table stays a table
        if self.preprocess_ is not None and fit_params.get("eval_set") is not None:
            fit_params["eval_set"] = [(self.preprocess_.transform(Xv), yv) for Xv, yv in fit_params["eval_set"]]
        self.localizer_.fit(Xt, y, floor=floor, building=building, **fit_params)
        self.n_features_in_ = self.localizer_.n_features_in_
        self.target_ndim_ = self.localizer_.target_ndim_
        return self

    def localize(self, X) -> Prediction:
        self._check_fitted()
        return self.localizer_.localize(X if self.preprocess_ is None else self.preprocess_.transform(X))
