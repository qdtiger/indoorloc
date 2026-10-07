"""scikit-learn fingerprinting localizers: SVM, random forest, extra trees, gradient boosting.

Each localizer regresses the position with a scikit-learn regressor and, when the training
data carry floor or building labels, classifies them with the matching scikit-learn
classifier (a single label needs no classifier). scikit-learn is imported inside the
functions that use it (extra ``[sklearn]``), so this module, the registry and ``clone`` work
without it and a missing install names the extra.

Determinism: forests are averaged tree by tree in index order, so predictions (and
``Prediction.spread``, the spread of the trees) do not depend on ``n_jobs``; every random
choice goes through ``random_state``, which is fixed (0) by default.

Persistence: fitted scikit-learn objects are converted to plain arrays and scalars by
``sklearn_to_state`` (trees as their node tables), so ``save``/``load_model`` never use
pickle. Loading rebuilds only scikit-learn classes (allow-listed by module) and warns, as
scikit-learn does, when the installed version differs from the one that saved the model.
"""
from __future__ import annotations

import importlib

import numpy as np

from ..core import Estimator, Prediction, requires
from .base import BaseLocalizer

# --------------------------------------------------------------------------- persistence
_ESTIMATOR, _TREE, _OBJECT, _LOSS, _RNG, _OBJECTS = (
    "__sklearn__", "__sklearn_tree__", "__sklearn_object__", "__sklearn_loss__", "__numpy_rng__", "__object_array__")
# Plain (non-estimator) scikit-learn objects whose whole state is their __dict__.
_PLAIN_OBJECTS = frozenset({"sklearn.ensemble._hist_gradient_boosting.predictor:TreePredictor"})
_BIT_GENERATORS = frozenset({"MT19937", "PCG64", "PCG64DXSM", "Philox", "SFC64"})


def _path(obj) -> str:
    cls = type(obj)
    return f"{cls.__module__}:{cls.__qualname__}"


def _sklearn_class(path: str):
    module, _, qualname = path.partition(":")
    if module != "sklearn" and not module.startswith("sklearn."):
        raise ValueError(f"refusing to rebuild {path!r}: only scikit-learn classes are restored")
    obj = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def sklearn_to_state(obj):
    """Fitted scikit-learn objects -> nested dicts/lists of arrays and scalars (no pickle).

    Estimators are stored through their ``__getstate__``; decision trees as their node and
    value tables; indoorloc Estimators, arrays and scalars pass through unchanged (the core
    persistence handles them). The loss object of a gradient-boosting model is not stored
    but rebuilt from the model's parameters on load. Unknown object types raise TypeError.
    """
    if obj is None or isinstance(obj, (bool, int, float, str, np.generic, Estimator)):
        return obj
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind != "O":
            return obj
        return {_OBJECTS: [sklearn_to_state(v) for v in obj.ravel()], "shape": list(obj.shape)}
    if isinstance(obj, (list, tuple)):
        return type(obj)(sklearn_to_state(v) for v in obj)
    if isinstance(obj, dict):
        return {k: sklearn_to_state(v) for k, v in obj.items()}
    if isinstance(obj, np.random.Generator):
        return {_RNG: "Generator", "state": obj.bit_generator.state}
    if isinstance(obj, np.random.RandomState):
        return {_RNG: "RandomState", "state": obj.get_state(legacy=False)}
    if type(obj).__module__.startswith("sklearn."):
        base = requires("sklearn.base", "sklearn")
        if isinstance(obj, base.BaseEstimator):
            state = dict(obj.__getstate__())
            if type(state.get("_loss")).__module__.startswith("sklearn._loss") and hasattr(obj, "_get_loss"):
                state["_loss"] = {_LOSS: True}  # rebuilt by _get_loss on load (prediction needs only the link)
            return {_ESTIMATOR: _path(obj), "state": sklearn_to_state(state)}
        tree = requires("sklearn.tree._tree", "sklearn")
        if type(obj) is tree.Tree:
            _, (n_features, n_classes, n_outputs), state = obj.__reduce__()
            return {_TREE: [int(n_features), np.asarray(n_classes), int(n_outputs)], "state": sklearn_to_state(state)}
        if _path(obj) in _PLAIN_OBJECTS:
            return {_OBJECT: _path(obj), "state": sklearn_to_state(dict(vars(obj)))}
    raise TypeError(f"cannot store a {_path(obj)} without pickle")


def sklearn_from_state(node):
    """Inverse of ``sklearn_to_state``: rebuilds only allow-listed scikit-learn classes."""
    if isinstance(node, (list, tuple)):
        return type(node)(sklearn_from_state(v) for v in node)
    if not isinstance(node, dict):
        return node
    if _OBJECTS in node:
        items = [sklearn_from_state(v) for v in node[_OBJECTS]]
        out = np.empty(len(items), dtype=object)
        for i, item in enumerate(items):  # element-wise: an array item must not broadcast
            out[i] = item
        return out.reshape(node["shape"])
    if _RNG in node:
        state = sklearn_from_state(node["state"])
        if node[_RNG] == "RandomState":
            rng = np.random.RandomState()
            rng.set_state(state)
            return rng
        name = state["bit_generator"]
        if name not in _BIT_GENERATORS:
            raise ValueError(f"unknown bit generator {name!r}")
        bit_generator = getattr(np.random, name)()
        bit_generator.state = state
        return np.random.Generator(bit_generator)
    if _ESTIMATOR in node:
        base = requires("sklearn.base", "sklearn")
        cls = _sklearn_class(node[_ESTIMATOR])
        if not (isinstance(cls, type) and issubclass(cls, base.BaseEstimator)):
            raise TypeError(f"{node[_ESTIMATOR]} is not a scikit-learn estimator")
        state = sklearn_from_state(node["state"])
        rebuild_loss = isinstance(state.get("_loss"), dict) and _LOSS in state["_loss"]
        if rebuild_loss:
            del state["_loss"]
        obj = cls.__new__(cls)
        obj.__setstate__(state)  # scikit-learn warns here if the saving version differs
        if rebuild_loss:
            obj._loss = obj._get_loss(sample_weight=None)
        return obj
    if _TREE in node:
        tree = requires("sklearn.tree._tree", "sklearn")
        n_features, n_classes, n_outputs = node[_TREE]
        obj = tree.Tree(int(n_features), np.ascontiguousarray(n_classes, dtype=np.intp), int(n_outputs))
        obj.__setstate__(sklearn_from_state(node["state"]))
        return obj
    if _OBJECT in node:
        if node[_OBJECT] not in _PLAIN_OBJECTS:
            raise ValueError(f"refusing to rebuild {node[_OBJECT]!r}")
        cls = _sklearn_class(node[_OBJECT])
        obj = cls.__new__(cls)
        obj.__dict__.update(sklearn_from_state(node["state"]))
        return obj
    return {k: sklearn_from_state(v) for k, v in node.items()}


# --------------------------------------------------------------------------- localizers
def _forest_mean(forest, X32: np.ndarray, proba: bool = False) -> np.ndarray:
    """Tree outputs averaged in index order (what scikit-learn computes with n_jobs=1)."""
    total = None
    for tree in forest.estimators_:
        out = tree.predict_proba(X32, check_input=False) if proba else tree.predict(X32, check_input=False)
        total = out.copy() if total is None else total + out
    return total / len(forest.estimators_)


class _SklearnLocalizer(BaseLocalizer):
    """Position regressor + floor/building classifiers built from scikit-learn estimators.

    Subclasses define ``_regressor()`` and ``_classifier()`` (unfitted estimators built from
    the constructor parameters). ``_joint = True``: one regressor predicts every coordinate
    axis (forests); otherwise one regressor per axis. ``_standardize``: z-score the target
    axes before fitting (so SVR's ``C``/``epsilon`` are unit free). N-D inputs are flattened.
    """

    _joint = False
    _standardize = False
    _forest = False

    def _regressor(self):
        raise NotImplementedError

    def _classifier(self):
        raise NotImplementedError

    def _fit(self, X, pos, floor, building):
        X = X.reshape(len(X), -1)
        center, scale = np.zeros(pos.shape[1]), np.ones(pos.shape[1])
        if self._standardize:
            center, spread = pos.mean(axis=0), pos.std(axis=0)
            scale = np.where(spread > 0, spread, 1.0)
        target = (pos - center) / scale if self._standardize else pos
        if self._joint:
            regressors = [self._regressor().fit(X, target if target.shape[1] > 1 else target[:, 0])]
        else:
            regressors = [self._regressor().fit(X, target[:, d]) for d in range(target.shape[1])]
        self.regressors_ = regressors
        self.pos_center_, self.pos_scale_ = center, scale
        self.floor_model_, self.floor_labels_ = self._fit_labels(X, floor)
        self.building_model_, self.building_labels_ = self._fit_labels(X, building)

    def _fit_labels(self, X, labels):
        if labels is None:
            return None, None
        classes = np.unique(labels)
        if len(classes) == 1:  # one floor / building: nothing to classify
            return None, classes
        return self._classifier().fit(X, labels), classes

    def _predict_labels(self, model, classes, X, X32):
        if classes is None:
            return None
        if model is None:
            return np.full(len(X), classes[0], dtype=np.int64)
        if self._forest:
            return model.classes_[np.argmax(_forest_mean(model, X32, proba=True), axis=1)].astype(np.int64)
        return np.asarray(model.predict(X), dtype=np.int64)

    def _localize(self, X):
        X = X.reshape(len(X), -1)
        X32 = np.ascontiguousarray(X, dtype=np.float32) if self._forest else None
        spread = None
        if self._forest:
            forest = self.regressors_[0]
            per_tree = [tree.predict(X32, check_input=False).reshape(len(X), -1) for tree in forest.estimators_]
            pos = per_tree[0].copy()
            for out in per_tree[1:]:
                pos += out
            pos /= len(per_tree)
            var = np.zeros(len(X))
            for out in per_tree:
                var += np.square(out - pos).sum(axis=1)
            spread = np.sqrt(var / len(per_tree))
        else:
            pos = np.column_stack([np.asarray(r.predict(X), dtype=np.float64).reshape(len(X), -1)
                                   for r in self.regressors_])
            pos = pos * self.pos_scale_ + self.pos_center_ if self._standardize else pos
        return Prediction(pos,
                          self._predict_labels(self.floor_model_, self.floor_labels_, X, X32),
                          self._predict_labels(self.building_model_, self.building_labels_, X, X32),
                          spread=spread)

    def _get_state(self) -> dict:
        return sklearn_to_state(super()._get_state())

    def _set_state(self, state: dict) -> None:
        super()._set_state(sklearn_from_state(state))


class SVMLocalizer(_SklearnLocalizer):
    """Support-vector fingerprinting: one epsilon-SVR per coordinate axis, SVC for labels.

    Each coordinate axis is z-scored and regressed by its own RBF (or other kernel) SVR, so
    ``C`` and ``epsilon`` are in units of that axis's standard deviation; floor and building
    use a one-vs-one SVC with the same kernel, ``C`` and ``gamma``. The SVM is deterministic
    (no ``random_state``). Training cost grows roughly quadratically with the number of
    scans (libsvm); ``cache_size`` (MB) trades memory for speed.

    Parameters: ``C`` penalty, ``epsilon`` half-width of the insensitive tube in standardized
    units (0.01 = 1 % of the axis's standard deviation: 1.2 m and 0.7 m on UJIIndoorLoc's two
    axes, about 0.3 m along a uniformly covered 100 m corridor), ``kernel``, ``gamma`` ("scale" =
    1 / (n_features * X.var())), ``cache_size``. The defaults were chosen on a split of the
    UJIIndoorLoc training file (25 % of reference positions held out), never on its
    validation file.

    References
    ----------
    Smola, A. J., Schölkopf, B., "A tutorial on support vector regression", Statistics and
    Computing 14(3), 2004. DOI 10.1023/B:STCO.0000035301.49549.88
    Brunato, M., Battiti, R., "Statistical learning theory for location fingerprinting in
    wireless LANs", Computer Networks 47(6), 2005. DOI 10.1016/j.comnet.2004.09.004
    Chang, C.-C., Lin, C.-J., "LIBSVM: A library for support vector machines", ACM TIST 2(3),
    2011. DOI 10.1145/1961189.1961199
    """

    _standardize = True

    def __init__(self, C: float = 1.0, epsilon: float = 0.01, kernel: str = "rbf", gamma="scale",
                 cache_size: float = 500.0):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.gamma = gamma
        self.cache_size = cache_size

    def _regressor(self):
        svm = requires("sklearn.svm", "sklearn")
        return svm.SVR(C=self.C, epsilon=self.epsilon, kernel=self.kernel, gamma=self.gamma,
                       cache_size=self.cache_size)

    def _classifier(self):
        svm = requires("sklearn.svm", "sklearn")
        return svm.SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, cache_size=self.cache_size)


class RandomForestLocalizer(_SklearnLocalizer):
    """Random-forest fingerprinting: one multi-output forest for the position, forests for labels.

    A single ``RandomForestRegressor`` predicts every coordinate axis jointly (each leaf holds a
    mean position); floor and building use ``RandomForestClassifier`` with the same
    parameters. Trees are averaged in index order, so the result does not depend on
    ``n_jobs``. ``Prediction.spread`` is the RMS distance of the individual trees'
    positions from the forest's estimate.

    Parameters (as in scikit-learn): ``n_estimators``, ``max_depth``, ``min_samples_leaf``,
    ``max_features`` ("sqrt" by default for both tasks, which suits hundreds of sparse AP
    features), ``bootstrap``, ``n_jobs`` and ``random_state`` (0: reproducible by default).

    References
    ----------
    Breiman, L., "Random forests", Machine Learning 45(1), 2001. DOI 10.1023/A:1010933404324
    Jedari, E., Wu, Z., Rashidzadeh, R., Saif, M., "Wi-Fi based indoor location positioning
    employing random forest classifier", IPIN 2015. DOI 10.1109/IPIN.2015.7346754
    """

    _joint = True
    _forest = True
    _names = ("RandomForestRegressor", "RandomForestClassifier")

    def __init__(self, n_estimators: int = 100, max_depth=None, min_samples_leaf: int = 1, max_features="sqrt",
                 bootstrap: bool = True, n_jobs=None, random_state=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _make(self, name):
        ensemble = requires("sklearn.ensemble", "sklearn")
        return getattr(ensemble, name)(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                       min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                                       bootstrap=self.bootstrap, n_jobs=self.n_jobs,
                                       random_state=self.random_state)

    def _regressor(self):
        return self._make(self._names[0])

    def _classifier(self):
        return self._make(self._names[1])


class ExtraTreesLocalizer(RandomForestLocalizer):
    """Extremely randomized trees: RandomForestLocalizer with random split thresholds.

    Same parameters and behaviour as ``RandomForestLocalizer``, but each split draws its
    threshold at random and, by default, every tree sees the whole training set
    (``bootstrap=False``), which lowers variance on dense fingerprint maps.

    References
    ----------
    Geurts, P., Ernst, D., Wehenkel, L., "Extremely randomized trees", Machine Learning 63(1),
    2006. DOI 10.1007/s10994-006-6226-1
    """

    _names = ("ExtraTreesRegressor", "ExtraTreesClassifier")

    def __init__(self, n_estimators: int = 100, max_depth=None, min_samples_leaf: int = 1, max_features="sqrt",
                 bootstrap: bool = False, n_jobs=None, random_state=0):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                         max_features=max_features, bootstrap=bootstrap, n_jobs=n_jobs, random_state=random_state)


class GradientBoostingLocalizer(_SklearnLocalizer):
    """Histogram gradient-boosted trees: one booster per coordinate axis, boosted classifiers for labels.

    Uses scikit-learn's ``HistGradientBoostingRegressor``/``Classifier`` (LightGBM-style
    binned trees), which is orders of magnitude faster than exact boosting on tens of
    thousands of scans and handles missing readings natively: NaN input is accepted (each
    split learns which side "not heard" goes to), so no ``FillMissing`` is needed.
    ``early_stopping`` is off by default so that the model uses all training data and
    does not depend on a random validation split.

    Parameters (as in scikit-learn): ``max_iter`` (boosting rounds), ``learning_rate``,
    ``max_leaf_nodes``, ``max_depth``, ``min_samples_leaf``, ``l2_regularization``,
    ``max_bins``, ``early_stopping`` and ``random_state``. scikit-learn sums gradients with
    OpenMP threads, so the last bits of the result can depend on the thread count.

    References
    ----------
    Friedman, J. H., "Greedy function approximation: A gradient boosting machine", Annals of
    Statistics 29(5), 2001. DOI 10.1214/aos/1013203451
    Ke, G., Meng, Q., Finley, T., et al., "LightGBM: A highly efficient gradient boosting
    decision tree", NeurIPS 2017.
    URL https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf
    """

    _allow_nan = True

    def __init__(self, max_iter: int = 100, learning_rate: float = 0.1, max_leaf_nodes: int = 31, max_depth=None,
                 min_samples_leaf: int = 20, l2_regularization: float = 0.0, max_bins: int = 255,
                 early_stopping: bool = False, random_state=0):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_bins = max_bins
        self.early_stopping = early_stopping
        self.random_state = random_state

    def _make(self, name):
        ensemble = requires("sklearn.ensemble", "sklearn")
        return getattr(ensemble, name)(max_iter=self.max_iter, learning_rate=self.learning_rate,
                                       max_leaf_nodes=self.max_leaf_nodes, max_depth=self.max_depth,
                                       min_samples_leaf=self.min_samples_leaf,
                                       l2_regularization=self.l2_regularization, max_bins=self.max_bins,
                                       early_stopping=self.early_stopping, random_state=self.random_state)

    def _fit(self, X, pos, floor, building):
        super()._fit(X, pos, floor, building)
        self.n_iter_ = max(r.n_iter_ for r in self.regressors_)  # boosting rounds run (sklearn convention)

    def _regressor(self):
        return self._make("HistGradientBoostingRegressor")

    def _classifier(self):
        return self._make("HistGradientBoostingClassifier")


__all__ = ["ExtraTreesLocalizer", "GradientBoostingLocalizer", "RandomForestLocalizer", "SVMLocalizer",
           "sklearn_from_state", "sklearn_to_state"]
