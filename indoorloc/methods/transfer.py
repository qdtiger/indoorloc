"""Unsupervised domain adaptation for fingerprints: CORAL, TCA and the MMD statistic (numpy only).

A radio map surveyed with one phone (the labelled *source* domain) is used to localize scans from
other phones or a later month (the unlabelled *target* domain). The adapters here learn from both
domains at fit time and are L2-style transforms (``signals.transforms.Transform``) with the extra
data declared in ``fit(X, y=None, *, target=None)``, so they slot into a ``LocalizerPipeline``::

    model = LocalizerPipeline(CORAL(), KNNLocalizer(k=5))
    model.fit(X_source, pos_source, preprocess__target=X_target)   # X_target: unlabelled scans
    model.predict(X_target_new)                                    # deployment = target domain

Shared rules:

* ``X`` and ``target`` are ``(N, F)`` arrays (or SampleTables) without NaN: fill missing readings
  first, on both domains, e.g. ``FillMissing(-104)``. A NaN is an error, never silently ignored.
* ``transform`` is applied to deployment data, i.e. it treats its input as target-domain data;
  ``fit_transform(X_source, target=X_target)`` (the call ``LocalizerPipeline.fit`` makes) returns
  the adapted *source* samples. This is the convention of skada's adapters. An ``eval_set`` passed
  through a pipeline is therefore mapped as target-domain data.
* Deterministic: the only randomness is TCA's sub-sampling, driven by ``random_state``.

``SkadaAdapter`` exposes any feature-level adapter of the skada library the same way (optional
extra ``[transfer]``; skada is imported lazily).
"""
from __future__ import annotations

import numpy as np

from ..core import NotFittedError, SampleTable, requires
from ..signals.transforms import Transform

__all__ = ["CORAL", "TCA", "SkadaAdapter", "mmd"]


# --------------------------------------------------------------------------------------------- helpers
def _matrix(X, name: str, n_features: int | None = None) -> np.ndarray:
    """``(N, F)`` float64 copy of an array or a SampleTable's X; NaN/inf and bad shapes are errors."""
    if X is None:
        raise ValueError(f"{name} is required: pass the unlabelled target-domain scans, e.g. "
                         "fit(X_source, target=X_target) or LocalizerPipeline.fit(..., preprocess__target=X_target)")
    if hasattr(X, "toarray"):
        raise TypeError("sparse input is not supported; pass a dense array (X.toarray())")
    arr = np.asarray(X.X if isinstance(X, SampleTable) else X)
    if arr.dtype.kind == "c":
        raise ValueError(f"{name} is complex; convert CSI to real features (amplitude, phase) first")
    arr = arr.astype(np.float64)
    if arr.ndim != 2 or 0 in arr.shape:
        raise ValueError(f"{name} must be a non-empty (N, F) array, got shape {arr.shape}")
    if n_features is not None and arr.shape[1] != n_features:
        raise ValueError(f"{name} has {arr.shape[1]} features, expected {n_features} (the source's)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf (missing readings?); fill them first on both domains, "
                         "e.g. FillMissing(-104)")
    return arr


def _like(X, values: np.ndarray):
    """Return ``values`` in the container ``X`` came in (a SampleTable keeps its other columns)."""
    return X.replace(X=values) if isinstance(X, SampleTable) else values


def _sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    d2 = np.einsum("ij,ij->i", A, A)[:, None] + np.einsum("ij,ij->i", B, B)[None, :] - 2.0 * (A @ B.T)
    return np.maximum(d2, 0.0, out=d2)


def _median_gamma(*samples: np.ndarray, max_rows: int = 500) -> float:
    """Median heuristic (Gretton et al., 2012): ``gamma = 1 / (2 median ||x - y||^2)`` over the pooled
    rows (at most ``max_rows`` evenly spaced rows per sample, so no randomness)."""
    rows = [s[np.linspace(0, len(s) - 1, min(len(s), max_rows)).astype(int)] for s in samples]
    pooled = np.concatenate(rows)
    d2 = _sq_dists(pooled, pooled)[np.triu_indices(len(pooled), k=1)]
    med = float(np.median(d2)) if d2.size else 0.0
    return 1.0 / (2.0 * med) if med > 0 else 1.0


def _kernel(A: np.ndarray, B: np.ndarray, kind: str, gamma: float | None) -> np.ndarray:
    if kind == "linear":
        return A @ B.T
    if kind == "rbf":
        return np.exp(-gamma * _sq_dists(A, B))
    raise ValueError(f"kernel must be 'linear' or 'rbf', got {kind!r}")


def _sym_power(C: np.ndarray, power: float, name: str) -> np.ndarray:
    """``C ** power`` for a symmetric positive (semi)definite matrix, via its eigendecomposition."""
    w, V = np.linalg.eigh((C + C.T) / 2)
    tol = w.max(initial=0.0) * len(C) * np.finfo(np.float64).eps
    if power < 0 and w.min() <= tol:
        raise ValueError(f"the {name} covariance is singular (constant or collinear features); use reg > 0")
    return (V * np.maximum(w, 0.0) ** power) @ V.T


# ------------------------------------------------------------------------------------------------ MMD
def mmd(X, Y, *, kernel: str = "rbf", gamma: float | None = None, unbiased: bool = False,
        chunk_size: int = 1024) -> float:
    """Squared maximum mean discrepancy between two samples, ``MMD^2(X, Y)``.

    ``MMD^2 = E k(x, x') + E k(y, y') - 2 E k(x, y)`` (Gretton et al., 2012): zero iff the two
    distributions coincide (for a characteristic kernel such as the RBF), and the quantity TCA minimises.

    Parameters
    ----------
    X, Y : (n, F) and (m, F) arrays (or SampleTables); no NaN.
    kernel : ``"rbf"`` (``exp(-gamma ||x - y||^2)``) or ``"linear"`` (``x . y``; then the biased
        estimate is exactly ``||mean(X) - mean(Y)||^2``).
    gamma : RBF width (> 0); None = median heuristic ``1 / (2 median ||x - y||^2)`` over the pooled
        rows (at most 500 evenly spaced rows of each sample, so it is deterministic).
    unbiased : False = biased V-statistic (always >= 0); True = unbiased U-statistic (excludes the
        ``i = j`` terms; may be slightly negative).
    chunk_size : rows per kernel block: memory is O(chunk_size * (n + m)), never O((n + m)^2).

    Returns
    -------
    float, the estimate of MMD squared.

    References
    ----------
    A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schoelkopf and A. Smola, "A Kernel Two-Sample
    Test", Journal of Machine Learning Research 13:723-773, 2012. https://jmlr.org/papers/v13/gretton12a.html
    """
    X = _matrix(X, "X")
    Y = _matrix(Y, "Y", X.shape[1])
    n, m = len(X), len(Y)
    if unbiased and min(n, m) < 2:
        raise ValueError("the unbiased estimate needs at least 2 samples in each set")
    if not (isinstance(chunk_size, (int, np.integer)) and chunk_size >= 1):
        raise ValueError(f"chunk_size must be an int >= 1, got {chunk_size!r}")
    if gamma is not None and not gamma > 0:
        raise ValueError(f"gamma must be > 0 (or None for the median heuristic), got {gamma!r}")
    if kernel == "linear":  # closed form, O((n + m) F)
        sx, sy = X.sum(0), Y.sum(0)
        if not unbiased:
            diff = sx / n - sy / m
            return float(diff @ diff)
        kxx = (sx @ sx - np.einsum("ij,ij->", X, X)) / (n * (n - 1))
        kyy = (sy @ sy - np.einsum("ij,ij->", Y, Y)) / (m * (m - 1))
        return float(kxx + kyy - 2.0 * (sx @ sy) / (n * m))
    if kernel != "rbf":
        raise ValueError(f"kernel must be 'linear' or 'rbf', got {kernel!r}")
    g = _median_gamma(X, Y) if gamma is None else float(gamma)

    def block_sum(A, B):
        return sum(float(_kernel(A[s:s + chunk_size], B, "rbf", g).sum()) for s in range(0, len(A), chunk_size))

    sxx, syy, sxy = block_sum(X, X), block_sum(Y, Y), block_sum(X, Y)
    if unbiased:  # k(x, x) = 1 on the diagonal of an RBF kernel
        return (sxx - n) / (n * (n - 1)) + (syy - m) / (m * (m - 1)) - 2.0 * sxy / (n * m)
    return sxx / n ** 2 + syy / m ** 2 - 2.0 * sxy / (n * m)


# ---------------------------------------------------------------------------------------------- CORAL
class CORAL(Transform):
    """CORrelation ALignment: re-colour the source features with the target covariance.

    With ``Cs = cov(X_source) + reg I`` and ``Ct = cov(X_target) + reg I`` the source samples are
    whitened and re-coloured, ``x -> (x - mu_s) Cs^(-1/2) Ct^(1/2) + mu``, so that their covariance
    becomes ``Ct`` (exactly when ``reg = 0``). ``mu`` is the source mean (``align_mean=False``,
    second-order alignment only, as in the paper) or the target mean (``align_mean=True``, which also
    removes a constant per-AP offset between devices). ``transform`` receives target-domain data and
    returns it unchanged (as float64); the source is mapped by ``fit_transform`` and
    ``transform_source``. A model trained on the re-coloured source is then applied to raw target data.

    Parameters
    ----------
    reg : float, default 1.0
        Ridge added to both covariances (the paper's ``+ eye``). It is in squared feature units: 1 is
        small for dBm features (variances of tens to hundreds) and dominant for features in [0, 1].
    align_mean : bool, default False
        Also move the source mean onto the target mean.

    Attributes
    ----------
    coef_ : (F, F) ``Cs^(-1/2) Ct^(1/2)``.  source_mean_, target_mean_ : (F,).

    Deviation from the paper: Algorithm 1 multiplies the raw features, ``X_s Cs^(-1/2) Ct^(1/2)``;
    here the map is applied to centred features (as skada's ``CORALAdapter`` also does). The two
    coincide for zero-mean features. For uncentred features such as dBm the raw product would move
    every feature's mean (the -104 dBm fill value of an AP heard by one phone only would land far
    from -104), which the centred form avoids.

    References
    ----------
    B. Sun, J. Feng and K. Saenko, "Return of Frustratingly Easy Domain Adaptation", Proceedings of
    the AAAI Conference on Artificial Intelligence 30(1), 2016. https://doi.org/10.1609/aaai.v30i1.10306
    """

    _requires_fit = True

    def __init__(self, reg: float = 1.0, align_mean: bool = False):
        self.reg = reg
        self.align_mean = align_mean

    def fit(self, X, y=None, *, target=None):
        """Learn the map from source ``X`` (N, F) and unlabelled ``target`` (M, F) scans."""
        Xs = _matrix(X, "X")
        Xt = _matrix(target, "target", Xs.shape[1])
        if self.reg < 0:
            raise ValueError(f"reg must be >= 0, got {self.reg}")
        if min(len(Xs), len(Xt)) < 2:
            raise ValueError("CORAL needs at least 2 samples in each domain to estimate a covariance")
        if len(Xt) <= Xs.shape[1]:
            import warnings

            warnings.warn(f"CORAL: {len(Xt)} target samples for {Xs.shape[1]} features: the target covariance is "
                          "rank-deficient and poorly estimated, so the re-colouring can hurt; raise reg, reduce the "
                          "features (e.g. APSelect) or collect more target scans", UserWarning, stacklevel=2)
        eye = self.reg * np.eye(Xs.shape[1])
        cs = np.atleast_2d(np.cov(Xs, rowvar=False)) + eye
        ct = np.atleast_2d(np.cov(Xt, rowvar=False)) + eye
        coef = _sym_power(cs, -0.5, "source") @ _sym_power(ct, 0.5, "target")
        super().fit(Xs)  # n_features_in_; only now is the transform fitted
        self.coef_, self.source_mean_, self.target_mean_ = coef, Xs.mean(0), Xt.mean(0)
        return self

    def fit_transform(self, X, y=None, *, target=None):
        """Fit, then return the adapted source samples (what the localizer is trained on)."""
        return self.fit(X, y, target=target).transform_source(X)

    def transform_source(self, X):
        """Map more source-domain samples (e.g. a source validation split) into the target colouring."""
        self._check_fitted("coef_")
        Xs = _matrix(X, "X", self.n_features_in_)
        shift = self.target_mean_ if self.align_mean else self.source_mean_
        return _like(X, (Xs - self.source_mean_) @ self.coef_ + shift)

    def _transform(self, x):  # target-domain data: CORAL leaves it as it is
        self._check_fitted("coef_")
        return np.array(x, dtype=np.float64)


# ------------------------------------------------------------------------------------------------ TCA
class TCA(Transform):
    """Transfer Component Analysis: a shared low-dimensional embedding in which the domains match.

    TCA finds ``n_components`` directions ``W`` in a kernel feature space that keep the variance of
    the pooled data while making the embedded domain means coincide: ``W`` holds the leading
    eigenvectors of ``(K L K + mu I)^(-1) K H K`` (Pan et al., 2011), where ``K`` is the kernel
    matrix of the pooled source and target samples, ``H`` the centring matrix and
    ``L = e e^T`` (``e_i = 1/n_s`` for source rows, ``-1/n_t`` for target rows) turns
    ``tr(W^T K L K W)`` into the squared MMD of the embedding. The map is the same for both domains,
    ``z = k(x, X_fit) W``, so ``transform`` applies to source and target scans alike.

    Parameters
    ----------
    n_components : int, default 30
        Embedding dimension (``m`` in the paper).
    kernel : {"linear", "rbf"}, default "linear"
    gamma : float or None
        RBF width; None = median heuristic on the fitted samples (stored as ``gamma_``).
    mu : float, default 1.0
        Trade-off between matching the domains (small ``mu``) and keeping variance (large ``mu``;
        ``mu -> inf`` is kernel PCA). It is compared with squared kernel values, so its effect depends
        on the kernel scale: with an RBF kernel (entries <= 1) the paper's range applies; with a linear
        kernel on raw dBm, ``mu = 1`` is already the ``mu -> 0`` limit (exactly equal domain means).
    max_samples : int or None, default 1000
        At most this many samples *per domain* are used to learn the embedding (TCA is O(n^3) in time
        and O(n^2) in memory); None uses all. Drawn without replacement with ``random_state``.
    random_state : int or None, default 0

    Attributes
    ----------
    components_ : (F, n_components) linear kernel only: ``z = (x - mean_) @ components_``.
    X_fit_, dual_coef_, K_fit_rows_, K_fit_all_ : RBF kernel only: the fitted samples, ``W`` and the
        kernel means used to centre new rows (the names of sklearn's ``KernelCenterer``).
    eigenvalues_ : (n_components,) values of the TCA objective, in decreasing order.
    gamma_, n_source_fit_, n_target_fit_.

    Deviations from the paper, both so that the ``mu -> inf`` limit is exactly (kernel) PCA: the
    kernel is centred in feature space as in kernel PCA (for the linear kernel: the features are
    centred on the pooled mean), where Pan et al. use the uncentred ``K``; and each component is
    scaled to a unit-length direction in feature space (``w^T K w = 1``; for the linear kernel the
    columns of ``components_`` are unit vectors), where the paper's constraint ``W^T K H K W = I``
    would whiten the embedding. The generalized symmetric eigenproblem is solved exactly through
    ``B^(-1/2)``, using ``B = K L K + mu I = mu I + v v^T`` (ridge plus rank one, ``v = K e``); each
    component's sign is fixed so that its largest-magnitude dual coefficient is positive.

    References
    ----------
    S. J. Pan, I. W. Tsang, J. T. Kwok and Q. Yang, "Domain Adaptation via Transfer Component
    Analysis", IEEE Transactions on Neural Networks 22(2):199-210, 2011.
    https://doi.org/10.1109/TNN.2010.2091281 (its experiments include cross-domain WiFi localization)
    """

    _requires_fit = True
    _block = 2048  # rows per kernel block in transform

    def __init__(self, n_components: int = 30, kernel: str = "linear", gamma: float | None = None,
                 mu: float = 1.0, max_samples: int | None = 1000, random_state: int | None = 0):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.mu = mu
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X, y=None, *, target=None):
        """Learn the embedding from source ``X`` (N, F) and unlabelled ``target`` (M, F) scans."""
        Xs = _matrix(X, "X")
        Xt = _matrix(target, "target", Xs.shape[1])
        if self.kernel not in ("linear", "rbf"):
            raise ValueError(f"kernel must be 'linear' or 'rbf', got {self.kernel!r}")
        if not self.mu > 0:
            raise ValueError(f"mu must be > 0, got {self.mu}")
        if self.gamma is not None and not self.gamma > 0:
            raise ValueError(f"gamma must be > 0 (or None for the median heuristic), got {self.gamma!r}")
        rng = np.random.default_rng(self.random_state)
        if self.max_samples is not None:
            if self.max_samples < 1:
                raise ValueError(f"max_samples must be >= 1 or None, got {self.max_samples}")
            pick = lambda A: A if len(A) <= self.max_samples else A[np.sort(  # noqa: E731
                rng.choice(len(A), self.max_samples, replace=False))]
            Xs, Xt = pick(Xs), pick(Xt)
        ns, nt = len(Xs), len(Xt)
        Xf = np.concatenate([Xs, Xt])
        n = ns + nt
        if not 1 <= self.n_components <= n:
            raise ValueError(f"n_components={self.n_components} must be between 1 and the number of "
                             f"fitted samples ({n})")
        gamma = None
        if self.kernel == "linear":
            mean = Xf.mean(0)
            Kc = (Xf - mean) @ (Xf - mean).T  # centred linear kernel
        else:
            gamma = _median_gamma(Xs, Xt) if self.gamma is None else float(self.gamma)
            K = _kernel(Xf, Xf, "rbf", gamma)
            col = K.mean(0)
            Kc = K - col[None, :] - col[:, None] + col.mean()  # H K H
        Kc = (Kc + Kc.T) / 2
        e = np.concatenate([np.full(ns, 1.0 / ns), np.full(nt, -1.0 / nt)])
        A = Kc @ Kc  # K H K for the centred kernel (H Kc = Kc)
        v = Kc @ e
        # B^(-1/2) = mu^(-1/2) (I + c u u^T) for B = mu I + v v^T, u = v / |v|
        vv = float(v @ v)
        if vv > 0:
            u = v / np.sqrt(vv)
            c = np.sqrt(self.mu / (self.mu + vv)) - 1.0
            a = A @ u
            s = float(u @ a)
            M = (A + c * (np.outer(u, a) + np.outer(a, u)) + c * c * s * np.outer(u, u)) / self.mu
            project = lambda Y: (Y + c * np.outer(u, u @ Y)) / np.sqrt(self.mu)  # noqa: E731
        else:  # the domains already coincide in feature space: plain (kernel) PCA
            M = A / self.mu
            project = lambda Y: Y / np.sqrt(self.mu)  # noqa: E731
        w, Y = np.linalg.eigh((M + M.T) / 2)
        top = np.argsort(w, kind="stable")[::-1][:self.n_components]
        W = project(Y[:, top])
        norms = np.einsum("ij,ij->j", W, Kc @ W)  # squared feature-space length of each direction
        rayleigh = norms / np.einsum("ij,ij->j", W, W)  # 0 for a direction in the null space of K
        if np.any(rayleigh <= 1e-10 * max(np.trace(Kc), np.finfo(np.float64).tiny)):
            raise ValueError(f"n_components={self.n_components} exceeds the rank of the kernel matrix; "
                             "use fewer components")
        W /= np.sqrt(norms)
        rows = np.argmax(np.abs(W), axis=0)
        W *= np.where(W[rows, np.arange(W.shape[1])] < 0, -1.0, 1.0)
        super().fit(Xs)  # n_features_in_; only now is the transform fitted
        self.eigenvalues_ = w[top]
        self.gamma_, self.n_source_fit_, self.n_target_fit_ = gamma, ns, nt
        if self.kernel == "linear":
            self.mean_ = mean
            self.components_ = (Xf - mean).T @ W  # z = (x - mean) Xc^T W
        else:
            self.X_fit_, self.dual_coef_ = Xf, W
            self.K_fit_rows_, self.K_fit_all_ = col, float(col.mean())  # for centring new rows
        return self

    def transform(self, X):
        """Embed scans of either domain: ``(N, F) -> (N, n_components)`` (a table gets new feature names)."""
        out = super().transform(X)
        if isinstance(out, SampleTable):
            names = tuple(f"tca{j}" for j in range(out.X.shape[-1]))
            out = out.replace(meta={**out.meta, "feature_names": names})
        return out

    def _transform(self, x):
        self._check_fitted("eigenvalues_")
        x = np.asarray(x, dtype=np.float64)
        if not np.all(np.isfinite(x)):
            raise ValueError("X contains NaN or inf (missing readings?); fill them first, e.g. FillMissing(-104)")
        rows = x[None] if x.ndim == 1 else x
        if self.kernel == "linear":
            z = (rows - self.mean_) @ self.components_
        else:  # in blocks of rows: memory O(block * n_fit), not O(N * n_fit)
            z = np.empty((len(rows), self.dual_coef_.shape[1]))
            for s in range(0, len(rows), self._block):
                k = _kernel(rows[s:s + self._block], self.X_fit_, "rbf", self.gamma_)
                k = k - self.K_fit_rows_[None, :] - k.mean(1, keepdims=True) + self.K_fit_all_  # KernelCenterer
                z[s:s + self._block] = k @ self.dual_coef_
        return z[0] if x.ndim == 1 else z

    def fit_transform(self, X, y=None, *, target=None):
        return self.fit(X, y, target=target).transform(X)


# ---------------------------------------------------------------------------------------------- skada
class SkadaAdapter(Transform):
    """Any feature-level adapter of the skada library as an L2 transform (``fit(X, target=...)``).

    The source and target scans are stacked with skada's ``sample_domain`` convention (positive =
    source, negative = target); ``fit_transform`` returns the adapted source rows and ``transform``
    maps new scans as target-domain data. Reweighting adapters (which output sample weights rather
    than features) are refused. The fitted skada object is not an array, so this transform cannot be
    saved with ``save``/``load_model``.

    Status: skada is not a dependency of the test environment, so the default test run checks this
    class against a stand-in with skada's adapter API (``fit_transform(X, y, sample_domain=...)``,
    ``transform(X, sample_domain=...)``). It was also run against skada 0.6.0 installed in an isolated
    directory: ``CORALAdapter`` (equal to ``CORAL(reg=0, align_mean=True)`` up to skada's conventions),
    ``SubspaceAlignmentAdapter`` and ``TransferComponentAnalysisAdapter`` work, and
    ``KMMReweightAdapter`` is refused. Note that skada's CORAL centres the target too.

    Parameters
    ----------
    adapter : str or skada adapter instance, default "CORALAdapter"
        A class name in the ``skada`` namespace (e.g. ``"SubspaceAlignmentAdapter"``,
        ``"TransferComponentAnalysisAdapter"``) or an unfitted adapter (it is cloned).
    params : dict or None
        Constructor arguments when ``adapter`` is a name.

    References
    ----------
    skada, "Scikit-learn-compatible domain adaptation", https://github.com/scikit-adaptation/skada;
    Y. Lalou et al., "SKADA-Bench: Benchmarking Unsupervised Domain Adaptation Methods with Realistic
    Validation", arXiv:2407.11676, 2024.
    """

    _requires_fit = True

    def __init__(self, adapter="CORALAdapter", params: dict | None = None):
        self.adapter = adapter
        self.params = params

    def _make(self):
        if isinstance(self.adapter, str):
            skada = requires("skada", "transfer")
            if not hasattr(skada, self.adapter):
                raise ValueError(f"skada has no adapter named {self.adapter!r}")
            return getattr(skada, self.adapter)(**(self.params or {}))
        return requires("sklearn.base", "transfer").clone(self.adapter)

    def fit(self, X, y=None, *, target=None):
        self.fit_transform(X, y, target=target)
        return self

    def _get_state(self) -> dict:
        if hasattr(self, "adapter_"):
            raise TypeError("a fitted SkadaAdapter cannot be saved: the skada object is not an array and "
                            "save() never pickles; use CORAL or TCA (array state) or refit after loading")
        return super()._get_state()

    def fit_transform(self, X, y=None, *, target=None):
        Xs = _matrix(X, "X")
        Xt = _matrix(target, "target", Xs.shape[1])
        adapter = self._make()
        domain = np.concatenate([np.ones(len(Xs), dtype=np.int64), np.full(len(Xt), -2, dtype=np.int64)])
        out = adapter.fit_transform(np.concatenate([Xs, Xt]), None, sample_domain=domain)
        if not isinstance(out, np.ndarray):
            raise TypeError(f"{type(adapter).__name__} returned {type(out).__name__}, not adapted features "
                            "(a reweighting adapter?); only feature-level adapters work as a transform")
        super().fit(Xs)
        self.adapter_ = adapter
        return _like(X, np.asarray(out[:len(Xs)], dtype=np.float64))

    def transform(self, X):
        out = super().transform(X)
        if isinstance(out, SampleTable) and out.X.shape[-1] != self.n_features_in_:
            out = out.replace(meta={**out.meta, "feature_names": tuple(f"skada{j}" for j in range(out.X.shape[-1]))})
        return out

    def _transform(self, x):
        if not hasattr(self, "adapter_"):
            raise NotFittedError("SkadaAdapter is not fitted yet; call fit first")
        rows = np.asarray(x, dtype=np.float64)
        one = rows.ndim == 1
        rows = rows[None] if one else rows
        z = np.asarray(self.adapter_.transform(rows, sample_domain=np.full(len(rows), -2, dtype=np.int64)),
                       dtype=np.float64)
        return z[0] if one else z
