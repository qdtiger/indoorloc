"""Angle-of-arrival localization: MUSIC bearings from array snapshots, then triangulation (numpy only).

``X`` is either bearings or raw array snapshots, with the anchor geometry as parameters:

    X real (N, A)            angle of arrival at each anchor, radians, in the anchor frame
                             (CONTRACTS.md ``aoa`` modality); NaN = missing.
    X complex (N, A, M, K)   K snapshots of an M-element uniform linear array (ULA) at each
                             anchor (e.g. CSI with K = packets x subcarriers); (N, A, M) = K = 1.

Array convention (every function here): element ``m`` sits at ``m * spacing`` wavelengths
along the array axis, which points 90 degrees counter-clockwise from the boresight; an
angle ``theta`` in (-pi/2, pi/2) is measured counter-clockwise from the boresight, so the
steering vector is ``a_m(theta) = exp(+2j pi spacing m sin(theta))``. The global bearing of
anchor ``i`` is ``orientations[i] + theta``. A ULA cannot tell front from back: sources are
assumed to lie in front of each array. Reverse the element order if your hardware indexes
the elements the other way.

A single-AP CSI table ``(N, n_rx, n_tx, n_sub)`` becomes snapshots with
``csi.reshape(N, 1, n_rx, n_tx * n_sub)``. Subcarriers are treated as narrowband snapshots
at the carrier wavelength (the usual approximation for 20-40 MHz channels).
"""
from __future__ import annotations

import warnings

import numpy as np

from ..core import Prediction
from .base import _unpack
from .geometric import (GeometricLocalizer, _as_anchors, _calibration, _exact_fit, _far_field, _inv_trace,
                        _masked_median, _meta_anchors, _refine, _residual_sigma, _solve_sym)

#: ``fit`` with positions warns when an anchor's angles disagree with them by a median of more than this
#: (radians, 45 degrees): the angles are then almost certainly in another frame (orientations missing).
FRAME_CHECK = np.pi / 4


def ula_steering(angles, n_elements: int, spacing: float = 0.5) -> np.ndarray:
    """Steering vectors ``exp(2j pi spacing m sin(theta))``, shape ``angles.shape + (n_elements,)``."""
    theta = np.asarray(angles, dtype=np.float64)
    m = np.arange(int(n_elements))
    return np.exp(2j * np.pi * float(spacing) * np.sin(theta)[..., None] * m)


def spatial_covariance(snapshots, *, subarray=None, forward_backward: bool = False) -> np.ndarray:
    """Sample covariance ``Y Y^H / K`` of ``(..., M, K)`` snapshots, optionally smoothed.

    ``subarray=L`` averages the covariances of the ``M - L + 1`` overlapping forward
    subarrays of length ``L`` (spatial smoothing), which restores the rank of the signal
    covariance of up to ``M - L + 1`` coherent paths; ``forward_backward=True`` also averages
    with the conjugate-reversed array. Returns ``(..., L, L)`` (``L = M`` without smoothing).

    References: T.-J. Shan, M. Wax, T. Kailath, "On spatial smoothing for direction-of-arrival
    estimation of coherent signals", IEEE Transactions on Acoustics, Speech, and Signal
    Processing 33(4), 1985. DOI 10.1109/TASSP.1985.1164649. S. U. Pillai, B. H. Kwon,
    "Forward/backward spatial smoothing techniques for coherent signal identification", IEEE
    Transactions on Acoustics, Speech, and Signal Processing 37(1), 1989. DOI 10.1109/29.17496.
    """
    Y = np.asarray(snapshots)
    if Y.ndim < 2:
        raise ValueError(f"snapshots must be (..., M, K), got shape {Y.shape}")
    Y = Y.astype(np.complex128)
    M, K = Y.shape[-2:]
    R = np.einsum("...ik,...jk->...ij", Y, Y.conj()) / K
    L = M if subarray is None else int(subarray)
    if not 1 <= L <= M:
        raise ValueError(f"subarray must be between 1 and the number of elements ({M}), got {subarray}")
    if L < M:
        R = sum(R[..., s:s + L, s:s + L] for s in range(M - L + 1)) / (M - L + 1)
    if forward_backward:
        R = 0.5 * (R + np.conj(R[..., ::-1, ::-1]))
    return R


def music_spectrum(covariance, angles, n_sources: int = 1, spacing: float = 0.5) -> np.ndarray:
    """MUSIC pseudo-spectrum ``1 / |E_n^H a(theta)|^2`` of ``(..., L, L)`` covariances at ``angles``.

    ``E_n`` spans the ``L - n_sources`` eigenvectors with the smallest eigenvalues (noise
    subspace). Returns ``(..., len(angles))``.

    References: R. O. Schmidt, "Multiple emitter location and signal parameter estimation",
    IEEE Transactions on Antennas and Propagation 34(3), 1986. DOI 10.1109/TAP.1986.1143830.
    """
    R = np.asarray(covariance, dtype=np.complex128)
    L = R.shape[-1]
    if not 1 <= n_sources < L:
        raise ValueError(f"n_sources must be between 1 and {L - 1} for {L}x{L} covariances, got {n_sources}")
    _, V = np.linalg.eigh(R)
    En = V[..., : L - n_sources]
    S = ula_steering(angles, L, spacing)  # (G, L)
    proj = np.einsum("gl,...lk->...gk", S.conj(), En)
    return 1.0 / np.maximum(np.sum(np.abs(proj) ** 2, axis=-1), 1e-300)


def _pick_peaks(P: np.ndarray, grid: np.ndarray, n: int) -> np.ndarray:
    """The ``n`` highest local maxima of each row of ``P`` (B, G), refined by a parabola
    through the log spectrum; sorted by height, NaN when a row has fewer peaks."""
    y = 10.0 * np.log10(P)
    G = y.shape[1]
    left = np.concatenate([np.full((len(y), 1), -np.inf), y[:, :-1]], axis=1)
    right = np.concatenate([y[:, 1:], np.full((len(y), 1), -np.inf)], axis=1)
    peak = (y > left) & (y >= right)
    score = np.where(peak, y, -np.inf)
    order = np.argsort(-score, axis=1, kind="stable")[:, :n]
    top = np.take_along_axis(score, order, axis=1)
    g = order
    gl, gr = np.clip(g - 1, 0, G - 1), np.clip(g + 1, 0, G - 1)
    y0, yl, yr = (np.take_along_axis(y, k, axis=1) for k in (g, gl, gr))
    denom = yl - 2.0 * y0 + yr
    interior = (g > 0) & (g < G - 1) & (denom < 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = np.where(interior, np.clip(0.5 * (yl - yr) / np.where(interior, denom, 1.0), -0.5, 0.5), 0.0)
    step = grid[1] - grid[0] if G > 1 else 0.0
    out = grid[g] + delta * step
    return np.where(np.isfinite(top), out, np.nan)


def music(snapshots, n_sources: int = 1, *, spacing: float = 0.5, subarray=None, forward_backward: bool = False,
          n_grid: int = 1801, chunk_size: int = 256) -> np.ndarray:
    """Directions of arrival (radians) from ``(..., M, K)`` ULA snapshots with MUSIC.

    Scans ``n_grid`` angles uniformly over [-pi/2, pi/2] (1801 = 0.1 degree), takes the
    ``n_sources`` highest peaks of the pseudo-spectrum and refines each by parabolic
    interpolation. Returns ``(..., n_sources)`` sorted by peak height (strongest first);
    NaN when fewer peaks exist. See ``spatial_covariance`` for ``subarray`` and
    ``forward_backward`` (needed for coherent multipath).

    References: R. O. Schmidt, "Multiple emitter location and signal parameter estimation",
    IEEE Transactions on Antennas and Propagation 34(3), 1986. DOI 10.1109/TAP.1986.1143830.
    """
    R = spatial_covariance(snapshots, subarray=subarray, forward_backward=forward_backward)
    lead = R.shape[:-2]
    R = R.reshape((-1,) + R.shape[-2:])
    grid = np.linspace(-np.pi / 2, np.pi / 2, int(n_grid))
    out = np.empty((len(R), int(n_sources)))
    for s in range(0, len(R), chunk_size):
        P = music_spectrum(R[s:s + chunk_size], grid, n_sources, spacing)
        out[s:s + chunk_size] = _pick_peaks(P, grid, int(n_sources))
    return out.reshape(lead + (int(n_sources),))


def triangulate(bearings, anchors) -> np.ndarray:
    """Least-squares intersection of bearing lines (Stansfield's estimator, equal weights).

    Minimises the sum of squared perpendicular distances from ``x`` to the lines through
    each anchor with global bearing ``phi_i`` (counter-clockwise from +x): a linear problem
    ``sum n_i n_i^T x = sum n_i n_i^T a_i`` with the line normals ``n_i``. NaN bearings are
    dropped; rows with fewer than two non-parallel bearings return NaN.

    bearings (N, A) radians, anchors (A, 2) -> (N, 2).

    References: R. G. Stansfield, "Statistical theory of d.f. fixing", Journal of the IEE -
    Part IIIA 94(15), 1947. DOI 10.1049/ji-3a-2.1947.0096.
    """
    anchors = _as_anchors(anchors, "triangulate", dims=(2,))
    phi = np.asarray(bearings, dtype=np.float64)
    if phi.ndim != 2 or phi.shape[1] != len(anchors):
        raise ValueError(f"bearings must be (N, {len(anchors)}), got shape {phi.shape}")
    origin = anchors.mean(axis=0)
    a = anchors - origin
    valid = np.isfinite(phi)
    p = np.where(valid, phi, 0.0)
    n = np.stack([-np.sin(p), np.cos(p)], axis=2) * valid[..., None]  # (N, A, 2)
    H = np.einsum("nai,naj->nij", n, n)
    b = np.einsum("nai,na->ni", n, np.einsum("naj,aj->na", n, a))
    x = np.full((len(phi), 2), np.nan)
    ok = valid.sum(axis=1) >= 2
    if ok.any():
        x[ok] = _solve_sym(H[ok], b[ok], rcond=1e-10)
    return x + origin


def _wrap(angle):
    return np.angle(np.exp(1j * angle))


def _circular_calibration(residuals: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-anchor angle offset (N, A) -> (A,) and the pooled noise scale, on the circle.

    A plain median of wrapped residuals fails for an offset near +-pi (an array mounted
    facing the other way): the residuals straddle the cut and split between +pi and -pi.
    Each column is therefore first centred on its circular mean, where the residuals are
    contiguous, and the median bias and MAD scale are taken there.
    """
    valid = np.isfinite(residuals)
    centre = np.angle(np.sum(np.where(valid, np.exp(1j * np.where(valid, residuals, 0.0)), 0.0), axis=0))
    bias, sigma = _calibration(_wrap(residuals - centre))
    return _wrap(bias + centre), sigma


def _bearing_model(anchors: np.ndarray, phi: np.ndarray):
    def model(x, rows, jac):
        diff = x[:, None, :] - anchors[None]
        pr = phi[rows]
        valid = np.isfinite(pr)
        e = np.where(valid, _wrap(pr - np.arctan2(diff[..., 1], diff[..., 0])), 0.0)
        if not jac:
            return e, valid
        rho2 = np.maximum(np.sum(diff * diff, axis=2), 1e-300)
        J = np.stack([-diff[..., 1], diff[..., 0]], axis=2) / rho2[..., None]
        return e, np.where(valid[..., None], J, 0.0), valid

    return model


class AoALocalizer(GeometricLocalizer):
    """2-D position from angles of arrival at anchors with known positions and orientations.

    Bearings come either directly from ``X`` (radians, anchor frame) or from MUSIC on complex
    ULA snapshots ``X`` (N, A, M, K) (Schmidt 1986; optional spatial smoothing, Shan et al.
    1985, and forward-backward averaging, Pillai & Kwon 1989; the strongest of ``n_paths``
    peaks is taken as the direct path). Global bearings ``orientations + theta`` are
    intersected by least squares (``solver="linear"``, Stansfield 1947) or by maximum
    likelihood under Gaussian angle noise (``solver="gauss_newton"``, started from the linear
    solution and from the anchors' centroid, keeping the lower cost). Rows with fewer than two
    valid bearings return NaN, and so do Gauss-Newton estimates farther than
    ``geometric.FAR_FIELD`` anchor spreads from the anchors (a run into the far field, which the
    geometry cannot resolve) unless the data vouch for them: an exact fit with redundancy, or a
    known noise scale (``sigma`` or the calibrated ``sigma_``) whose CRLB there is smaller than
    the distance. So a precisely measured target outside a small cluster of arrays is kept.

    ``Prediction.spread`` = ``sigma * sqrt(trace((J^T J)^-1))`` with ``sigma`` the bearing std
    in radians (``sigma``, else the calibrated ``sigma_``, else estimated from the residuals when
    more than two bearings are available, else NaN). With ``calibrate=True``,
    ``fit(X, positions)`` learns a per-anchor angle offset ``bias_`` (orientation error; the
    circular median, so it also recovers a whole unknown orientation, e.g. an array facing
    -x with ``orientations=None``) and ``sigma_``. Otherwise ``localize`` works without ``fit``
    and ``fit(X)`` needs no positions.

    The angles of the ``aoa`` modality are in each anchor's frame: without ``orientations``
    they are read as global bearings, and a table's local angles then give errors of tens of
    metres. ``AoALocalizer.from_meta(table.meta)`` takes ``meta["anchor_orientations"]`` with
    the anchors. ``fit`` with positions (and ``calibrate=False``) warns when an anchor's angles
    disagree with the labelled positions by a median of more than 45 degrees (``FRAME_CHECK``),
    which noise alone does not produce; nothing is checked without labels.

    Parameters
    ----------
    anchors : (A, 2) array of array (anchor) positions.
    orientations : (A,) boresight direction of each array, radians counter-clockwise from +x;
        None = 0 (``X`` then holds global bearings). A table's ``meta["anchor_orientations"]``.
    solver : "linear" or "gauss_newton".
    spacing : ULA element spacing in wavelengths (snapshot input).
    n_paths : signal-subspace dimension for MUSIC (number of paths per array).
    subarray : spatial-smoothing subarray length (None = no smoothing).
    forward_backward : forward-backward averaging of the covariance.
    n_grid : MUSIC scan points over [-pi/2, pi/2].
    sigma : bearing noise std in radians, for ``spread``.
    calibrate : learn ``bias_`` and ``sigma_`` in ``fit``.
    max_iter, tol : Gauss-Newton iterations and relative step tolerance.

    References
    ----------
    R. O. Schmidt, "Multiple emitter location and signal parameter estimation", IEEE
    Transactions on Antennas and Propagation 34(3), 1986. DOI 10.1109/TAP.1986.1143830.
    T.-J. Shan, M. Wax, T. Kailath, "On spatial smoothing for direction-of-arrival estimation
    of coherent signals", IEEE Transactions on Acoustics, Speech, and Signal Processing 33(4),
    1985. DOI 10.1109/TASSP.1985.1164649.
    S. U. Pillai, B. H. Kwon, "Forward/backward spatial smoothing techniques for coherent
    signal identification", IEEE Transactions on Acoustics, Speech, and Signal Processing
    37(1), 1989. DOI 10.1109/29.17496.
    R. G. Stansfield, "Statistical theory of d.f. fixing", Journal of the IEE - Part IIIA
    94(15), 1947. DOI 10.1049/ji-3a-2.1947.0096.
    S. Gavish, A. J. Weiss, "Performance analysis of bearing-only target location
    algorithms", IEEE Transactions on Aerospace and Electronic Systems 28(3), 1992.
    DOI 10.1109/7.256302.
    """

    _allow_complex = True

    def __init__(self, anchors=None, orientations=None, solver: str = "gauss_newton", spacing: float = 0.5,
                 n_paths: int = 1, subarray=None, forward_backward: bool = False, n_grid: int = 1801,
                 sigma=None, calibrate: bool = False, max_iter: int = 50, tol: float = 1e-10):
        self.anchors = anchors
        self.orientations = orientations
        self.solver = solver
        self.spacing = spacing
        self.n_paths = n_paths
        self.subarray = subarray
        self.forward_backward = forward_backward
        self.n_grid = n_grid
        self.sigma = sigma
        self.calibrate = calibrate
        self.max_iter = max_iter
        self.tol = tol

    @classmethod
    def from_meta(cls, meta, **params) -> AoALocalizer:
        """A localizer configured from the ``meta`` of an ``aoa`` table: ``meta["anchors"]`` and
        ``meta["anchor_orientations"]`` (the angles of the modality are in each anchor's frame).
        Without orientations in ``meta``, pass ``orientations=`` explicitly (``None`` if ``X`` holds
        global bearings). ``params`` override or add constructor arguments, e.g.
        ``AoALocalizer.from_meta(meta, solver="linear")``."""
        kw = {"anchors": _meta_anchors(meta, "aoa", cls.__name__)}
        if meta.get("anchor_orientations") is not None:
            kw["orientations"] = np.asarray(meta["anchor_orientations"], dtype=np.float64)
        elif "orientations" not in params:
            raise ValueError("meta has no anchor_orientations: pass orientations= (the boresight bearing of each "
                             "array), or orientations=None if X holds global bearings")
        kw.update(params)
        return cls(**kw)

    def _geometry(self):
        anchors = _as_anchors(self.anchors, type(self).__name__, dims=(2,))
        o = np.zeros(len(anchors)) if self.orientations is None else np.asarray(self.orientations, np.float64)
        if o.shape != (len(anchors),) or not np.all(np.isfinite(o)):
            raise ValueError(f"orientations must be a finite ({len(anchors)},) array of radians, got {o.shape}")
        return anchors, o

    def _check_input(self, X):
        anchors, _ = self._geometry()
        if self.solver not in ("linear", "gauss_newton"):
            raise ValueError(f"solver must be 'linear' or 'gauss_newton', got {self.solver!r}")
        if X.dtype.kind == "c":
            if X.ndim not in (3, 4) or X.shape[1] != len(anchors):
                raise ValueError(f"complex X must be ULA snapshots (N, {len(anchors)}, M, K) or (N, "
                                 f"{len(anchors)}, M); got {X.shape}")
        elif X.ndim != 2 or X.shape[1] != len(anchors):
            raise ValueError(f"X must hold one angle per anchor, shape (N, {len(anchors)}); got {X.shape}")

    def local_angles(self, X) -> np.ndarray:
        """Angle of arrival (N, A) in each anchor's frame: ``X`` itself, or MUSIC on snapshots.
        ``X`` is an array or a SampleTable."""
        X = np.asarray(_unpack(X)[0])
        if X.dtype.kind != "c":
            return X.astype(np.float64)
        Y = X[..., None] if X.ndim == 3 else X
        missing = ~np.all(np.isfinite(Y), axis=(2, 3))
        Y = np.where(missing[..., None, None], 0.0, Y)
        theta = music(Y, int(self.n_paths), spacing=self.spacing, subarray=self.subarray,
                      forward_backward=self.forward_backward, n_grid=self.n_grid)[..., 0]
        return np.where(missing, np.nan, theta)

    def bearings(self, X) -> np.ndarray:
        """Global bearings (N, A), radians counter-clockwise from +x, after calibration."""
        X = self._validate(_unpack(X)[0])
        _, o = self._geometry()
        return _wrap(self.local_angles(X) + o - getattr(self, "bias_", 0.0))

    def _fit(self, X, pos, floor, building):
        anchors, o = self._geometry()
        self._check_labels(pos, anchors)
        self._clear_calibration()
        if pos is None:
            return
        true = np.arctan2(pos[:, None, 1] - anchors[None, :, 1], pos[:, None, 0] - anchors[None, :, 0])
        if self.calibrate:
            self.bias_, self.sigma_ = _circular_calibration(_wrap(self.local_angles(X) + o - true))
        elif X.dtype.kind != "c":  # snapshots would need MUSIC over the training set: not checked
            self._check_frame(_wrap(np.asarray(X, np.float64) + o - true))

    def _check_frame(self, residual) -> None:
        """Warn when some anchor's global bearings disagree with the labelled positions by a
        median of more than ``FRAME_CHECK``: local angles read without their orientations."""
        valid = np.isfinite(residual)
        median = _masked_median(np.abs(residual).T, valid.T)
        with np.errstate(invalid="ignore"):
            off = np.flatnonzero((valid.sum(axis=0) >= 3) & (median > FRAME_CHECK))
        if off.size:
            worst = np.rad2deg(median[off])
            warnings.warn(f"{type(self).__name__}: the angles of anchor(s) {off.tolist()} disagree with the labelled "
                          f"positions by a median of {worst.min():.0f}-{worst.max():.0f} degrees. X holds angles in "
                          "each anchor's frame (the 'aoa' modality): pass orientations= (a table's "
                          "meta['anchor_orientations'], or use AoALocalizer.from_meta), or calibrate=True to learn "
                          "the offsets", UserWarning, stacklevel=4)

    def _localize(self, X):
        anchors, o = self._geometry()
        phi = _wrap(self.local_angles(X) + o - getattr(self, "bias_", 0.0))
        x = triangulate(phi, anchors)
        model = _bearing_model(anchors, phi)
        if self.solver == "gauss_newton":
            centre = np.where(np.all(np.isfinite(x), axis=1, keepdims=True), anchors.mean(axis=0), np.nan)
            x = _refine([x, centre], model, max_iter=self.max_iter, tol=self.tol)
        spread = np.full(len(phi), np.nan)
        rows = np.flatnonzero(np.all(np.isfinite(x), axis=1))
        if rows.size:
            e, J, valid = model(x[rows], rows, True)
            w = valid.astype(np.float64)
            gdop = np.sqrt(_inv_trace(np.einsum("nm,nmi,nmj->nij", w, J, J)))  # position CRLB / sigma
            spread[rows] = self._noise(_residual_sigma(e, w, 2)) * gdop
            if self.solver == "gauss_newton":
                far = rows[_far_field(x[rows], anchors, gdop, self._noise(np.full(len(rows), np.nan)),
                                      _exact_fit(e, w, 2, 1.0))]
                x[far], spread[far] = np.nan, np.nan
        return Prediction(x, spread=spread)


__all__ = ["AoALocalizer", "music", "music_spectrum", "spatial_covariance", "triangulate", "ula_steering"]
