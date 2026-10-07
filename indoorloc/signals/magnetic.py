"""Magnetometer science: field components, tilt-compensated heading, hard/soft-iron calibration.

Pure numpy functions over 3-axis magnetometer readings ``(N, 3)`` (or one reading ``(3,)``)
in any unit (smartphones report microtesla, uT), plus two transforms:

    components   ``field_magnitude``, ``field_components`` (horizontal, vertical),
                 ``inclination``, ``magnetic_features`` -> the ``magnetic`` table layout
    heading      ``heading`` (tilt-compensated, with declination)
    calibration  ``fit_ellipsoid`` (hard-iron offset + soft-iron matrix), ``apply_calibration``
    transforms   ``MagneticFeatures`` (imu table or array -> ``magnetic`` table),
                 ``MagnetometerCalibration`` (fit/transform wrapper of ``fit_ellipsoid``)

Frames and conventions
    Readings are in the sensor (device) frame. ``gravity`` is the gravity vector as a phone
    reports it when at rest: the accelerometer's specific force, or Android's
    ``TYPE_GRAVITY``, i.e. about +9.81 m/s^2 along the device axis that points **up**
    (``(0, 0, 9.81)`` for a phone lying flat, screen up). Only its direction is used. When
    ``gravity`` is None the device z axis is taken as up. The world frame is East-North-Up
    (x east, y north, z up), as everywhere in IndoorLoc; headings are radians
    counter-clockwise from (true) east, the library-wide convention (CONTRACTS.md, section 7).
    A compass azimuth (clockwise from north) is ``pi / 2 - heading``.

``magnetic`` modality (the fingerprint layout this module produces)
    ``X`` ``(N, 3)`` float64 in uT with ``meta["feature_names"] = ("B", "B_h", "B_v")``:
    ``B = |b|`` (total intensity, orientation invariant), ``B_h`` the horizontal intensity and
    ``B_v = b . up`` the vertical component, **positive up** (so negative in the northern
    hemisphere, where the field dips downward). ``meta["units"] = "uT"``. Sequences (walks)
    are tables with one row per time step plus ``groups["trajectory"]`` and ``groups["time"]``
    (and ``meta["rate_hz"]``), as for every time series. ``B_h`` and ``B_v`` need the vertical,
    i.e. a tilt-compensated device or a device held flat; ``B`` alone is valid for any
    orientation. These three are the orientation-free features used by magnetic
    fingerprinting systems (e.g. Li et al. 2012; the magnitude alone in LocateMe, Subbu et
    al. 2013).

References
    B. Li, T. Gallagher, A. G. Dempster, C. Rizos, "How feasible is the use of magnetic field alone
    for indoor positioning?", IPIN 2012. https://doi.org/10.1109/IPIN.2012.6418880
    K. P. Subbu, B. Gozick, R. Dantu, "LocateMe: magnetic-fields-based indoor localization using
    smartphones", ACM Transactions on Intelligent Systems and Technology 4(4), 2013.
    https://doi.org/10.1145/2508037.2508054
"""
from __future__ import annotations

import numpy as np

from ..core import SampleTable
from .imu import moving_average
from .transforms import Transform, _features

FEATURE_NAMES = ("B", "B_h", "B_v")
_MAG = ("mag_x", "mag_y", "mag_z")
_GRAV = ("grav_x", "grav_y", "grav_z")
_ACC = ("acc_x", "acc_y", "acc_z")


# ---------------------------------------------------------------------------------- helpers
def _vectors(x, name: str) -> tuple[np.ndarray, bool]:
    """``(N, 3)`` float64 and whether the input was a single ``(3,)`` vector."""
    a = np.asarray(x)
    if a.dtype.kind == "c":
        raise ValueError(f"{name} must be real-valued")
    a = a.astype(np.float64)
    single = a.ndim == 1
    a = a[None] if single else a
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"{name} must be (N, 3) or (3,), got shape {np.shape(x)}")
    return a, single


def _up(gravity, n: int) -> np.ndarray:
    """``(n, 3)`` unit up vectors from gravity readings (device z when None)."""
    if gravity is None:
        return np.broadcast_to(np.array([0.0, 0.0, 1.0]), (n, 3))
    g, _ = _vectors(gravity, "gravity")
    if len(g) not in (1, n):
        raise ValueError(f"gravity has {len(g)} rows, expected 1 or {n}")
    norm = np.linalg.norm(g, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        up = np.where(norm > 0, g / norm, np.nan)
    return np.broadcast_to(up, (n, 3))


def _out(values, single: bool):
    return values[0] if single else values


def wrap_angle(angle) -> np.ndarray:
    """Wrap radians to ``[-pi, pi)``."""
    return (np.asarray(angle, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


# ------------------------------------------------------------------------------- components
def field_magnitude(mag) -> np.ndarray:
    """Total intensity ``|b|`` of ``(N, 3)`` readings (``(N,)``; a scalar for one reading).
    Invariant to the device orientation but not to hard/soft-iron distortion."""
    m, single = _vectors(mag, "mag")
    return _out(np.linalg.norm(m, axis=1), single)


def field_components(mag, gravity=None) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal intensity and vertical component ``(B_h, B_v)`` of the field.

    ``B_v = b . up`` (positive up, negative where the field dips downward) and
    ``B_h = sqrt(|b|^2 - B_v^2)``, with ``up`` the unit gravity reading (tilt compensation;
    see the module docstring for the sign of ``gravity``). ``gravity=None``: the device is
    flat, ``B_h = hypot(b_x, b_y)`` and ``B_v = b_z``.
    """
    m, single = _vectors(mag, "mag")
    up = _up(gravity, len(m))
    vertical = np.einsum("ij,ij->i", m, up)
    horizontal = np.sqrt(np.maximum(np.einsum("ij,ij->i", m, m) - vertical * vertical, 0.0))
    return _out(horizontal, single), _out(vertical, single)


def inclination(mag, gravity=None) -> np.ndarray:
    """Magnetic inclination (dip) in radians, ``atan2(-B_v, B_h)``: positive when the field
    points below the horizon (northern hemisphere), the geomagnetic sign convention."""
    h, v = field_components(mag, gravity)
    return np.arctan2(-v, h)


def magnetic_features(mag, gravity=None) -> np.ndarray:
    """``(N, 3)`` float64 ``[B, B_h, B_v]``: the ``magnetic`` modality layout (module docstring)."""
    m, single = _vectors(mag, "mag")
    h, v = field_components(m, gravity)
    return _out(np.column_stack([np.linalg.norm(m, axis=1), h, v]), single)


def heading(mag, gravity=None, *, forward=(0.0, 1.0, 0.0), declination: float = 0.0) -> np.ndarray:
    """Tilt-compensated compass heading of the device ``forward`` axis, radians in ``[-pi, pi)``.

    The field is projected on the horizontal plane (normal = up, from ``gravity``) to give
    magnetic north ``n``; magnetic east is ``e = n x up``; the heading of the horizontal
    projection of ``forward`` is ``atan2(forward . n, forward . e)``, counter-clockwise from
    magnetic east. ``declination`` (radians, positive when magnetic north lies east of true
    north, as in the IGRF/WMM tables) turns it into a heading from true east:
    ``heading_true = heading_magnetic - declination``. ``forward`` defaults to the device +y
    axis (the top edge of a phone held in portrait), as in ``apps.pdr.magnetic_heading``.
    Readings must be calibrated (``fit_ellipsoid``): a hard-iron offset of a few uT already
    biases the heading by several degrees. Indoors, steel distorts the direction of the
    field too, so the heading is locally biased near large anomalies.

    References
        M. J. Caruso, "Applications of magnetic sensors for low cost compass systems", IEEE
        Position Location and Navigation Symposium (PLANS) 2000, pp. 177-184.
        https://doi.org/10.1109/PLANS.2000.838300
    """
    m, single = _vectors(mag, "mag")
    up = _up(gravity, len(m))
    north = m - np.einsum("ij,ij->i", m, up)[:, None] * up
    north = north / np.maximum(np.linalg.norm(north, axis=1, keepdims=True), 1e-300)
    east = np.cross(north, up)
    f = np.asarray(forward, dtype=np.float64)
    if f.shape != (3,) or not np.any(f):
        raise ValueError(f"forward must be a non-zero 3-vector, got {forward!r}")
    h = np.arctan2(north @ f, east @ f) - float(declination)
    return _out(wrap_angle(h), single)


# ------------------------------------------------------------------------------ calibration
def _sqrtm_sym(A: np.ndarray) -> np.ndarray:
    ev, V = np.linalg.eigh(A)
    return (V * np.sqrt(np.maximum(ev, 0.0))) @ V.T


def _sym(w: np.ndarray) -> np.ndarray:
    """Symmetric 3x3 matrix from ``[w11, w22, w33, w12, w13, w23]``."""
    return np.array([[w[0], w[3], w[4]], [w[3], w[1], w[5]], [w[4], w[5], w[2]]])


def _refine(m, b, W, F, max_iter: int, tol: float):
    """Gauss-Newton on ``r_i = |W (m_i - b)| - F`` over ``b`` (3) and symmetric ``W`` (6), with
    step halving so the cost never increases; ``F`` fixed (it sets the scale of ``W``)."""
    theta = np.concatenate([b, W[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]])

    def residual(t):
        u = (m - t[:3]) @ _sym(t[3:]).T
        return np.linalg.norm(u, axis=1) - F, u

    r, u = residual(theta)
    cost = r @ r
    for _ in range(int(max_iter)):
        d = m - theta[:3]
        un = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-300)
        Wm = _sym(theta[3:])
        J = np.empty((len(m), 9))
        J[:, :3] = -un @ Wm                                                   # d|u|/db = -u^T W / |u|
        J[:, 3], J[:, 4], J[:, 5] = un[:, 0] * d[:, 0], un[:, 1] * d[:, 1], un[:, 2] * d[:, 2]
        J[:, 6] = un[:, 0] * d[:, 1] + un[:, 1] * d[:, 0]                     # w12 enters u_x and u_y
        J[:, 7] = un[:, 0] * d[:, 2] + un[:, 2] * d[:, 0]
        J[:, 8] = un[:, 1] * d[:, 2] + un[:, 2] * d[:, 1]
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        t = 1.0
        for _ in range(40):
            cand = theta + t * step
            rc, uc = residual(cand)
            if rc @ rc <= cost:
                break
            t *= 0.5
        else:
            break
        moved = np.linalg.norm(cand - theta)
        theta, r, u, cost = cand, rc, uc, rc @ rc
        if moved <= tol * (1.0 + np.linalg.norm(theta)):
            break
    return theta[:3], _sym(theta[3:])


def fit_ellipsoid(mag, *, field_strength: float | None = None, refine: bool = True, max_iter: int = 100,
                  tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Hard-iron offset and soft-iron correction from readings taken in many orientations.

    Model: ``m = A b_true + h + noise`` with ``|b_true| = F`` constant (a small region, no
    disturbance), ``h`` the hard-iron offset and ``A`` the soft-iron (and scale-factor,
    misalignment) distortion. The readings then lie on the ellipsoid
    ``(m - h)^T Q (m - h) = 1``. Returned: ``offset = h`` ``(3,)`` and the symmetric
    correction ``matrix = W`` ``(3, 3)`` such that ``b = W (m - h)`` has constant norm ``F``
    (``apply_calibration``). ``W`` equals ``A^-1`` up to a rotation, which readings alone cannot
    identify; the symmetric choice is returned (for a symmetric ``A``, exactly ``A^-1``).

    Steps: (1) algebraic least-squares quadric fit on centred, scaled data, the unit-norm
    parameter vector minimising ``|D theta|`` (the smallest right singular vector of the
    design matrix; Gander, Golub & Strebel 1994), converted to centre and shape;
    (2) with ``refine=True``, Gauss-Newton on the magnitude residuals
    ``|W (m_i - h)| - F`` (the geometric cost in the calibrated space), which removes the
    bias of the algebraic fit under noise. ``field_strength`` fixes ``F`` (e.g. the local
    IGRF/WMM total intensity, in the unit of ``mag``); None normalises ``W`` to unit
    determinant (the calibrated magnitude is then the geometric mean of the ellipsoid's
    semi-axes).

    The readings must span all three axes (e.g. a figure-eight motion): a raise, not a
    guess, when the smallest principal spread is below 5 % of the largest, or the fitted
    quadric is not an ellipsoid. Walking with the phone held flat only rotates it about
    the vertical and does not determine the ellipsoid. NaN rows are dropped.

    Deviation from Vasconcelos et al. (2011): their maximum-likelihood cost weights the
    residuals by the sensor noise model; the refinement here is the unweighted magnitude
    residual, which coincides with it for isotropic noise and a nearly spherical ``A``.

    References
        W. Gander, G. H. Golub, R. Strebel, "Least-squares fitting of circles and ellipses", BIT
        Numerical Mathematics 34(4):558-578, 1994. https://doi.org/10.1007/BF01934268
        V. Renaudin, M. H. Afzal, G. Lachapelle, "Complete triaxis magnetometer calibration in the
        magnetic domain", Journal of Sensors 2010, Article 967245. https://doi.org/10.1155/2010/967245
        J. F. Vasconcelos, G. Elkaim, C. Silvestre, P. Oliveira, B. Cardeira, "Geometric approach
        to strapdown magnetometer calibration in sensor frame", IEEE Transactions on Aerospace and
        Electronic Systems 47(2):1293-1306, 2011. https://doi.org/10.1109/TAES.2011.5751259
    """
    m, _ = _vectors(mag, "mag")
    m = m[np.all(np.isfinite(m), axis=1)]
    if len(m) < 9:
        raise ValueError(f"need at least 9 finite readings to fit an ellipsoid, got {len(m)}")
    if field_strength is not None and not float(field_strength) > 0:
        raise ValueError(f"field_strength must be positive, got {field_strength}")
    mu = m.mean(axis=0)
    spread = np.linalg.eigvalsh(np.cov((m - mu).T))
    if not spread[0] > 0.05 ** 2 * spread[-1]:
        raise ValueError("the readings do not span all three axes (nearly planar); rotate the sensor through "
                         "many orientations (e.g. a figure-eight) before calibrating")
    s = np.sqrt(np.mean(np.sum((m - mu) ** 2, axis=1)))
    x, y, z = ((m - mu) / s).T
    D = np.column_stack([x * x, y * y, z * z, 2 * y * z, 2 * x * z, 2 * x * y, 2 * x, 2 * y, 2 * z,
                         np.ones_like(x)])
    v = np.linalg.svd(D, full_matrices=False)[2][-1]
    M = np.array([[v[0], v[5], v[4]], [v[5], v[1], v[3]], [v[4], v[3], v[2]]])
    ev = np.linalg.eigvalsh(M)
    if not (np.all(ev > 0) or np.all(ev < 0)):
        raise ValueError("the readings do not fit an ellipsoid (indefinite quadric); check for disturbances "
                         "and rotate the sensor through more orientations")
    centre = -np.linalg.solve(M, v[6:9])
    k = centre @ M @ centre - v[9]
    Q = M / (k * s * s)                                                      # (m - h)^T Q (m - h) = 1
    offset = mu + s * centre
    if not np.all(np.linalg.eigvalsh(Q) > 0):
        raise ValueError("the fitted quadric is not an ellipsoid")
    F = float(field_strength) if field_strength is not None else float(np.linalg.det(Q)) ** (-1.0 / 6.0)
    W = F * _sqrtm_sym(Q)
    if refine:
        offset, W = _refine(m, offset, W, F, max_iter, tol)
        if field_strength is None:  # keep the unit-determinant normalisation
            W = W / np.cbrt(np.linalg.det(W))
    return offset, W


def apply_calibration(mag, offset, matrix) -> np.ndarray:
    """Calibrated readings ``W (m - h)`` (``(N, 3)``, or ``(3,)`` for one reading)."""
    m, single = _vectors(mag, "mag")
    W = np.asarray(matrix, dtype=np.float64)
    if W.shape != (3, 3):
        raise ValueError(f"matrix must be (3, 3), got {W.shape}")
    return _out((m - np.asarray(offset, dtype=np.float64)) @ W.T, single)


# ------------------------------------------------------------------------------- transforms
def _channels(table: SampleTable, names, who: str) -> np.ndarray:
    channels = list(table.meta.get("channels", ()))
    missing = [c for c in names if c not in channels]
    if missing:
        raise ValueError(f"{who}: the table's meta['channels'] lacks {missing} (have {channels})")
    return np.asarray(table.X)[:, [channels.index(c) for c in names]].astype(np.float64)


class MagneticFeatures(Transform):
    """Per-sample magnetic fingerprint features ``[B, B_h, B_v]`` (the ``magnetic`` layout).

    Input:

    * an ``imu`` SampleTable whose ``meta["channels"]`` include ``mag_x/y/z``; the up
      direction comes from ``grav_x/y/z`` if present, else from ``acc_x/y/z`` averaged over
      ``smoothing`` seconds (a centred moving average per ``groups["trajectory"]``, which
      removes the walking accelerations; needs ``meta["rate_hz"]``), else from the device z
      axis. The output table keeps pos, groups, ids and ``rate_hz`` and gets
      ``modality="magnetic"``, ``units="uT"``, ``feature_names=("B", "B_h", "B_v")``;
      ``channels`` is dropped. A ``magnetic`` table passes through unchanged;
    * an array ``(..., 3)`` (magnetometer, device z up) or ``(..., 6)``
      (``[mag_x, mag_y, mag_z, g_x, g_y, g_z]`` with a gravity reading per row).

    Stateless. See ``magnetic_features`` for the definitions and the module docstring for
    the sign conventions.

    References
        B. Li, T. Gallagher, A. G. Dempster, C. Rizos, "How feasible is the use of magnetic field
        alone for indoor positioning?", IPIN 2012. https://doi.org/10.1109/IPIN.2012.6418880
    """

    _requires_fit = False

    def __init__(self, smoothing: float = 1.0):
        self.smoothing = smoothing

    def _transform(self, x):
        x = np.asarray(x)
        if x.shape[-1] not in (3, 6):
            raise ValueError(f"MagneticFeatures needs (..., 3) magnetometer or (..., 6) magnetometer + gravity "
                             f"rows, got shape {x.shape}")
        flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
        out = magnetic_features(flat[:, :3], None if flat.shape[1] == 3 else flat[:, 3:])
        return out.reshape(x.shape[:-1] + (3,))

    def _up_from_table(self, table: SampleTable):
        channels = table.meta.get("channels", ())
        if all(c in channels for c in _GRAV):
            return _channels(table, _GRAV, "MagneticFeatures")
        if not all(c in channels for c in _ACC):
            return None
        acc = _channels(table, _ACC, "MagneticFeatures")
        if not self.smoothing:
            return acc
        rate = table.meta.get("rate_hz")
        if not rate:
            raise ValueError("MagneticFeatures: smoothing the accelerometer needs meta['rate_hz'] "
                             "(or pass smoothing=0 to use the raw acceleration as the up direction)")
        n = max(1, int(round(float(self.smoothing) * float(rate))))
        n += 1 - n % 2  # centred window: odd length
        walks = table.groups.get("trajectory")
        if walks is None:
            return moving_average(acc, n)
        out = np.empty_like(acc)
        for w in np.unique(walks):
            rows = np.flatnonzero(walks == w)
            out[rows] = moving_average(acc[rows], n)
        return out

    def transform(self, X):
        if not isinstance(X, SampleTable):
            self._check_features(X)
            return self._transform(_features(X))
        if X.meta.get("modality") == "magnetic":
            return X
        out = magnetic_features(_channels(X, _MAG, "MagneticFeatures"), self._up_from_table(X))
        meta = {k: v for k, v in X.meta.items() if k != "channels"}
        meta.update(modality="magnetic", units="uT", feature_names=FEATURE_NAMES)
        return X.replace(X=out, meta=meta)

    def fit(self, X, y=None):
        if isinstance(X, SampleTable):  # the output layout does not depend on the input width
            return self
        return super().fit(X, y)


class MagnetometerCalibration(Transform):
    """Hard-iron and soft-iron calibration learned from readings in many orientations.

    ``fit`` runs ``fit_ellipsoid`` on ``(N, 3)`` readings (or the ``mag_x/y/z`` channels of an
    ``imu`` table) and stores ``offset_`` ``(3,)``, ``matrix_`` ``(3, 3)`` and
    ``field_strength_`` (the calibrated magnitude: ``field_strength`` if given, else the
    unit-determinant value); ``transform`` returns ``W (m - h)`` for arrays ``(..., 3)`` and,
    for an ``imu`` table, replaces the three magnetometer channels (dtype kept). Fit it on a
    calibration session (a figure-eight), not on a walk with the phone held flat.

    Parameters
        field_strength  known total intensity (same unit as the readings), or None.
        refine          Gauss-Newton refinement of the algebraic fit (see ``fit_ellipsoid``).

    References
        J. F. Vasconcelos, G. Elkaim, C. Silvestre, P. Oliveira, B. Cardeira, "Geometric approach
        to strapdown magnetometer calibration in sensor frame", IEEE Transactions on Aerospace and
        Electronic Systems 47(2):1293-1306, 2011. https://doi.org/10.1109/TAES.2011.5751259
        V. Renaudin, M. H. Afzal, G. Lachapelle, "Complete triaxis magnetometer calibration in the
        magnetic domain", Journal of Sensors 2010, Article 967245. https://doi.org/10.1155/2010/967245
    """

    _requires_fit = True

    def __init__(self, field_strength: float | None = None, refine: bool = True):
        self.field_strength = field_strength
        self.refine = refine

    @staticmethod
    def _imu_table(X) -> bool:
        """True for a table of named IMU channels; raises for a ``magnetic`` feature table."""
        if not isinstance(X, SampleTable):
            return False
        if X.meta.get("modality") == "magnetic":
            raise ValueError("MagnetometerCalibration works on raw magnetometer readings (an imu table or (N, 3) "
                             "arrays), not on a 'magnetic' table of [B, B_h, B_v] features: calibrate first, then "
                             "apply MagneticFeatures")
        return "channels" in X.meta

    @classmethod
    def _readings(cls, X) -> np.ndarray:
        if cls._imu_table(X):
            return _channels(X, _MAG, "MagnetometerCalibration")
        x = np.asarray(_features(X), dtype=np.float64)
        if x.shape[-1] != 3:
            raise ValueError(f"MagnetometerCalibration needs (..., 3) readings, got shape {x.shape}")
        return x.reshape(-1, 3)

    def fit(self, X, y=None):
        m = self._readings(X)
        offset, W = fit_ellipsoid(m, field_strength=self.field_strength, refine=self.refine)
        self.offset_, self.matrix_ = offset, W
        self.field_strength_ = (float(self.field_strength) if self.field_strength is not None
                                else float(np.nanmedian(np.linalg.norm(apply_calibration(m, offset, W), axis=1))))
        self.n_features_in_ = 3
        return self

    def transform(self, X):
        self._check_fitted("offset_")
        if self._imu_table(X):
            channels = list(X.meta["channels"])
            cols = [channels.index(c) for c in _MAG] if all(c in channels for c in _MAG) else None
            if cols is None:
                raise ValueError(f"MagnetometerCalibration: the table's meta['channels'] lacks {list(_MAG)} "
                                 f"(have {channels})")
            out = np.array(X.X, copy=True)
            out[:, cols] = apply_calibration(X.X[:, cols], self.offset_, self.matrix_).astype(out.dtype)
            return X.replace(X=out)
        x = np.asarray(_features(X), dtype=np.float64)
        if x.shape[-1] != 3:
            raise ValueError(f"MagnetometerCalibration needs (..., 3) readings, got shape {x.shape}")
        out = apply_calibration(x.reshape(-1, 3), self.offset_, self.matrix_).reshape(x.shape)
        return X.replace(X=out) if isinstance(X, SampleTable) else out

    def _transform(self, x):  # pragma: no cover - transform is overridden
        return apply_calibration(x, self.offset_, self.matrix_)


__all__ = ["FEATURE_NAMES", "MagneticFeatures", "MagnetometerCalibration", "apply_calibration", "field_components",
           "field_magnitude", "fit_ellipsoid", "heading", "inclination", "magnetic_features", "wrap_angle"]
