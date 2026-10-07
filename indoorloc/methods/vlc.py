"""Visible light positioning from received optical power (RSS) with the Lambertian LOS model.

``X`` ``(N, A)`` holds the received optical power of each LED (watts, or any unit
proportional to it when ``calibrate=True`` learns the per-LED emitted power); NaN or a value
``<= min_power`` = LED not seen. The LED geometry is a constructor parameter
(CONTRACTS.md, section 5): positions ``anchors`` ``(A, 3)`` in metres, emission axes
``normals`` and the Lambertian order. The model (``signals.vlc``, Kahn & Barry 1997)::

    P_i(x) = Pt_i (m_i + 1) A / (2 pi d_i^2) cos^m_i(phi_i) Ts g cos(psi_i),   psi_i <= FOV

This module is numpy only (it shares the batched Gauss-Newton solver of ``methods.geometric``).

Example (simulated data)::

    train, test = load_dataset("synthetic_office", modality="vlc")
    model = LambertianLocalizer.from_meta(train.meta)      # LED geometry and receiver from the table
    model.evaluate(test)                                    # no fit needed; fit(train) with calibrate=True
"""
from __future__ import annotations

import numpy as np

from ..core import Prediction
from .geometric import GeometricLocalizer, _as_anchors, _gauss_newton, _residual_sigma, _spread, \
    linear_trilateration

_TINY = 1e-300


def _unit_rows(v, n: int, name: str) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    a = np.broadcast_to(a, (n, 3)) if a.ndim == 1 else a
    if a.shape != (n, 3):
        raise ValueError(f"{name} must be (3,) or ({n}, 3), got shape {np.shape(v)}")
    norm = np.linalg.norm(a, axis=1, keepdims=True)
    if not np.all(norm > 0):
        raise ValueError(f"{name} must be non-zero vectors")
    return a / norm


class LambertianLocalizer(GeometricLocalizer):
    """Visible light positioning by received-power ranging or model fitting (Lambertian LOS).

    ``solver``:

    * ``"ranges"``: each power is inverted to a distance with the vertical-link formula
      ``d = (C h^(m+1) / P)^(1/(m+3))`` (``signals.vlc.power_to_distance``; exact for LEDs
      facing down and a receiver facing up at the known ``receiver_height``), converted to
      a horizontal radius and trilaterated in 2-D (``linear_trilateration``, then
      Gauss-Newton on the radii): the classic RSS VLP pipeline (Zhang et al. 2014; Zhuang
      et al. 2018, section on RSS);
    * ``"nls"`` (default): maximum likelihood under additive Gaussian power noise, i.e.
      Gauss-Newton on ``sum_i (P_i - P_i(x))^2`` with the full model (any LED and receiver
      orientation, FOV cut-off), started from the ``"ranges"`` solution (2-D) and the
      power-weighted centroid; the lower cost wins. With ``receiver_height=None`` the height
      is estimated too (3-D output), from starts at the centroid and at the strongest LED,
      each at the height at which the strongest reading would be received directly below.

    With ``receiver_height`` given the output is 2-D ``(x, y)`` at that height (the anchors'
    z frame); otherwise 3-D. Rows seeing fewer than D + 1 LEDs return NaN. Missing readings
    are ignored (no censoring model). LEDs that all lie on one line (a single row of corridor
    lights) cannot tell a point from its mirror image across that line: both solvers return
    NaN there (singular trilateration; singular information matrix at the ``"nls"``
    solution, which the symmetry pins to the line), so ``evaluate`` counts such rows in
    ``n_failed`` instead of scoring a biased point. ``Prediction.spread`` =
    ``sigma sqrt(trace((J^T J)^-1))`` at the estimate (the Cramer-Rao-shaped error scale of
    the power model), ``sigma`` being ``sigma`` (W), else the calibrated ``sigma_``, else the
    per-sample residual std (NaN without redundancy); for ``"ranges"``, ``sigma`` is taken as a
    range noise std in metres.

    ``calibrate=True``: ``fit(X, positions)`` learns the emitted power of every LED seen at
    least ``min_readings`` times, ``tx_power_ = sum P H / sum H^2`` (least squares through
    the origin with the known geometry, which also absorbs the unknown photodiode
    responsivity or ADC scale), and the residual noise std ``sigma_``. Otherwise the model
    needs no fit: ``localize`` works straight away and ``fit`` only records the input size
    (``fit(X)`` needs no positions).

    Parameters
    ----------
    anchors : (A, 3) LED positions in metres.
    normals : (3,) or (A, 3) LED emission axes; None = straight down.
    order : Lambertian order ``m`` (scalar or (A,)); ``half_power_angle`` (radians) instead
        sets ``m = -ln 2 / ln cos(angle)``; neither = 1 (a 60 degree half-power angle).
    tx_power : emitted optical power per LED (W), scalar or (A,).
    area, fov, filter_gain, concentrator_gain : photodiode area (m^2), field-of-view
        semi-angle (radians), optical filter and concentrator gains.
    receiver_normal : (3,) receiver axis; default straight up.
    receiver_height : known receiver z in the anchors' frame (2-D output), or None (3-D).
    solver : "nls" or "ranges" (``"ranges"`` needs ``receiver_height``).
    min_power : readings ``<= min_power`` count as not seen.
    sigma : noise std for ``spread`` (W for "nls", m for "ranges"); None = estimated.
    calibrate, min_readings : learn ``tx_power_`` and ``sigma_`` in ``fit``.
    max_iter, tol : Gauss-Newton iterations and relative step tolerance.

    References
    ----------
    Y. Zhuang, L. Hua, L. Qi, J. Yang, P. Cao, Y. Cao, Y. Wu, J. Thompson, H. Haas, "A survey of
    positioning systems using visible LED lights", IEEE Communications Surveys & Tutorials
    20(3):1963-1988, 2018. DOI 10.1109/COMST.2018.2806558.
    W. Zhang, M. I. S. Chowdhury, M. Kavehrad, "Asynchronous indoor positioning system based on
    visible light communications", Optical Engineering 53(4):045105, 2014. DOI 10.1117/1.OE.53.4.045105.
    J. M. Kahn, J. R. Barry, "Wireless infrared communications", Proceedings of the IEEE 85(2):265-298,
    1997. DOI 10.1109/5.554222.
    T. Komine, M. Nakagawa, "Fundamental analysis for visible-light communication system using LED
    lights", IEEE Transactions on Consumer Electronics 50(1):100-107, 2004. DOI 10.1109/TCE.2004.1277847.
    """

    def __init__(self, anchors=None, normals=None, order=None, half_power_angle=None, tx_power=1.0,
                 area: float = 1e-4, fov: float = np.pi / 2, filter_gain: float = 1.0, concentrator_gain: float = 1.0,
                 receiver_normal=(0.0, 0.0, 1.0), receiver_height=None, solver: str = "nls", min_power: float = 0.0,
                 sigma=None, calibrate: bool = False, min_readings: int = 3, max_iter: int = 50, tol: float = 1e-10):
        self.anchors = anchors
        self.normals = normals
        self.order = order
        self.half_power_angle = half_power_angle
        self.tx_power = tx_power
        self.area = area
        self.fov = fov
        self.filter_gain = filter_gain
        self.concentrator_gain = concentrator_gain
        self.receiver_normal = receiver_normal
        self.receiver_height = receiver_height
        self.solver = solver
        self.min_power = min_power
        self.sigma = sigma
        self.calibrate = calibrate
        self.min_readings = min_readings
        self.max_iter = max_iter
        self.tol = tol

    _META_KEYS = {"anchors": "led_positions", "normals": "anchor_normals", "order": "lambertian_order",
                  "tx_power": "tx_power_w", "area": "receiver_area_m2", "fov": "receiver_fov_rad",
                  "filter_gain": "filter_gain", "concentrator_gain": "concentrator_gain"}

    @classmethod
    def from_meta(cls, meta, **params) -> LambertianLocalizer:
        """A localizer configured from the ``meta`` of a ``vlc`` table (e.g. ``SyntheticOffice``).

        Reads ``led_positions`` (A, 3), ``anchor_normals``, ``lambertian_order``,
        ``tx_power_w``, ``receiver_area_m2``, ``receiver_fov_rad``, ``filter_gain`` and
        ``concentrator_gain`` (the keys present). For 2-D positions on a single storey
        (``pos_names`` of length 2, ``floors == (0,)``) the receiver height is
        ``meta["device_height"]``; 2-D positions on several storeys need an explicit
        ``receiver_height`` (or 3-D positions). ``params`` override or add constructor arguments,
        e.g. ``LambertianLocalizer.from_meta(train.meta, solver="ranges")``.
        """
        if meta.get("modality", "vlc") != "vlc":
            raise ValueError(f"from_meta needs the meta of a 'vlc' table, got modality {meta.get('modality')!r}")
        if "led_positions" not in meta:
            raise ValueError("from_meta needs meta['led_positions'] (A, 3), the LED positions in metres")
        kw = {arg: meta[key] for arg, key in cls._META_KEYS.items() if key in meta}
        if len(meta.get("pos_names", ("x", "y", "z"))) == 2 and "receiver_height" not in params:
            if tuple(meta.get("floors", (0,))) != (0,) or "device_height" not in meta:
                raise ValueError("2-D positions: give receiver_height (the receiver z in the LEDs' frame); "
                                 "meta has no single-storey device_height")
            kw["receiver_height"] = float(meta["device_height"])
        kw.update(params)
        return cls(**kw)

    # -- geometry and model ------------------------------------------------------------------

    def _leds(self) -> np.ndarray:
        return _as_anchors(self.anchors, type(self).__name__, dims=(3,))

    def _orders(self, A: int) -> np.ndarray:
        if self.order is not None and self.half_power_angle is not None:
            raise ValueError("give order or half_power_angle, not both")
        if self.half_power_angle is not None:
            phi = np.asarray(self.half_power_angle, dtype=np.float64)
            if np.any(~(phi > 0) | ~(phi < np.pi / 2)):
                raise ValueError(f"half_power_angle must be in (0, pi/2) radians, got {self.half_power_angle}")
            m = -np.log(2.0) / np.log(np.cos(phi))
        else:
            m = np.asarray(1.0 if self.order is None else self.order, dtype=np.float64)
        m = np.broadcast_to(m, (A,)).astype(np.float64)
        if np.any(~(m >= 0)):
            raise ValueError(f"the Lambertian order must be non-negative, got {self.order}")
        return m

    def _per_led(self, value, A: int, name: str) -> np.ndarray:
        v = np.asarray(value, dtype=np.float64)
        if v.ndim == 0:
            return np.full(A, float(v))
        if v.shape != (A,):
            raise ValueError(f"{name} must be a scalar or an ({A},) array, got shape {v.shape}")
        return v.copy()

    def _geometry(self):
        """(leds, normals, orders, gain constants C (A,) without Pt, receiver normal, cos FOV)."""
        leds = self._leds()
        A = len(leds)
        nt = _unit_rows((0.0, 0.0, -1.0) if self.normals is None else self.normals, A, "normals")
        nr = _unit_rows(self.receiver_normal, 1, "receiver_normal")[0]
        m = self._orders(A)
        if not float(self.area) > 0 or not 0 < float(self.fov) <= np.pi / 2:
            raise ValueError(f"need area > 0 and 0 < fov <= pi/2, got area={self.area}, fov={self.fov}")
        C = (m + 1.0) * float(self.area) * float(self.filter_gain) * float(self.concentrator_gain) / (2.0 * np.pi)
        return leds, nt, m, C, nr, float(np.cos(float(self.fov)))

    def _power_rates(self, pts, leds, nt, m, C, nr, cos_fov, jac: bool):
        """Model power per unit emitted power ``H`` (n, A) at 3-D points (n, 3) and dH/dp (n, A, 3)."""
        v = pts[:, None, :] - leds[None]                                  # LED -> receiver
        d2 = np.sum(v * v, axis=2)
        d = np.sqrt(d2)
        u = np.einsum("nad,ad->na", v, nt)                                # d cos(phi)
        w = -(v @ nr)                                                     # d cos(psi)
        seen = (d > 0) & (u > 0) & (w > 0) & (w >= (cos_fov - 1e-15) * d)
        us, ws = np.where(seen, u, 1.0), np.where(seen, w, 1.0)
        ds2 = np.where(seen, d2, 1.0)
        H = np.where(seen, C * us ** m * ws / ds2 ** ((m + 3.0) / 2.0), 0.0)
        if not jac:
            return H, None
        # dH/dp = H (m n_t / u - n_r / w - (m + 3) v / d^2)
        G = (m[None, :, None] * nt[None] / us[..., None] - nr[None, None, :] / ws[..., None]
             - (m + 3.0)[None, :, None] * v / ds2[..., None])
        return H, H[..., None] * G

    def _check_input(self, X):
        leds = self._leds()
        if X.dtype.kind == "c":
            raise ValueError("received power must be real-valued")
        if X.ndim != 2 or X.shape[1] != len(leds):
            raise ValueError(f"X must hold one received power per LED, shape (N, {len(leds)}); got {X.shape}")

    def _check_options(self):
        if self.solver not in ("nls", "ranges"):
            raise ValueError(f"solver must be 'nls' or 'ranges', got {self.solver!r}")
        if self.solver == "ranges" and self.receiver_height is None:
            raise ValueError("solver='ranges' inverts power to distance with a known receiver height: "
                             "give receiver_height, or use solver='nls'")

    def _points(self, pos) -> np.ndarray:
        """3-D receiver points from labels: (N, 3) as they are, (N, 2) at ``receiver_height``."""
        if pos.shape[1] == 3:
            return pos
        if pos.shape[1] == 2 and self.receiver_height is not None:
            return np.column_stack([pos, np.full(len(pos), float(self.receiver_height))])
        raise ValueError(f"positions must be (N, 3), or (N, 2) with receiver_height given; got {pos.shape}")

    # -- fitting -----------------------------------------------------------------------------

    def _fit(self, X, pos, floor, building):
        self._check_options()
        leds, nt, m, C, nr, cos_fov = self._geometry()
        self._clear_calibration()
        for name in ("tx_power_", "calibrated_"):
            self.__dict__.pop(name, None)
        tx = self._per_led(self.tx_power, len(leds), "tx_power")
        if pos is None:  # fit(X) without positions (calibrate=False): nothing to learn
            return
        pts = self._points(pos)
        if not self.calibrate:
            return
        z = np.asarray(X, dtype=np.float64)
        H, _ = self._power_rates(pts, leds, nt, m, C, nr, cos_fov, False)
        ok = np.isfinite(z) & (z > float(self.min_power)) & (H > 0)
        count = ok.sum(axis=0)
        num = np.sum(np.where(ok, z * H, 0.0), axis=0)
        den = np.sum(np.where(ok, H * H, 0.0), axis=0)
        learned = count >= max(int(self.min_readings), 1)
        tx = np.where(learned, num / np.where(learned, den, 1.0), tx)
        resid = np.where(ok, z - tx * H, np.nan)
        dof = int(ok.sum()) - int(learned.sum())
        self.tx_power_ = tx
        self.calibrated_ = learned
        self.sigma_ = float(np.sqrt(np.nansum(resid * resid) / dof)) if dof > 0 else np.nan

    # -- localization ------------------------------------------------------------------------

    def _tx(self, A: int) -> np.ndarray:
        return self.tx_power_ if hasattr(self, "tx_power_") else self._per_led(self.tx_power, A, "tx_power")

    def _range_start(self, z, leds, m, C, tx):
        """Horizontal radii from the vertical-link inversion and their linear trilateration."""
        h = leds[:, 2] - float(self.receiver_height)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(z > 0, (tx * C * np.abs(h) ** (m + 1.0) / np.where(z > 0, z, 1.0)) ** (1.0 / (m + 3.0)),
                         np.nan)
            r = np.sqrt(np.maximum(d * d - h * h, 0.0))
        r = np.where(np.isfinite(z) & (h > 0), r, np.nan)
        return r, linear_trilateration(r, leds[:, :2])

    def _localize(self, X):
        self._check_options()
        leds, nt, m, C, nr, cos_fov = self._geometry()
        A = len(leds)
        tx = self._tx(A)
        z = np.asarray(X, dtype=np.float64)
        z = np.where(np.isfinite(z) & (z > float(self.min_power)), z, np.nan)
        heard = np.isfinite(z)
        planar = self.receiver_height is not None
        D = 2 if planar else 3
        enough = heard.sum(axis=1) >= D + 1
        top = np.max(np.where(heard, z, -np.inf), axis=1)
        top = np.where(np.isfinite(top), top, np.nan)
        scale = np.where(np.isfinite(top) & (top > 0), top, 1.0)          # per-row scale: conditioning only
        w = np.where(heard, z, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            centroid = (w @ leds) / w.sum(axis=1, keepdims=True)
        strongest = np.argmax(np.where(heard, z, -np.inf), axis=1)

        if self.solver == "ranges":
            from .geometric import _range_model

            r, x = self._range_start(z, leds, m, C, tx)
            model = _range_model(leds[:, :2], r)
            x = np.where(enough[:, None], x, np.nan)
            x = _gauss_newton(x, model, max_iter=self.max_iter, tol=self.tol)
            spread = np.full(len(z), np.nan)
            rows = np.flatnonzero(np.all(np.isfinite(x), axis=1))
            if rows.size:
                e, J, valid = model(x[rows], rows, True)
                wv = valid.astype(np.float64)
                sig = (np.full(rows.size, float(self.sigma)) if self.sigma is not None
                       else _residual_sigma(e, wv, D))                    # metres: sigma_ is in watts
                spread[rows] = _spread(J, wv, sig)
            return Prediction(x, spread=spread)

        zh = float(self.receiver_height) if planar else 0.0

        def lift(x):
            return np.column_stack([x, np.full(len(x), zh)]) if planar else x

        def model(x, rows, jac):
            H, dH = self._power_rates(lift(x), leds, nt, m, C, nr, cos_fov, jac)
            zr, sr = z[rows], scale[rows, None]
            valid = np.isfinite(zr)
            e = np.where(valid, (zr - tx * H) / sr, 0.0)
            if not jac:
                return e, valid
            J = tx[None, :, None] * dH[..., :D] / sr[..., None]
            return e, np.where(valid[..., None], J, 0.0), valid

        if planar:
            _, x_ranges = self._range_start(z, leds, m, C, tx)
            starts = [x_ranges, centroid[:, :2]]
        else:
            # height at which the strongest reading would be received straight below its LED
            k = strongest
            with np.errstate(invalid="ignore", divide="ignore"):
                h0 = np.sqrt(tx[k] * C[k] / np.where(np.isfinite(top), top, np.nan))
            z0 = leds[k, 2] - h0
            starts = [np.column_stack([centroid[:, :2], z0]), np.column_stack([leds[k, :2], z0])]
        best = np.full((len(z), D), np.nan)
        best_cost = np.full(len(z), np.inf)
        for x0 in starts:
            x0 = np.where(enough[:, None] & np.all(np.isfinite(x0), axis=1, keepdims=True), x0, np.nan)
            x = _gauss_newton(x0, model, max_iter=self.max_iter, tol=self.tol)
            rows = np.flatnonzero(np.all(np.isfinite(x), axis=1))
            if rows.size == 0:
                continue
            e, _ = model(x[rows], rows, False)
            cost = np.sum(e * e, axis=1)
            better = cost < best_cost[rows]  # strict: ties keep the earlier start
            best[rows[better]] = x[rows[better]]
            best_cost[rows[better]] = cost[better]
        spread = np.full(len(z), np.nan)
        rows = np.flatnonzero(np.all(np.isfinite(best), axis=1))
        if rows.size:
            e, J, valid = model(best[rows], rows, True)
            sr = scale[rows]
            e, J = e * sr[:, None], J * sr[:, None, None]                 # back to watts
            wv = valid.astype(np.float64)
            spread[rows] = _spread(J, wv, self._noise(_residual_sigma(e, wv, D)))
            # singular information matrix: the position is not identifiable (collinear LEDs)
            lost = rows[~np.isfinite(_spread(J, wv, np.ones(rows.size)))]
            best[lost] = np.nan
            spread[lost] = np.nan
        return Prediction(best, spread=spread)

    def predict_power(self, positions) -> np.ndarray:
        """Model received power ``(N, A)`` at ``positions`` ((N, 3), or (N, 2) at ``receiver_height``)."""
        if self.calibrate:
            self._check_fitted("tx_power_")
        leds, nt, m, C, nr, cos_fov = self._geometry()
        pts = self._points(np.atleast_2d(np.asarray(positions, dtype=np.float64)))
        H, _ = self._power_rates(pts, leds, nt, m, C, nr, cos_fov, False)
        return self._tx(len(leds)) * H


__all__ = ["LambertianLocalizer"]
