"""Cross-device RSSI calibration: map one device's readings onto a reference device's scale.

Different phones report different RSSI for the same signal (antenna, front end, driver),
which breaks a radio map recorded with another device. Across devices the relation is
close to linear in dBm (Haeberlen et al. 2004; Kjaergaard 2011; Figuera et al. 2011), so
an offset or a line learned from a little data largely removes it.
"""
from __future__ import annotations

import numpy as np

from ..core import NotFittedError
from . import functional as F
from .transforms import Transform, _features


class DeviceCalibration(Transform):
    """Learn ``reference ~ g(device)`` and apply ``g`` to the device's readings.

    ``fit(X, reference=R)``: ``X`` holds readings of the device to calibrate, ``R`` those of
    the reference device (the one that recorded the radio map). Arrays, SampleTables or
    scan views; NaN (not heard) is ignored.

    ``method``
        ``"offset"``    ``g(x) = x + b``, the mean difference (slope fixed to 1);
        ``"linear"``    ``g(x) = a x + b`` by least squares;
        ``"quantile"``  a monotone piecewise-linear map that matches the two RSS
                        distributions (histogram / CDF matching; no linearity assumption).
    ``paired``
        True: ``X`` and ``R`` have the same shape and element ``[i, j]`` of both was
        measured at the same place for the same AP (collected side by side); the fit uses
        the pairs heard by both devices (Haeberlen et al. 2004; Figuera et al. 2011).
        False (default): the two sets are unrelated samples of the same environment, e.g.
        the radio map and a few minutes of the new device's online scans. Offset: difference
        of means; linear: least squares through the two devices' matched quantiles (a Q-Q
        line), our operationalisation of the histogram-based self-calibration of Laoudias et
        al. (2013), which also fits a linear map from the two RSS histograms; quantile:
        always distribution matching. Like Tsui et al. (2009), no location labels are needed
        for the new device.
    ``n_quantiles``
        Probability levels (evenly spaced in [0, 1]) for the unpaired fits.

    Learned: ``coef_`` (a) and ``intercept_`` (b) for offset/linear; ``quantiles_`` and
    ``reference_quantiles_`` (the knots of ``g``) for quantile. Beyond the knots ``g``
    continues with slope 1 (a constant offset), so no reading is clipped.

    The calibration is global (one map for all APs), as in the cited works. Unpaired
    matching assumes the two sample sets cover similar places; a less sensitive device also
    hears fewer weak APs, which censors its histogram at the low end, so prefer paired data
    when available. Use it standalone (calibrate the new device's scans, then localize with
    a model trained on the reference data), not as a ``LocalizerPipeline`` step, which would
    also apply ``g`` to the reference training data.

    References
        A. Haeberlen, E. Flannery, A. M. Ladd, A. Rudys, D. S. Wallach, L. E. Kavraki, "Practical
        robust localization over large-scale 802.11 wireless networks", ACM MobiCom 2004, pp. 70-84.
        https://doi.org/10.1145/1023720.1023728
        A. W. Tsui, Y.-H. Chuang, H.-H. Chu, "Unsupervised learning for solving RSS hardware variance
        problem in WiFi localization", Mobile Networks and Applications 14(5):677-691, 2009.
        https://doi.org/10.1007/s11036-008-0139-0
        C. Figuera, J. L. Rojo-Alvarez, I. Mora-Jimenez, A. Guerrero-Curieses, M. Wilby, J. Ramos-Lopez,
        "Time-space sampling and mobile device calibration for WiFi indoor location systems", IEEE
        Transactions on Mobile Computing 10(7):913-926, 2011. https://doi.org/10.1109/TMC.2011.84
        M. B. Kjaergaard, "Indoor location fingerprinting with heterogeneous clients", Pervasive and
        Mobile Computing 7(1):31-43, 2011. https://doi.org/10.1016/j.pmcj.2010.04.005
        C. Laoudias, R. Piche, C. G. Panayiotou, "Device self-calibration in location systems using
        signal strength histograms", Journal of Location Based Services 7(3):165-181, 2013.
        https://doi.org/10.1080/17489725.2013.816792
    """

    _requires_fit = True
    _real_only = True
    _methods = ("offset", "linear", "quantile")

    def __init__(self, method: str = "linear", paired: bool = False, n_quantiles: int = 101):
        self.method = method
        self.paired = paired
        self.n_quantiles = n_quantiles

    def fit(self, X, y=None, *, reference=None):
        if self.method not in self._methods:
            raise ValueError(f"method must be one of {self._methods}, got {self.method!r}")
        if reference is None:
            raise TypeError("DeviceCalibration.fit needs the reference device's readings: fit(X, reference=R)")
        x = F._real(_features(X), "DeviceCalibration").astype(np.float64)
        r = F._real(_features(reference), "DeviceCalibration").astype(np.float64)
        super().fit(X, y)
        if self.paired:
            if x.shape != r.shape:
                raise ValueError(f"paired=True needs X and reference of the same shape, got {x.shape} and {r.shape}")
            both = ~np.isnan(x) & ~np.isnan(r)
            xs, rs = x[both], r[both]
        else:
            xs, rs = x[~np.isnan(x)], r[~np.isnan(r)]
        need = 1 if self.method == "offset" else 2
        if min(len(xs), len(rs)) < need:
            raise ValueError(f"DeviceCalibration(method={self.method!r}) needs at least {need} heard "
                             f"reading(s) from each device, got {len(xs)} and {len(rs)}")
        for name in ("coef_", "intercept_", "quantiles_", "reference_quantiles_"):
            self.__dict__.pop(name, None)  # refitting with another method leaves no stale state
        if self.method == "offset":
            self.coef_ = 1.0
            self.intercept_ = float(np.mean(rs - xs)) if self.paired else float(np.mean(rs) - np.mean(xs))
        elif self.method == "linear":
            if self.paired:
                a, b = _line(xs, rs)
            else:
                a, b = _line(*self._matched_quantiles(xs, rs))
            if not a > 0:
                raise ValueError(f"fitted slope {a:.3g} is not positive: the two devices' readings are not "
                                 "positively related (check the pairing or use more data)")
            self.coef_, self.intercept_ = a, b
        else:
            qx, qr = self._matched_quantiles(xs, rs)
            knots, inverse = np.unique(qx, return_inverse=True)  # integer dBm readings repeat
            self.quantiles_ = knots
            self.reference_quantiles_ = np.bincount(inverse, weights=qr) / np.bincount(inverse)
        return self

    def _matched_quantiles(self, xs, rs):
        n = int(self.n_quantiles)
        if n < 2:
            raise ValueError(f"n_quantiles must be >= 2, got {self.n_quantiles}")
        levels = np.linspace(0.0, 1.0, n)
        return np.quantile(xs, levels), np.quantile(rs, levels)

    def _transform(self, x):
        x = F._real(x, "DeviceCalibration")
        dtype = F._out_dtype(x)
        v = x.astype(np.float64)
        if self.method == "quantile":
            if not hasattr(self, "quantiles_"):
                raise NotFittedError("DeviceCalibration must be fitted first: fit(X, reference=R)")
            q, qr = self.quantiles_, self.reference_quantiles_
            out = np.interp(v, q, qr)
            out = np.where(v < q[0], v + (qr[0] - q[0]), out)
            out = np.where(v > q[-1], v + (qr[-1] - q[-1]), out)
            out[np.isnan(v)] = np.nan
        else:
            if not hasattr(self, "coef_"):
                raise NotFittedError("DeviceCalibration must be fitted first: fit(X, reference=R)")
            out = self.coef_ * v + self.intercept_
        return out.astype(dtype)


def _line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares ``y ~ a x + b`` (centred, so it is exact for exactly linear data)."""
    mx, my = x.mean(), y.mean()
    sxx = np.sum((x - mx) ** 2)
    if sxx == 0:
        raise ValueError("cannot fit a line: the device's readings are all equal")
    a = float(np.sum((x - mx) * (y - my)) / sxx)
    return a, float(my - a * mx)
