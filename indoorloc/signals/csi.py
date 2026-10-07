"""WiFi channel state information (CSI): pure functions and fit/transform classes.

Layout (CONTRACTS.md): one CSI sample is ``(n_rx, n_tx, n_sub)`` complex, a batch is
``(N, ..., n_sub)`` with samples first and subcarriers last; any axes in between are
antenna chains. Arrays are always read as batches; a 1-D array is one chain. Tables
carry ``meta["modality"]`` (``"csi"`` complex, ``"csi_amp"`` amplitude, ``"csi_phase"``
sanitized phase) and optionally ``meta["subcarriers"]``, the OFDM subcarrier indices of
the last axis (e.g. ``INTEL5300_SUBCARRIERS_20MHZ``). Transforms that change the
representation update ``meta["modality"]``.

Functions: ``amplitude``, ``phase``, ``sanitize_phase``, ``conjugate_multiply``, ``csi_ratio``.
Transforms: ``CSIAmplitude``, ``CSIPhaseSanitize``, ``SubcarrierSelect``.
"""
from __future__ import annotations

import numpy as np

from ..core import SampleTable
from . import functional as F
from ._scan import ScanView
from .transforms import Transform, _take_columns

# IEEE 802.11n grouping Ng = 2 at 20 MHz: the 30 subcarriers reported by the Intel 5300 CSI tool
# (Halperin et al., SIGCOMM CCR 41(1), 2011, https://doi.org/10.1145/1925861.1925870).
INTEL5300_SUBCARRIERS_20MHZ = np.array([-28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, -1,
                                        1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 28])


def amplitude(H, db: bool = False) -> np.ndarray:
    """``|H|`` (float32 for complex64 input), or ``20 log10 |H|`` in dB with ``db=True``.
    A zero amplitude (typically a padded or null subcarrier) is NaN in dB."""
    amp = np.abs(np.asarray(H))
    if not db:
        return amp
    amp = amp.astype(F._out_dtype(amp))
    out = np.full(amp.shape, np.nan, dtype=amp.dtype)
    np.log10(amp, out=out, where=amp > 0)
    return 20 * out


def phase(H, unwrap: bool = False) -> np.ndarray:
    """Phase in radians; ``unwrap=True`` unwraps it across subcarriers (last axis)."""
    ph = np.angle(np.asarray(H))
    return np.unwrap(ph, axis=-1) if unwrap else ph


def sanitize_phase(ph, subcarriers=None, joint: bool = False, method: str = "lstsq") -> np.ndarray:
    """Remove the linear phase error of CSI across subcarriers.

    ``ph`` is a batch ``(N, ..., n_sub)`` of phases in radians (wrapped or not; it is
    unwrapped along the last axis first). A sampling-time offset adds ``a * k`` and the
    carrier/packet offsets add ``b`` to the phase of subcarrier index ``k``; this returns
    ``unwrap(ph) - a * k - b``. ``subcarriers``: the indices ``k`` (default ``0..n_sub-1``;
    use the real indices, e.g. ``INTEL5300_SUBCARRIERS_20MHZ``, which are not evenly spaced).

    ``method``: ``"lstsq"`` least-squares line; ``"endpoints"`` the transformation of Sen et
    al. (2012): ``a = (ph[-1] - ph[0]) / (k[-1] - k[0])``, ``b = mean(ph)``.
    ``joint``: False fits each antenna chain on its own (fingerprinting, PhaseFi); True fits one
    slope per sample shared by all chains (SpotFi's STO removal) and removes one common offset
    (the circular mean of the chains' offsets), which preserves the phase differences between
    antennas that carry angle-of-arrival information. A 1-D input is one chain. A NaN
    subcarrier makes its chain NaN (its whole sample with ``joint=True``): select or fill first.
    """
    if method not in ("lstsq", "endpoints"):
        raise ValueError(f"method must be 'lstsq' or 'endpoints', got {method!r}")
    ph = np.unwrap(np.asarray(ph, dtype=np.float64), axis=-1)
    n_sub = ph.shape[-1]
    k = np.arange(n_sub, dtype=np.float64) if subcarriers is None else np.asarray(subcarriers, dtype=np.float64)
    if k.shape != (n_sub,):
        raise ValueError(f"{k.size} subcarrier indices for {n_sub} subcarriers")
    if n_sub < 2 or k[-1] == k[0]:
        raise ValueError("need at least two distinct subcarrier indices to remove a linear phase trend")
    axes = tuple(range(1, ph.ndim - 1)) if joint and ph.ndim > 2 else ()
    if method == "lstsq":
        kc = k - k.mean()
        a = np.sum(ph * kc, axis=-1, keepdims=True) / np.sum(kc * kc)
    else:
        a = (ph[..., -1:] - ph[..., :1]) / (k[-1] - k[0])
    if axes:
        a = np.mean(a, axis=axes, keepdims=True)  # same k on every chain: the pooled slope
    out = ph - a * k
    offset = np.mean(out if method == "lstsq" else ph, axis=-1, keepdims=True)  # per chain
    out -= offset
    if axes:
        # Each chain's offset is known only modulo 2 pi (unwrapping starts from a wrapped angle), so
        # the shared offset is their circular mean and each chain keeps its wrapped deviation from it:
        # the inter-antenna phase differences survive and the output is free of 2 pi ambiguity.
        common = np.angle(np.sum(np.exp(1j * offset), axis=axes, keepdims=True))
        out += np.angle(np.exp(1j * (offset - common)))
    return out


def conjugate_multiply(H, ref: int = 0, axis: int = 1) -> np.ndarray:
    """``H * conj(H_ref)``: each antenna times the conjugate of antenna ``ref`` along ``axis``.

    Offsets common to all antennas of one receiver (CFO, SFO, packet detection delay) cancel,
    leaving the phase differences between antennas. The ``ref`` slot becomes ``|H_ref|**2``.
    ``axis=1`` is the receive-antenna axis of a ``(N, n_rx, n_tx, n_sub)`` batch.

    References: X. Li et al., "IndoTrack: device-free indoor human tracking with commodity
    Wi-Fi", Proc. ACM IMWUT 1(3), 2017. https://doi.org/10.1145/3130940
    """
    H = np.asarray(H)
    return H * np.conj(np.take(H, [ref], axis=axis))


def csi_ratio(H, ref: int = 0, axis: int = 1) -> np.ndarray:
    """``H / H_ref`` along ``axis``: cancels common phase offsets and common amplitude noise
    (automatic gain control). Division by a zero reference is NaN.

    References: Y. Zeng et al., "FarSense: pushing the range limit of WiFi-based respiration
    sensing with CSI ratio of two antennas", Proc. ACM IMWUT 3(3), 2019. https://doi.org/10.1145/3351279
    """
    H = np.asarray(H)
    den = np.take(H, [ref], axis=axis)
    den = np.broadcast_to(den, H.shape)
    out = np.full(np.broadcast_shapes(H.shape, den.shape), np.nan, dtype=np.result_type(H, np.complex64))
    np.divide(H, den, out=out, where=den != 0)
    return out


def _csi_input(X, who: str):
    if isinstance(X, ScanView):
        raise TypeError(f"{who} works on CSI arrays (N, ..., n_sub) or SampleTables, not on RSSI scans")
    return X.X if isinstance(X, SampleTable) else np.asarray(X)


def _with_meta(X, out: np.ndarray, **meta):
    if isinstance(X, SampleTable):
        return X.replace(X=out, meta={**X.meta, **meta})
    return out


class CSIAmplitude(Transform):
    """CSI amplitude ``|H|``, or ``20 log10 |H|`` dB with ``db=True`` (zero amplitude -> NaN).

    Tables change modality ``"csi"`` -> ``"csi_amp"`` (and ``units`` -> ``"dB"`` with
    ``db=True``); the shape is kept. Amplitude is the most common CSI fingerprint feature
    (e.g. DeepFi). A real-valued input is an amplitude already: it passes through
    unchanged (values are not rectified, so a dB amplitude keeps its sign), and
    ``db=True`` converts it only if it is linear (a table whose ``units`` are ``"dB"`` or
    negative values are an error rather than a silent double conversion).

    References
        X. Wang, L. Gao, S. Mao, S. Pandey, "CSI-based fingerprinting for indoor localization: a deep
        learning approach", IEEE Transactions on Vehicular Technology 66(1):763-776, 2017.
        https://doi.org/10.1109/TVT.2016.2545523
    """

    def __init__(self, db: bool = False):
        self.db = db

    def transform(self, X):
        self._check_features(X)
        H = _csi_input(X, "CSIAmplitude")
        if H.dtype.kind == "c":
            out = amplitude(H, self.db)
        elif not self.db:
            out = F._as_float(H)  # already an amplitude (linear or dB): keep it as it is
        else:
            if isinstance(X, SampleTable) and X.meta.get("units") == "dB":
                raise ValueError("CSIAmplitude(db=True): the table's amplitudes are already in dB (meta['units'])")
            if np.any(H < 0):
                raise ValueError("CSIAmplitude(db=True): real input with negative values is not a linear "
                                 "amplitude (already in dB?); use CSIAmplitude() to keep it")
            out = amplitude(H, True)
        return _with_meta(X, out, modality="csi_amp", **({"units": "dB"} if self.db else {}))


class CSIPhaseSanitize(Transform):
    """Unwrap CSI phase across subcarriers and remove its linear trend (STO/CFO offsets).

    The raw CSI phase of commodity NICs is corrupted by a slope across subcarriers (sampling
    time offset, packet detection delay) and a constant (carrier frequency and phase offsets)
    that change from packet to packet. Following Sen et al. (2012) the phase is unwrapped along
    subcarriers and a line in the subcarrier index is subtracted; ``joint=True`` shares the
    slope between all antenna chains of a sample, as in SpotFi (Kotaru et al. 2015,
    Algorithm 1), which keeps the inter-antenna phase differences. See ``sanitize_phase``.

    Parameters
        subcarriers  subcarrier indices of the last axis; None = ``meta["subcarriers"]`` of a
                     table, else ``0..n_sub-1``.
        joint        one slope per sample (True) or a line per antenna chain (False).
        method       ``"lstsq"`` (least squares) or ``"endpoints"`` (Sen et al.'s slope from
                     the first and last subcarrier and offset = mean phase; also PhaseFi).
        output       ``"complex"``: ``|H| exp(j phase)`` (modality stays ``"csi"``) or
                     ``"phase"``: the sanitized phase in radians (modality ``"csi_phase"``).
    Input: complex CSI, or real phases in radians (then only ``output="phase"``). A zero CSI
    value (padding) has no phase: mark it NaN or drop it (SubcarrierSelect) first.

    Deviation: SpotFi subtracts only the fitted slope and keeps the offset; ``joint=True``
    also subtracts one offset common to all antennas (the circular mean of the per-chain
    offsets, which are defined only modulo 2 pi after unwrapping), so no phase difference,
    AoA or ToF estimate changes while the output becomes comparable across packets.

    References
        S. Sen, B. Radunovic, R. R. Choudhury, T. Minka, "You are facing the Mona Lisa: spot
        localization using PHY layer information", ACM MobiSys 2012, pp. 183-196.
        https://doi.org/10.1145/2307636.2307654
        M. Kotaru, K. Joshi, D. Bharadia, S. Katti, "SpotFi: decimeter level localization using WiFi",
        ACM SIGCOMM 2015, pp. 269-282. https://doi.org/10.1145/2785956.2787487
        X. Wang, L. Gao, S. Mao, "CSI phase fingerprinting for indoor localization with a deep learning
        approach", IEEE Internet of Things Journal 3(6):1113-1123, 2016. https://doi.org/10.1109/JIOT.2016.2558659
    """

    def __init__(self, subcarriers=None, joint: bool = False, method: str = "lstsq", output: str = "complex"):
        self.subcarriers = subcarriers
        self.joint = joint
        self.method = method
        self.output = output

    def transform(self, X):
        self._check_features(X)
        if self.output not in ("complex", "phase"):
            raise ValueError(f"output must be 'complex' or 'phase', got {self.output!r}")
        H = _csi_input(X, "CSIPhaseSanitize")
        k = self.subcarriers
        if k is None and isinstance(X, SampleTable):
            k = X.meta.get("subcarriers")
        is_complex = H.dtype.kind == "c"
        if not is_complex and self.output == "complex":
            raise ValueError("real input is read as phase in radians; it has no amplitude, so use output='phase'")
        ph = sanitize_phase(np.angle(H) if is_complex else H, k, self.joint, self.method)
        if self.output == "phase":
            real = np.float32 if H.dtype in (np.complex64, np.float32) else np.float64
            return _with_meta(X, ph.astype(real), modality="csi_phase", units="rad")
        out = (np.abs(H) * np.exp(1j * ph)).astype(H.dtype)
        return _with_meta(X, out, modality="csi")


class SubcarrierSelect(Transform):
    """Keep the subcarriers at positions ``indices`` of the last axis, in the given order.

    E.g. drop edge or pilot subcarriers, or decimate (``np.arange(0, 56, 2)``). Tables keep
    ``meta["subcarriers"]`` and ``meta["feature_names"]`` aligned when they describe the last
    axis. Works for complex CSI, amplitudes and phases. Stateless. ``indices`` is a list
    or an integer array of positions (negative ones count from the end), or a boolean
    mask with one entry per subcarrier.

    References
        D. Halperin, W. Hu, A. Sheth, D. Wetherall, "Tool release: gathering 802.11n traces with channel
        state information", ACM SIGCOMM Computer Communication Review 41(1):53, 2011 (the 30 grouped
        subcarriers of ``INTEL5300_SUBCARRIERS_20MHZ``). https://doi.org/10.1145/1925861.1925870
    """

    def __init__(self, indices):
        self.indices = indices

    def transform(self, X):
        self._check_features(X)
        x = _csi_input(X, "SubcarrierSelect")
        n = x.shape[-1]
        idx = np.asarray(self.indices)
        if idx.dtype == bool:
            if idx.shape != (n,):
                raise ValueError(f"a boolean mask needs one entry per subcarrier ({n}), got shape {idx.shape}")
            idx = np.flatnonzero(idx)
        elif idx.size and idx.dtype.kind not in "iu":
            raise TypeError(f"indices must be integer positions or a boolean mask, got dtype {idx.dtype}")
        idx = idx.astype(np.intp).reshape(-1)
        if idx.size == 0 or np.any(idx >= n) or np.any(idx < -n):
            raise ValueError(f"indices must be non-empty positions in [-{n}, {n}), got {self.indices!r}")
        return _take_columns(X, np.where(idx < 0, idx + n, idx))
