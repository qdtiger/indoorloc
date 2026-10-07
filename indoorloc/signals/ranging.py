"""Ranging helpers: times and signal strength to metres, bias calibration, simple NLOS flags.

Pure numpy functions over ``ranges`` tables (``(N, n_anchor)`` metres, CONTRACTS.md) and
the raw measurements they come from. Position estimation from ranges (trilateration,
TDoA, ...) is an L3 method; these helpers only prepare and screen the measurements.

Contents
    conversions   ``toa_to_distance``, ``rtt_to_distance``, ``rssi_to_distance``, ``distance_to_rssi``
    two-way       ``twr_tof`` (single-sided), ``ds_twr_tof`` (asymmetric double-sided, clock-drift
                  tolerant), ``twr_distance``, ``ds_twr_distance`` (UWB, or ultrasound with ``speed=``)
    ultrasound    ``speed_of_sound`` (temperature), ``ultrasound_tof_to_distance``,
                  ``rf_ultrasound_distance`` (RF + ultrasound arrival difference, Cricket / Active Bat)
    bias          ``fit_range_bias``, ``correct_range_bias`` (linear error model per anchor)
    NLOS flags    ``nlos_flags_power`` (first-path power), ``nlos_flags_std`` (range jitter),
                  ``nlos_flags_geometry`` (triangle inequality between anchors)
"""
from __future__ import annotations

import numpy as np

SPEED_OF_LIGHT = 299_792_458.0  # m/s, exact (SI definition of the metre)
SPEED_OF_SOUND_0C = 331.3       # m/s, dry air at 0 degC and sea-level pressure (ideal-gas value)
ZERO_CELSIUS = 273.15           # K


def toa_to_distance(toa_s) -> np.ndarray:
    """One-way time of flight (s) -> distance (m): ``c * t``."""
    return SPEED_OF_LIGHT * np.asarray(toa_s, dtype=np.float64)


def rtt_to_distance(rtt_s, turnaround_s: float = 0.0) -> np.ndarray:
    """Round-trip time (s) -> distance (m): ``c * (rtt - turnaround) / 2``.

    ``turnaround_s`` is the responder's known processing delay (two-way ranging); IEEE
    802.11mc fine timing measurement already reports RTT without it (``turnaround_s=0``).
    """
    return SPEED_OF_LIGHT * (np.asarray(rtt_s, dtype=np.float64) - float(turnaround_s)) / 2.0


def twr_tof(round_s, reply_s) -> np.ndarray:
    """Single-sided two-way ranging: time of flight ``(T_round - T_reply) / 2`` seconds.

    The initiator measures ``T_round`` (poll sent to response received) with its clock, the
    responder ``T_reply`` (poll received to response sent) with its own. A relative clock
    error ``e`` of the responder biases the result by about ``e * T_reply / 2``: 20 ppm on a
    300 us reply time is 3 ns, i.e. 0.9 m at the speed of light; use ``ds_twr_tof`` then.

    References
        D. Neirynck, E. Luk, M. McLaughlin, "An alternative double-sided two-way ranging method",
        13th Workshop on Positioning, Navigation and Communications (WPNC), 2016.
        https://doi.org/10.1109/WPNC.2016.7822844 (single- and double-sided error analysis)
    """
    return (np.asarray(round_s, dtype=np.float64) - np.asarray(reply_s, dtype=np.float64)) / 2.0


def ds_twr_tof(round1_s, reply1_s, round2_s, reply2_s) -> np.ndarray:
    """Asymmetric double-sided two-way ranging: time of flight in seconds.

    ``tof = (Tround1 Tround2 - Treply1 Treply2) / (Tround1 + Tround2 + Treply1 + Treply2)``,
    where exchange 1 is poll/response (``Tround1`` on the initiator, ``Treply1`` on the
    responder) and exchange 2 is response/final (``Tround2`` on the responder, ``Treply2`` on
    the initiator). Exact without clock errors, and with clock errors ``e_A``, ``e_B`` the
    error is only about ``tof * (e_A + e_B) / 2`` whatever the reply times, unlike
    ``twr_tof``; the reply times need not be equal.

    References
        D. Neirynck, E. Luk, M. McLaughlin, "An alternative double-sided two-way ranging method",
        13th Workshop on Positioning, Navigation and Communications (WPNC), 2016.
        https://doi.org/10.1109/WPNC.2016.7822844
    """
    r1, p1, r2, p2 = (np.asarray(v, dtype=np.float64) for v in (round1_s, reply1_s, round2_s, reply2_s))
    return (r1 * r2 - p1 * p2) / (r1 + r2 + p1 + p2)


def twr_distance(round_s, reply_s, speed: float = SPEED_OF_LIGHT) -> np.ndarray:
    """``speed * twr_tof(round, reply)`` metres (single-sided two-way ranging)."""
    return float(speed) * twr_tof(round_s, reply_s)


def ds_twr_distance(round1_s, reply1_s, round2_s, reply2_s, speed: float = SPEED_OF_LIGHT) -> np.ndarray:
    """``speed * ds_twr_tof(...)`` metres (asymmetric double-sided two-way ranging)."""
    return float(speed) * ds_twr_tof(round1_s, reply1_s, round2_s, reply2_s)


def speed_of_sound(temperature_c=20.0) -> np.ndarray:
    """Speed of sound in dry air, ``331.3 sqrt(1 + T / 273.15)`` m/s at ``T`` degrees Celsius.

    The ideal-gas law (``c ~ sqrt(T_kelvin)``) scaled to 331.3 m/s at 0 degC: 343.2 m/s at
    20 degC. A 1 degC error changes ranges by ``1 / (2 T_kelvin)``, about 0.17 % at room
    temperature (1.7 cm at 10 m), which is why ultrasound positioning systems measure the
    temperature. Humidity raises the speed
    by a few tenths of a percent at room temperature (Cramer 1993); it is not modelled here.

    References
        L. E. Kinsler, A. R. Frey, A. B. Coppens, J. V. Sanders, "Fundamentals of Acoustics", 4th ed.,
        Wiley, 2000 (speed of sound in an ideal gas). ISBN 978-0-471-84789-2.
        O. Cramer, "The variation of the specific heat ratio and the speed of sound in air with
        temperature, pressure, humidity, and CO2 concentration", Journal of the Acoustical Society
        of America 93(5):2510-2516, 1993. https://doi.org/10.1121/1.405827
    """
    t = np.asarray(temperature_c, dtype=np.float64)
    if np.any(t <= -ZERO_CELSIUS):
        raise ValueError(f"temperature must be above absolute zero (-273.15 degC), got {temperature_c}")
    return SPEED_OF_SOUND_0C * np.sqrt(1.0 + t / ZERO_CELSIUS)


def ultrasound_tof_to_distance(tof_s, temperature_c=20.0) -> np.ndarray:
    """One-way ultrasonic time of flight (s) -> distance (m): ``speed_of_sound(T) * t``.

    References
        A. Harter, A. Hopper, P. Steggles, A. Ward, P. Webster, "The anatomy of a context-aware
        application", ACM MobiCom 1999, pp. 59-68 (Active Bat). https://doi.org/10.1145/313451.313457
    """
    return speed_of_sound(temperature_c) * np.asarray(tof_s, dtype=np.float64)


def rf_ultrasound_distance(delay_s, temperature_c=20.0, *, exact: bool = True) -> np.ndarray:
    """Distance from the arrival-time difference of a simultaneous RF and ultrasound pulse.

    The RF pulse arrives after ``d / c_light``, the ultrasound after ``d / c_sound``, so
    ``delay = d (1 / c_sound - 1 / c_light)`` and ``d = delay / (1 / c_sound - 1 / c_light)``
    (``exact=True``). Cricket neglects the RF flight time, ``d = c_sound * delay``
    (``exact=False``): a relative error of ``c_sound / c_light``, about 1e-6.

    References
        N. B. Priyantha, A. Chakraborty, H. Balakrishnan, "The Cricket location-support system",
        ACM MobiCom 2000, pp. 32-43. https://doi.org/10.1145/345910.345917
    """
    c = speed_of_sound(temperature_c)
    dt = np.asarray(delay_s, dtype=np.float64)
    return dt / (1.0 / c - 1.0 / SPEED_OF_LIGHT) if exact else c * dt


def _check_path_loss(n: float, d0_m: float) -> None:
    if not float(n) > 0:
        raise ValueError(f"the path loss exponent n must be positive, got {n}")
    if not float(d0_m) > 0:
        raise ValueError(f"the reference distance d0_m must be positive, got {d0_m}")


def distance_to_rssi(distance_m, p0_dbm: float, n: float = 2.0, d0_m: float = 1.0) -> np.ndarray:
    """Log-distance path loss model: ``p0 - 10 n log10(d / d0)`` dBm (``p0`` at ``d0``)."""
    _check_path_loss(n, d0_m)
    d = np.asarray(distance_m, dtype=np.float64)
    return float(p0_dbm) - 10.0 * float(n) * np.log10(d / float(d0_m))


def rssi_to_distance(rssi_dbm, p0_dbm: float, n: float = 2.0, d0_m: float = 1.0) -> np.ndarray:
    """Inverse of the log-distance model: ``d0 * 10 ** ((p0 - rssi) / (10 n))`` metres.

    ``p0_dbm`` is the RSSI at the reference distance ``d0_m`` (an iBeacon's "measured
    power" is ``p0`` at 1 m) and ``n`` the path loss exponent (2 in free space; Rappaport
    lists 1.6-1.8 for line of sight and 4-6 for obstructed paths inside buildings). NaN
    stays NaN. Shadowing makes the result a rough estimate: a 4 dB error at n = 2 is a
    factor of 1.6 in distance.

    References
        T. S. Rappaport, "Wireless Communications: Principles and Practice", 2nd ed., Prentice Hall,
        2002, sec. 4.9 and table 4.2.
        S. Y. Seidel, T. S. Rappaport, "914 MHz path loss prediction models for indoor wireless
        communications in multifloored buildings", IEEE Transactions on Antennas and Propagation
        40(2):207-217, 1992. https://doi.org/10.1109/8.127405
    """
    _check_path_loss(n, d0_m)
    r = np.asarray(rssi_dbm, dtype=np.float64)
    return float(d0_m) * np.power(10.0, (float(p0_dbm) - r) / (10.0 * float(n)))


def fit_range_bias(measured, true, per_anchor: bool = True):
    """Least-squares linear error model ``measured ~ scale * true + offset``.

    ``measured`` and ``true`` are ``(N,)`` or ``(N, n_anchor)`` ranges in metres; NaN pairs
    are ignored. ``per_anchor=True`` fits each column on its own (anchor-specific antenna
    delay), otherwise one model for all. Returns ``(scale, offset)`` as floats or ``(A,)``
    arrays; undo it with ``correct_range_bias``. With ``scale`` fixed to 1 this reduces to
    the constant offset that ToF ranging (UWB antenna delay, 802.11mc FTM) needs per device.

    References
        M. Ibrahim et al., "Verification: accuracy evaluation of WiFi fine time measurements on an
        open platform", ACM MobiCom 2018, pp. 417-427. https://doi.org/10.1145/3241539.3241555
    """
    m = np.asarray(measured, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    if m.shape != t.shape:
        raise ValueError(f"measured {m.shape} and true {t.shape} must have the same shape")
    if not per_anchor or m.ndim == 1:
        m, t = m.reshape(-1, 1), t.reshape(-1, 1)
        scalar = True
    else:
        scalar = False
    scale = np.empty(m.shape[1])
    offset = np.empty(m.shape[1])
    for j in range(m.shape[1]):
        ok = ~np.isnan(m[:, j]) & ~np.isnan(t[:, j])
        tj, mj = t[ok, j], m[ok, j]
        if len(tj) < 2 or np.all(tj == tj[0]):
            raise ValueError(f"column {j}: need at least two distinct true ranges to fit scale and offset")
        tc = tj - tj.mean()
        scale[j] = np.sum(tc * (mj - mj.mean())) / np.sum(tc * tc)
        offset[j] = mj.mean() - scale[j] * tj.mean()
    return (float(scale[0]), float(offset[0])) if scalar else (scale, offset)


def correct_range_bias(measured, scale, offset) -> np.ndarray:
    """``(measured - offset) / scale``: the ranges with the fitted linear bias removed."""
    return (np.asarray(measured, dtype=np.float64) - np.asarray(offset)) / np.asarray(scale)


def nlos_flags_power(rx_power_dbm, first_path_power_dbm, threshold_db: float) -> np.ndarray:
    """Flag NLOS when the total received power exceeds the first-path power by more than
    ``threshold_db``.

    Under line of sight most energy arrives in the first path; behind obstacles the first
    path is attenuated relative to later reflections. UWB radios such as the DW1000 report
    both powers. The threshold depends on hardware and site: calibrate it on labelled
    measurements. NaN inputs give False.

    References
        Decawave, "APS006 Part 3 application note: DW1000 metrics for estimation of non line of sight
        operating conditions", version 1.1, 2016.
        M. Kolakowski, J. Modelski, "First path component power based NLOS mitigation in UWB
        positioning system", TELFOR 2017. https://doi.org/10.1109/TELFOR.2017.8249313
    """
    rx = np.asarray(rx_power_dbm, dtype=np.float64)
    fp = np.asarray(first_path_power_dbm, dtype=np.float64)
    return (rx - fp) > float(threshold_db)


def nlos_flags_std(ranges, window: int, threshold_m: float, degree: int = 1) -> np.ndarray:
    """Flag NLOS when the range jitter over the last ``window`` samples exceeds ``threshold_m``.

    Wylie and Holtzman (1996): NLOS ranges scatter far more than the LOS measurement noise.
    For each time step ``t`` (rows of ``ranges``, ``(T,)`` or ``(T, n_anchor)``, one
    trajectory in time order) a polynomial of ``degree`` 0 or 1 in time is fitted to the
    samples ``t - window + 1 .. t`` (removing the motion of the target) and the residual
    standard deviation is compared with ``threshold_m``, e.g. 2-3 times the LOS ranging
    noise std. Causal; the first ``window - 1`` rows and windows with NaN are not flagged.

    References
        M. P. Wylie, J. Holtzman, "The non-line of sight problem in mobile location estimation",
        Proc. IEEE ICUPC 1996, vol. 2, pp. 827-831. https://doi.org/10.1109/ICUPC.1996.562692
    """
    r = np.asarray(ranges, dtype=np.float64)
    w = int(window)
    if degree not in (0, 1):
        raise ValueError(f"degree must be 0 or 1, got {degree}")
    if w < degree + 2:
        raise ValueError(f"window must be at least {degree + 2} samples to estimate a residual std")
    flags = np.zeros(r.shape, dtype=bool)
    if len(r) < w:
        return flags
    win = np.lib.stride_tricks.sliding_window_view(r, w, axis=0)  # (T - w + 1, ..., w)
    resid = win - win.mean(axis=-1, keepdims=True)
    if degree == 1:
        tc = np.arange(w) - (w - 1) / 2.0
        slope = np.sum(resid * tc, axis=-1, keepdims=True) / np.sum(tc * tc)
        resid = resid - slope * tc
    std = np.sqrt(np.sum(resid * resid, axis=-1) / (w - degree - 1))
    flags[w - 1:] = std > float(threshold_m)
    return flags


def nlos_flags_geometry(ranges, anchors, tol_m: float = 0.0) -> np.ndarray:
    """Flag ranges that violate the triangle inequality between anchors.

    For LOS ranges ``d_i``, ``d_j`` from one point to anchors ``a_i``, ``a_j``,
    ``|d_i - d_j| <= ||a_i - a_j||``. NLOS only lengthens a range, so when
    ``d_i - d_j > ||a_i - a_j|| + tol_m`` the longer range ``d_i`` is flagged. No position
    is needed; biases too small to break the inequality go undetected. ``ranges``
    ``(N, A)`` or ``(A,)``, ``anchors`` ``(A, D)``; returns bool of the shape of ``ranges``.
    """
    d = np.asarray(ranges, dtype=np.float64)
    single = d.ndim == 1
    d = d[None] if single else d
    a = np.asarray(anchors, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != d.shape[1]:
        raise ValueError(f"anchors must be (A, D) with A={d.shape[1]}, got shape {a.shape}")
    baseline = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1)  # (A, A)
    excess = d[:, :, None] - d[:, None, :] - baseline - float(tol_m)  # (N, A, A)
    flags = np.any(excess > 0, axis=2)
    return flags[0] if single else flags
