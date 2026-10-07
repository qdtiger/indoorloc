"""Visible light positioning: the Lambertian line-of-sight channel, its inversion and receiver noise.

Pure numpy functions (no transform: received power is already the measurement).

    emitter        ``lambertian_order``, ``half_power_angle``
    receiver       ``concentrator_gain``
    channel        ``channel_gain``, ``received_power`` (any LED and receiver orientation)
    inversion      ``power_to_distance``, ``distance_to_power`` (LED facing down, receiver facing up)
    noise          ``noise_variance`` (shot + thermal noise of a PIN photodiode receiver)

The line-of-sight DC gain of an LED with Lambertian order ``m`` seen by a photodiode of area
``A`` at distance ``d`` is (Kahn & Barry 1997; Komine & Nakagawa 2004)::

    H = (m + 1) A / (2 pi d^2) cos^m(phi) Ts g(psi) cos(psi)    for 0 <= psi <= FOV, else 0

with ``phi`` the irradiance angle (from the LED axis), ``psi`` the incidence angle (from the
receiver axis), ``Ts`` the optical filter gain, ``g`` the concentrator gain (``n^2 / sin^2 FOV``
for an ideal non-imaging concentrator of refractive index ``n``) and
``m = -ln 2 / ln cos(Phi_1/2)`` for a half-power semi-angle ``Phi_1/2`` (60 degrees gives
``m = 1``). The received optical power is ``P_r = P_t H``. Only the line-of-sight path is
modelled: wall reflections add power, most near walls and corners, and bias RSS ranging
there (Gu et al. 2016). Units: metres, radians, watts, square metres.

For an LED facing straight down at height ``h`` above a receiver facing straight up,
``cos phi = cos psi = h / d``, so ``P_r = C h^(m+1) / d^(m+3)`` with
``C = P_t (m + 1) A Ts g / (2 pi)``: the received power fixes the distance,
``d = (C h^(m+1) / P_r)^(1 / (m + 3))`` (the RSS ranging of VLP systems, Zhuang et al. 2018).

References
    J. M. Kahn, J. R. Barry, "Wireless infrared communications", Proceedings of the IEEE 85(2):265-298,
    1997. https://doi.org/10.1109/5.554222
    T. Komine, M. Nakagawa, "Fundamental analysis for visible-light communication system using LED
    lights", IEEE Transactions on Consumer Electronics 50(1):100-107, 2004.
    https://doi.org/10.1109/TCE.2004.1277847
    Y. Zhuang, L. Hua, L. Qi, J. Yang, P. Cao, Y. Cao, Y. Wu, J. Thompson, H. Haas, "A survey of
    positioning systems using visible LED lights", IEEE Communications Surveys & Tutorials
    20(3):1963-1988, 2018. https://doi.org/10.1109/COMST.2018.2806558
    W. Gu, M. Aminikashani, P. Deng, M. Kavehrad, "Impact of multipath reflections on the performance
    of indoor visible light positioning systems", Journal of Lightwave Technology 34(10):2578-2587,
    2016. https://doi.org/10.1109/JLT.2016.2541659
"""
from __future__ import annotations

import numpy as np

ELEMENTARY_CHARGE = 1.602176634e-19  # C, exact (SI 2019)
BOLTZMANN = 1.380649e-23             # J/K, exact (SI 2019)


def lambertian_order(half_power_angle) -> np.ndarray:
    """``m = -ln 2 / ln cos(Phi_1/2)`` for a half-power semi-angle in radians, ``0 < Phi < pi/2``."""
    phi = np.asarray(half_power_angle, dtype=np.float64)
    if np.any(~(phi > 0) | ~(phi < np.pi / 2)):
        raise ValueError(f"the half-power angle must be in (0, pi/2) radians, got {half_power_angle}")
    return -np.log(2.0) / np.log(np.cos(phi))


def half_power_angle(order) -> np.ndarray:
    """Inverse of ``lambertian_order``: ``arccos(2^(-1/m))`` radians."""
    m = np.asarray(order, dtype=np.float64)
    if np.any(~(m > 0)):
        raise ValueError(f"the Lambertian order must be positive, got {order}")
    return np.arccos(np.power(2.0, -1.0 / m))


def concentrator_gain(refractive_index: float, fov: float) -> float:
    """Gain ``n^2 / sin^2(FOV)`` of an ideal non-imaging concentrator (Kahn & Barry 1997)."""
    n, f = float(refractive_index), float(fov)
    if not n > 0 or not 0 < f <= np.pi / 2:
        raise ValueError(f"need refractive_index > 0 and 0 < fov <= pi/2, got {refractive_index}, {fov}")
    return n * n / np.sin(f) ** 2


def _unit(v, n: int, name: str) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    a = np.broadcast_to(a, (n, 3)) if a.ndim == 1 else a
    if a.shape != (n, 3):
        raise ValueError(f"{name} must be (3,) or ({n}, 3), got shape {np.shape(v)}")
    norm = np.linalg.norm(a, axis=1, keepdims=True)
    if not np.all(norm > 0):
        raise ValueError(f"{name} must be non-zero vectors")
    return a / norm


def _points(x, name: str) -> np.ndarray:
    a = np.atleast_2d(np.asarray(x, dtype=np.float64))
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"{name} must be (N, 3) positions in metres, got shape {np.shape(x)}")
    return a


def channel_gain(positions, leds, *, order=1.0, area: float = 1e-4, fov: float = np.pi / 2,
                 led_normals=(0.0, 0.0, -1.0), receiver_normals=(0.0, 0.0, 1.0), filter_gain: float = 1.0,
                 concentrator_gain: float = 1.0) -> np.ndarray:
    """Line-of-sight DC gain ``H`` ``(N, A)`` from ``A`` LEDs to receivers at ``positions`` ``(N, 3)``.

    ``leds`` ``(A, 3)``; ``order`` scalar or ``(A,)``; ``led_normals`` ``(3,)`` or ``(A, 3)``
    (emission axes, default straight down); ``receiver_normals`` ``(3,)`` or ``(N, 3)``
    (default straight up); ``area`` in m^2, ``fov`` the receiver's field-of-view semi-angle
    in radians. ``H = 0`` behind the LED (``phi >= 90 deg``), behind the receiver and outside
    its field of view (``psi > fov``). See the module docstring for the formula.
    """
    p, a = _points(positions, "positions"), _points(leds, "leds")
    nt = _unit(led_normals, len(a), "led_normals")
    nr = _unit(receiver_normals, len(p), "receiver_normals")
    m = np.broadcast_to(np.asarray(order, dtype=np.float64), (len(a),))
    if np.any(~(m >= 0)):
        raise ValueError(f"the Lambertian order must be non-negative, got {order}")
    v = p[:, None, :] - a[None]                                   # LED -> receiver
    d = np.sqrt(np.sum(v * v, axis=2))
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_phi = np.einsum("nad,ad->na", v, nt) / d
        cos_psi = -np.einsum("nad,nd->na", v, nr) / d
    seen = (d > 0) & (cos_phi > 0) & (cos_psi > 0) & (cos_psi >= np.cos(float(fov)) - 1e-15)
    cp, cs = np.where(seen, cos_phi, 1.0), np.where(seen, cos_psi, 0.0)
    dd = np.where(seen, d, 1.0)
    H = (m + 1.0) * float(area) / (2.0 * np.pi * dd * dd) * cp ** m * float(filter_gain) * float(concentrator_gain) * cs
    return np.where(seen, H, 0.0)


def received_power(positions, leds, tx_power=1.0, **channel) -> np.ndarray:
    """Received optical power ``P_t H`` ``(N, A)`` in watts; ``tx_power`` scalar or ``(A,)``;
    ``channel`` = the keyword arguments of ``channel_gain``."""
    return np.asarray(tx_power, dtype=np.float64) * channel_gain(positions, leds, **channel)


def _vertical_constant(tx_power, order, area, filter_gain, concentrator_gain):
    m = np.asarray(order, dtype=np.float64)
    if np.any(~(m >= 0)) or not float(area) > 0:
        raise ValueError("need order >= 0 and area > 0")
    return m, np.asarray(tx_power, dtype=np.float64) * (m + 1.0) * float(area) * float(filter_gain) \
        * float(concentrator_gain) / (2.0 * np.pi)


def distance_to_power(distance, height, *, tx_power=1.0, order=1.0, area: float = 1e-4, filter_gain: float = 1.0,
                      concentrator_gain: float = 1.0) -> np.ndarray:
    """``P_r = C h^(m+1) / d^(m+3)`` for an LED facing down ``height`` metres above a receiver
    facing up, at 3-D ``distance`` ``d >= h`` (inside the field of view; the FOV cut-off is
    the caller's concern)."""
    m, C = _vertical_constant(tx_power, order, area, filter_gain, concentrator_gain)
    d = np.asarray(distance, dtype=np.float64)
    h = np.asarray(height, dtype=np.float64)
    return C * h ** (m + 1.0) / d ** (m + 3.0)


def power_to_distance(power, height, *, tx_power=1.0, order=1.0, area: float = 1e-4, filter_gain: float = 1.0,
                      concentrator_gain: float = 1.0, horizontal: bool = False) -> np.ndarray:
    """Distance from received power for an LED facing down ``height`` metres above a receiver
    facing up: ``d = (C h^(m+1) / P_r)^(1/(m+3))``, ``C = P_t (m+1) A Ts g / (2 pi)``.

    ``horizontal=True`` returns ``sqrt(d^2 - h^2)`` instead (0 when noise pushes ``d`` below
    ``h``), the radius used by 2-D trilateration with a known receiver height. ``power`` in
    watts, broadcast against ``height``, ``tx_power`` and ``order`` (e.g. ``(N, A)`` powers
    with ``(A,)`` heights); ``power <= 0`` gives inf, NaN stays NaN. Tilting the receiver or
    the LED breaks the ``cos phi = cos psi = h / d`` identity this inversion relies on;
    ``methods.vlc.LambertianLocalizer`` fits the full model instead.

    References
        W. Zhang, M. I. S. Chowdhury, M. Kavehrad, "Asynchronous indoor positioning system based on
        visible light communications", Optical Engineering 53(4):045105, 2014.
        https://doi.org/10.1117/1.OE.53.4.045105
    """
    m, C = _vertical_constant(tx_power, order, area, filter_gain, concentrator_gain)
    P = np.asarray(power, dtype=np.float64)
    h = np.asarray(height, dtype=np.float64)
    if np.any(h <= 0):
        raise ValueError("the LED must be above the receiver (height > 0)")
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(P > 0, (C * h ** (m + 1.0) / np.where(P > 0, P, 1.0)) ** (1.0 / (m + 3.0)),
                     np.where(np.isnan(P), np.nan, np.inf))
    if horizontal:
        with np.errstate(invalid="ignore"):
            return np.sqrt(np.maximum(d * d - h * h, 0.0))
    return d


def noise_variance(power, *, responsivity: float = 0.54, bandwidth: float = 100e6,
                   background_current: float = 5100e-6, area: float = 1e-4, temperature: float = 295.0,
                   open_loop_gain: float = 10.0, capacitance_per_area: float = 1.12e-6,
                   channel_noise_factor: float = 1.5, transconductance: float = 30e-3,
                   noise_bandwidth_factors=(0.562, 0.0868)) -> np.ndarray:
    """Receiver noise variance (A^2) at received optical ``power`` (W): shot plus thermal noise.

    Komine & Nakagawa (2004) for a PIN photodiode with a FET preamplifier::

        shot    = 2 q R P B + 2 q I_bg I2 B
        thermal = 8 pi k T eta A I2 B^2 / G + 16 pi^2 k T Gamma eta^2 A^2 I3 B^3 / g_m

    (``R`` responsivity A/W, ``B`` noise bandwidth Hz, ``I_bg`` background photocurrent A,
    ``eta`` capacitance per area F/m^2, ``G`` open-loop voltage gain, ``Gamma`` FET channel
    noise factor, ``g_m`` FET transconductance S, ``I2``/``I3`` noise-bandwidth factors).
    The photocurrent is ``R P``, so a reading of optical power has standard deviation
    ``sqrt(noise_variance(P)) / R`` watts. The defaults are the receiver parameters usually
    quoted from Komine & Nakagawa's Table I (1 cm^2 detector, 100 MHz bandwidth,
    5100 uA background current from direct sunlight, 112 pF/cm^2); indoor ambient light
    gives a much smaller background current. Positioning receivers average over many
    samples, which lowers the effective bandwidth ``B``.

    References
        T. Komine, M. Nakagawa, "Fundamental analysis for visible-light communication system using
        LED lights", IEEE Transactions on Consumer Electronics 50(1):100-107, 2004.
        https://doi.org/10.1109/TCE.2004.1277847
    """
    P = np.maximum(np.asarray(power, dtype=np.float64), 0.0)
    q, k = ELEMENTARY_CHARGE, BOLTZMANN
    B, T, eta = float(bandwidth), float(temperature), float(capacitance_per_area)
    i2, i3 = (float(v) for v in noise_bandwidth_factors)
    shot = 2.0 * q * float(responsivity) * P * B + 2.0 * q * float(background_current) * i2 * B
    thermal = (8.0 * np.pi * k * T * eta * float(area) * i2 * B ** 2 / float(open_loop_gain)
               + 16.0 * np.pi ** 2 * k * T * float(channel_noise_factor) * eta ** 2 * float(area) ** 2 * i3 * B ** 3
               / float(transconductance))
    return shot + thermal


__all__ = ["BOLTZMANN", "ELEMENTARY_CHARGE", "channel_gain", "concentrator_gain", "distance_to_power",
           "half_power_angle", "lambertian_order", "noise_variance", "power_to_distance", "received_power"]
