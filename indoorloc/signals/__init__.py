"""L2: signal views, pure functions and fit/transform estimators (numpy only).

Transforms (sklearn transformer contract; each takes a scan ``(F,)``, a batch
``(N, ...)``, a SampleTable or a scan view and returns the same kind of object):

    missing / scaling   FillMissing, RSSINormalize, APFilter
    AP selection        APSelect
    representations     PositiveRepresentation, ExponentialRepresentation, PowedRepresentation
    robust filtering    HampelFilter
    augmentation        GaussianNoise, APDropout             (training data only)
    device calibration  DeviceCalibration
    CSI                 CSIAmplitude, CSIPhaseSanitize, SubcarrierSelect
    magnetometer        MagneticFeatures (imu -> ``magnetic`` table), MagnetometerCalibration
    composition         Compose

Views: WiFiSignal, BLESignal (one scan). Functions: ``functional`` (RSSI), ``csi``,
``ranging`` (ToA/RTT/two-way/ultrasound), ``imu``, ``magnetic`` (field components, heading,
ellipsoid calibration) and ``vlc`` (Lambertian channel, power <-> distance, receiver noise).
"""
from __future__ import annotations

from . import csi, functional, imu, magnetic, ranging, vlc
from .augment import APDropout, Augmentation, GaussianNoise
from .ble import BLESignal
from .calibration import DeviceCalibration
from .csi import CSIAmplitude, CSIPhaseSanitize, SubcarrierSelect
from .magnetic import MagneticFeatures, MagnetometerCalibration
from .transforms import (APFilter, APSelect, Compose, ExponentialRepresentation, FillMissing, HampelFilter,
                         PositiveRepresentation, PowedRepresentation, RSSINormalize, Transform)
from .wifi import WiFiSignal

__all__ = ["APDropout", "APFilter", "APSelect", "Augmentation", "BLESignal", "CSIAmplitude", "CSIPhaseSanitize",
           "Compose", "DeviceCalibration", "ExponentialRepresentation", "FillMissing", "GaussianNoise",
           "HampelFilter", "MagneticFeatures", "MagnetometerCalibration", "PositiveRepresentation",
           "PowedRepresentation", "RSSINormalize", "SubcarrierSelect", "Transform", "WiFiSignal", "csi", "functional",
           "imu", "magnetic", "ranging", "vlc"]
