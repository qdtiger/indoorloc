"""
VLC (Visible Light Communication) Signal Implementation

Provides VLC signal representations for indoor localization based on
LED-based visible light positioning systems.
"""
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import torch

from .base import BaseSignal, SignalMetadata
from ..registry import SIGNALS


@dataclass
class LEDTransmitter:
    """Represents a single LED transmitter measurement.

    Attributes:
        led_id: Unique identifier for the LED transmitter.
        position: 3D position (x, y, z) in meters.
        received_power: Received optical power (arbitrary units or lux).
        phase_difference: Phase difference measurement (for angle estimation).
        angle_of_arrival: Estimated angle of arrival in radians.
        tx_power: Transmitted optical power.
        timestamp: Measurement timestamp.
    """
    led_id: str
    position: Optional[Tuple[float, float, float]] = None
    received_power: float = 0.0
    phase_difference: Optional[float] = None
    angle_of_arrival: Optional[float] = None
    tx_power: Optional[float] = None
    timestamp: Optional[float] = None


@SIGNALS.register_module()
class VLCSignal(BaseSignal):
    """VLC signal for indoor localization.

    Represents optical signals from LED transmitters for visible light
    positioning (VLP) systems.

    Args:
        led_ids: List of LED transmitter identifiers.
        received_power: Array of received optical power values.
        led_positions: Dictionary mapping LED IDs to (x, y, z) positions.
        phase_difference: Phase difference measurements (optional).
        aoa_measurements: Angle-of-Arrival measurements (optional).
        leds: List of LEDTransmitter objects (alternative to arrays).
        timestamps: Timestamps for each measurement.
        metadata: Signal metadata.

    Example:
        >>> # Received signal strength from 4 LEDs
        >>> led_ids = ['LED1', 'LED2', 'LED3', 'LED4']
        >>> received_power = np.array([0.8, 0.5, 0.3, 0.6])  # Normalized
        >>> led_pos = {
        ...     'LED1': (0.0, 0.0, 3.0),
        ...     'LED2': (5.0, 0.0, 3.0),
        ...     'LED3': (5.0, 5.0, 3.0),
        ...     'LED4': (0.0, 5.0, 3.0),
        ... }
        >>> signal = VLCSignal(
        ...     led_ids=led_ids,
        ...     received_power=received_power,
        ...     led_positions=led_pos
        ... )
    """

    # Typical range for normalized received power
    MIN_POWER = 0.0
    MAX_POWER = 1.0

    def __init__(
        self,
        led_ids: Optional[List[str]] = None,
        received_power: Optional[np.ndarray] = None,
        led_positions: Optional[Dict[str, Tuple[float, float, float]]] = None,
        phase_difference: Optional[np.ndarray] = None,
        aoa_measurements: Optional[np.ndarray] = None,
        leds: Optional[List[LEDTransmitter]] = None,
        timestamps: Optional[np.ndarray] = None,
        metadata: Optional[SignalMetadata] = None,
        **kwargs
    ):
        # Store LED information
        self.led_positions = led_positions or {}
        self.timestamps = timestamps
        self.phase_difference = phase_difference
        self.aoa_measurements = aoa_measurements

        # Initialize from either arrays or LED list
        if leds is not None:
            self.leds = leds
            self.led_ids = [led.led_id for led in leds]
            self.received_power = np.array(
                [led.received_power for led in leds],
                dtype=np.float32
            )
            if led_positions is None and any(led.position for led in leds):
                self.led_positions = {
                    led.led_id: led.position
                    for led in leds if led.position
                }
        elif led_ids is not None and received_power is not None:
            self.led_ids = led_ids
            self.received_power = np.array(received_power, dtype=np.float32)
            self.leds = [
                LEDTransmitter(
                    led_id=led_id,
                    received_power=power,
                    position=self.led_positions.get(led_id)
                )
                for led_id, power in zip(led_ids, received_power)
            ]
        else:
            self.led_ids = []
            self.received_power = np.array([], dtype=np.float32)
            self.leds = []

        # Pass data to parent class
        super().__init__(self.received_power, metadata)

    @property
    def signal_type(self) -> str:
        """Return the signal type identifier."""
        return 'vlc'

    @property
    def feature_dim(self) -> int:
        """Return the feature dimension."""
        dim = len(self.received_power)

        # Add phase difference and AOA if available
        if self.phase_difference is not None:
            dim += len(self.phase_difference)
        if self.aoa_measurements is not None:
            dim += len(self.aoa_measurements)

        return dim

    @property
    def num_leds(self) -> int:
        """Return the number of LED transmitters."""
        return len(self.led_ids)

    def estimate_distance(
        self,
        led_id: str,
        lambertian_order: int = 1
    ) -> Optional[float]:
        """Estimate distance to LED based on received power using simplified Lambertian model.

        Args:
            led_id: LED transmitter identifier.
            lambertian_order: Lambertian order (1 for Lambertian, higher for directional).

        Returns:
            Estimated distance in meters, or None if LED not found.

        Note:
            This is a simplified model. Real VLP systems require calibration.
        """
        try:
            idx = self.led_ids.index(led_id)
        except ValueError:
            return None

        power = self.received_power[idx]

        # Simplified inverse-square law with Lambertian model
        # P_r = P_t * (m+1) * A / (2π * d²) * cos^m(φ) * cos(ψ)
        # Assuming normal incidence: φ = ψ = 0, cos(0) = 1
        # P_r ∝ 1/d²  =>  d = sqrt(k / P_r)

        # Use empirical constant (requires calibration in practice)
        k = 1.0  # Calibration constant
        if power > 0:
            distance = np.sqrt(k / power)
            return float(distance)

        return None

    def to_tensor(self, device: str = 'cpu') -> torch.Tensor:
        """Convert signal to PyTorch tensor.

        Args:
            device: Device to place tensor on ('cpu' or 'cuda').

        Returns:
            Received power measurements as tensor.
        """
        features = [self.received_power]

        if self.phase_difference is not None:
            features.append(self.phase_difference)

        if self.aoa_measurements is not None:
            features.append(self.aoa_measurements)

        # Concatenate all features
        data = np.concatenate(features)
        return torch.from_numpy(data.astype(np.float32)).to(device)

    def to_numpy(self) -> np.ndarray:
        """Convert signal to NumPy array.

        Returns:
            Received power measurements as NumPy array.
        """
        features = [self.received_power]

        if self.phase_difference is not None:
            features.append(self.phase_difference)

        if self.aoa_measurements is not None:
            features.append(self.aoa_measurements)

        return np.concatenate(features).astype(np.float32)

    def normalize(self, method: str = 'minmax') -> 'VLCSignal':
        """Normalize received power measurements.

        Args:
            method: Normalization method ('minmax', 'standard').

        Returns:
            New VLCSignal with normalized power values.
        """
        data = self.received_power

        if method == 'minmax':
            # Normalize to [0, 1]
            data_min = data.min()
            data_max = data.max()
            if data_max > data_min:
                normalized = (data - data_min) / (data_max - data_min)
            else:
                normalized = data

        elif method == 'standard':
            # Z-score normalization
            mean = data.mean()
            std = data.std()
            if std > 0:
                normalized = (data - mean) / std
            else:
                normalized = data

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return VLCSignal(
            led_ids=self.led_ids,
            received_power=normalized,
            led_positions=self.led_positions,
            phase_difference=self.phase_difference,
            aoa_measurements=self.aoa_measurements,
            timestamps=self.timestamps,
            metadata=self.metadata
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'VLCSignal':
        """Create VLCSignal from dictionary.

        Args:
            d: Dictionary containing signal data.

        Returns:
            VLCSignal instance.
        """
        led_ids = d.get('led_ids')
        received_power = d.get('received_power')
        led_positions = d.get('led_positions')

        if received_power is not None and isinstance(received_power, list):
            received_power = np.array(received_power, dtype=np.float32)

        phase_diff = d.get('phase_difference')
        if phase_diff is not None and isinstance(phase_diff, list):
            phase_diff = np.array(phase_diff, dtype=np.float32)

        aoa = d.get('aoa_measurements')
        if aoa is not None and isinstance(aoa, list):
            aoa = np.array(aoa, dtype=np.float32)

        timestamps = d.get('timestamps')
        if timestamps is not None and isinstance(timestamps, list):
            timestamps = np.array(timestamps, dtype=np.float32)

        return cls(
            led_ids=led_ids,
            received_power=received_power,
            led_positions=led_positions,
            phase_difference=phase_diff,
            aoa_measurements=aoa,
            timestamps=timestamps
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert VLCSignal to dictionary.

        Returns:
            Dictionary representation of signal.
        """
        result = {
            'signal_type': self.signal_type,
            'led_ids': self.led_ids,
            'received_power': self.received_power.tolist(),
            'num_leds': self.num_leds
        }

        if self.led_positions:
            result['led_positions'] = self.led_positions

        if self.phase_difference is not None:
            result['phase_difference'] = self.phase_difference.tolist()

        if self.aoa_measurements is not None:
            result['aoa_measurements'] = self.aoa_measurements.tolist()

        if self.timestamps is not None:
            result['timestamps'] = self.timestamps.tolist()

        return result

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"VLCSignal(num_leds={self.num_leds}, "
            f"power_range=[{self.received_power.min():.2f}, {self.received_power.max():.2f}])"
        )
