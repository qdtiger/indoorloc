"""
Ultrasound Signal Implementation

Provides ultrasound signal representations for indoor localization based on
Time-of-Flight (TOF) acoustic measurements.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Any
import numpy as np
import torch

from .base import BaseSignal, SignalMetadata
from ..registry import SIGNALS


@dataclass
class UltrasoundTransmitter:
    """Represents a single ultrasound transmitter measurement.

    Attributes:
        transmitter_id: Unique identifier for the transmitter.
        position: 3D position (x, y, z) in meters.
        tof: Time-of-Flight measurement in seconds.
        distance: Measured distance in meters.
        snr: Signal-to-Noise Ratio.
        timestamp: Measurement timestamp.
    """
    transmitter_id: str
    position: Optional[Tuple[float, float, float]] = None
    tof: float = 0.0
    distance: Optional[float] = None
    snr: Optional[float] = None
    timestamp: Optional[float] = None


@SIGNALS.register_module()
class UltrasoundSignal(BaseSignal):
    """Ultrasound signal for indoor localization.

    Represents acoustic Time-of-Flight measurements from ultrasound
    transmitters for indoor positioning.

    Args:
        tof_measurements: Dictionary mapping transmitter IDs to TOF (seconds).
        speed_of_sound: Speed of sound in m/s (default 343.0 at 20°C).
        transmitter_positions: Dictionary mapping transmitter IDs to (x, y, z) positions.
        transmitters: List of UltrasoundTransmitter objects (alternative to dict).
        timestamps: Timestamps for each measurement.
        temperature: Ambient temperature in Celsius (affects speed of sound).
        metadata: Signal metadata.

    Example:
        >>> # TOF measurements from 4 transmitters
        >>> tof = {
        ...     'US1': 0.0145,  # seconds (~5 meters)
        ...     'US2': 0.0218,  # seconds (~7.5 meters)
        ...     'US3': 0.0093,  # seconds (~3.2 meters)
        ...     'US4': 0.0178,  # seconds (~6.1 meters)
        ... }
        >>> tx_pos = {
        ...     'US1': (0.0, 0.0, 2.5),
        ...     'US2': (10.0, 0.0, 2.5),
        ...     'US3': (10.0, 10.0, 2.5),
        ...     'US4': (0.0, 10.0, 2.5),
        ... }
        >>> signal = UltrasoundSignal(
        ...     tof_measurements=tof,
        ...     transmitter_positions=tx_pos,
        ...     temperature=20.0
        ... )
    """

    # Default speed of sound in air at 20°C
    DEFAULT_SPEED_OF_SOUND = 343.0  # m/s

    # Typical range for ultrasound indoor positioning
    MIN_DISTANCE = 0.1  # meters
    MAX_DISTANCE = 20.0  # meters

    def __init__(
        self,
        tof_measurements: Optional[Dict[str, float]] = None,
        speed_of_sound: float = DEFAULT_SPEED_OF_SOUND,
        transmitter_positions: Optional[Dict[str, Tuple[float, float, float]]] = None,
        transmitters: Optional[List[UltrasoundTransmitter]] = None,
        timestamps: Optional[np.ndarray] = None,
        temperature: Optional[float] = None,
        metadata: Optional[SignalMetadata] = None,
        **kwargs
    ):
        # Adjust speed of sound based on temperature if provided
        if temperature is not None:
            # Speed of sound approximation: v = 331.3 + 0.606 * T (°C)
            self.speed_of_sound = 331.3 + 0.606 * temperature
        else:
            self.speed_of_sound = speed_of_sound

        self.transmitter_positions = transmitter_positions or {}
        self.timestamps = timestamps
        self.temperature = temperature

        # Initialize from either TOF dict or transmitters list
        if transmitters is not None:
            self.transmitters = transmitters
            self.tof_measurements = {
                tx.transmitter_id: tx.tof for tx in transmitters
            }
            if transmitter_positions is None and any(tx.position for tx in transmitters):
                self.transmitter_positions = {
                    tx.transmitter_id: tx.position
                    for tx in transmitters if tx.position
                }
        elif tof_measurements is not None:
            self.tof_measurements = tof_measurements
            self.transmitters = [
                UltrasoundTransmitter(
                    transmitter_id=tx_id,
                    tof=tof,
                    position=self.transmitter_positions.get(tx_id),
                    distance=self.tof_to_distance(tof)
                )
                for tx_id, tof in tof_measurements.items()
            ]
        else:
            self.tof_measurements = {}
            self.transmitters = []

        # Pass data to parent class
        super().__init__(self.tof_measurements, metadata)

    @property
    def signal_type(self) -> str:
        """Return the signal type identifier."""
        return 'ultrasound'

    @property
    def feature_dim(self) -> int:
        """Return the feature dimension."""
        return len(self.tof_measurements)

    @property
    def num_transmitters(self) -> int:
        """Return the number of ultrasound transmitters."""
        return len(self.transmitters)

    def tof_to_distance(self, tof: float) -> float:
        """Convert Time-of-Flight to distance.

        Args:
            tof: Time-of-Flight in seconds.

        Returns:
            Distance in meters.
        """
        return tof * self.speed_of_sound

    def distance_to_tof(self, distance: float) -> float:
        """Convert distance to Time-of-Flight.

        Args:
            distance: Distance in meters.

        Returns:
            Time-of-Flight in seconds.
        """
        return distance / self.speed_of_sound

    def get_distances(self) -> Dict[str, float]:
        """Get distances to all transmitters.

        Returns:
            Dictionary mapping transmitter IDs to distances (meters).
        """
        return {
            tx_id: self.tof_to_distance(tof)
            for tx_id, tof in self.tof_measurements.items()
        }

    def to_tensor(self, device: str = 'cpu') -> torch.Tensor:
        """Convert signal to PyTorch tensor.

        Args:
            device: Device to place tensor on ('cpu' or 'cuda').

        Returns:
            Distance measurements as tensor.
        """
        # Convert TOF to distances and return as tensor
        distances = self.get_distances()
        sorted_ids = sorted(distances.keys())
        data = np.array([distances[tx_id] for tx_id in sorted_ids], dtype=np.float32)

        return torch.from_numpy(data).to(device)

    def to_numpy(self) -> np.ndarray:
        """Convert signal to NumPy array.

        Returns:
            Distance measurements as NumPy array.
        """
        distances = self.get_distances()
        sorted_ids = sorted(distances.keys())
        return np.array([distances[tx_id] for tx_id in sorted_ids], dtype=np.float32)

    def normalize(self, method: str = 'minmax') -> 'UltrasoundSignal':
        """Normalize TOF/distance measurements.

        Args:
            method: Normalization method ('minmax', 'standard').

        Returns:
            New UltrasoundSignal with normalized measurements.
        """
        # Get distances for normalization
        distances = self.get_distances()
        sorted_ids = sorted(distances.keys())
        data = np.array([distances[tx_id] for tx_id in sorted_ids])

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

        # Convert normalized distances back to TOF
        new_tof = {
            tx_id: self.distance_to_tof(normalized[i])
            for i, tx_id in enumerate(sorted_ids)
        }

        return UltrasoundSignal(
            tof_measurements=new_tof,
            speed_of_sound=self.speed_of_sound,
            transmitter_positions=self.transmitter_positions,
            timestamps=self.timestamps,
            temperature=self.temperature,
            metadata=self.metadata
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'UltrasoundSignal':
        """Create UltrasoundSignal from dictionary.

        Args:
            d: Dictionary containing signal data.

        Returns:
            UltrasoundSignal instance.
        """
        tof_measurements = d.get('tof_measurements')
        speed_of_sound = d.get('speed_of_sound', cls.DEFAULT_SPEED_OF_SOUND)
        transmitter_positions = d.get('transmitter_positions')
        temperature = d.get('temperature')

        timestamps = d.get('timestamps')
        if timestamps is not None and isinstance(timestamps, list):
            timestamps = np.array(timestamps, dtype=np.float32)

        return cls(
            tof_measurements=tof_measurements,
            speed_of_sound=speed_of_sound,
            transmitter_positions=transmitter_positions,
            timestamps=timestamps,
            temperature=temperature
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert UltrasoundSignal to dictionary.

        Returns:
            Dictionary representation of signal.
        """
        result = {
            'signal_type': self.signal_type,
            'tof_measurements': self.tof_measurements,
            'speed_of_sound': self.speed_of_sound,
            'distances': self.get_distances(),
            'num_transmitters': self.num_transmitters
        }

        if self.transmitter_positions:
            result['transmitter_positions'] = self.transmitter_positions

        if self.temperature is not None:
            result['temperature'] = self.temperature

        if self.timestamps is not None:
            result['timestamps'] = self.timestamps.tolist()

        return result

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"UltrasoundSignal(num_transmitters={self.num_transmitters}, "
            f"speed_of_sound={self.speed_of_sound:.1f}m/s)"
        )
