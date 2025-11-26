"""
UWB (Ultra-Wideband) Signal Implementation

Provides UWB ranging signal representations for indoor localization based on
Time-of-Flight (TOF) and Time-Difference-of-Arrival (TDOA) measurements.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Any, Union
import numpy as np
import torch

from .base import BaseSignal, SignalMetadata
from ..registry import SIGNALS


@dataclass
class UWBAnchor:
    """Represents a single UWB anchor measurement.

    Attributes:
        anchor_id: Unique identifier for the anchor.
        position: 3D position (x, y, z) in meters.
        distance: Measured distance to anchor in meters.
        distance_std: Standard deviation of distance measurement.
        nlos: Non-Line-of-Sight indicator (True if obstructed).
        timestamp: Measurement timestamp.
    """
    anchor_id: str
    position: Optional[Tuple[float, float, float]] = None
    distance: float = 0.0
    distance_std: Optional[float] = None
    nlos: Optional[bool] = None
    timestamp: Optional[float] = None


@SIGNALS.register_module()
class UWBSignal(BaseSignal):
    """UWB signal for indoor localization.

    Supports both Time-of-Flight (TOF) distance measurements and
    Time-Difference-of-Arrival (TDOA) measurements.

    Args:
        distances: Dictionary mapping anchor IDs to distances (meters).
        anchor_positions: Dictionary mapping anchor IDs to (x, y, z) positions.
        tdoa_measurements: TDOA measurements array.
        anchors: List of UWBAnchor objects (alternative to distances dict).
        timestamps: Timestamps for each measurement.
        metadata: Signal metadata.

    Example:
        >>> # TOF mode - distance measurements
        >>> distances = {'A0': 5.0, 'A1': 7.5, 'A2': 3.2, 'A3': 6.1}
        >>> anchor_pos = {
        ...     'A0': (0.0, 0.0, 2.5),
        ...     'A1': (10.0, 0.0, 2.5),
        ...     'A2': (10.0, 10.0, 2.5),
        ...     'A3': (0.0, 10.0, 2.5),
        ... }
        >>> signal = UWBSignal(distances=distances, anchor_positions=anchor_pos)

        >>> # TDOA mode
        >>> tdoa = np.array([0.1, 0.2, 0.3])  # TDOA values in seconds
        >>> signal = UWBSignal(tdoa_measurements=tdoa)
    """

    # Speed of light for TOF to distance conversion
    SPEED_OF_LIGHT = 299792458.0  # m/s

    # Default minimum and maximum measurable distances
    MIN_DISTANCE = 0.1  # meters
    MAX_DISTANCE = 100.0  # meters

    def __init__(
        self,
        distances: Optional[Union[Dict[str, float], np.ndarray]] = None,
        anchor_positions: Optional[Dict[str, Tuple[float, float, float]]] = None,
        tdoa_measurements: Optional[np.ndarray] = None,
        anchors: Optional[List[UWBAnchor]] = None,
        timestamps: Optional[np.ndarray] = None,
        metadata: Optional[SignalMetadata] = None,
        **kwargs
    ):

        # Store anchor information
        self.anchor_positions = anchor_positions or {}
        self.timestamps = timestamps

        # Initialize from either distances dict or anchors list
        if anchors is not None:
            self.anchors = anchors
            self.distances = {a.anchor_id: a.distance for a in anchors}
            if anchor_positions is None and any(a.position for a in anchors):
                self.anchor_positions = {
                    a.anchor_id: a.position for a in anchors if a.position
                }
        elif isinstance(distances, dict):
            self.distances = distances
            self.anchors = [
                UWBAnchor(
                    anchor_id=aid,
                    distance=dist,
                    position=self.anchor_positions.get(aid)
                )
                for aid, dist in distances.items()
            ]
        elif isinstance(distances, np.ndarray):
            # Dense array mode - assume sequential anchor IDs
            self.distances = {
                f'A{i}': dist for i, dist in enumerate(distances)
            }
            self.anchors = [
                UWBAnchor(anchor_id=f'A{i}', distance=dist)
                for i, dist in enumerate(distances)
            ]
        else:
            self.distances = {}
            self.anchors = []

        # TDOA measurements
        self.tdoa_measurements = tdoa_measurements

        # Pass data to parent class
        if self.tdoa_measurements is not None:
            data = self.tdoa_measurements
        elif self.distances:
            data = self.distances
        else:
            data = {}

        super().__init__(data, metadata)

    @property
    def signal_type(self) -> str:
        """Return the signal type identifier."""
        return 'uwb'

    @property
    def feature_dim(self) -> int:
        """Return the feature dimension."""
        if self.tdoa_measurements is not None:
            return len(self.tdoa_measurements)
        return len(self.distances)

    @property
    def num_anchors(self) -> int:
        """Return the number of UWB anchors."""
        return len(self.anchors)

    def to_tensor(self, device: str = 'cpu') -> torch.Tensor:
        """Convert signal to PyTorch tensor.

        Args:
            device: Device to place tensor on ('cpu' or 'cuda').

        Returns:
            Distance measurements as tensor.
        """
        if self.tdoa_measurements is not None:
            data = self.tdoa_measurements
        else:
            # Convert distances dict to array (sorted by anchor ID)
            sorted_anchors = sorted(self.distances.keys())
            data = np.array([self.distances[aid] for aid in sorted_anchors])

        return torch.from_numpy(data.astype(np.float32)).to(device)

    def to_numpy(self) -> np.ndarray:
        """Convert signal to NumPy array.

        Returns:
            Distance measurements as NumPy array.
        """
        if self.tdoa_measurements is not None:
            return self.tdoa_measurements

        # Convert distances dict to array (sorted by anchor ID)
        sorted_anchors = sorted(self.distances.keys())
        return np.array([self.distances[aid] for aid in sorted_anchors], dtype=np.float32)

    def normalize(self, method: str = 'minmax') -> 'UWBSignal':
        """Normalize distance measurements.

        Args:
            method: Normalization method ('minmax', 'standard').

        Returns:
            New UWBSignal with normalized distances.
        """
        data = self.to_numpy()

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

        # Create new signal with normalized distances
        if isinstance(self.distances, dict):
            sorted_anchors = sorted(self.distances.keys())
            new_distances = {aid: normalized[i] for i, aid in enumerate(sorted_anchors)}
        else:
            new_distances = normalized

        return UWBSignal(
            distances=new_distances,
            anchor_positions=self.anchor_positions,
            tdoa_measurements=self.tdoa_measurements,
            timestamps=self.timestamps,
            metadata=self.metadata
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'UWBSignal':
        """Create UWBSignal from dictionary.

        Args:
            d: Dictionary containing signal data.

        Returns:
            UWBSignal instance.
        """
        distances = d.get('distances')
        anchor_positions = d.get('anchor_positions')
        tdoa = d.get('tdoa_measurements')
        timestamps = d.get('timestamps')

        if tdoa is not None and isinstance(tdoa, list):
            tdoa = np.array(tdoa, dtype=np.float32)

        if timestamps is not None and isinstance(timestamps, list):
            timestamps = np.array(timestamps, dtype=np.float32)

        return cls(
            distances=distances,
            anchor_positions=anchor_positions,
            tdoa_measurements=tdoa,
            timestamps=timestamps
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert UWBSignal to dictionary.

        Returns:
            Dictionary representation of signal.
        """
        result = {
            'signal_type': self.signal_type,
            'distances': self.distances,
            'anchor_positions': self.anchor_positions,
            'num_anchors': self.num_anchors
        }

        if self.tdoa_measurements is not None:
            result['tdoa_measurements'] = self.tdoa_measurements.tolist()

        if self.timestamps is not None:
            result['timestamps'] = self.timestamps.tolist()

        return result

    def estimate_position_trilateration(self) -> Optional[Tuple[float, float, float]]:
        """Estimate 3D position using trilateration (requires ≥4 anchors with known positions).

        Returns:
            Estimated (x, y, z) position, or None if insufficient anchors.

        Note:
            This is a simple least-squares trilateration implementation.
            For production use, consider more robust methods.
        """
        # Need at least 4 anchors with known positions for 3D trilateration
        valid_anchors = [
            a for a in self.anchors
            if a.position is not None and a.distance > 0
        ]

        if len(valid_anchors) < 4:
            return None

        # Prepare matrices for least-squares solution
        # System: ||P - Ai||² = di² for each anchor i
        # Linearize: 2(A1-An)·P = d1² - dn² + ||An||² - ||A1||²

        ref_anchor = valid_anchors[-1]  # Use last anchor as reference
        ref_pos = np.array(ref_anchor.position)
        ref_dist = ref_anchor.distance

        A = []
        b = []

        for anchor in valid_anchors[:-1]:
            pos = np.array(anchor.position)
            dist = anchor.distance

            # Build row of A matrix
            A.append(2 * (pos - ref_pos))

            # Build b vector
            b_val = (
                dist**2 - ref_dist**2 +
                np.sum(ref_pos**2) - np.sum(pos**2)
            )
            b.append(b_val)

        A = np.array(A)
        b = np.array(b)

        # Solve least-squares: A·P = b
        try:
            position = np.linalg.lstsq(A, b, rcond=None)[0]
            return tuple(position.tolist())
        except np.linalg.LinAlgError:
            return None

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"UWBSignal(num_anchors={self.num_anchors}, "
            f"distances={list(self.distances.values())[:3]}...)"
        )
