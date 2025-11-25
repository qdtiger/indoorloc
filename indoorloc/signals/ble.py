"""
BLE (Bluetooth Low Energy) Signal Implementation

Provides BLE signal representations for indoor localization based on
Bluetooth beacons, including iBeacon and Eddystone protocols.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np

from .base import BaseSignal
from ..registry import SIGNALS


@dataclass
class BLEBeacon:
    """Represents a single BLE beacon measurement.

    Attributes:
        uuid: Beacon UUID (for iBeacon) or namespace (for Eddystone).
        major: Major value (iBeacon).
        minor: Minor value (iBeacon).
        rssi: Received Signal Strength Indicator in dBm.
        tx_power: Transmitted power at 1m (for distance estimation).
        mac_address: Beacon MAC address.
        timestamp: Measurement timestamp.
    """
    uuid: Optional[str] = None
    major: Optional[int] = None
    minor: Optional[int] = None
    rssi: float = -100.0
    tx_power: Optional[float] = None
    mac_address: Optional[str] = None
    timestamp: Optional[float] = None

    def estimated_distance(self) -> Optional[float]:
        """Estimate distance to beacon using log-distance path loss model.

        Returns:
            Estimated distance in meters, or None if tx_power is not available.
        """
        if self.tx_power is None:
            return None

        # Log-distance path loss model
        # RSSI = tx_power - 10 * n * log10(d)
        # where n is path loss exponent (typically 2-4 for indoor)
        n = 2.0  # Default path loss exponent
        ratio = (self.tx_power - self.rssi) / (10 * n)
        return 10 ** ratio

    @property
    def beacon_id(self) -> str:
        """Return a unique identifier for this beacon."""
        if self.uuid and self.major is not None and self.minor is not None:
            return f"{self.uuid}:{self.major}:{self.minor}"
        elif self.mac_address:
            return self.mac_address
        else:
            return "unknown"


@SIGNALS.register_module()
class BLESignal(BaseSignal):
    """BLE signal for indoor localization.

    Supports both dense (fixed beacon array) and sparse (variable beacons) modes.

    Args:
        beacons: List of BLEBeacon measurements.
        rssi_values: Dense RSSI array (alternative to beacons list).
        beacon_ids: Beacon identifiers corresponding to rssi_values.
        timestamp: Signal measurement timestamp.
        scan_duration: Duration of the BLE scan in seconds.

    Example:
        >>> # Sparse mode - variable number of beacons
        >>> beacons = [
        ...     BLEBeacon(uuid="xxx", major=1, minor=1, rssi=-65),
        ...     BLEBeacon(uuid="xxx", major=1, minor=2, rssi=-72),
        ... ]
        >>> signal = BLESignal(beacons=beacons)

        >>> # Dense mode - fixed beacon array
        >>> rssi = np.array([-65, -72, -100, -100])  # 4 beacons
        >>> signal = BLESignal(rssi_values=rssi)
    """

    NOT_DETECTED_VALUE = -100.0
    MIN_RSSI = -100.0
    MAX_RSSI = 0.0

    def __init__(
        self,
        beacons: Optional[List[BLEBeacon]] = None,
        rssi_values: Optional[np.ndarray] = None,
        beacon_ids: Optional[List[str]] = None,
        timestamp: Optional[float] = None,
        scan_duration: Optional[float] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.beacons = beacons or []
        self.rssi_values = rssi_values
        self.beacon_ids = beacon_ids
        self.timestamp = timestamp
        self.scan_duration = scan_duration

        # Validate inputs
        if rssi_values is not None:
            self.rssi_values = np.asarray(rssi_values, dtype=np.float32)

    @property
    def signal_type(self) -> str:
        return 'ble'

    @property
    def is_sparse(self) -> bool:
        """Check if signal is in sparse mode (beacon list)."""
        return len(self.beacons) > 0

    @property
    def num_beacons(self) -> int:
        """Return number of detected beacons."""
        if self.is_sparse:
            return len(self.beacons)
        elif self.rssi_values is not None:
            return int(np.sum(self.rssi_values != self.NOT_DETECTED_VALUE))
        return 0

    def to_dense(self, beacon_order: Optional[List[str]] = None) -> 'BLESignal':
        """Convert sparse beacon list to dense RSSI array.

        Args:
            beacon_order: Ordered list of beacon IDs. If None, uses detected order.

        Returns:
            New BLESignal with dense representation.
        """
        if not self.is_sparse:
            return self

        # Build beacon ID to RSSI mapping
        rssi_map = {b.beacon_id: b.rssi for b in self.beacons}

        if beacon_order is None:
            beacon_order = list(rssi_map.keys())

        # Create dense array
        rssi_values = np.full(len(beacon_order), self.NOT_DETECTED_VALUE, dtype=np.float32)
        for i, bid in enumerate(beacon_order):
            if bid in rssi_map:
                rssi_values[i] = rssi_map[bid]

        return BLESignal(
            rssi_values=rssi_values,
            beacon_ids=beacon_order,
            timestamp=self.timestamp,
            scan_duration=self.scan_duration
        )

    def to_tensor(self, device: str = 'cpu'):
        """Convert to PyTorch tensor.

        Args:
            device: Target device ('cpu' or 'cuda').

        Returns:
            torch.Tensor of shape (num_beacons,) or (num_rssi_values,).
        """
        import torch

        if self.rssi_values is not None:
            return torch.tensor(self.rssi_values, dtype=torch.float32, device=device)
        elif self.is_sparse:
            rssi = np.array([b.rssi for b in self.beacons], dtype=np.float32)
            return torch.tensor(rssi, dtype=torch.float32, device=device)
        else:
            return torch.tensor([], dtype=torch.float32, device=device)

    def normalize(self, method: str = 'minmax') -> 'BLESignal':
        """Normalize RSSI values.

        Args:
            method: Normalization method ('minmax', 'positive', 'standard').

        Returns:
            New BLESignal with normalized values.
        """
        if self.rssi_values is None:
            # Convert to dense first
            dense = self.to_dense()
            return dense.normalize(method)

        rssi = self.rssi_values.copy()
        mask = rssi != self.NOT_DETECTED_VALUE

        if method == 'minmax':
            # Scale to [0, 1]
            rssi[mask] = (rssi[mask] - self.MIN_RSSI) / (self.MAX_RSSI - self.MIN_RSSI)
            rssi[~mask] = 0.0

        elif method == 'positive':
            # Shift to positive range
            rssi[mask] = rssi[mask] + abs(self.MIN_RSSI)
            rssi[~mask] = 0.0

        elif method == 'standard':
            # Z-score normalization on detected values
            if mask.sum() > 0:
                mean_val = rssi[mask].mean()
                std_val = rssi[mask].std()
                if std_val > 0:
                    rssi[mask] = (rssi[mask] - mean_val) / std_val
                else:
                    rssi[mask] = 0.0
            rssi[~mask] = 0.0

        return BLESignal(
            rssi_values=rssi,
            beacon_ids=self.beacon_ids,
            timestamp=self.timestamp,
            scan_duration=self.scan_duration
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            'signal_type': self.signal_type,
            'num_beacons': self.num_beacons,
            'timestamp': self.timestamp,
            'scan_duration': self.scan_duration,
        }

        if self.rssi_values is not None:
            result['rssi_values'] = self.rssi_values.tolist()
            result['beacon_ids'] = self.beacon_ids

        if self.beacons:
            result['beacons'] = [
                {
                    'uuid': b.uuid,
                    'major': b.major,
                    'minor': b.minor,
                    'rssi': b.rssi,
                    'tx_power': b.tx_power,
                    'mac_address': b.mac_address,
                }
                for b in self.beacons
            ]

        return result

    def __repr__(self) -> str:
        return f"BLESignal(num_beacons={self.num_beacons}, sparse={self.is_sparse})"
