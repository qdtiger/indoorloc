"""
WiFi Signal Implementation

Handles WiFi RSSI (Received Signal Strength Indicator) fingerprints
for indoor localization.
"""
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
import torch

from .base import BaseSignal, SignalMetadata
from ..registry import SIGNALS


@dataclass
class APInfo:
    """
    Access Point information.

    Attributes:
        bssid: MAC address of the AP
        ssid: Network name (optional)
        frequency: Frequency in MHz (optional)
        channel: WiFi channel number (optional)
    """
    bssid: str
    ssid: Optional[str] = None
    frequency: Optional[int] = None
    channel: Optional[int] = None


@SIGNALS.register_module()
class WiFiSignal(BaseSignal):
    """
    WiFi RSSI Signal for fingerprint-based localization.

    Supports both sparse representation (dict of BSSID -> RSSI)
    and dense representation (fixed-length vector).

    Attributes:
        NOT_DETECTED_VALUE: Default value for undetected APs (100 in UJIndoorLoc)
        MIN_RSSI: Typical minimum RSSI value (-104 dBm)
        MAX_RSSI: Typical maximum RSSI value (0 dBm)
    """

    NOT_DETECTED_VALUE = 100
    MIN_RSSI = -104
    MAX_RSSI = 0

    def __init__(
        self,
        rssi_values: Union[Dict[str, float], np.ndarray, torch.Tensor, List[float]],
        ap_list: Optional[List[str]] = None,
        ap_info: Optional[Dict[str, APInfo]] = None,
        metadata: Optional[SignalMetadata] = None
    ):
        """
        Initialize a WiFi signal.

        Args:
            rssi_values: RSSI values, can be:
                - Dict[str, float]: Sparse format {bssid: rssi}
                - np.ndarray/List: Dense format (requires ap_list)
            ap_list: List of AP identifiers for dense format
            ap_info: Additional AP information
            metadata: Signal metadata
        """
        self._sparse_data: Optional[Dict[str, float]] = None
        self._dense_data: Optional[np.ndarray] = None
        self._ap_list = ap_list
        self._ap_info = ap_info or {}

        # Handle different input formats
        if isinstance(rssi_values, dict):
            self._sparse_data = rssi_values
            data = rssi_values
        else:
            if isinstance(rssi_values, list):
                rssi_values = np.array(rssi_values, dtype=np.float32)
            elif isinstance(rssi_values, torch.Tensor):
                rssi_values = rssi_values.numpy()
            self._dense_data = rssi_values.astype(np.float32)
            data = self._dense_data

        super().__init__(data, metadata)

    @property
    def signal_type(self) -> str:
        return 'wifi'

    @property
    def feature_dim(self) -> int:
        if self._dense_data is not None:
            return len(self._dense_data)
        return len(self._sparse_data)

    @property
    def ap_list(self) -> Optional[List[str]]:
        """Get the list of AP identifiers."""
        return self._ap_list

    @property
    def num_detected_aps(self) -> int:
        """Get the number of detected APs (excluding NOT_DETECTED)."""
        if self._sparse_data is not None:
            return len(self._sparse_data)
        return int(np.sum(self._dense_data != self.NOT_DETECTED_VALUE))

    @property
    def is_sparse(self) -> bool:
        """Check if signal is in sparse format."""
        return self._sparse_data is not None

    @property
    def is_dense(self) -> bool:
        """Check if signal is in dense format."""
        return self._dense_data is not None

    def get_rssi(self, bssid: str) -> Optional[float]:
        """
        Get RSSI value for a specific AP.

        Args:
            bssid: AP identifier

        Returns:
            RSSI value or None if not detected
        """
        if self._sparse_data is not None:
            return self._sparse_data.get(bssid)

        if self._ap_list and bssid in self._ap_list:
            idx = self._ap_list.index(bssid)
            val = self._dense_data[idx]
            return None if val == self.NOT_DETECTED_VALUE else float(val)

        return None

    def to_dense(
        self,
        ap_list: Optional[List[str]] = None,
        fill_value: Optional[float] = None
    ) -> 'WiFiSignal':
        """
        Convert to dense (fixed-length) representation.

        Args:
            ap_list: Target AP list for indexing
            fill_value: Value for undetected APs (default: NOT_DETECTED_VALUE)

        Returns:
            New WiFiSignal in dense format
        """
        if fill_value is None:
            fill_value = self.NOT_DETECTED_VALUE

        if self._dense_data is not None and ap_list is None:
            return self  # Already dense

        target_ap_list = ap_list or self._ap_list
        if target_ap_list is None:
            raise ValueError("ap_list required for dense conversion")

        dense = np.full(len(target_ap_list), fill_value, dtype=np.float32)

        if self._sparse_data:
            for bssid, rssi in self._sparse_data.items():
                if bssid in target_ap_list:
                    idx = target_ap_list.index(bssid)
                    dense[idx] = rssi

        return WiFiSignal(
            rssi_values=dense,
            ap_list=target_ap_list,
            ap_info=self._ap_info,
            metadata=self._metadata
        )

    def to_sparse(self) -> 'WiFiSignal':
        """
        Convert to sparse (dict) representation.

        Returns:
            New WiFiSignal in sparse format
        """
        if self._sparse_data is not None:
            return self  # Already sparse

        if self._ap_list is None:
            raise ValueError("Cannot convert to sparse without ap_list")

        sparse = {}
        for i, bssid in enumerate(self._ap_list):
            val = self._dense_data[i]
            if val != self.NOT_DETECTED_VALUE:
                sparse[bssid] = float(val)

        return WiFiSignal(
            rssi_values=sparse,
            ap_list=self._ap_list,
            ap_info=self._ap_info,
            metadata=self._metadata
        )

    def to_tensor(self, device: str = 'cpu') -> torch.Tensor:
        """Convert to PyTorch tensor."""
        if self._dense_data is None:
            raise ValueError("Convert to dense format first using to_dense()")
        return torch.tensor(self._dense_data, dtype=torch.float32, device=device)

    def to_numpy(self) -> np.ndarray:
        """Convert to NumPy array."""
        if self._dense_data is None:
            raise ValueError("Convert to dense format first using to_dense()")
        return self._dense_data.copy()

    def normalize(self, method: str = 'minmax') -> 'WiFiSignal':
        """
        Normalize RSSI values.

        Args:
            method: Normalization method
                - 'minmax': Scale to [0, 1] based on MIN_RSSI and MAX_RSSI
                - 'positive': Convert to positive values (UJIndoorLoc style)
                - 'standard': Z-score normalization

        Returns:
            New normalized WiFiSignal
        """
        if self._dense_data is None:
            raise ValueError("Convert to dense format first using to_dense()")

        data = self._dense_data.copy()

        if method == 'minmax':
            # Replace NOT_DETECTED with MIN_RSSI
            data[data == self.NOT_DETECTED_VALUE] = self.MIN_RSSI
            # Scale to [0, 1]
            data = (data - self.MIN_RSSI) / (self.MAX_RSSI - self.MIN_RSSI)

        elif method == 'positive':
            # UJIndoorLoc style: NOT_DETECTED -> 0, others -> abs value
            data = np.where(
                data == self.NOT_DETECTED_VALUE,
                0,
                np.abs(data)
            )

        elif method == 'standard':
            # Z-score normalization (excluding NOT_DETECTED)
            mask = data != self.NOT_DETECTED_VALUE
            if np.any(mask):
                mean = data[mask].mean()
                std = data[mask].std() + 1e-8
                data[mask] = (data[mask] - mean) / std
                data[~mask] = 0  # Set NOT_DETECTED to 0

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return WiFiSignal(
            rssi_values=data,
            ap_list=self._ap_list,
            ap_info=self._ap_info,
            metadata=self._metadata
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'WiFiSignal':
        """Create WiFiSignal from dictionary."""
        metadata = SignalMetadata(
            timestamp=d.get('metadata', {}).get('timestamp', 0.0),
            device_id=d.get('metadata', {}).get('device_id'),
            device_type=d.get('metadata', {}).get('device_type'),
            venue_id=d.get('metadata', {}).get('venue_id'),
            floor_id=d.get('metadata', {}).get('floor_id'),
            extra=d.get('metadata', {}).get('extra', {}),
        )

        return cls(
            rssi_values=d['data'],
            ap_list=d.get('ap_list'),
            metadata=metadata
        )

    def __len__(self) -> int:
        return self.feature_dim


__all__ = ['WiFiSignal', 'APInfo']
