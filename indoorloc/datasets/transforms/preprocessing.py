"""
Preprocessing Transforms for Indoor Localization

Provides signal preprocessing transforms for WiFi/BLE fingerprints.
"""
from typing import Optional
import numpy as np

from ...registry import TRANSFORMS
from .core import BaseTransform


@TRANSFORMS.register_module()
class RSSINormalize(BaseTransform):
    """Normalize RSSI values using various methods.

    Args:
        method: Normalization method:
            - 'minmax': Scale to [0, 1] range
            - 'positive': Shift to positive values only
            - 'standard': Zero mean, unit variance

    Example:
        >>> transform = RSSINormalize(method='minmax')
        >>> normalized_signal = transform(signal)
    """

    def __init__(self, method: str = 'minmax'):
        if method not in ('minmax', 'positive', 'standard'):
            raise ValueError(f"Unknown method: {method}. Use 'minmax', 'positive', or 'standard'")
        self.method = method

    def __call__(self, signal):
        """Apply RSSI normalization to a signal.

        Args:
            signal: WiFiSignal or compatible signal object with normalize() method,
                    or numpy array of RSSI values.

        Returns:
            Normalized signal.
        """
        # If signal has normalize method (WiFiSignal), use it
        if hasattr(signal, 'normalize'):
            return signal.normalize(method=self.method)

        # Otherwise treat as numpy array
        if isinstance(signal, np.ndarray):
            return self._normalize_array(signal)

        return signal

    def _normalize_array(self, data: np.ndarray) -> np.ndarray:
        """Normalize a numpy array of RSSI values."""
        # Typical WiFi RSSI range
        NOT_DETECTED = 100
        MIN_RSSI = -104
        MAX_RSSI = 0

        # Create mask for detected values
        detected = data != NOT_DETECTED

        if self.method == 'minmax':
            # Scale detected values to [0, 1]
            result = np.zeros_like(data, dtype=np.float32)
            if detected.any():
                result[detected] = (data[detected] - MIN_RSSI) / (MAX_RSSI - MIN_RSSI)
            return result

        elif self.method == 'positive':
            # Shift to positive: add abs(MIN_RSSI)
            result = data.copy().astype(np.float32)
            result[detected] = data[detected] + abs(MIN_RSSI)
            result[~detected] = 0
            return result

        elif self.method == 'standard':
            # Zero mean, unit variance for detected values
            result = np.zeros_like(data, dtype=np.float32)
            if detected.any():
                detected_values = data[detected]
                mean = np.mean(detected_values)
                std = np.std(detected_values)
                if std > 0:
                    result[detected] = (detected_values - mean) / std
            return result

        return data

    def __repr__(self) -> str:
        return f"RSSINormalize(method='{self.method}')"


@TRANSFORMS.register_module()
class APFilter(BaseTransform):
    """Filter weak access points by RSSI threshold.

    Sets RSSI values below the threshold to the NOT_DETECTED value,
    effectively filtering out weak or unreliable AP readings.

    Args:
        threshold: RSSI threshold in dBm (e.g., -90). Values below this
                   will be marked as not detected.

    Example:
        >>> transform = APFilter(threshold=-90)
        >>> filtered_signal = transform(signal)
    """

    def __init__(self, threshold: float = -90):
        self.threshold = threshold

    def __call__(self, signal):
        """Apply AP filtering to a signal.

        Args:
            signal: WiFiSignal or numpy array of RSSI values.

        Returns:
            Filtered signal with weak APs removed.
        """
        # Handle WiFiSignal objects
        if hasattr(signal, '_dense_data') and signal._dense_data is not None:
            NOT_DETECTED = getattr(signal, 'NOT_DETECTED_VALUE', 100)
            data = signal._dense_data.copy()

            # Values below threshold (but not already NOT_DETECTED) -> NOT_DETECTED
            mask = (data < self.threshold) & (data != NOT_DETECTED)
            data[mask] = NOT_DETECTED

            # Create new signal with filtered data
            signal_class = type(signal)
            return signal_class(
                rssi_values=data,
                ap_list=getattr(signal, '_ap_list', None),
                ap_info=getattr(signal, '_ap_info', None),
                metadata=getattr(signal, '_metadata', None)
            )

        # Handle numpy arrays
        if isinstance(signal, np.ndarray):
            return self._filter_array(signal)

        return signal

    def _filter_array(self, data: np.ndarray) -> np.ndarray:
        """Filter a numpy array of RSSI values."""
        NOT_DETECTED = 100
        result = data.copy()
        mask = (result < self.threshold) & (result != NOT_DETECTED)
        result[mask] = NOT_DETECTED
        return result

    def __repr__(self) -> str:
        return f"APFilter(threshold={self.threshold})"


@TRANSFORMS.register_module()
class APSelect(BaseTransform):
    """Select a subset of access points by index.

    Args:
        indices: List of AP indices to keep.

    Example:
        >>> transform = APSelect(indices=[0, 10, 20, 30])
        >>> selected_signal = transform(signal)
    """

    def __init__(self, indices: list):
        self.indices = indices

    def __call__(self, signal):
        """Select specific APs from a signal.

        Args:
            signal: WiFiSignal or numpy array.

        Returns:
            Signal with only selected APs.
        """
        if hasattr(signal, '_dense_data') and signal._dense_data is not None:
            data = signal._dense_data[self.indices]
            ap_list = None
            if signal._ap_list is not None:
                ap_list = [signal._ap_list[i] for i in self.indices]

            signal_class = type(signal)
            return signal_class(
                rssi_values=data,
                ap_list=ap_list,
                ap_info=getattr(signal, '_ap_info', None),
                metadata=getattr(signal, '_metadata', None)
            )

        if isinstance(signal, np.ndarray):
            return signal[self.indices]

        return signal

    def __repr__(self) -> str:
        return f"APSelect(indices=[{len(self.indices)} APs])"


@TRANSFORMS.register_module()
class GaussianNoise(BaseTransform):
    """Add Gaussian noise to RSSI values for data augmentation.

    Args:
        std: Standard deviation of Gaussian noise (in dBm).

    Example:
        >>> transform = GaussianNoise(std=2.0)
        >>> noisy_signal = transform(signal)
    """

    def __init__(self, std: float = 2.0):
        self.std = std

    def __call__(self, signal):
        """Add Gaussian noise to signal.

        Args:
            signal: WiFiSignal or numpy array.

        Returns:
            Signal with added noise.
        """
        if hasattr(signal, '_dense_data') and signal._dense_data is not None:
            NOT_DETECTED = getattr(signal, 'NOT_DETECTED_VALUE', 100)
            data = signal._dense_data.copy()

            # Only add noise to detected values
            detected = data != NOT_DETECTED
            noise = np.random.normal(0, self.std, data.shape).astype(np.float32)
            data[detected] = data[detected] + noise[detected]

            signal_class = type(signal)
            return signal_class(
                rssi_values=data,
                ap_list=getattr(signal, '_ap_list', None),
                ap_info=getattr(signal, '_ap_info', None),
                metadata=getattr(signal, '_metadata', None)
            )

        if isinstance(signal, np.ndarray):
            noise = np.random.normal(0, self.std, signal.shape).astype(np.float32)
            return signal + noise

        return signal

    def __repr__(self) -> str:
        return f"GaussianNoise(std={self.std})"
