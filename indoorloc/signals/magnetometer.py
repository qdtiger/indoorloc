"""
Magnetometer Signal Implementation

Provides magnetometer (geomagnetic field) signal representations for
indoor localization based on magnetic field fingerprinting.
"""
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch

from .base import BaseSignal, SignalMetadata
from ..registry import SIGNALS


@SIGNALS.register_module()
class MagnetometerSignal(BaseSignal):
    """Magnetometer signal for indoor localization.

    Represents 3-axis magnetic field measurements for geomagnetic
    positioning and navigation.

    Args:
        magnetic_field: 3-axis magnetic field data.
            Shape: (N, 3) for N measurements of (mx, my, mz) in μT.
            Or (3,) for single measurement.
        timestamps: Timestamps for each measurement.
        sampling_rate: Sampling rate in Hz.
        calibration_bias: Calibration bias (mx, my, mz).
        calibration_scale: Calibration scale factors (mx, my, mz).
        metadata: Signal metadata.

    Example:
        >>> # Single measurement
        >>> mag_data = np.array([30.5, 45.2, -12.3])  # μT
        >>> signal = MagnetometerSignal(magnetic_field=mag_data)

        >>> # Time series measurements
        >>> mag_series = np.random.randn(100, 3) * 10 + 40  # 100 samples
        >>> signal = MagnetometerSignal(
        ...     magnetic_field=mag_series,
        ...     sampling_rate=50.0
        ... )
    """

    # Earth's magnetic field typical range in μT (microtesla)
    MIN_FIELD = -100.0  # μT
    MAX_FIELD = 100.0   # μT

    def __init__(
        self,
        magnetic_field: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        sampling_rate: Optional[float] = None,
        calibration_bias: Optional[Tuple[float, float, float]] = None,
        calibration_scale: Optional[Tuple[float, float, float]] = None,
        metadata: Optional[SignalMetadata] = None,
        **kwargs
    ):
        # Ensure magnetic_field is 2D (N, 3)
        if magnetic_field.ndim == 1:
            if magnetic_field.shape[0] != 3:
                raise ValueError("Single measurement must have shape (3,)")
            magnetic_field = magnetic_field.reshape(1, 3)
        elif magnetic_field.ndim != 2 or magnetic_field.shape[1] != 3:
            raise ValueError(
                f"magnetic_field must have shape (N, 3), got {magnetic_field.shape}"
            )

        self.magnetic_field = magnetic_field.astype(np.float32)
        self.timestamps = timestamps
        self.sampling_rate = sampling_rate
        self.calibration_bias = calibration_bias
        self.calibration_scale = calibration_scale

        # Pass data to parent class
        super().__init__(self.magnetic_field, metadata)

    @property
    def signal_type(self) -> str:
        """Return the signal type identifier."""
        return 'magnetometer'

    @property
    def feature_dim(self) -> int:
        """Return the feature dimension."""
        # For single measurement: 3 (mx, my, mz)
        # For time series: N * 3
        return self.magnetic_field.size

    @property
    def num_samples(self) -> int:
        """Return the number of time samples."""
        return len(self.magnetic_field)

    def compute_magnitude(self) -> np.ndarray:
        """Compute magnetic field magnitude for each measurement.

        Returns:
            Array of magnitudes with shape (N,).
        """
        return np.linalg.norm(self.magnetic_field, axis=1)

    def compute_heading(self) -> np.ndarray:
        """Compute heading from horizontal magnetic field components.

        Returns:
            Array of headings in radians with shape (N,).
            Heading is measured clockwise from magnetic north.
        """
        # Heading from horizontal components (mx, my)
        # heading = atan2(my, mx)
        mx = self.magnetic_field[:, 0]
        my = self.magnetic_field[:, 1]
        return np.arctan2(my, mx)

    def apply_calibration(self) -> 'MagnetometerSignal':
        """Apply calibration bias and scale to magnetic field data.

        Returns:
            New MagnetometerSignal with calibrated data.
        """
        if self.calibration_bias is None and self.calibration_scale is None:
            return self

        calibrated = self.magnetic_field.copy()

        # Apply bias correction
        if self.calibration_bias is not None:
            bias = np.array(self.calibration_bias)
            calibrated = calibrated - bias

        # Apply scale correction
        if self.calibration_scale is not None:
            scale = np.array(self.calibration_scale)
            calibrated = calibrated * scale

        return MagnetometerSignal(
            magnetic_field=calibrated,
            timestamps=self.timestamps,
            sampling_rate=self.sampling_rate,
            metadata=self.metadata
        )

    def to_tensor(self, device: str = 'cpu') -> torch.Tensor:
        """Convert signal to PyTorch tensor.

        Args:
            device: Device to place tensor on ('cpu' or 'cuda').

        Returns:
            Magnetic field data as tensor with shape (N, 3).
        """
        return torch.from_numpy(self.magnetic_field).to(device)

    def to_numpy(self) -> np.ndarray:
        """Convert signal to NumPy array.

        Returns:
            Magnetic field data as NumPy array with shape (N, 3).
        """
        return self.magnetic_field

    def normalize(self, method: str = 'standard') -> 'MagnetometerSignal':
        """Normalize magnetic field measurements.

        Args:
            method: Normalization method.
                - 'standard': Z-score normalization (mean=0, std=1)
                - 'minmax': Min-max normalization to [0, 1]
                - 'magnitude': Normalize by magnitude (unit vectors)

        Returns:
            New MagnetometerSignal with normalized data.
        """
        data = self.magnetic_field

        if method == 'standard':
            # Z-score normalization per axis
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            normalized = (data - mean) / (std + 1e-8)

        elif method == 'minmax':
            # Min-max normalization per axis
            data_min = data.min(axis=0)
            data_max = data.max(axis=0)
            normalized = (data - data_min) / (data_max - data_min + 1e-8)

        elif method == 'magnitude':
            # Normalize each vector to unit length
            magnitudes = np.linalg.norm(data, axis=1, keepdims=True)
            normalized = data / (magnitudes + 1e-8)

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return MagnetometerSignal(
            magnetic_field=normalized,
            timestamps=self.timestamps,
            sampling_rate=self.sampling_rate,
            calibration_bias=self.calibration_bias,
            calibration_scale=self.calibration_scale,
            metadata=self.metadata
        )

    def get_window(self, start_idx: int, end_idx: int) -> 'MagnetometerSignal':
        """Extract a time window from the signal.

        Args:
            start_idx: Start index (inclusive).
            end_idx: End index (exclusive).

        Returns:
            New MagnetometerSignal with windowed data.
        """
        windowed_field = self.magnetic_field[start_idx:end_idx]
        windowed_timestamps = (
            self.timestamps[start_idx:end_idx]
            if self.timestamps is not None
            else None
        )

        return MagnetometerSignal(
            magnetic_field=windowed_field,
            timestamps=windowed_timestamps,
            sampling_rate=self.sampling_rate,
            calibration_bias=self.calibration_bias,
            calibration_scale=self.calibration_scale,
            metadata=self.metadata
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MagnetometerSignal':
        """Create MagnetometerSignal from dictionary.

        Args:
            d: Dictionary containing signal data.

        Returns:
            MagnetometerSignal instance.
        """
        magnetic_field = np.array(d['magnetic_field'], dtype=np.float32)

        timestamps = d.get('timestamps')
        if timestamps is not None:
            timestamps = np.array(timestamps, dtype=np.float32)

        return cls(
            magnetic_field=magnetic_field,
            timestamps=timestamps,
            sampling_rate=d.get('sampling_rate'),
            calibration_bias=d.get('calibration_bias'),
            calibration_scale=d.get('calibration_scale')
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert MagnetometerSignal to dictionary.

        Returns:
            Dictionary representation of signal.
        """
        result = {
            'signal_type': self.signal_type,
            'magnetic_field': self.magnetic_field.tolist(),
            'num_samples': self.num_samples
        }

        if self.timestamps is not None:
            result['timestamps'] = self.timestamps.tolist()

        if self.sampling_rate is not None:
            result['sampling_rate'] = self.sampling_rate

        if self.calibration_bias is not None:
            result['calibration_bias'] = self.calibration_bias

        if self.calibration_scale is not None:
            result['calibration_scale'] = self.calibration_scale

        return result

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MagnetometerSignal(num_samples={self.num_samples}, "
            f"sampling_rate={self.sampling_rate}Hz)"
        )
