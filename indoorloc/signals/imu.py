"""
IMU (Inertial Measurement Unit) Signal Implementation

Provides IMU signal representations for indoor localization, including
accelerometer, gyroscope, and magnetometer data.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from .base import BaseSignal
from ..registry import SIGNALS


@dataclass
class IMUReading:
    """Single IMU reading at a point in time.

    Attributes:
        timestamp: Unix timestamp in seconds.
        accelerometer: (ax, ay, az) in m/s².
        gyroscope: (gx, gy, gz) in rad/s.
        magnetometer: (mx, my, mz) in μT (optional).
        orientation: (roll, pitch, yaw) in radians (optional, computed).
    """
    timestamp: float
    accelerometer: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    gyroscope: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    magnetometer: Optional[Tuple[float, float, float]] = None
    orientation: Optional[Tuple[float, float, float]] = None

    def to_array(self, include_mag: bool = True, include_orient: bool = False) -> np.ndarray:
        """Convert reading to numpy array.

        Args:
            include_mag: Include magnetometer data.
            include_orient: Include orientation data.

        Returns:
            Numpy array of IMU values.
        """
        values = list(self.accelerometer) + list(self.gyroscope)

        if include_mag and self.magnetometer is not None:
            values.extend(self.magnetometer)

        if include_orient and self.orientation is not None:
            values.extend(self.orientation)

        return np.array(values, dtype=np.float32)


@SIGNALS.register_module()
class IMUSignal(BaseSignal):
    """IMU signal for Pedestrian Dead Reckoning (PDR) and motion-based localization.

    Stores a sequence of IMU readings over time for trajectory estimation.

    Args:
        readings: List of IMUReading objects.
        accelerometer: Array of shape (N, 3) for accelerometer data.
        gyroscope: Array of shape (N, 3) for gyroscope data.
        magnetometer: Array of shape (N, 3) for magnetometer data (optional).
        timestamps: Array of timestamps for each reading.
        sampling_rate: IMU sampling rate in Hz.

    Example:
        >>> # From arrays
        >>> accel = np.random.randn(100, 3)  # 100 samples, 3 axes
        >>> gyro = np.random.randn(100, 3)
        >>> signal = IMUSignal(accelerometer=accel, gyroscope=gyro, sampling_rate=100)

        >>> # From readings
        >>> readings = [IMUReading(timestamp=i/100, accelerometer=(0,0,9.8)) for i in range(100)]
        >>> signal = IMUSignal(readings=readings)
    """

    # Gravity constant
    GRAVITY = 9.81

    def __init__(
        self,
        readings: Optional[List[IMUReading]] = None,
        accelerometer: Optional[np.ndarray] = None,
        gyroscope: Optional[np.ndarray] = None,
        magnetometer: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        sampling_rate: Optional[float] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.readings = readings or []
        self.sampling_rate = sampling_rate

        # Store raw arrays
        if accelerometer is not None:
            self.accelerometer = np.asarray(accelerometer, dtype=np.float32)
        else:
            self.accelerometer = None

        if gyroscope is not None:
            self.gyroscope = np.asarray(gyroscope, dtype=np.float32)
        else:
            self.gyroscope = None

        if magnetometer is not None:
            self.magnetometer = np.asarray(magnetometer, dtype=np.float32)
        else:
            self.magnetometer = None

        if timestamps is not None:
            self.timestamps = np.asarray(timestamps, dtype=np.float64)
        else:
            self.timestamps = None

        # If readings provided, extract arrays
        if self.readings and self.accelerometer is None:
            self._extract_from_readings()

    def _extract_from_readings(self) -> None:
        """Extract arrays from readings list."""
        n = len(self.readings)

        self.timestamps = np.array([r.timestamp for r in self.readings], dtype=np.float64)
        self.accelerometer = np.array([r.accelerometer for r in self.readings], dtype=np.float32)
        self.gyroscope = np.array([r.gyroscope for r in self.readings], dtype=np.float32)

        if self.readings[0].magnetometer is not None:
            self.magnetometer = np.array([r.magnetometer for r in self.readings], dtype=np.float32)

        # Estimate sampling rate
        if n > 1:
            dt = np.diff(self.timestamps)
            self.sampling_rate = 1.0 / np.mean(dt)

    @property
    def signal_type(self) -> str:
        return 'imu'

    @property
    def num_samples(self) -> int:
        """Return number of samples in the signal."""
        if self.accelerometer is not None:
            return len(self.accelerometer)
        return len(self.readings)

    @property
    def duration(self) -> float:
        """Return signal duration in seconds."""
        if self.timestamps is not None and len(self.timestamps) > 1:
            return self.timestamps[-1] - self.timestamps[0]
        elif self.sampling_rate is not None and self.num_samples > 0:
            return self.num_samples / self.sampling_rate
        return 0.0

    @property
    def has_magnetometer(self) -> bool:
        """Check if magnetometer data is available."""
        return self.magnetometer is not None

    def to_tensor(self, device: str = 'cpu'):
        """Convert to PyTorch tensor.

        Args:
            device: Target device.

        Returns:
            torch.Tensor of shape (N, C) where C is number of channels.
        """
        import torch

        # Stack available sensor data
        arrays = [self.accelerometer, self.gyroscope]
        if self.has_magnetometer:
            arrays.append(self.magnetometer)

        data = np.concatenate(arrays, axis=1)
        return torch.tensor(data, dtype=torch.float32, device=device)

    def normalize(self, method: str = 'standard') -> 'IMUSignal':
        """Normalize IMU data.

        Args:
            method: Normalization method ('standard', 'minmax', 'gravity').

        Returns:
            New IMUSignal with normalized values.
        """
        accel = self.accelerometer.copy() if self.accelerometer is not None else None
        gyro = self.gyroscope.copy() if self.gyroscope is not None else None
        mag = self.magnetometer.copy() if self.magnetometer is not None else None

        if method == 'standard':
            # Z-score normalization
            if accel is not None:
                accel = (accel - accel.mean(axis=0)) / (accel.std(axis=0) + 1e-8)
            if gyro is not None:
                gyro = (gyro - gyro.mean(axis=0)) / (gyro.std(axis=0) + 1e-8)
            if mag is not None:
                mag = (mag - mag.mean(axis=0)) / (mag.std(axis=0) + 1e-8)

        elif method == 'minmax':
            # Scale to [0, 1]
            def minmax(arr):
                min_val = arr.min(axis=0)
                max_val = arr.max(axis=0)
                return (arr - min_val) / (max_val - min_val + 1e-8)

            if accel is not None:
                accel = minmax(accel)
            if gyro is not None:
                gyro = minmax(gyro)
            if mag is not None:
                mag = minmax(mag)

        elif method == 'gravity':
            # Normalize accelerometer by gravity
            if accel is not None:
                accel = accel / self.GRAVITY

        return IMUSignal(
            accelerometer=accel,
            gyroscope=gyro,
            magnetometer=mag,
            timestamps=self.timestamps.copy() if self.timestamps is not None else None,
            sampling_rate=self.sampling_rate
        )

    def get_window(self, start_idx: int, window_size: int) -> 'IMUSignal':
        """Extract a window of IMU data.

        Args:
            start_idx: Starting index.
            window_size: Number of samples in window.

        Returns:
            New IMUSignal containing the window.
        """
        end_idx = start_idx + window_size

        return IMUSignal(
            accelerometer=self.accelerometer[start_idx:end_idx] if self.accelerometer is not None else None,
            gyroscope=self.gyroscope[start_idx:end_idx] if self.gyroscope is not None else None,
            magnetometer=self.magnetometer[start_idx:end_idx] if self.magnetometer is not None else None,
            timestamps=self.timestamps[start_idx:end_idx] if self.timestamps is not None else None,
            sampling_rate=self.sampling_rate
        )

    def detect_steps(self, threshold: float = 1.0) -> List[int]:
        """Detect steps using accelerometer magnitude peaks.

        Simple peak detection for step counting.

        Args:
            threshold: Minimum peak prominence for step detection.

        Returns:
            List of step indices.
        """
        if self.accelerometer is None:
            return []

        # Compute acceleration magnitude
        mag = np.linalg.norm(self.accelerometer, axis=1)

        # Simple peak detection
        steps = []
        for i in range(1, len(mag) - 1):
            if mag[i] > mag[i-1] and mag[i] > mag[i+1]:
                if mag[i] - min(mag[i-1], mag[i+1]) > threshold:
                    steps.append(i)

        return steps

    def compute_heading(self) -> Optional[np.ndarray]:
        """Compute heading from magnetometer data.

        Returns:
            Array of heading angles in radians, or None if no magnetometer.
        """
        if not self.has_magnetometer:
            return None

        # Simple heading from horizontal magnetometer components
        heading = np.arctan2(self.magnetometer[:, 1], self.magnetometer[:, 0])
        return heading

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            'signal_type': self.signal_type,
            'num_samples': self.num_samples,
            'duration': self.duration,
            'sampling_rate': self.sampling_rate,
            'has_magnetometer': self.has_magnetometer,
        }

        if self.accelerometer is not None:
            result['accelerometer'] = self.accelerometer.tolist()
        if self.gyroscope is not None:
            result['gyroscope'] = self.gyroscope.tolist()
        if self.magnetometer is not None:
            result['magnetometer'] = self.magnetometer.tolist()
        if self.timestamps is not None:
            result['timestamps'] = self.timestamps.tolist()

        return result

    def __repr__(self) -> str:
        return (
            f"IMUSignal(samples={self.num_samples}, "
            f"duration={self.duration:.2f}s, "
            f"rate={self.sampling_rate}Hz)"
        )
