"""
Coordinate System Definitions

Provides coordinate abstractions for indoor localization.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class Coordinate:
    """
    Coordinate representation for indoor positioning.

    Supports multiple coordinate systems:
    - Local (x, y, z): Relative to building origin
    - Geographic (lat, lon, altitude): WGS84 coordinates
    - UTM: Universal Transverse Mercator projection

    Attributes:
        x: Local X coordinate in meters
        y: Local Y coordinate in meters
        z: Local Z coordinate (height) in meters
        latitude: WGS84 latitude (optional)
        longitude: WGS84 longitude (optional)
        altitude: Altitude in meters (optional)
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: Optional[float] = None

    @property
    def xy(self) -> Tuple[float, float]:
        """Get (x, y) tuple."""
        return (self.x, self.y)

    @property
    def xyz(self) -> Tuple[float, float, float]:
        """Get (x, y, z) tuple."""
        return (self.x, self.y, self.z)

    @property
    def latlon(self) -> Optional[Tuple[float, float]]:
        """Get (latitude, longitude) tuple if available."""
        if self.latitude is not None and self.longitude is not None:
            return (self.latitude, self.longitude)
        return None

    def distance_to(self, other: 'Coordinate') -> float:
        """
        Calculate 2D Euclidean distance to another coordinate.

        Args:
            other: Target coordinate

        Returns:
            Distance in meters
        """
        return float(np.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2
        ))

    def distance_3d_to(self, other: 'Coordinate') -> float:
        """
        Calculate 3D Euclidean distance to another coordinate.

        Args:
            other: Target coordinate

        Returns:
            Distance in meters
        """
        return float(np.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        ))

    def to_array(self, include_z: bool = False) -> np.ndarray:
        """
        Convert to NumPy array.

        Args:
            include_z: Whether to include z coordinate

        Returns:
            Array of shape (2,) or (3,)
        """
        if include_z:
            return np.array([self.x, self.y, self.z], dtype=np.float32)
        return np.array([self.x, self.y], dtype=np.float32)

    @classmethod
    def from_array(
        cls,
        arr: np.ndarray,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None
    ) -> 'Coordinate':
        """
        Create coordinate from NumPy array.

        Args:
            arr: Array of shape (2,) or (3,)
            latitude: Optional latitude
            longitude: Optional longitude

        Returns:
            Coordinate instance
        """
        x, y = arr[0], arr[1]
        z = arr[2] if len(arr) > 2 else 0.0
        return cls(
            x=float(x),
            y=float(y),
            z=float(z),
            latitude=latitude,
            longitude=longitude
        )

    def __add__(self, other: 'Coordinate') -> 'Coordinate':
        """Add two coordinates."""
        return Coordinate(
            x=self.x + other.x,
            y=self.y + other.y,
            z=self.z + other.z
        )

    def __sub__(self, other: 'Coordinate') -> 'Coordinate':
        """Subtract two coordinates."""
        return Coordinate(
            x=self.x - other.x,
            y=self.y - other.y,
            z=self.z - other.z
        )

    def __mul__(self, scalar: float) -> 'Coordinate':
        """Multiply coordinate by scalar."""
        return Coordinate(
            x=self.x * scalar,
            y=self.y * scalar,
            z=self.z * scalar
        )

    def __repr__(self) -> str:
        return f"Coordinate(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})"


__all__ = ['Coordinate']
