"""
Location Data Structure

Provides complete location representation including spatial and semantic information.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np

from .coordinate import Coordinate


@dataclass
class Location:
    """
    Complete location information for indoor positioning.

    Combines spatial coordinates with semantic information
    (floor, building, room) and confidence metrics.

    Attributes:
        coordinate: Spatial coordinates
        floor: Floor number (0-indexed or building-specific)
        building_id: Building identifier
        space_id: Space/room identifier
        space_type: Type of space ('office', 'corridor', 'lobby', etc.)
        confidence: Overall location confidence [0, 1]
        position_uncertainty: Position uncertainty in meters
        floor_confidence: Floor prediction confidence [0, 1]
        timestamp: Unix timestamp of the location
        extra: Additional metadata
    """
    coordinate: Coordinate
    floor: Optional[int] = None
    building_id: Optional[str] = None
    space_id: Optional[str] = None
    space_type: Optional[str] = None
    confidence: float = 1.0
    position_uncertainty: float = 0.0
    floor_confidence: float = 1.0
    timestamp: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def x(self) -> float:
        """Get x coordinate."""
        return self.coordinate.x

    @property
    def y(self) -> float:
        """Get y coordinate."""
        return self.coordinate.y

    @property
    def z(self) -> float:
        """Get z coordinate."""
        return self.coordinate.z

    @property
    def xy(self):
        """Get (x, y) tuple."""
        return self.coordinate.xy

    def distance_to(self, other: 'Location') -> float:
        """
        Calculate 2D distance to another location.

        Args:
            other: Target location

        Returns:
            Distance in meters
        """
        return self.coordinate.distance_to(other.coordinate)

    def distance_3d_to(self, other: 'Location') -> float:
        """
        Calculate 3D distance to another location.

        Args:
            other: Target location

        Returns:
            Distance in meters
        """
        return self.coordinate.distance_3d_to(other.coordinate)

    def floor_matches(self, other: 'Location') -> bool:
        """
        Check if floor matches another location.

        Args:
            other: Target location

        Returns:
            True if floors match or if floor info is unavailable
        """
        if self.floor is None or other.floor is None:
            return True  # Unknown floors are considered matching
        return self.floor == other.floor

    def building_matches(self, other: 'Location') -> bool:
        """
        Check if building matches another location.

        Args:
            other: Target location

        Returns:
            True if buildings match or if building info is unavailable
        """
        if self.building_id is None or other.building_id is None:
            return True
        return self.building_id == other.building_id

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize location to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            'x': self.coordinate.x,
            'y': self.coordinate.y,
            'z': self.coordinate.z,
            'latitude': self.coordinate.latitude,
            'longitude': self.coordinate.longitude,
            'floor': self.floor,
            'building_id': self.building_id,
            'space_id': self.space_id,
            'space_type': self.space_type,
            'confidence': self.confidence,
            'position_uncertainty': self.position_uncertainty,
            'floor_confidence': self.floor_confidence,
            'timestamp': self.timestamp,
            'extra': self.extra,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Location':
        """
        Create location from dictionary.

        Args:
            d: Dictionary representation

        Returns:
            Location instance
        """
        coordinate = Coordinate(
            x=d.get('x', 0.0),
            y=d.get('y', 0.0),
            z=d.get('z', 0.0),
            latitude=d.get('latitude'),
            longitude=d.get('longitude')
        )

        return cls(
            coordinate=coordinate,
            floor=d.get('floor'),
            building_id=d.get('building_id'),
            space_id=d.get('space_id'),
            space_type=d.get('space_type'),
            confidence=d.get('confidence', 1.0),
            position_uncertainty=d.get('position_uncertainty', 0.0),
            floor_confidence=d.get('floor_confidence', 1.0),
            timestamp=d.get('timestamp'),
            extra=d.get('extra', {}),
        )

    @classmethod
    def from_coordinates(
        cls,
        x: float,
        y: float,
        z: float = 0.0,
        floor: Optional[int] = None,
        building_id: Optional[str] = None,
        **kwargs
    ) -> 'Location':
        """
        Create location from coordinate values.

        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate
            floor: Floor number
            building_id: Building identifier
            **kwargs: Additional arguments

        Returns:
            Location instance
        """
        return cls(
            coordinate=Coordinate(x=x, y=y, z=z),
            floor=floor,
            building_id=building_id,
            **kwargs
        )

    def to_array(self, include_z: bool = False) -> np.ndarray:
        """
        Convert coordinate to NumPy array.

        Args:
            include_z: Whether to include z coordinate

        Returns:
            Array of shape (2,) or (3,)
        """
        return self.coordinate.to_array(include_z=include_z)

    def __repr__(self) -> str:
        parts = [f"Location(x={self.x:.2f}, y={self.y:.2f}"]
        if self.floor is not None:
            parts.append(f", floor={self.floor}")
        if self.building_id is not None:
            parts.append(f", building={self.building_id}")
        parts.append(")")
        return "".join(parts)


@dataclass
class LocalizationResult:
    """
    Result of a localization prediction.

    Wraps a Location with additional prediction metadata.

    Attributes:
        location: Predicted location
        raw_output: Raw model output (logits, probabilities, etc.)
        latency_ms: Prediction latency in milliseconds
    """
    location: Location
    raw_output: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

    @property
    def coordinate(self) -> Coordinate:
        """Get predicted coordinate."""
        return self.location.coordinate

    @property
    def floor(self) -> Optional[int]:
        """Get predicted floor."""
        return self.location.floor

    @property
    def building(self) -> Optional[str]:
        """Get predicted building."""
        return self.location.building_id

    @property
    def confidence(self) -> float:
        """Get prediction confidence."""
        return self.location.confidence

    @property
    def x(self) -> float:
        return self.location.x

    @property
    def y(self) -> float:
        return self.location.y


__all__ = ['Location', 'LocalizationResult']
