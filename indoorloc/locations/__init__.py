"""
IndoorLoc Locations Module

Provides location and coordinate abstractions for indoor localization.
"""
from .coordinate import Coordinate
from .location import Location, LocalizationResult

__all__ = [
    'Coordinate',
    'Location',
    'LocalizationResult',
]
