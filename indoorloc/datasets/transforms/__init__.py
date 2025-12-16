"""
Data Transforms for Indoor Localization

Provides data augmentation and preprocessing transforms for indoor localization signals.

Built-in transforms:
    - Compose: Chain multiple transforms
    - RSSINormalize: Normalize RSSI values
    - APFilter: Filter weak access points
    - APSelect: Select specific access points
    - GaussianNoise: Add noise for augmentation

Example:
    >>> import indoorloc as iloc
    >>>
    >>> # Create transform pipeline
    >>> transform = iloc.Compose([
    ...     iloc.APFilter(threshold=-90),
    ...     iloc.RSSINormalize(method='minmax'),
    ... ])
    >>>
    >>> # Apply to dataset
    >>> dataset = iloc.UJIndoorLoc(transform=transform)
"""
from .core import BaseTransform, Compose, Identity
from .preprocessing import RSSINormalize, APFilter, APSelect, GaussianNoise

__all__ = [
    # Core
    'BaseTransform',
    'Compose',
    'Identity',
    # Preprocessing
    'RSSINormalize',
    'APFilter',
    'APSelect',
    'GaussianNoise',
]
