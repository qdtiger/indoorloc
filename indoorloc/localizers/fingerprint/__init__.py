"""
Fingerprint-based Localization Algorithms

Traditional machine learning methods for WiFi fingerprint localization.
"""
from .knn import KNNLocalizer, WKNNLocalizer

__all__ = [
    'KNNLocalizer',
    'WKNNLocalizer',
]
