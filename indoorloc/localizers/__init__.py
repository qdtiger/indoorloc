"""
IndoorLoc Localizers Module

Provides localization algorithms for indoor positioning.
"""
from .base import BaseLocalizer, TraditionalLocalizer
from .fingerprint import KNNLocalizer, WKNNLocalizer, SVMLocalizer, RandomForestLocalizer
from .fusion import EnsembleLocalizer, StackingLocalizer
from .transfer import TransferLocalizer, CORALLocalizer, TCALocalizer, KMMLocalizer

__all__ = [
    'BaseLocalizer',
    'TraditionalLocalizer',
    # Fingerprint-based
    'KNNLocalizer',
    'WKNNLocalizer',
    'SVMLocalizer',
    'RandomForestLocalizer',
    # Fusion
    'EnsembleLocalizer',
    'StackingLocalizer',
    # Transfer Learning
    'TransferLocalizer',
    'CORALLocalizer',
    'TCALocalizer',
    'KMMLocalizer',
]
