"""
IndoorLoc: A Unified Framework for Indoor Localization

A comprehensive library for indoor positioning supporting multiple algorithms,
datasets, and sensor modalities.

Example:
    >>> import indoorloc as iloc
    >>> model = iloc.create_model('knn', k=5)
    >>> model.fit(train_signals, train_locations)
    >>> result = model.predict(test_signal)
    >>> print(f"Position: ({result.x:.2f}, {result.y:.2f})")
"""
from .version import __version__, __version_info__

from .registry import (
    Registry,
    SIGNALS,
    DATASETS,
    TRANSFORMS,
    LOCALIZERS,
    FUSIONS,
    METRICS,
    BACKBONES,
    HEADS,
    TRAINERS,
    VISUALIZERS,
)

from .signals import (
    BaseSignal,
    SignalMetadata,
    WiFiSignal,
    APInfo,
    BLESignal,
    BLEBeacon,
    IMUSignal,
    IMUReading,
)

from .locations import (
    Coordinate,
    Location,
    LocalizationResult,
)

from .localizers import (
    BaseLocalizer,
    TraditionalLocalizer,
    KNNLocalizer,
    WKNNLocalizer,
)

from .utils import (
    Config,
    load_config,
    merge_configs,
    get_data_home,
)

from .datasets import (
    UJIndoorLocDataset,
    UJIndoorLoc,
    SODIndoorLocDataset,
    SODIndoorLoc,
)


# Convenience functions
def create_model(
    model_type: str = None,
    config: str = None,
    **kwargs
) -> BaseLocalizer:
    """
    Create a localization model.

    Args:
        model_type: Model type name (e.g., 'knn', 'wknn')
        config: Path to configuration file
        **kwargs: Model-specific arguments

    Returns:
        Localizer instance

    Example:
        >>> model = iloc.create_model('knn', k=5)
        >>> model = iloc.create_model(config='configs/wifi/knn.yaml')
    """
    if config is not None:
        cfg = load_config(config)
        model_cfg = cfg.get('model', cfg.to_dict())
        return LOCALIZERS.build(model_cfg)

    if model_type is not None:
        return LOCALIZERS.build({'type': model_type, **kwargs})

    raise ValueError("Must specify either model_type or config")


def build_model(cfg: dict) -> BaseLocalizer:
    """
    Build a model from configuration dictionary.

    Args:
        cfg: Model configuration dict with 'type' key

    Returns:
        Localizer instance

    Example:
        >>> model = iloc.build_model({'type': 'KNNLocalizer', 'k': 5})
    """
    return LOCALIZERS.build(cfg)


def list_models() -> list:
    """
    List all available model types.

    Returns:
        List of model type names
    """
    return LOCALIZERS.list_modules()


def list_datasets() -> list:
    """
    List all available dataset types.

    Returns:
        List of dataset type names
    """
    return DATASETS.list_modules()


__all__ = [
    # Version
    '__version__',
    '__version_info__',

    # Registry
    'Registry',
    'SIGNALS',
    'DATASETS',
    'TRANSFORMS',
    'LOCALIZERS',
    'FUSIONS',
    'METRICS',
    'BACKBONES',
    'HEADS',
    'TRAINERS',
    'VISUALIZERS',

    # Signals
    'BaseSignal',
    'SignalMetadata',
    'WiFiSignal',
    'APInfo',
    'BLESignal',
    'BLEBeacon',
    'IMUSignal',
    'IMUReading',

    # Locations
    'Coordinate',
    'Location',
    'LocalizationResult',

    # Localizers
    'BaseLocalizer',
    'TraditionalLocalizer',
    'KNNLocalizer',
    'WKNNLocalizer',

    # Utils
    'Config',
    'load_config',
    'merge_configs',
    'get_data_home',

    # Datasets
    'UJIndoorLocDataset',
    'UJIndoorLoc',
    'SODIndoorLocDataset',
    'SODIndoorLoc',

    # Convenience functions
    'create_model',
    'build_model',
    'list_models',
    'list_datasets',
]
