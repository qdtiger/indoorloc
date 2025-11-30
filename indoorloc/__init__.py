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

from .models import (
    # Backbones
    BaseBackbone,
    InputAdapter,
    TimmBackbone,
    # Heads
    BaseHead,
    RegressionHead,
    MultiScaleRegressionHead,
    ClassificationHead,
    FloorHead,
    BuildingHead,
    HybridHead,
    HierarchicalHead,
    # Localizers
    DeepLocalizer,
)

from .utils import (
    Config,
    load_config,
    merge_configs,
    get_data_home,
    print_config_help,
    get_default_config,
    explain_model,
    explain_dataset,
    explain_config,
)

from .datasets import (
    # HuggingFace-style API (top-level convenience)
    load_dataset,
    list_datasets as list_available_datasets,
    dataset_info,
    # Base classes
    BaseDataset,
    WiFiDataset,
    BLEDataset,
    UWBDataset,
    HybridDataset,
    MagneticDataset,
    UJIndoorLocDataset,
    UJIndoorLoc,
    SODIndoorLocDataset,
    SODIndoorLoc,
    LongTermWiFiDataset,
    LongTermWiFi,
    TampereDataset,
    Tampere,
    WLANRSSIDataset,
    WLANRSSI,
    TUJI1Dataset,
    TUJI1,
    iBeaconRSSIDataset,
    iBeaconRSSI,
    BLEIndoorDataset,
    BLEIndoor,
    BLERSSIUCIDataset,
    BLERSSIU_UCI,
    WiFiIMUHybridDataset,
    WiFiIMUHybrid,
    WiFiMagneticHybridDataset,
    WiFiMagneticHybrid,
    MultiModalIndoorDataset,
    MultiModalIndoor,
    SensorFusionDataset,
    SensorFusion,
    UWBIndoorDataset,
    UWBIndoor,
    UWBRangingDataset,
    UWBRanging,
    MagneticIndoorDataset,
    MagneticIndoor,
    VLCIndoorDataset,
    VLCIndoor,
    UltrasoundIndoorDataset,
    UltrasoundIndoor,
    RSSBasedDataset,
    RSSBased,
    CSIIndoorDataset,
    CSIIndoor,
    RFIDIndoorDataset,
    RFIDIndoor,
    # New CSI datasets
    CSIFingerprintDataset,
    CSIFingerprint,
    HWILDDataset,
    HWILD,
    CSUIndoorLocDataset,
    CSUIndoorLoc,
    WILDv2Dataset,
    WILDv2,
    OpenCSIDataset,
    OpenCSI,
    HALOCDataset,
    HALOC,
    CSIBenchDataset,
    CSIBench,
    MaMIMOCSIDataset,
    MaMIMOCSI,
    DICHASUSDataset,
    DICHASUS,
    ESPARGOSDataset,
    ESPARGOS,
    CSI2PosDataset,
    CSI2Pos,
    CSI2TAoADataset,
    CSI2TAoA,
    DeepMIMODataset,
    DeepMIMO,
    MaMIMOUAVDataset,
    MaMIMOUAV,
    WiFiCSID2DDataset,
    WiFiCSID2D,
)


# Known traditional model aliases
_TRADITIONAL_MODELS = {
    'knn': 'KNNLocalizer',
    'wknn': 'WKNNLocalizer',
    # Future: 'svm', 'rf', 'gp', etc.
}


def _is_timm_model(name: str) -> bool:
    """Check if a name corresponds to a timm model."""
    # Common model name patterns
    common_patterns = [
        'resnet', 'efficientnet', 'mobilenet', 'convnext',
        'vit_', 'swin', 'deit', 'beit', 'densenet', 'inception',
        'regnet', 'resnext', 'wide_resnet', 'vgg', 'alexnet',
        'nfnet', 'efficientformer', 'poolformer', 'pvt',
        'maxvit', 'coatnet', 'mixer', 'mlp_mixer', 'crossvit',
    ]
    name_lower = name.lower()

    # First check pattern match (works whether timm is installed or not)
    if any(name_lower.startswith(p) or p in name_lower for p in common_patterns):
        return True

    # If timm is installed, also check exact match
    try:
        import timm
        return name in timm.list_models()
    except ImportError:
        return False


# Convenience functions
def create_model(
    model_type: str = None,
    dataset: 'BaseDataset' = None,
    config: str = None,
    input_dim: int = None,
    num_coords: int = 2,
    num_floors: int = None,
    num_buildings: int = None,
    pretrained: bool = True,
    **kwargs
) -> BaseLocalizer:
    """
    Create a localization model.

    Supports three modes:
    1. **Auto mode**: `create_model('auto', dataset=train)` - automatically selects
       model type based on dataset size and configures dimensions.
    2. **Explicit mode**: `create_model('knn')` or `create_model('resnet18')`
    3. **Config mode**: `create_model(config='configs/wifi/knn.yaml')`

    Args:
        model_type: Model type name
            - 'auto': Automatically select based on dataset (requires dataset param)
            - Traditional: 'knn', 'wknn'
            - Deep learning (timm): 'resnet18', 'efficientnet_b0', 'mobilenetv3_small', etc.
        dataset: Dataset for auto-configuration. When provided:
            - input_dim is extracted from dataset.input_dim
            - num_floors/num_buildings from dataset.output_dim
            - For 'auto' mode: selects traditional (≤5000 samples) or deep learning
        config: Path to configuration file
        input_dim: Input dimension (auto-inferred from dataset if provided)
        num_coords: Number of coordinate outputs (default: 2 for x, y)
        num_floors: Number of floors for classification (None to skip)
        num_buildings: Number of buildings for classification (None to skip)
        pretrained: Whether to use pretrained weights for deep learning models
        **kwargs: Model-specific arguments

    Returns:
        Localizer instance (BaseLocalizer or DeepLocalizer)

    Example:
        >>> import indoorloc as iloc
        >>> train, test = iloc.load_dataset("ujindoorloc")
        >>>
        >>> # Auto mode (recommended for beginners)
        >>> model = iloc.create_model('auto', dataset=train)
        >>>
        >>> # Traditional ML
        >>> model = iloc.create_model('knn', k=5)
        >>> model = iloc.create_model('wknn', dataset=train)  # auto-configure dims
        >>>
        >>> # Deep Learning
        >>> model = iloc.create_model('resnet18', dataset=train)
        >>> model = iloc.create_model('efficientnet_b0', pretrained=True)
        >>>
        >>> # With config file
        >>> model = iloc.create_model(config='configs/wifi/knn.yaml')
    """
    # Config file takes precedence
    if config is not None:
        cfg = load_config(config)
        model_cfg = cfg.get('model', cfg.to_dict())
        return LOCALIZERS.build(model_cfg)

    # Extract configuration from dataset if provided
    if dataset is not None:
        # Get input dimension
        if input_dim is None:
            ds_input_dim = dataset.input_dim
            input_dim = ds_input_dim[0] if len(ds_input_dim) == 1 else ds_input_dim

        # Get output dimensions
        ds_output_dim = dataset.output_dim
        if num_floors is None and ds_output_dim.get('num_floors', 1) > 1:
            num_floors = ds_output_dim['num_floors']
        if num_buildings is None and ds_output_dim.get('num_buildings', 1) > 1:
            num_buildings = ds_output_dim['num_buildings']
        if 'num_coords' in ds_output_dim:
            num_coords = ds_output_dim['num_coords']

    if model_type is None:
        raise ValueError("Must specify either model_type or config")

    model_type_lower = model_type.lower()

    # Handle 'auto' mode
    if model_type_lower == 'auto':
        if dataset is None:
            raise ValueError(
                "model_type='auto' requires dataset parameter.\n"
                "Usage: create_model('auto', dataset=train_dataset)"
            )
        # Select model based on dataset size
        # Traditional ML for small datasets, deep learning for large
        if len(dataset) <= 5000:
            model_type_lower = 'wknn'  # Default traditional model
        else:
            model_type_lower = 'resnet18'  # Default deep learning model

    # Check if it's a traditional ML model
    if model_type_lower in _TRADITIONAL_MODELS:
        registered_name = _TRADITIONAL_MODELS[model_type_lower]
        return LOCALIZERS.build({'type': registered_name, **kwargs})

    # Check if it's a registered localizer (e.g., 'KNNLocalizer')
    if model_type_lower in [m.lower() for m in LOCALIZERS.list_modules()]:
        # Find exact case match
        for registered in LOCALIZERS.list_modules():
            if registered.lower() == model_type_lower:
                return LOCALIZERS.build({'type': registered, **kwargs})

    # Check if it's a deep learning model (timm)
    if _is_timm_model(model_type_lower) or _is_timm_model(model_type):
        actual_model_name = model_type_lower if _is_timm_model(model_type_lower) else model_type

        # Build head configuration
        head_config = {
            'type': 'HybridHead' if (num_floors or num_buildings) else 'RegressionHead',
            'num_coords': num_coords,
        }
        if num_floors:
            head_config['num_floors'] = num_floors
        if num_buildings:
            head_config['num_buildings'] = num_buildings

        # Build backbone configuration
        backbone_config = {
            'type': 'TimmBackbone',
            'model_name': actual_model_name,
            'pretrained': pretrained,
            'input_type': '1d',  # Default for RSSI fingerprints
        }
        if input_dim is not None:
            backbone_config['input_size'] = input_dim

        return DeepLocalizer(
            backbone=backbone_config,
            head=head_config,
            **kwargs
        )

    raise ValueError(
        f"Unknown model type: '{model_type}'\n"
        f"Available options:\n"
        f"  - 'auto': Auto-select based on dataset (requires dataset param)\n"
        f"  - Traditional: {list(_TRADITIONAL_MODELS.keys())}\n"
        f"  - Deep learning: timm model names (e.g., 'resnet18', 'efficientnet_b0')"
    )


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

    # ===== HuggingFace-style API (Recommended) =====
    'load_dataset',           # load_dataset("ujindoorloc") → (train, test)
    'list_available_datasets',  # list available dataset names
    'dataset_info',           # get dataset metadata
    'create_model',           # create_model('auto', dataset=train)
    'build_model',            # build from config dict
    'list_models',            # list available model types
    'list_datasets',          # list registered dataset classes (old API)

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

    # Models - Backbones
    'BaseBackbone',
    'InputAdapter',
    'TimmBackbone',

    # Models - Heads
    'BaseHead',
    'RegressionHead',
    'MultiScaleRegressionHead',
    'ClassificationHead',
    'FloorHead',
    'BuildingHead',
    'HybridHead',
    'HierarchicalHead',

    # Models - Deep Localizers
    'DeepLocalizer',

    # Utils
    'Config',
    'load_config',
    'merge_configs',
    'get_data_home',
    'explain_model',
    'explain_dataset',
    'explain_config',

    # Datasets
    'BaseDataset',
    'WiFiDataset',
    'BLEDataset',
    'UWBDataset',
    'HybridDataset',
    'MagneticDataset',
    'UJIndoorLocDataset',
    'UJIndoorLoc',
    'SODIndoorLocDataset',
    'SODIndoorLoc',
    'LongTermWiFiDataset',
    'LongTermWiFi',
    'TampereDataset',
    'Tampere',
    'WLANRSSIDataset',
    'WLANRSSI',
    'TUJI1Dataset',
    'TUJI1',
    'iBeaconRSSIDataset',
    'iBeaconRSSI',
    'BLEIndoorDataset',
    'BLEIndoor',
    'BLERSSIUCIDataset',
    'BLERSSIU_UCI',
    'WiFiIMUHybridDataset',
    'WiFiIMUHybrid',
    'WiFiMagneticHybridDataset',
    'WiFiMagneticHybrid',
    'MultiModalIndoorDataset',
    'MultiModalIndoor',
    'SensorFusionDataset',
    'SensorFusion',
    'UWBIndoorDataset',
    'UWBIndoor',
    'UWBRangingDataset',
    'UWBRanging',
    'MagneticIndoorDataset',
    'MagneticIndoor',
    'VLCIndoorDataset',
    'VLCIndoor',
    'UltrasoundIndoorDataset',
    'UltrasoundIndoor',
    'RSSBasedDataset',
    'RSSBased',
    'CSIIndoorDataset',
    'CSIIndoor',
    'RFIDIndoorDataset',
    'RFIDIndoor',
    # New CSI datasets
    'CSIFingerprintDataset',
    'CSIFingerprint',
    'HWILDDataset',
    'HWILD',
    'CSUIndoorLocDataset',
    'CSUIndoorLoc',
    'WILDv2Dataset',
    'WILDv2',
    'OpenCSIDataset',
    'OpenCSI',
    'HALOCDataset',
    'HALOC',
    'CSIBenchDataset',
    'CSIBench',
    'MaMIMOCSIDataset',
    'MaMIMOCSI',
    'DICHASUSDataset',
    'DICHASUS',
    'ESPARGOSDataset',
    'ESPARGOS',
    'CSI2PosDataset',
    'CSI2Pos',
    'CSI2TAoADataset',
    'CSI2TAoA',
    'DeepMIMODataset',
    'DeepMIMO',
    'MaMIMOUAVDataset',
    'MaMIMOUAV',
    'WiFiCSID2DDataset',
    'WiFiCSID2D',
]
