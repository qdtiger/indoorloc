"""
IndoorLoc Datasets Module

Provides dataset implementations for indoor localization.

Example:
    >>> import indoorloc as iloc
    >>> train, test = iloc.load_dataset("ujindoorloc")
"""
from .base import (
    BaseDataset,
    WiFiDataset,
    BLEDataset,
    UWBDataset,
    HybridDataset,
    MagneticDataset,
    CSIDataset,
)

# HuggingFace-style loading API
from .loading import load_dataset, list_datasets, dataset_info
from .catalog import DATASET_ALIASES, DATASETS_BY_SIGNAL
from .ujindoorloc import UJIndoorLocDataset, UJIndoorLoc
from .sodindoorloc import SODIndoorLocDataset, SODIndoorLoc
from .longtermwifi import LongTermWiFiDataset, LongTermWiFi
from .tampere import TampereDataset, Tampere
from .wlanrssi import WLANRSSIDataset, WLANRSSI
from .tuji1 import TUJI1Dataset, TUJI1
from .ibeacon_rssi import iBeaconRSSIDataset, iBeaconRSSI
from .ble_indoor import BLEIndoorDataset, BLEIndoor
from .ble_rssi_uci import BLERSSIUCIDataset, BLERSSIU_UCI

# CSI datasets with real data
from .csi_fingerprint import CSIFingerprintDataset, CSIFingerprint
from .hwild import HWILDDataset, HWILD
from .haloc import HALOCDataset, HALOC

__all__ = [
    # HuggingFace-style API
    'load_dataset',
    'list_datasets',
    'dataset_info',
    'DATASET_ALIASES',
    'DATASETS_BY_SIGNAL',
    # Base classes
    'BaseDataset',
    'WiFiDataset',
    'BLEDataset',
    'UWBDataset',
    'HybridDataset',
    'MagneticDataset',
    'CSIDataset',
    # WiFi RSSI datasets
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
    # BLE datasets
    'iBeaconRSSIDataset',
    'iBeaconRSSI',
    'BLEIndoorDataset',
    'BLEIndoor',
    'BLERSSIUCIDataset',
    'BLERSSIU_UCI',
    # CSI datasets
    'CSIFingerprintDataset',
    'CSIFingerprint',
    'HWILDDataset',
    'HWILD',
    'HALOCDataset',
    'HALOC',
]
