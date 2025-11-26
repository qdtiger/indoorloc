"""
IndoorLoc Datasets Module

Provides dataset implementations for indoor localization.
"""
from .base import (
    BaseDataset,
    WiFiDataset,
    BLEDataset,
    UWBDataset,
    HybridDataset,
    MagneticDataset,
)
from .ujindoorloc import UJIndoorLocDataset, UJIndoorLoc
from .sodindoorloc import SODIndoorLocDataset, SODIndoorLoc
from .longtermwifi import LongTermWiFiDataset, LongTermWiFi
from .tampere import TampereDataset, Tampere
from .wlanrssi import WLANRSSIDataset, WLANRSSI
from .tuji1 import TUJI1Dataset, TUJI1
from .ibeacon_rssi import iBeaconRSSIDataset, iBeaconRSSI
from .ble_indoor import BLEIndoorDataset, BLEIndoor
from .ble_rssi_uci import BLERSSIUCIDataset, BLERSSIU_UCI
from .wifi_imu_hybrid import WiFiIMUHybridDataset, WiFiIMUHybrid
from .wifi_magnetic_hybrid import WiFiMagneticHybridDataset, WiFiMagneticHybrid
from .multimodal_indoor import MultiModalIndoorDataset, MultiModalIndoor
from .sensor_fusion import SensorFusionDataset, SensorFusion
from .uwb_indoor import UWBIndoorDataset, UWBIndoor
from .uwb_ranging import UWBRangingDataset, UWBRanging
from .magnetic_indoor import MagneticIndoorDataset, MagneticIndoor
from .vlc_indoor import VLCIndoorDataset, VLCIndoor
from .ultrasound_indoor import UltrasoundIndoorDataset, UltrasoundIndoor
from .rss_based import RSSBasedDataset, RSSBased
from .csi_indoor import CSIIndoorDataset, CSIIndoor
from .rfid_indoor import RFIDIndoorDataset, RFIDIndoor

__all__ = [
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
]
