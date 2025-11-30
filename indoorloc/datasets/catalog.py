"""
Dataset Catalog for IndoorLoc

Provides short name aliases for all datasets, enabling HuggingFace-style loading:
    >>> train, test = load_dataset("ujindoorloc")
"""
from typing import Dict, List

# Short name -> Registry class name mapping
DATASET_ALIASES: Dict[str, str] = {
    # ========== WiFi RSSI Datasets ==========
    'ujindoorloc': 'UJIndoorLocDataset',
    'uji': 'UJIndoorLocDataset',
    'sodindoorloc': 'SODIndoorLocDataset',
    'sod': 'SODIndoorLocDataset',
    'tampere': 'TampereDataset',
    'wlanrssi': 'WLANRSSIDataset',
    'wlan': 'WLANRSSIDataset',
    'tuji1': 'TUJI1Dataset',
    'longtermwifi': 'LongTermWiFiDataset',
    'longterm': 'LongTermWiFiDataset',

    # ========== BLE Datasets ==========
    'ble_indoor': 'BLEIndoorDataset',
    'ble': 'BLEIndoorDataset',
    'ibeacon_rssi': 'iBeaconRSSIDataset',
    'ibeacon': 'iBeaconRSSIDataset',
    'ble_rssi_uci': 'BLERSSIUCIDataset',

    # ========== CSI Datasets ==========
    'csi_indoor': 'CSIIndoorDataset',
    'csi': 'CSIIndoorDataset',
    'csi_fingerprint': 'CSIFingerprintDataset',
    'hwild': 'HWILDDataset',
    'csu_indoorloc': 'CSUIndoorLocDataset',
    'csu': 'CSUIndoorLocDataset',
    'wild_v2': 'WILDv2Dataset',
    'wildv2': 'WILDv2Dataset',
    'opencsi': 'OpenCSIDataset',
    'haloc': 'HALOCDataset',
    'csi_bench': 'CSIBenchDataset',
    'csibench': 'CSIBenchDataset',
    'mamimo_csi': 'MaMIMOCSIDataset',
    'mamimo': 'MaMIMOCSIDataset',
    'dichasus': 'DICHASUSDataset',
    'espargos': 'ESPARGOSDataset',
    'csi2pos': 'CSI2PosDataset',
    'csi2taoa': 'CSI2TAoADataset',
    'deepmimo': 'DeepMIMODataset',
    'mamimo_uav': 'MaMIMOUAVDataset',
    'wifi_csi_d2d': 'WiFiCSID2DDataset',

    # ========== UWB Datasets ==========
    'uwb_indoor': 'UWBIndoorDataset',
    'uwb': 'UWBIndoorDataset',
    'uwb_ranging': 'UWBRangingDataset',

    # ========== Hybrid/Multi-modal Datasets ==========
    'wifi_imu': 'WiFiIMUHybridDataset',
    'wifi_magnetic': 'WiFiMagneticHybridDataset',
    'multimodal': 'MultiModalIndoorDataset',
    'sensor_fusion': 'SensorFusionDataset',

    # ========== Other Signal Types ==========
    'magnetic_indoor': 'MagneticIndoorDataset',
    'magnetic': 'MagneticIndoorDataset',
    'vlc_indoor': 'VLCIndoorDataset',
    'vlc': 'VLCIndoorDataset',
    'ultrasound': 'UltrasoundIndoorDataset',
    'rfid_indoor': 'RFIDIndoorDataset',
    'rfid': 'RFIDIndoorDataset',
    'rss_based': 'RSSBasedDataset',
}

# Group datasets by signal type for filtering
DATASETS_BY_SIGNAL: Dict[str, List[str]] = {
    'wifi': [
        'ujindoorloc', 'sodindoorloc', 'tampere',
        'wlanrssi', 'tuji1', 'longtermwifi'
    ],
    'ble': [
        'ble_indoor', 'ibeacon_rssi', 'ble_rssi_uci'
    ],
    'csi': [
        'csi_indoor', 'csi_fingerprint', 'hwild', 'csu_indoorloc',
        'wild_v2', 'opencsi', 'haloc', 'csi_bench', 'mamimo_csi',
        'dichasus', 'espargos', 'csi2pos', 'csi2taoa', 'deepmimo',
        'mamimo_uav', 'wifi_csi_d2d'
    ],
    'uwb': [
        'uwb_indoor', 'uwb_ranging'
    ],
    'hybrid': [
        'wifi_imu', 'wifi_magnetic', 'multimodal', 'sensor_fusion'
    ],
    'magnetic': [
        'magnetic_indoor'
    ],
    'other': [
        'vlc_indoor', 'ultrasound', 'rfid_indoor', 'rss_based'
    ],
}


def get_all_dataset_names() -> List[str]:
    """Get all unique dataset short names (excluding aliases)."""
    # Get primary names (first occurrence of each class)
    seen_classes = set()
    primary_names = []
    for name, cls_name in DATASET_ALIASES.items():
        if cls_name not in seen_classes:
            seen_classes.add(cls_name)
            primary_names.append(name)
    return sorted(primary_names)


def get_dataset_class_name(name: str) -> str:
    """
    Resolve dataset short name to registry class name.

    Args:
        name: Short name like 'ujindoorloc', 'uji', 'tampere'

    Returns:
        Registry class name like 'UJIndoorLocDataset'

    Raises:
        ValueError: If name not found
    """
    name_lower = name.lower().replace('-', '_').replace(' ', '_')

    if name_lower in DATASET_ALIASES:
        return DATASET_ALIASES[name_lower]

    # Try fuzzy match
    for alias, cls_name in DATASET_ALIASES.items():
        if name_lower in alias or alias in name_lower:
            return cls_name

    raise ValueError(
        f"Unknown dataset: '{name}'\n"
        f"Available datasets: {get_all_dataset_names()[:10]}...\n"
        f"Use list_datasets() to see all options."
    )


__all__ = [
    'DATASET_ALIASES',
    'DATASETS_BY_SIGNAL',
    'get_all_dataset_names',
    'get_dataset_class_name',
]
