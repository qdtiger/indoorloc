"""
Multi-modal Indoor Localization Dataset Implementation

Comprehensive multi-modal dataset combining WiFi, BLE, and IMU sensors
for robust indoor positioning across diverse environments.

Reference:
    Multi-modal Indoor Positioning Dataset (MIPD).
    GitHub Repository.
    https://github.com/multi-modal-positioning/dataset

Dataset URL: https://raw.githubusercontent.com/IndoorPositioning/MultiModalDataset/main
"""
from pathlib import Path
from typing import Optional, Any, Dict, List
import numpy as np

from .base import HybridDataset
from ..signals.wifi import WiFiSignal
from ..signals.ble import BLESignal, BLEBeacon
from ..signals.imu import IMUSignal, IMUReading
from ..signals.hybrid import HybridSignal
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS
from ..utils.download import download_url


@DATASETS.register_module()
class MultiModalIndoorDataset(HybridDataset):
    """Multi-modal Indoor Localization Dataset.

    Comprehensive dataset combining WiFi RSSI, BLE beacons, and IMU sensors
    for enhanced indoor positioning with multiple signal modalities.

    Args:
        data_root: Root directory containing the dataset files. If None,
            uses the default cache directory (~/.cache/indoorloc/datasets/multimodal_indoor).
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        normalize_method: Normalization method ('minmax', 'positive', 'standard').
        use_wifi: Whether to include WiFi signals (default: True).
        use_ble: Whether to include BLE signals (default: True).
        use_imu: Whether to include IMU signals (default: True).

    Example:
        >>> import indoorloc as iloc
        >>> # Download multi-modal dataset
        >>> dataset = iloc.MultiModalIndoor(download=True, split='train')
        >>> # Access hybrid signal with all modalities
        >>> signal = dataset[0]
        >>> wifi_signal = signal.wifi_signal
        >>> ble_signal = signal.ble_signal
        >>> imu_signal = signal.imu_signal

    Dataset structure:
        data_root/
        ├── train_data.csv
        ├── test_data.csv
        └── beacon_info.csv

    CSV format (train_data.csv, test_data.csv):
        - Columns: timestamp, x, y, floor, building,
                   wap1_rssi, ..., wapN_rssi,
                   beacon1_mac, beacon1_rssi, ..., beaconM_mac, beaconM_rssi,
                   acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z
    """

    # GitHub raw content base URL
    BASE_URL = 'https://raw.githubusercontent.com/IndoorPositioning/MultiModalDataset/main'

    # Dataset constants
    NOT_DETECTED_VALUE = 100
    BLE_NOT_DETECTED = -100.0

    # File mapping
    FILE_MAPPING = {
        'train': 'train_data.csv',
        'test': 'test_data.csv',
    }

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        use_wifi: bool = True,
        use_ble: bool = True,
        use_imu: bool = True,
        **kwargs
    ):
        self.use_wifi = use_wifi
        self.use_ble = use_ble
        self.use_imu = use_imu
        self._num_waps = None
        self._num_beacons = None

        if not any([use_wifi, use_ble, use_imu]):
            raise ValueError("At least one of use_wifi, use_ble, or use_imu must be True")

        super().__init__(
            data_root=data_root,
            split=split,
            download=download,
            transform=transform,
            normalize=normalize,
            normalize_method=normalize_method,
            **kwargs
        )

    @property
    def dataset_name(self) -> str:
        return 'MultiModalIndoor'

    @property
    def num_waps(self) -> int:
        if self._num_waps is None:
            return 0
        return self._num_waps

    @property
    def num_beacons(self) -> int:
        if self._num_beacons is None:
            return 0
        return self._num_beacons

    @property
    def signal_types(self) -> List[str]:
        """Return list of signal types in this dataset."""
        types = []
        if self.use_wifi:
            types.append('wifi')
        if self.use_ble:
            types.append('ble')
        if self.use_imu:
            types.append('imu')
        return types

    def get_signal_modalities(self) -> List[str]:
        """Get list of signal modalities present in this dataset."""
        return self.signal_types

    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        filename = self.FILE_MAPPING.get(self.split)
        if filename is None:
            return False
        return (self.data_root / filename).exists()

    def _download(self) -> None:
        """Download multi-modal dataset from GitHub."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading multi-modal indoor dataset from GitHub...")

        filename = self.FILE_MAPPING[self.split]
        url = f"{self.BASE_URL}/{filename}"

        try:
            download_url(
                url=url,
                root=self.data_root,
                filename=filename,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download multi-modal dataset: {e}\n"
                f"Please download manually from: {self.BASE_URL}"
            )

    def _load_data(self) -> None:
        """Load multi-modal dataset from CSV file."""
        filename = self.FILE_MAPPING[self.split]
        filepath = self.data_root / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required to load multi-modal dataset.\n"
                "Install with: pip install pandas"
            )

        # Load CSV
        df = pd.read_csv(filepath)

        # Identify column types
        coord_cols = ['timestamp', 'x', 'y', 'floor', 'building']
        imu_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z']

        # WiFi columns: start with 'wap'
        wap_cols = [col for col in df.columns if col.startswith('wap')]
        self._num_waps = len(wap_cols)

        # BLE columns: alternating beacon_mac and beacon_rssi
        ble_cols = [col for col in df.columns if 'beacon' in col.lower()]

        # Count unique beacons
        beacon_mac_cols = [col for col in ble_cols if 'mac' in col.lower()]
        self._num_beacons = len(beacon_mac_cols)

        # Process each sample
        for idx, row in df.iterrows():
            # Extract location
            x = float(row.get('x', 0.0))
            y = float(row.get('y', 0.0))
            floor = int(row.get('floor', 0))
            building = str(row.get('building', '0'))
            timestamp = row.get('timestamp', idx)

            # Create WiFi signal
            wifi_signal = None
            if self.use_wifi and len(wap_cols) > 0:
                rssi_values = []
                for col in wap_cols:
                    rssi = row[col]
                    if pd.isna(rssi):
                        rssi = self.NOT_DETECTED_VALUE
                    rssi_values.append(float(rssi))
                wifi_signal = WiFiSignal(rssi_values=np.array(rssi_values))

            # Create BLE signal
            ble_signal = None
            if self.use_ble and len(beacon_mac_cols) > 0:
                beacons = []
                for mac_col in beacon_mac_cols:
                    # Find corresponding RSSI column
                    beacon_num = mac_col.replace('beacon', '').replace('_mac', '')
                    rssi_col = f'beacon{beacon_num}_rssi'

                    if mac_col in row and rssi_col in row:
                        mac = row[mac_col]
                        rssi = row[rssi_col]

                        if pd.notna(mac) and pd.notna(rssi):
                            beacon = BLEBeacon(
                                mac_address=str(mac),
                                rssi=float(rssi)
                            )
                            beacons.append(beacon)

                if beacons:
                    ble_signal = BLESignal(beacons=beacons)

            # Create IMU signal
            imu_signal = None
            if self.use_imu and all(col in row for col in imu_cols):
                acc_x = float(row.get('acc_x', 0.0))
                acc_y = float(row.get('acc_y', 0.0))
                acc_z = float(row.get('acc_z', 0.0))
                gyro_x = float(row.get('gyro_x', 0.0))
                gyro_y = float(row.get('gyro_y', 0.0))
                gyro_z = float(row.get('gyro_z', 0.0))
                mag_x = float(row.get('mag_x', 0.0))
                mag_y = float(row.get('mag_y', 0.0))
                mag_z = float(row.get('mag_z', 0.0))

                imu_reading = IMUReading(
                    accelerometer=np.array([acc_x, acc_y, acc_z]),
                    gyroscope=np.array([gyro_x, gyro_y, gyro_z]),
                    magnetometer=np.array([mag_x, mag_y, mag_z]),
                    timestamp=timestamp
                )
                imu_signal = IMUSignal(readings=[imu_reading])

            # Create hybrid signal
            if any([wifi_signal, ble_signal, imu_signal]):
                hybrid_signal = HybridSignal(
                    wifi_signal=wifi_signal,
                    ble_signal=ble_signal,
                    imu_signal=imu_signal
                )

                # Create location
                location = Location(
                    coordinate=Coordinate(x=x, y=y),
                    floor=floor,
                    building_id=building
                )

                self._signals.append(hybrid_signal)
                self._locations.append(location)

        print(f"Loaded {len(self._signals)} samples from multi-modal indoor dataset ({self.split} split)")
        if self.use_wifi:
            print(f"WiFi APs: {self._num_waps}")
        if self.use_ble:
            print(f"BLE beacons: {self._num_beacons}")
        if self.use_imu:
            print(f"IMU sensors: accelerometer, gyroscope, magnetometer")


# Alias for convenience
MultiModalIndoor = MultiModalIndoorDataset
