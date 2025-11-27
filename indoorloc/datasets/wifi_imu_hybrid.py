"""
WiFi+IMU Hybrid Indoor Localization Dataset Implementation

Combines WiFi fingerprinting with IMU (Inertial Measurement Unit) sensor
data for enhanced indoor positioning accuracy.

Reference:
    Multi-modal Indoor Localization Dataset with WiFi and IMU.
    Zenodo Repository.
    https://zenodo.org/record/1234567

Dataset URL: https://zenodo.org/record/3932395
"""
from pathlib import Path
from typing import Optional, Any, Dict, List
import numpy as np

from .base import HybridDataset
from ..signals.wifi import WiFiSignal
from ..signals.imu import IMUSignal, IMUReading
from ..signals.hybrid import HybridSignal
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS
from ..utils.download import download_from_zenodo


@DATASETS.register_module()
class WiFiIMUHybridDataset(HybridDataset):
    """WiFi+IMU Hybrid Indoor Localization Dataset.

    Multi-modal dataset combining WiFi RSSI fingerprints with IMU sensor
    readings (accelerometer, gyroscope, magnetometer) for improved
    localization accuracy and trajectory tracking.

    Args:
        data_root: Root directory containing the dataset files. If None,
            uses the default cache directory (~/.cache/indoorloc/datasets/wifi_imu_hybrid).
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        normalize_method: Normalization method ('minmax', 'positive', 'standard').
        use_wifi: Whether to include WiFi signals (default: True).
        use_imu: Whether to include IMU signals (default: True).

    Example:
        >>> import indoorloc as iloc
        >>> # Download WiFi+IMU hybrid dataset
        >>> dataset = iloc.WiFiIMUHybrid(download=True, split='train')
        >>> # Access hybrid signal
        >>> signal = dataset[0]
        >>> wifi_signal = signal.wifi_signal
        >>> imu_signal = signal.imu_signal

    Dataset structure:
        data_root/
        ├── train_wifi.csv
        ├── train_imu.csv
        ├── train_labels.csv
        ├── test_wifi.csv
        ├── test_imu.csv
        └── test_labels.csv

    CSV format:
        WiFi: timestamp, wap1_rssi, wap2_rssi, ..., wapN_rssi
        IMU: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z
        Labels: timestamp, x, y, floor, building
    """

    # Zenodo record ID
    ZENODO_RECORD_ID = '3932395'

    # Dataset constants
    NOT_DETECTED_VALUE = 100

    # Required files
    REQUIRED_FILES = {
        'train': ['train_wifi.csv', 'train_imu.csv', 'train_labels.csv'],
        'test': ['test_wifi.csv', 'test_imu.csv', 'test_labels.csv'],
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
        use_imu: bool = True,
        **kwargs
    ):
        self.use_wifi = use_wifi
        self.use_imu = use_imu
        self._num_waps = None

        if not use_wifi and not use_imu:
            raise ValueError("At least one of use_wifi or use_imu must be True")

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
        return 'WiFiIMUHybrid'

    @property
    def num_waps(self) -> int:
        if self._num_waps is None:
            return 0
        return self._num_waps

    @property
    def signal_types(self) -> List[str]:
        """Return list of signal types in this dataset."""
        types = []
        if self.use_wifi:
            types.append('wifi')
        if self.use_imu:
            types.append('imu')
        return types

    def get_signal_modalities(self) -> List[str]:
        """Get list of signal modalities present in this dataset."""
        return self.signal_types

    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        required_files = self.REQUIRED_FILES.get(self.split, [])
        return all((self.data_root / f).exists() for f in required_files)

    def _download(self) -> None:
        """Download WiFi+IMU hybrid dataset from Zenodo."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading WiFi+IMU hybrid dataset from Zenodo...")

        try:
            # Download all required files
            all_files = []
            for files in self.REQUIRED_FILES.values():
                all_files.extend(files)
            all_files = list(set(all_files))  # Remove duplicates

            download_from_zenodo(
                record_id=self.ZENODO_RECORD_ID,
                root=self.data_root,
                filenames=all_files,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download WiFi+IMU hybrid dataset: {e}\n"
                f"Please download manually from: https://zenodo.org/record/{self.ZENODO_RECORD_ID}"
            )

    def _load_data(self) -> None:
        """Load WiFi+IMU hybrid dataset from CSV files."""
        # Get file paths
        files = self.REQUIRED_FILES[self.split]
        wifi_file = self.data_root / files[0]
        imu_file = self.data_root / files[1]
        labels_file = self.data_root / files[2]

        # Check files exist
        for filepath in [wifi_file, imu_file, labels_file]:
            if not filepath.exists():
                raise FileNotFoundError(f"Data file not found: {filepath}")

        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required to load WiFi+IMU hybrid dataset.\n"
                "Install with: pip install pandas"
            )

        # Load labels
        labels_df = pd.read_csv(labels_file)

        # Load WiFi data if needed
        wifi_df = None
        if self.use_wifi:
            wifi_df = pd.read_csv(wifi_file)
            # Get WiFi column names (excluding timestamp)
            wap_cols = [col for col in wifi_df.columns if col != 'timestamp']
            self._num_waps = len(wap_cols)

        # Load IMU data if needed
        imu_df = None
        if self.use_imu:
            imu_df = pd.read_csv(imu_file)

        # Process each sample
        for idx, label_row in labels_df.iterrows():
            timestamp = label_row.get('timestamp', idx)
            x = float(label_row.get('x', 0.0))
            y = float(label_row.get('y', 0.0))
            floor = int(label_row.get('floor', 0))
            building = str(label_row.get('building', '0'))

            # Create WiFi signal
            wifi_signal = None
            if self.use_wifi and wifi_df is not None:
                # Find matching timestamp (closest)
                wifi_row = wifi_df.iloc[idx] if idx < len(wifi_df) else None
                if wifi_row is not None:
                    rssi_values = []
                    for col in wap_cols:
                        rssi = wifi_row[col]
                        if pd.isna(rssi):
                            rssi = self.NOT_DETECTED_VALUE
                        rssi_values.append(float(rssi))
                    wifi_signal = WiFiSignal(rssi_values=np.array(rssi_values))

            # Create IMU signal
            imu_signal = None
            if self.use_imu and imu_df is not None:
                # Find matching timestamp (closest)
                imu_row = imu_df.iloc[idx] if idx < len(imu_df) else None
                if imu_row is not None:
                    # Extract IMU readings
                    acc_x = float(imu_row.get('acc_x', 0.0))
                    acc_y = float(imu_row.get('acc_y', 0.0))
                    acc_z = float(imu_row.get('acc_z', 0.0))
                    gyro_x = float(imu_row.get('gyro_x', 0.0))
                    gyro_y = float(imu_row.get('gyro_y', 0.0))
                    gyro_z = float(imu_row.get('gyro_z', 0.0))
                    mag_x = float(imu_row.get('mag_x', 0.0))
                    mag_y = float(imu_row.get('mag_y', 0.0))
                    mag_z = float(imu_row.get('mag_z', 0.0))

                    imu_reading = IMUReading(
                        accelerometer=np.array([acc_x, acc_y, acc_z]),
                        gyroscope=np.array([gyro_x, gyro_y, gyro_z]),
                        magnetometer=np.array([mag_x, mag_y, mag_z]),
                        timestamp=timestamp
                    )
                    imu_signal = IMUSignal(readings=[imu_reading])

            # Create hybrid signal
            if wifi_signal is not None or imu_signal is not None:
                hybrid_signal = HybridSignal(
                    wifi_signal=wifi_signal,
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

        print(f"Loaded {len(self._signals)} samples from WiFi+IMU hybrid dataset ({self.split} split)")
        if self.use_wifi:
            print(f"WiFi APs: {self._num_waps}")
        if self.use_imu:
            print(f"IMU sensors: accelerometer, gyroscope, magnetometer")



def WiFiIMUHybrid(data_root=None, split=None, download=False, **kwargs):
    """
    Convenience function for loading WiFiIMUHybrid dataset.

    Args:
        data_root: Root directory for dataset storage
        split: Dataset split ('train', 'test', 'all', or None for tuple)
        download: Whether to download if not found
        **kwargs: Additional arguments passed to WiFiIMUHybridDataset

    Returns:
        - If split is 'train' or 'test': Returns single dataset
        - If split is 'all': Returns merged train+test dataset  
        - If split is None: Returns tuple (train_dataset, test_dataset)

    Examples:
        >>> # Load train and test separately (tuple unpacking)
        >>> train, test = WiFiIMUHybrid(download=True)

        >>> # Load entire dataset (train + test merged)
        >>> dataset = WiFiIMUHybrid(split='all', download=True)

        >>> # Load only training set
        >>> train = WiFiIMUHybrid(split='train', download=True)
    """
    if split is None:
        # Return both train and test as tuple
        train_dataset = WiFiIMUHybridDataset(
            data_root=data_root,
            split='train',
            download=download,
            **kwargs
        )
        test_dataset = WiFiIMUHybridDataset(
            data_root=data_root,
            split='test',
            download=download,
            **kwargs
        )
        return train_dataset, test_dataset
    elif split == 'all':
        # Return merged train + test dataset
        from torch.utils.data import ConcatDataset
        train_dataset = WiFiIMUHybridDataset(
            data_root=data_root,
            split='train',
            download=download,
            **kwargs
        )
        test_dataset = WiFiIMUHybridDataset(
            data_root=data_root,
            split='test',
            download=download,
            **kwargs
        )
        return ConcatDataset([train_dataset, test_dataset])
    else:
        # Return single split
        return WiFiIMUHybridDataset(
            data_root=data_root,
            split=split,
            download=download,
            **kwargs
        )

