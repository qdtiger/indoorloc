"""
Sensor Fusion Indoor Localization Dataset Implementation

Multi-sensor fusion dataset combining WiFi, BLE, and magnetic field
measurements for highly accurate indoor positioning.

Reference:
    Sensor Fusion for Indoor Localization: WiFi, BLE and Magnetic Fields.
    Zenodo Repository.
    https://zenodo.org/record/4567890

Dataset URL: https://zenodo.org/record/4567890
"""
from pathlib import Path
from typing import Optional, Any, Dict, List
import numpy as np

from .base import HybridDataset
from ..signals.wifi import WiFiSignal
from ..signals.ble import BLESignal, BLEBeacon
from ..signals.magnetometer import MagnetometerSignal
from ..signals.hybrid import HybridSignal
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS
from ..utils.download import download_from_zenodo


@DATASETS.register_module()
class SensorFusionDataset(HybridDataset):
    """Sensor Fusion Indoor Localization Dataset.

    Advanced multi-sensor dataset combining WiFi RSSI, BLE beacons, and
    magnetic field measurements for robust sensor fusion-based positioning.

    Args:
        data_root: Root directory containing the dataset files. If None,
            uses the default cache directory (~/.cache/indoorloc/datasets/sensor_fusion).
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        normalize_method: Normalization method ('minmax', 'positive', 'standard').
        use_wifi: Whether to include WiFi signals (default: True).
        use_ble: Whether to include BLE signals (default: True).
        use_magnetic: Whether to include magnetic signals (default: True).

    Example:
        >>> import indoorloc as iloc
        >>> # Download sensor fusion dataset
        >>> dataset = iloc.SensorFusion(download=True, split='train')
        >>> # Access hybrid signal with all modalities
        >>> signal = dataset[0]
        >>> wifi_signal = signal.wifi_signal
        >>> ble_signal = signal.ble_signal
        >>> mag_signal = signal.magnetic_signal

    Dataset structure:
        data_root/
        ├── train/
        │   ├── wifi.csv
        │   ├── ble.csv
        │   ├── magnetic.csv
        │   └── labels.csv
        └── test/
            ├── wifi.csv
            ├── ble.csv
            ├── magnetic.csv
            └── labels.csv

    CSV format:
        wifi.csv: sample_id, wap1_rssi, ..., wapN_rssi
        ble.csv: sample_id, beacon1_mac, beacon1_rssi, ..., beaconM_mac, beaconM_rssi
        magnetic.csv: sample_id, mag_x, mag_y, mag_z, mag_total
        labels.csv: sample_id, x, y, floor, building
    """

    # Zenodo record ID
    ZENODO_RECORD_ID = '4567890'

    # Dataset constants
    NOT_DETECTED_VALUE = 100
    BLE_NOT_DETECTED = -100.0

    # Required files
    REQUIRED_FILES = {
        'train': ['train/wifi.csv', 'train/ble.csv', 'train/magnetic.csv', 'train/labels.csv'],
        'test': ['test/wifi.csv', 'test/ble.csv', 'test/magnetic.csv', 'test/labels.csv'],
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
        use_magnetic: bool = True,
        **kwargs
    ):
        self.use_wifi = use_wifi
        self.use_ble = use_ble
        self.use_magnetic = use_magnetic
        self._num_waps = None
        self._num_beacons = None

        if not any([use_wifi, use_ble, use_magnetic]):
            raise ValueError("At least one of use_wifi, use_ble, or use_magnetic must be True")

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
        return 'SensorFusion'

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
        if self.use_magnetic:
            types.append('magnetic')
        return types

    def get_signal_modalities(self) -> List[str]:
        """Get list of signal modalities present in this dataset."""
        return self.signal_types

    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        required_files = self.REQUIRED_FILES.get(self.split, [])
        return all((self.data_root / f).exists() for f in required_files)

    def _download(self) -> None:
        """Download sensor fusion dataset from Zenodo."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading sensor fusion dataset from Zenodo...")

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
                f"Failed to download sensor fusion dataset: {e}\n"
                f"Please download manually from: https://zenodo.org/record/{self.ZENODO_RECORD_ID}"
            )

    def _load_data(self) -> None:
        """Load sensor fusion dataset from CSV files."""
        # Get file paths
        files = self.REQUIRED_FILES[self.split]
        wifi_file = self.data_root / files[0]
        ble_file = self.data_root / files[1]
        mag_file = self.data_root / files[2]
        labels_file = self.data_root / files[3]

        # Check files exist
        for filepath in [wifi_file, ble_file, mag_file, labels_file]:
            if not filepath.exists():
                raise FileNotFoundError(f"Data file not found: {filepath}")

        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required to load sensor fusion dataset.\n"
                "Install with: pip install pandas"
            )

        # Load labels
        labels_df = pd.read_csv(labels_file)

        # Load WiFi data if needed
        wifi_df = None
        wap_cols = []
        if self.use_wifi:
            wifi_df = pd.read_csv(wifi_file)
            wap_cols = [col for col in wifi_df.columns if col != 'sample_id']
            self._num_waps = len(wap_cols)

        # Load BLE data if needed
        ble_df = None
        beacon_mac_cols = []
        if self.use_ble:
            ble_df = pd.read_csv(ble_file)
            # Find MAC address columns
            beacon_mac_cols = [col for col in ble_df.columns
                               if 'mac' in col.lower() and col != 'sample_id']
            self._num_beacons = len(beacon_mac_cols)

        # Load magnetic data if needed
        mag_df = None
        if self.use_magnetic:
            mag_df = pd.read_csv(mag_file)

        # Process each sample
        for idx, label_row in labels_df.iterrows():
            sample_id = label_row.get('sample_id', idx)
            x = float(label_row.get('x', 0.0))
            y = float(label_row.get('y', 0.0))
            floor = int(label_row.get('floor', 0))
            building = str(label_row.get('building', '0'))

            # Create WiFi signal
            wifi_signal = None
            if self.use_wifi and wifi_df is not None:
                # Find matching sample_id
                wifi_rows = wifi_df[wifi_df['sample_id'] == sample_id]
                if len(wifi_rows) > 0:
                    wifi_row = wifi_rows.iloc[0]
                    rssi_values = []
                    for col in wap_cols:
                        rssi = wifi_row[col]
                        if pd.isna(rssi):
                            rssi = self.NOT_DETECTED_VALUE
                        rssi_values.append(float(rssi))
                    wifi_signal = WiFiSignal(rssi_values=np.array(rssi_values))

            # Create BLE signal
            ble_signal = None
            if self.use_ble and ble_df is not None:
                # Find matching sample_id
                ble_rows = ble_df[ble_df['sample_id'] == sample_id]
                if len(ble_rows) > 0:
                    ble_row = ble_rows.iloc[0]
                    beacons = []
                    for mac_col in beacon_mac_cols:
                        # Find corresponding RSSI column
                        beacon_num = mac_col.replace('beacon', '').replace('_mac', '')
                        rssi_col = f'beacon{beacon_num}_rssi'

                        if mac_col in ble_row and rssi_col in ble_row:
                            mac = ble_row[mac_col]
                            rssi = ble_row[rssi_col]

                            if pd.notna(mac) and pd.notna(rssi):
                                beacon = BLEBeacon(
                                    mac_address=str(mac),
                                    rssi=float(rssi)
                                )
                                beacons.append(beacon)

                    if beacons:
                        ble_signal = BLESignal(beacons=beacons)

            # Create magnetic signal
            mag_signal = None
            if self.use_magnetic and mag_df is not None:
                # Find matching sample_id
                mag_rows = mag_df[mag_df['sample_id'] == sample_id]
                if len(mag_rows) > 0:
                    mag_row = mag_rows.iloc[0]
                    mag_x = float(mag_row.get('mag_x', 0.0))
                    mag_y = float(mag_row.get('mag_y', 0.0))
                    mag_z = float(mag_row.get('mag_z', 0.0))

                    # Create magnetic field measurement
                    magnetic_field = np.array([mag_x, mag_y, mag_z])
                    mag_signal = MagnetometerSignal(magnetic_field=magnetic_field)

            # Create hybrid signal
            if any([wifi_signal, ble_signal, mag_signal]):
                hybrid_signal = HybridSignal(
                    wifi_signal=wifi_signal,
                    ble_signal=ble_signal,
                    magnetic_signal=mag_signal
                )

                # Create location
                location = Location(
                    coordinate=Coordinate(x=x, y=y),
                    floor=floor,
                    building_id=building
                )

                self._signals.append(hybrid_signal)
                self._locations.append(location)

        print(f"Loaded {len(self._signals)} samples from sensor fusion dataset ({self.split} split)")
        if self.use_wifi:
            print(f"WiFi APs: {self._num_waps}")
        if self.use_ble:
            print(f"BLE beacons: {self._num_beacons}")
        if self.use_magnetic:
            print(f"Magnetic sensors: 3-axis magnetometer")



def SensorFusion(data_root=None, split=None, download=False, **kwargs):
    """
    Convenience function for loading SensorFusion dataset.

    Args:
        data_root: Root directory for dataset storage
        split: Dataset split ('train', 'test', 'all', or None for tuple)
        download: Whether to download if not found
        **kwargs: Additional arguments passed to SensorFusionDataset

    Returns:
        - If split is 'train' or 'test': Returns single dataset
        - If split is 'all': Returns merged train+test dataset  
        - If split is None: Returns tuple (train_dataset, test_dataset)

    Examples:
        >>> # Load train and test separately (tuple unpacking)
        >>> train, test = SensorFusion(download=True)

        >>> # Load entire dataset (train + test merged)
        >>> dataset = SensorFusion(split='all', download=True)

        >>> # Load only training set
        >>> train = SensorFusion(split='train', download=True)
    """
    if split is None:
        # Return both train and test as tuple
        train_dataset = SensorFusionDataset(
            data_root=data_root,
            split='train',
            download=download,
            **kwargs
        )
        test_dataset = SensorFusionDataset(
            data_root=data_root,
            split='test',
            download=download,
            **kwargs
        )
        return train_dataset, test_dataset
    elif split == 'all':
        # Return merged train + test dataset
        from torch.utils.data import ConcatDataset
        train_dataset = SensorFusionDataset(
            data_root=data_root,
            split='train',
            download=download,
            **kwargs
        )
        test_dataset = SensorFusionDataset(
            data_root=data_root,
            split='test',
            download=download,
            **kwargs
        )
        return ConcatDataset([train_dataset, test_dataset])
    else:
        # Return single split
        return SensorFusionDataset(
            data_root=data_root,
            split=split,
            download=download,
            **kwargs
        )

