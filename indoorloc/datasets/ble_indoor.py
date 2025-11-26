"""
BLE Indoor Localization Dataset Implementation

Bluetooth Low Energy indoor positioning dataset with comprehensive
beacon deployment and multi-floor coverage.

Reference:
    BLE Indoor Positioning Dataset. GitHub Repository.
    https://github.com/datasets/ble-indoor-positioning

Dataset URL: https://raw.githubusercontent.com/BLE-Indoor-Positioning/Dataset/main
"""
from pathlib import Path
from typing import Optional, Any, Dict, List
import numpy as np

from .base import BLEDataset
from ..signals.ble import BLESignal, BLEBeacon
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS
from ..utils.download import download_url


@DATASETS.register_module()
class BLEIndoorDataset(BLEDataset):
    """BLE Indoor Localization Dataset.

    Comprehensive BLE fingerprinting dataset with multi-floor coverage
    and systematic beacon placement.

    Args:
        data_root: Root directory containing the dataset files. If None,
            uses the default cache directory (~/.cache/indoorloc/datasets/ble_indoor).
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        floor: Floor number to load (None for all floors).
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize RSSI values.
        normalize_method: Normalization method ('minmax', 'positive', 'standard').

    Example:
        >>> import indoorloc as iloc
        >>> # Download from GitHub
        >>> dataset = iloc.BLEIndoor(download=True, floor=1)

    Dataset structure:
        data_root/
        ├── floor1_train.csv
        ├── floor1_test.csv
        ├── floor2_train.csv
        └── floor2_test.csv

    CSV format:
        - Columns: x, y, beacon_mac_1, rssi_1, beacon_mac_2, rssi_2, ...
        - RSSI values in dBm
        - MAC addresses as beacon identifiers
    """

    # GitHub raw content base URL
    BASE_URL = 'https://raw.githubusercontent.com/BLE-Indoor-Positioning/Dataset/main'

    # Dataset constants
    NOT_DETECTED_VALUE = -100.0

    # Available floors
    AVAILABLE_FLOORS = [1, 2, 3]

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        floor: Optional[int] = None,
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        **kwargs
    ):
        self.floor = floor
        self._num_beacons = None

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
        return 'BLEIndoor'

    @property
    def num_beacons(self) -> int:
        if self._num_beacons is None:
            return 0
        return self._num_beacons

    def _get_filenames(self) -> List[str]:
        """Get list of filenames to download/load."""
        if self.floor is not None:
            return [f'floor{self.floor}_{self.split}.csv']
        else:
            # All floors
            return [f'floor{f}_{self.split}.csv' for f in self.AVAILABLE_FLOORS]

    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        filenames = self._get_filenames()
        return all((self.data_root / f).exists() for f in filenames)

    def _download(self) -> None:
        """Download BLE Indoor dataset from GitHub."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading BLE Indoor dataset from GitHub...")

        filenames = self._get_filenames()

        for filename in filenames:
            url = f"{self.BASE_URL}/{filename}"
            try:
                download_url(
                    url=url,
                    root=self.data_root,
                    filename=filename,
                )
            except Exception as e:
                print(f"Warning: Failed to download {filename}: {e}")
                continue

    def _load_data(self) -> None:
        """Load BLE Indoor dataset from CSV files."""
        filenames = self._get_filenames()

        # Collect all unique MAC addresses across all files
        all_macs = set()
        all_samples = []

        for filename in filenames:
            filepath = self.data_root / filename

            if not filepath.exists():
                print(f"Warning: File not found: {filepath}")
                continue

            # Extract floor number from filename
            floor_num = int(filename.split('_')[0].replace('floor', ''))

            # Load CSV
            try:
                import pandas as pd
                df = pd.read_csv(filepath)
            except ImportError:
                raise ImportError(
                    "pandas is required to load BLE Indoor dataset.\n"
                    "Install with: pip install pandas"
                )

            # Parse data
            # Columns: x, y, beacon_mac_1, rssi_1, beacon_mac_2, rssi_2, ...
            coord_cols = ['x', 'y']
            existing_coord_cols = [col for col in coord_cols if col in df.columns]

            # Remaining columns are beacon MAC-RSSI pairs
            other_cols = [col for col in df.columns if col not in existing_coord_cols]

            for idx, row in df.iterrows():
                x = float(row['x']) if 'x' in row else 0.0
                y = float(row['y']) if 'y' in row else 0.0

                # Parse beacon measurements (alternating MAC, RSSI columns)
                beacon_measurements = {}
                i = 0
                while i < len(other_cols) - 1:
                    mac_col = other_cols[i]
                    rssi_col = other_cols[i + 1]

                    if 'mac' in mac_col.lower() or 'beacon' in mac_col.lower():
                        mac = row[mac_col]
                        rssi = row[rssi_col]

                        if pd.notna(mac) and pd.notna(rssi):
                            mac = str(mac).strip()
                            if mac and mac != 'nan':
                                all_macs.add(mac)
                                beacon_measurements[mac] = float(rssi)

                    i += 2

                all_samples.append({
                    'x': x,
                    'y': y,
                    'floor': floor_num,
                    'measurements': beacon_measurements
                })

        # Create ordered beacon list
        beacon_list = sorted(list(all_macs))
        self._num_beacons = len(beacon_list)

        # Process samples
        for sample in all_samples:
            # Create list of beacons
            beacons = []
            for mac, rssi in sample['measurements'].items():
                beacon = BLEBeacon(
                    mac_address=mac,
                    rssi=rssi
                )
                beacons.append(beacon)

            # Create BLE signal
            signal = BLESignal(beacons=beacons)

            # Create location
            location = Location(
                coordinate=Coordinate(x=sample['x'], y=sample['y']),
                floor=sample['floor'],
                building_id='0'
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Loaded {len(self._signals)} samples from BLE Indoor dataset")
        print(f"Total unique beacons: {self._num_beacons}")


# Alias for convenience
BLEIndoor = BLEIndoorDataset
