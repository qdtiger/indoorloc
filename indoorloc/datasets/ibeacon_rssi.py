"""
iBeacon RSSI Indoor Localization Dataset Implementation

BLE iBeacon-based indoor localization dataset with RSSI fingerprints
collected from multiple iBeacon transmitters in an indoor environment.

Reference:
    Faragher, R., & Harle, R. (2015). Location Fingerprinting With Bluetooth
    Low Energy Beacons. IEEE Journal on Selected Areas in Communications.

Dataset URL: https://zenodo.org/record/1066044
"""
from pathlib import Path
from typing import Optional, Any, Dict, List
import numpy as np

from .base import BLEDataset
from ..signals.ble import BLESignal, BLEBeacon
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS
from ..utils.download import download_from_zenodo


@DATASETS.register_module()
class iBeaconRSSIDataset(BLEDataset):
    """iBeacon RSSI Indoor Localization Dataset.

    BLE fingerprinting dataset using iBeacon technology with RSSI
    measurements from multiple beacon transmitters.

    Args:
        data_root: Root directory containing the dataset files. If None,
            uses the default cache directory (~/.cache/indoorloc/datasets/ibeacon_rssi).
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize RSSI values.
        normalize_method: Normalization method ('minmax', 'positive', 'standard').

    Example:
        >>> import indoorloc as iloc
        >>> # Download from Zenodo
        >>> dataset = iloc.iBeaconRSSI(download=True, split='train')

    Dataset structure:
        data_root/
        ├── train.csv
        └── test.csv

    CSV format:
        - Columns: x, y, floor, beacon_uuid, beacon_major, beacon_minor, rssi
        - Multiple rows per location (one per detected beacon)
        - RSSI values in dBm
    """

    # Zenodo record ID
    ZENODO_RECORD_ID = '1066044'

    # Dataset constants
    NOT_DETECTED_VALUE = -100.0

    # File mapping
    FILE_MAPPING = {
        'train': 'train.csv',
        'test': 'test.csv',
    }

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        **kwargs
    ):
        self._num_beacons = None  # Will be determined from data

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
        return 'iBeaconRSSI'

    @property
    def num_beacons(self) -> int:
        if self._num_beacons is None:
            return 0
        return self._num_beacons

    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        filename = self.FILE_MAPPING.get(self.split)
        if filename is None:
            return False
        return (self.data_root / filename).exists()

    def _download(self) -> None:
        """Download iBeacon RSSI dataset from Zenodo."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading iBeacon RSSI dataset from Zenodo...")

        try:
            filenames = list(self.FILE_MAPPING.values())
            download_from_zenodo(
                record_id=self.ZENODO_RECORD_ID,
                root=self.data_root,
                filenames=filenames,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download iBeacon RSSI dataset: {e}\n"
                f"Please download manually from: https://zenodo.org/record/{self.ZENODO_RECORD_ID}"
            )

    def _load_data(self) -> None:
        """Load iBeacon RSSI dataset from CSV file."""
        filename = self.FILE_MAPPING[self.split]
        filepath = self.data_root / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        # Load CSV file
        try:
            import pandas as pd
            df = pd.read_csv(filepath)
        except ImportError:
            raise ImportError(
                "pandas is required to load iBeacon RSSI dataset.\n"
                "Install with: pip install pandas"
            )

        # Expected columns: x, y, floor, beacon_uuid, beacon_major, beacon_minor, rssi
        # Multiple rows per location (one per beacon)

        # Group by location to aggregate beacon measurements
        location_cols = ['x', 'y', 'floor']
        beacon_cols = ['beacon_uuid', 'beacon_major', 'beacon_minor', 'rssi']

        # Find all unique beacons
        beacon_identifiers = []
        for _, row in df[['beacon_uuid', 'beacon_major', 'beacon_minor']].drop_duplicates().iterrows():
            uuid = row['beacon_uuid'] if 'beacon_uuid' in row else 'unknown'
            major = int(row['beacon_major']) if 'beacon_major' in row and pd.notna(row['beacon_major']) else 0
            minor = int(row['beacon_minor']) if 'beacon_minor' in row and pd.notna(row['beacon_minor']) else 0
            beacon_id = f"{uuid}:{major}:{minor}"
            beacon_identifiers.append(beacon_id)

        self._num_beacons = len(beacon_identifiers)
        beacon_to_idx = {bid: idx for idx, bid in enumerate(beacon_identifiers)}

        # Group by location
        if all(col in df.columns for col in location_cols):
            grouped = df.groupby(location_cols)
        else:
            # Fallback: create unique groups by row index
            df['_group_id'] = df.index
            grouped = df.groupby('_group_id')

        for location_key, group in grouped:
            # Extract location info
            if isinstance(location_key, tuple) and len(location_key) == 3:
                x, y, floor = location_key
            else:
                # Fallback
                x = group['x'].iloc[0] if 'x' in group else 0.0
                y = group['y'].iloc[0] if 'y' in group else 0.0
                floor = int(group['floor'].iloc[0]) if 'floor' in group else 0

            # Create list of beacons for this location
            beacons = []
            for _, row in group.iterrows():
                uuid = str(row['beacon_uuid']) if 'beacon_uuid' in row else 'unknown'
                major = int(row['beacon_major']) if 'beacon_major' in row and pd.notna(row['beacon_major']) else 0
                minor = int(row['beacon_minor']) if 'beacon_minor' in row and pd.notna(row['beacon_minor']) else 0
                rssi = float(row['rssi']) if 'rssi' in row else self.NOT_DETECTED_VALUE

                beacon = BLEBeacon(
                    uuid=uuid,
                    major=major,
                    minor=minor,
                    rssi=rssi
                )
                beacons.append(beacon)

            # Create BLE signal
            signal = BLESignal(beacons=beacons)

            # Create location
            location = Location(
                coordinate=Coordinate(x=float(x), y=float(y)),
                floor=int(floor),
                building_id='0'
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Loaded {len(self._signals)} samples from iBeacon RSSI dataset")
        print(f"Total unique beacons: {self._num_beacons}")



def iBeaconRSSI(data_root=None, split=None, download=False, **kwargs):
    """
    Convenience function for loading iBeaconRSSI dataset.

    Args:
        data_root: Root directory for dataset storage
        split: Dataset split ('train', 'test', 'all', or None for tuple)
        download: Whether to download if not found
        **kwargs: Additional arguments passed to iBeaconRSSIDataset

    Returns:
        - If split is 'train' or 'test': Returns single dataset
        - If split is 'all': Returns merged train+test dataset  
        - If split is None: Returns tuple (train_dataset, test_dataset)

    Examples:
        >>> # Load train and test separately (tuple unpacking)
        >>> train, test = iBeaconRSSI(download=True)

        >>> # Load entire dataset (train + test merged)
        >>> dataset = iBeaconRSSI(split='all', download=True)

        >>> # Load only training set
        >>> train = iBeaconRSSI(split='train', download=True)
    """
    if split is None:
        # Return both train and test as tuple
        train_dataset = iBeaconRSSIDataset(
            data_root=data_root,
            split='train',
            download=download,
            **kwargs
        )
        test_dataset = iBeaconRSSIDataset(
            data_root=data_root,
            split='test',
            download=download,
            **kwargs
        )
        return train_dataset, test_dataset
    elif split == 'all':
        # Return merged train + test dataset
        from torch.utils.data import ConcatDataset
        train_dataset = iBeaconRSSIDataset(
            data_root=data_root,
            split='train',
            download=download,
            **kwargs
        )
        test_dataset = iBeaconRSSIDataset(
            data_root=data_root,
            split='test',
            download=download,
            **kwargs
        )
        return ConcatDataset([train_dataset, test_dataset])
    else:
        # Return single split
        return iBeaconRSSIDataset(
            data_root=data_root,
            split=split,
            download=download,
            **kwargs
        )

