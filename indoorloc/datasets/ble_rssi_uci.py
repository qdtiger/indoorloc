"""
BLE RSSI Indoor Localization Dataset (UCI) Implementation

BLE-based indoor localization dataset from UCI Machine Learning Repository
with systematic RSSI measurements from multiple BLE beacons.

Reference:
    UCI Machine Learning Repository: BLE RSSI Dataset for Indoor Localization.
    https://archive.ics.uci.edu/

Dataset URL: https://archive.ics.uci.edu/dataset/519/ble+rssi+dataset+for+indoor+localization
"""
from pathlib import Path
from typing import Optional, Any, Dict, List
import numpy as np

from .base import BLEDataset
from ..signals.ble import BLESignal, BLEBeacon
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS
from ..utils.download import download_from_uci


@DATASETS.register_module()
class BLERSSIUCIDataset(BLEDataset):
    """BLE RSSI Indoor Localization Dataset from UCI.

    Systematic BLE RSSI fingerprinting dataset with multiple beacon
    measurements for indoor positioning research.

    Args:
        data_root: Root directory containing the dataset files. If None,
            uses the default cache directory (~/.cache/indoorloc/datasets/ble_rssi_uci).
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize RSSI values.
        normalize_method: Normalization method ('minmax', 'positive', 'standard').
        train_ratio: Ratio for train/test split (default: 0.7).

    Example:
        >>> import indoorloc as iloc
        >>> # Download from UCI repository
        >>> dataset = iloc.BLERSSIU CI(download=True)

    Dataset structure:
        data_root/
        └── iBeacon_RSSI_Labeled.csv

    CSV format:
        - Columns: date, location, b3001_rssi, b3002_rssi, ..., b3013_rssi
        - 13 iBeacon transmitters
        - RSSI values in dBm
        - Location labels as room identifiers
    """

    # UCI dataset name
    UCI_DATASET_NAME = 'ble-rssi-dataset-for-indoor-localization'

    # Dataset constants
    NOT_DETECTED_VALUE = -100.0
    NUM_BEACONS = 13  # This dataset has 13 iBeacons

    # Required files
    REQUIRED_FILES = ['iBeacon_RSSI_Labeled.csv']

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        train_ratio: float = 0.7,
        **kwargs
    ):
        self.train_ratio = train_ratio

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
        return 'BLERSSIU CI'

    @property
    def num_beacons(self) -> int:
        return self.NUM_BEACONS

    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        return all(
            (self.data_root / f).exists()
            for f in self.REQUIRED_FILES
        )

    def _download(self) -> None:
        """Download BLE RSSI UCI dataset."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading BLE RSSI UCI dataset...")

        try:
            download_from_uci(
                dataset_name=self.UCI_DATASET_NAME,
                root=self.data_root,
                filenames=self.REQUIRED_FILES,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download BLE RSSI UCI dataset: {e}\n"
                f"Please download manually from: "
                f"https://archive.ics.uci.edu/dataset/519/"
            )

    def _load_data(self) -> None:
        """Load BLE RSSI UCI dataset from CSV file."""
        filepath = self.data_root / 'iBeacon_RSSI_Labeled.csv'

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        # Load CSV
        try:
            import pandas as pd
            df = pd.read_csv(filepath)
        except ImportError:
            raise ImportError(
                "pandas is required to load BLE RSSI UCI dataset.\n"
                "Install with: pip install pandas"
            )

        # Expected columns: date, location, b3001, b3002, ..., b3013
        # 13 beacons with IDs b3001 through b3013

        # Find RSSI columns (beacons)
        rssi_cols = [col for col in df.columns if col.startswith('b30')]

        if len(rssi_cols) == 0:
            raise ValueError("No beacon RSSI columns found in dataset")

        # Create location mapping (room labels to coordinates)
        unique_locations = df['location'].unique() if 'location' in df.columns else []
        location_to_coord = {
            loc: (i * 5.0, 0.0)  # Simple grid layout
            for i, loc in enumerate(unique_locations)
        }

        # Split data
        num_train = int(len(df) * self.train_ratio)
        if self.split == 'train':
            df_split = df.iloc[:num_train]
        else:  # test
            df_split = df.iloc[num_train:]

        # Process each sample
        for idx, row in df_split.iterrows():
            # Get location
            location_label = row['location'] if 'location' in row else 'unknown'
            x, y = location_to_coord.get(location_label, (0.0, 0.0))

            # Create beacons
            beacons = []
            for beacon_col in rssi_cols:
                rssi = row[beacon_col]

                if pd.notna(rssi):
                    # Extract beacon ID from column name (e.g., 'b3001' -> '3001')
                    beacon_id = beacon_col.replace('b', '')

                    beacon = BLEBeacon(
                        mac_address=beacon_id,
                        rssi=float(rssi)
                    )
                    beacons.append(beacon)

            # Create BLE signal
            signal = BLESignal(beacons=beacons)

            # Create location
            location = Location(
                coordinate=Coordinate(x=x, y=y),
                floor=0,
                building_id='0'
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Loaded {len(self._signals)} samples from BLE RSSI UCI dataset ({self.split} split)")


# Alias for convenience
BLERSSIU_UCI = BLERSSIUCIDataset
