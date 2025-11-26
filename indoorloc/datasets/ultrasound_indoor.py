"""
Ultrasound Indoor Localization Dataset Implementation

Ultrasonic signal-based indoor positioning dataset using time-of-flight
measurements from ultrasound transmitters and receivers.

Reference:
    Ultrasound Indoor Localization System Dataset.
    UCI Machine Learning Repository.

Dataset URL: https://archive.ics.uci.edu/dataset/632/ultrasound+indoor+localization
"""
from pathlib import Path
from typing import Optional, Any, Dict, List
import numpy as np

from .base import BaseDataset
from ..signals.ultrasound import UltrasoundSignal, UltrasoundTransmitter
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS
from ..utils.download import download_from_uci


@DATASETS.register_module()
class UltrasoundIndoorDataset(BaseDataset):
    """Ultrasound Indoor Localization Dataset.

    Indoor positioning dataset using ultrasonic signals with time-of-flight
    measurements for accurate distance estimation.

    Args:
        data_root: Root directory containing the dataset files. If None,
            uses the default cache directory (~/.cache/indoorloc/datasets/ultrasound_indoor).
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        normalize_method: Normalization method ('minmax', 'positive', 'standard').
        train_ratio: Ratio for train/test split (default: 0.7).

    Example:
        >>> import indoorloc as iloc
        >>> # Download ultrasound indoor dataset
        >>> dataset = iloc.UltrasoundIndoor(download=True, split='train')
        >>> # Access ultrasound signal with ToF measurements
        >>> signal = dataset[0]
        >>> for beacon in signal.beacons:
        ...     print(f"Beacon {beacon.beacon_id}: ToF={beacon.tof}ms, distance={beacon.distance}m")

    Dataset structure:
        data_root/
        └── ultrasound_data.csv

    CSV format:
        - Columns: x, y, z, beacon1_id, beacon1_tof, beacon1_distance,
                   beacon2_id, beacon2_tof, beacon2_distance, ...,
                   beaconN_id, beaconN_tof, beaconN_distance
        - ToF (time-of-flight) in milliseconds
        - Distance in meters
        - 3D coordinates (x, y, z)
    """

    # UCI dataset name
    UCI_DATASET_NAME = 'ultrasound-indoor-localization'

    # Dataset constants
    SPEED_OF_SOUND = 343.0  # m/s at 20°C
    MAX_DISTANCE = 20.0  # Maximum ultrasound range in meters

    # Required files
    REQUIRED_FILES = ['ultrasound_data.csv']

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        transform: Optional[Any] = None,
        normalize: bool = False,
        normalize_method: str = 'minmax',
        train_ratio: float = 0.7,
        **kwargs
    ):
        self.train_ratio = train_ratio
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
        return 'UltrasoundIndoor'

    @property
    def signal_type(self) -> str:
        return 'ultrasound'

    @property
    def num_beacons(self) -> int:
        if self._num_beacons is None:
            return 0
        return self._num_beacons

    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        return all(
            (self.data_root / f).exists()
            for f in self.REQUIRED_FILES
        )

    def _download(self) -> None:
        """Download ultrasound indoor dataset from UCI."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading ultrasound indoor dataset from UCI...")

        try:
            download_from_uci(
                dataset_name=self.UCI_DATASET_NAME,
                root=self.data_root,
                filenames=self.REQUIRED_FILES,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download ultrasound indoor dataset: {e}\n"
                f"Please download manually from: "
                f"https://archive.ics.uci.edu/dataset/632/"
            )

    def _load_data(self) -> None:
        """Load ultrasound indoor dataset from CSV file."""
        filepath = self.data_root / 'ultrasound_data.csv'

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required to load ultrasound indoor dataset.\n"
                "Install with: pip install pandas"
            )

        # Load CSV
        df = pd.read_csv(filepath)

        # Identify beacon columns (beacon{N}_id, beacon{N}_tof, beacon{N}_distance)
        beacon_id_cols = [col for col in df.columns if 'beacon' in col.lower() and '_id' in col.lower()]

        # Count unique beacons
        all_beacon_ids = set()
        for col in beacon_id_cols:
            beacon_ids = df[col].dropna().unique()
            all_beacon_ids.update([str(bid) for bid in beacon_ids])
        self._num_beacons = len(all_beacon_ids)

        # Split data
        num_train = int(len(df) * self.train_ratio)
        if self.split == 'train':
            df_split = df.iloc[:num_train]
        else:  # test
            df_split = df.iloc[num_train:]

        # Process each sample
        for idx, row in df_split.iterrows():
            # Extract location
            x = float(row.get('x', 0.0))
            y = float(row.get('y', 0.0))
            z = float(row.get('z', 0.0))

            # Create ultrasound transmitters
            transmitters = []
            for beacon_id_col in beacon_id_cols:
                # Extract beacon number from column name
                beacon_num = beacon_id_col.replace('beacon', '').replace('_id', '')
                tof_col = f'beacon{beacon_num}_tof'
                distance_col = f'beacon{beacon_num}_distance'

                if beacon_id_col in row and tof_col in row and distance_col in row:
                    beacon_id = str(row[beacon_id_col])
                    tof = row[tof_col]
                    distance = row[distance_col]

                    if pd.notna(beacon_id) and pd.notna(tof) and pd.notna(distance):
                        tof = float(tof)
                        distance = float(distance)

                        transmitter = UltrasoundTransmitter(
                            transmitter_id=beacon_id,
                            tof=tof,
                            distance=distance
                        )
                        transmitters.append(transmitter)

            # Create ultrasound signal
            if transmitters:
                signal = UltrasoundSignal(transmitters=transmitters)

                # Create location
                location = Location(
                    coordinate=Coordinate(x=x, y=y, z=z),
                    floor=0,
                    building_id='0'
                )

                self._signals.append(signal)
                self._locations.append(location)

        print(f"Loaded {len(self._signals)} samples from ultrasound indoor dataset ({self.split} split)")
        print(f"Total ultrasound beacons: {self._num_beacons}")


# Alias for convenience
UltrasoundIndoor = UltrasoundIndoorDataset
