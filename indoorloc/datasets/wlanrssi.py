"""
WLAN RSSI Indoor Localization Dataset Implementation

WiFi-based indoor localization dataset collected in a controlled environment
with systematic coverage and multiple devices.

Reference:
    Rohra, J. G., et al. (2017). User Localization in an Indoor Environment
    Using Fuzzy Hybrid of Particle Swarm Optimization & Gravitational Search
    Algorithm with Neural Networks. Sixth International Conference on Soft
    Computing for Problem Solving.

Dataset URL: https://archive.ics.uci.edu/dataset/422/localization+data+for+person+activity
"""
from pathlib import Path
from typing import Optional, Any, Dict, List
import numpy as np

from .base import WiFiDataset
from ..signals.wifi import WiFiSignal
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS
from ..utils.download import download_from_uci


@DATASETS.register_module()
class WLANRSSIDataset(WiFiDataset):
    """WLAN RSSI Indoor Localization Dataset.

    WiFi fingerprinting dataset with RSSI measurements from multiple
    access points in a controlled indoor environment.

    Args:
        data_root: Root directory containing the dataset files. If None,
            uses the default cache directory (~/.cache/indoorloc/datasets/wlanrssi).
        split: Dataset split ('train' or 'test'). Note: this dataset
            may need manual splitting.
        download: Whether to download the dataset if not found.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize RSSI values.
        normalize_method: Normalization method ('minmax', 'positive', 'standard').

    Example:
        >>> import indoorloc as iloc
        >>> # Download from UCI repository
        >>> dataset = iloc.WLANRSSI(download=True)

    Dataset structure:
        data_root/
        └── dataset.txt

    File format:
        Each line: date time x y floor user_id phone_id rssi1 rssi2 rssi3 rssi4 rssi5 rssi6 rssi7
        - 7 WiFi access points
        - Coordinates in meters
        - RSSI in dBm
    """

    # UCI dataset name
    UCI_DATASET_NAME = 'localization-data-for-person-activity'

    # Dataset constants
    NOT_DETECTED_VALUE = -110  # Typical WiFi noise floor
    NUM_WAPS = 7  # This dataset has 7 access points

    # Required files
    REQUIRED_FILES = ['dataset.txt']

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        train_ratio: float = 0.7,  # Split ratio for train/test
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
        return 'WLANRSSI'

    @property
    def num_aps(self) -> int:
        return self.NUM_WAPS

    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        return all(
            (self.data_root / f).exists()
            for f in self.REQUIRED_FILES
        )

    def _download(self) -> None:
        """Download WLAN RSSI dataset from UCI repository."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading WLAN RSSI dataset from UCI...")

        # Note: UCI repository structure may vary
        # Try direct download first
        try:
            download_from_uci(
                dataset_name=self.UCI_DATASET_NAME,
                root=self.data_root,
                filenames=self.REQUIRED_FILES,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download WLAN RSSI dataset: {e}\n"
                f"Please download manually from: "
                f"https://archive.ics.uci.edu/dataset/422/"
            )

    def _load_data(self) -> None:
        """Load WLAN RSSI dataset from file."""
        filepath = self.data_root / 'dataset.txt'

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        # Load data file
        # Format: date time x y floor user_id phone_id rssi1 rssi2 ... rssi7
        try:
            data = []
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 7 + 7:  # 7 metadata + 7 RSSI values
                        continue
                    data.append(parts)

            if len(data) == 0:
                raise ValueError("No valid data found in file")

        except Exception as e:
            raise RuntimeError(f"Failed to parse dataset file: {e}")

        # Convert to arrays
        all_samples = []
        for parts in data:
            try:
                # Parse metadata (skip date/time for now)
                x = float(parts[2])
                y = float(parts[3])
                floor = int(float(parts[4]))

                # Parse RSSI values (last 7 columns)
                rssi_values = np.array([float(parts[i]) for i in range(7, 14)], dtype=np.float32)

                all_samples.append({
                    'x': x,
                    'y': y,
                    'floor': floor,
                    'rssi': rssi_values
                })
            except (ValueError, IndexError) as e:
                continue

        if len(all_samples) == 0:
            raise RuntimeError("No valid samples could be parsed")

        # Split into train/test
        num_train = int(len(all_samples) * self.train_ratio)

        if self.split == 'train':
            samples = all_samples[:num_train]
        else:  # test
            samples = all_samples[num_train:]

        # Process samples
        for sample in samples:
            # Create WiFi signal
            signal = WiFiSignal(rssi_values=sample['rssi'])

            # Create location
            location = Location(
                coordinate=Coordinate(x=sample['x'], y=sample['y']),
                floor=sample['floor'],
                building_id='0'
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Loaded {len(self._signals)} samples from WLAN RSSI dataset ({self.split} split)")


# Alias for convenience
WLANRSSI = WLANRSSIDataset
