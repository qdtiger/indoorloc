"""
UJIndoorLoc Dataset Implementation

The UJIndoorLoc dataset is a popular WiFi fingerprinting dataset collected at
Universitat Jaume I, Spain. It covers 3 buildings with 4-5 floors each.

Reference:
    Torres-Sospedra, J., et al. (2014). UJIndoorLoc: A new multi-building and
    multi-floor database for WLAN fingerprint-based indoor localization problems.
    In 2014 International Conference on Indoor Positioning and Indoor Navigation (IPIN).

Dataset URL: https://archive.ics.uci.edu/dataset/310/ujiindoorloc
"""
import csv
from pathlib import Path
from typing import Optional, Any, Dict, List

import numpy as np

from .base import WiFiDataset
from ..signals.wifi import WiFiSignal
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS


@DATASETS.register_module()
class UJIndoorLocDataset(WiFiDataset):
    """UJIndoorLoc WiFi Fingerprinting Dataset.

    This dataset contains WiFi RSSI fingerprints from 520 access points
    collected in 3 buildings at Universitat Jaume I.

    Args:
        data_root: Root directory containing the dataset files.
        split: Dataset split ('train' or 'test').
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize RSSI values.
        normalize_method: Normalization method ('minmax', 'positive', 'standard').

    Dataset structure:
        data_root/
        ├── trainingData.csv
        └── validationData.csv

    CSV columns (0-indexed):
        0-519: WAP001-WAP520 (RSSI values, 100 = not detected)
        520: LONGITUDE
        521: LATITUDE
        522: FLOOR
        523: BUILDINGID
        524: SPACEID
        525: RELATIVEPOSITION
        526: USERID
        527: PHONEID
        528: TIMESTAMP
    """

    # Dataset constants
    NUM_WAPS = 520
    NOT_DETECTED_VALUE = 100

    # Column indices
    COL_LONGITUDE = 520
    COL_LATITUDE = 521
    COL_FLOOR = 522
    COL_BUILDING = 523
    COL_SPACEID = 524
    COL_RELATIVEPOSITION = 525
    COL_USERID = 526
    COL_PHONEID = 527
    COL_TIMESTAMP = 528

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        **kwargs
    ):
        # Store kwargs before calling super().__init__
        self._kwargs = kwargs
        super().__init__(
            data_root=data_root,
            split=split,
            transform=transform,
            normalize=normalize,
            normalize_method=normalize_method,
            **kwargs
        )

    @property
    def dataset_name(self) -> str:
        return 'UJIndoorLoc'

    @property
    def num_aps(self) -> int:
        return self.NUM_WAPS

    def _load_data(self) -> None:
        """Load data from CSV files."""
        # Determine file to load
        if self.split == 'train':
            filename = 'trainingData.csv'
        elif self.split in ('test', 'val', 'validation'):
            filename = 'validationData.csv'
        else:
            raise ValueError(f"Unknown split: {self.split}. Use 'train' or 'test'.")

        filepath = self.data_root / filename

        if not filepath.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {filepath}\n"
                f"Please download UJIndoorLoc dataset from:\n"
                f"https://archive.ics.uci.edu/dataset/310/ujiindoorloc"
            )

        # Load CSV data
        with open(filepath, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header

            for row in reader:
                if len(row) < 529:
                    continue  # Skip incomplete rows

                # Parse RSSI values (columns 0-519)
                rssi_values = np.array(
                    [float(row[i]) for i in range(self.NUM_WAPS)],
                    dtype=np.float32
                )

                # Create WiFi signal
                signal = WiFiSignal(
                    rssi_values=rssi_values,
                    timestamp=float(row[self.COL_TIMESTAMP]) if row[self.COL_TIMESTAMP] else None
                )
                self._signals.append(signal)

                # Parse location
                longitude = float(row[self.COL_LONGITUDE])
                latitude = float(row[self.COL_LATITUDE])
                floor = int(row[self.COL_FLOOR])
                building_id = str(int(row[self.COL_BUILDING]))

                # UJIndoorLoc uses UTM coordinates (longitude/latitude are actually x/y in meters)
                # The coordinates are already in a local reference frame
                coordinate = Coordinate(
                    x=longitude,  # In UJIndoorLoc, LONGITUDE column is actually x coordinate
                    y=latitude,   # LATITUDE column is actually y coordinate
                    latitude=latitude,
                    longitude=longitude
                )

                location = Location(
                    coordinate=coordinate,
                    floor=floor,
                    building_id=building_id
                )
                self._locations.append(location)

                # Store metadata
                metadata = {
                    'space_id': int(row[self.COL_SPACEID]),
                    'relative_position': int(row[self.COL_RELATIVEPOSITION]),
                    'user_id': int(row[self.COL_USERID]),
                    'phone_id': int(row[self.COL_PHONEID]),
                    'timestamp': float(row[self.COL_TIMESTAMP]) if row[self.COL_TIMESTAMP] else None
                }
                self._metadata.append(metadata)

    def get_statistics(self) -> Dict[str, Any]:
        """Compute UJIndoorLoc-specific statistics."""
        stats = super().get_statistics()

        # User and phone statistics
        user_ids = [m['user_id'] for m in self._metadata]
        phone_ids = [m['phone_id'] for m in self._metadata]

        stats['num_users'] = len(set(user_ids))
        stats['num_phones'] = len(set(phone_ids))

        # Building-floor distribution
        building_floor_counts = {}
        for loc in self._locations:
            key = (loc.building_id, loc.floor)
            building_floor_counts[key] = building_floor_counts.get(key, 0) + 1

        stats['samples_per_building_floor'] = building_floor_counts

        return stats

    @classmethod
    def download(cls, data_root: str, force: bool = False) -> None:
        """Download UJIndoorLoc dataset.

        Args:
            data_root: Directory to save the dataset.
            force: Whether to overwrite existing files.

        Note:
            This method requires internet connection and the 'requests' package.
        """
        import zipfile
        import io

        try:
            import requests
        except ImportError:
            raise ImportError("Please install requests: pip install requests")

        data_root = Path(data_root)
        data_root.mkdir(parents=True, exist_ok=True)

        # Check if already downloaded
        train_file = data_root / 'trainingData.csv'
        test_file = data_root / 'validationData.csv'

        if train_file.exists() and test_file.exists() and not force:
            print(f"Dataset already exists at {data_root}")
            return

        # UCI ML Repository URL
        url = "https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip"

        print(f"Downloading UJIndoorLoc dataset from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Extract ZIP
        print("Extracting files...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            for member in zf.namelist():
                if member.endswith('.csv'):
                    # Extract to data_root with just the filename
                    filename = Path(member).name
                    target_path = data_root / filename

                    with zf.open(member) as source, open(target_path, 'wb') as target:
                        target.write(source.read())
                    print(f"  Extracted: {filename}")

        print(f"Dataset downloaded to {data_root}")


# Alias for convenience
UJIndoorLoc = UJIndoorLocDataset
