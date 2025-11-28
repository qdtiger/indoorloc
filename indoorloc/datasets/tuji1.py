"""
TUJI1 (TUT Indoor) WiFi Dataset Implementation

Indoor WiFi fingerprinting dataset collected at Tampere University of Technology
with high-quality ground truth and comprehensive coverage.

Reference:
    Torres-Sospedra, J., et al. (2022). The Smartphone-based Offline Indoor
    Location Competition at IPIN 2021. IEEE IPIN.

Dataset URL: https://github.com/IndoorLocation/IPIN2021-Competition-Track3-Dataset
"""
from pathlib import Path
from typing import Optional, Any, Dict, List, Union
import numpy as np

from .base import WiFiDataset
from ..signals.wifi import WiFiSignal
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS
from ..utils.download import download_url


@DATASETS.register_module()
class TUJI1Dataset(WiFiDataset):
    """TUJI1 (TUT Indoor) WiFi Fingerprinting Dataset.

    High-quality WiFi fingerprinting dataset from IPIN 2021 competition
    with precise ground truth and comprehensive spatial coverage.

    Args:
        data_root: Root directory containing the dataset files. If None,
            uses the default cache directory (~/.cache/indoorloc/datasets/tuji1).
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize RSSI values.
        normalize_method: Normalization method ('minmax', 'positive', 'standard').

    Example:
        >>> import indoorloc as iloc
        >>> # Download from GitHub
        >>> dataset = iloc.TUJI1(download=True)

    Dataset structure:
        data_root/
        ├── training_data.csv
        └── test_data.csv

    CSV format:
        - Columns: timestamp, x, y, floor, BSSID1, RSSI1, BSSID2, RSSI2, ...
        - RSSI values in dBm
        - Missing APs not included (sparse format)
    """

    # GitHub raw content base URL
    BASE_URL = 'https://raw.githubusercontent.com/IndoorLocation/IPIN2021-Competition-Track3-Dataset/main'

    # Dataset constants
    NOT_DETECTED_VALUE = -110

    # File mapping
    FILE_MAPPING = {
        'train': 'training_data.csv',
        'test': 'test_data.csv',
    }

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        floor: Union[int, List[int], str] = 'all',
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        **kwargs
    ):
        self._num_aps = None  # Will be determined from data
        self._ap_list = None  # List of all seen BSSIDs
        self._floor_param = floor
        self._available_floors: List[int] = []

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
        return 'TUJI1'

    @property
    def num_aps(self) -> int:
        if self._num_aps is None:
            return 0
        return self._num_aps

    @classmethod
    def list_floors(cls, data_root: Optional[str] = None) -> List[int]:
        """List all available floors in the dataset.

        Args:
            data_root: Root directory containing the dataset files.

        Returns:
            List of floor numbers.
        """
        from ..utils.download import get_data_home

        if data_root is None:
            root = get_data_home() / 'tuji1'
        else:
            root = Path(data_root)

        train_file = root / 'training_data.csv'
        if not train_file.exists():
            return []

        try:
            import pandas as pd
            df = pd.read_csv(train_file, usecols=['floor'])
            return sorted(df['floor'].astype(int).unique().tolist())
        except Exception:
            return []

    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        filename = self.FILE_MAPPING.get(self.split)
        if filename is None:
            return False
        return (self.data_root / filename).exists()

    def _download(self) -> None:
        """Download TUJI1 dataset from GitHub."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading TUJI1 dataset from GitHub...")

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
                f"Failed to download TUJI1 dataset: {e}\n"
                f"Please download manually from: {self.BASE_URL}"
            )

    def _load_data(self) -> None:
        """Load TUJI1 dataset from CSV file."""
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
                "pandas is required to load TUJI1 dataset.\n"
                "Install with: pip install pandas"
            )

        # Expected columns: timestamp, x, y, floor, then BSSID-RSSI pairs
        # Extract coordinate columns
        coord_cols = ['timestamp', 'x', 'y', 'floor']
        existing_coord_cols = [col for col in coord_cols if col in df.columns]

        # Remaining columns should be BSSID-RSSI pairs
        wifi_cols = [col for col in df.columns if col not in existing_coord_cols]

        # Parse WiFi data (assume alternating BSSID, RSSI columns)
        # Build a mapping of all unique BSSIDs
        all_bssids = set()
        samples_data = []

        for idx, row in df.iterrows():
            x = row['x'] if 'x' in row else 0.0
            y = row['y'] if 'y' in row else 0.0
            floor = int(row['floor']) if 'floor' in row else 0

            # Parse BSSID-RSSI pairs
            # Assuming columns are ordered as: BSSID1, RSSI1, BSSID2, RSSI2, ...
            ap_measurements = {}
            i = 0
            while i < len(wifi_cols) - 1:
                bssid_col = wifi_cols[i]
                rssi_col = wifi_cols[i + 1]

                if 'BSSID' in bssid_col or 'bssid' in bssid_col.lower():
                    bssid = row[bssid_col]
                    rssi = row[rssi_col]

                    if pd.notna(bssid) and pd.notna(rssi):
                        bssid = str(bssid).strip()
                        if bssid and bssid != 'nan':
                            all_bssids.add(bssid)
                            ap_measurements[bssid] = float(rssi)

                i += 2

            samples_data.append({
                'x': float(x),
                'y': float(y),
                'floor': floor,
                'measurements': ap_measurements
            })

        # Store available floors
        self._available_floors = sorted(list(set(s['floor'] for s in samples_data)))

        # Filter by floor parameter
        if self._floor_param != 'all':
            if isinstance(self._floor_param, int):
                floors_to_load = [self._floor_param]
            else:
                floors_to_load = list(self._floor_param)
            samples_data = [s for s in samples_data if s['floor'] in floors_to_load]

        if len(samples_data) == 0:
            raise ValueError(f"No data found for floor(s): {self._floor_param}")

        # Create ordered AP list
        self._ap_list = sorted(list(all_bssids))
        self._num_aps = len(self._ap_list)

        # Convert sparse measurements to dense format
        ap_to_idx = {bssid: idx for idx, bssid in enumerate(self._ap_list)}

        for sample in samples_data:
            # Create dense RSSI vector
            rssi_vector = np.full(self._num_aps, self.NOT_DETECTED_VALUE, dtype=np.float32)

            for bssid, rssi in sample['measurements'].items():
                if bssid in ap_to_idx:
                    rssi_vector[ap_to_idx[bssid]] = rssi

            # Create WiFi signal
            signal = WiFiSignal(rssi_values=rssi_vector, ap_list=self._ap_list)

            # Create location
            location = Location(
                coordinate=Coordinate(x=sample['x'], y=sample['y']),
                floor=sample['floor'],
                building_id='0'
            )

            self._signals.append(signal)
            self._locations.append(location)

        floor_info = f" (floor: {self._floor_param})" if self._floor_param != 'all' else ""
        print(f"Loaded {len(self._signals)} samples from TUJI1 dataset{floor_info}")
        print(f"Total unique APs: {self._num_aps}")



def TUJI1(data_root=None, split=None, download=False, floor='all', **kwargs):
    """
    Convenience function for loading TUJI1 dataset.

    Args:
        data_root: Root directory for dataset storage
        split: Dataset split ('train', 'test', 'all', or None for tuple)
        download: Whether to download if not found
        floor: Floor(s) to load. Can be:
            - 'all': Load all floors (default)
            - Single floor: 0, 1, 2, etc.
            - List of floors: [0, 1, 2]
        **kwargs: Additional arguments passed to TUJI1Dataset

    Returns:
        - If split is 'train' or 'test': Returns single dataset
        - If split is 'all': Returns merged train+test dataset
        - If split is None: Returns tuple (train_dataset, test_dataset)

    Examples:
        >>> # Load train and test separately (tuple unpacking)
        >>> train, test = TUJI1(download=True)

        >>> # Load entire dataset (train + test merged)
        >>> dataset = TUJI1(split='all', download=True)

        >>> # Load specific floor(s)
        >>> train = TUJI1(floor=[0, 1], split='train')

        >>> # List available floors
        >>> TUJI1.list_floors()
    """
    if split is None:
        # Return both train and test as tuple
        train_dataset = TUJI1Dataset(
            data_root=data_root,
            split='train',
            download=download,
            floor=floor,
            **kwargs
        )
        test_dataset = TUJI1Dataset(
            data_root=data_root,
            split='test',
            download=download,
            floor=floor,
            **kwargs
        )
        return train_dataset, test_dataset
    elif split == 'all':
        # Return merged train + test dataset
        from torch.utils.data import ConcatDataset
        train_dataset = TUJI1Dataset(
            data_root=data_root,
            split='train',
            download=download,
            floor=floor,
            **kwargs
        )
        test_dataset = TUJI1Dataset(
            data_root=data_root,
            split='test',
            download=download,
            floor=floor,
            **kwargs
        )
        return ConcatDataset([train_dataset, test_dataset])
    else:
        # Return single split
        return TUJI1Dataset(
            data_root=data_root,
            split=split,
            download=download,
            floor=floor,
            **kwargs
        )


# Attach class method to convenience function
TUJI1.list_floors = TUJI1Dataset.list_floors

