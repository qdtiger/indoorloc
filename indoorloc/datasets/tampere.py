"""
Tampere University WiFi Dataset Implementation

Large-scale WiFi fingerprinting dataset collected at Tampere University
covering multiple buildings and floors with high spatial resolution.

Reference:
    Lohan, E. S., et al. (2017). Wi-Fi Crowdsourced Fingerprinting Dataset
    for Indoor Positioning. Data, 2(4), 32.

Dataset URL: https://zenodo.org/record/889798
"""
from pathlib import Path
from typing import Optional, Any, Dict, List
import numpy as np

from .base import WiFiDataset
from ..signals.wifi import WiFiSignal
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS
from ..utils.download import download_from_zenodo


@DATASETS.register_module()
class TampereDataset(WiFiDataset):
    """Tampere University WiFi Fingerprinting Dataset.

    Large-scale WiFi dataset with crowdsourced fingerprints collected
    across multiple buildings at Tampere University.

    Args:
        data_root: Root directory containing the dataset files. If None,
            uses the default cache directory (~/.cache/indoorloc/datasets/tampere).
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize RSSI values.
        normalize_method: Normalization method ('minmax', 'positive', 'standard').

    Example:
        >>> import indoorloc as iloc
        >>> # Download from Zenodo
        >>> dataset = iloc.Tampere(download=True, split='train')

    Dataset structure:
        data_root/
        ├── train.csv
        └── test.csv

    CSV format:
        - Columns: timestamp, x, y, floor, building, WAP1_RSSI, WAP2_RSSI, ...
        - RSSI values in dBm, missing values represented as 100
    """

    # Zenodo record ID for Tampere dataset
    ZENODO_RECORD_ID = '1066041'

    # Dataset constants
    NOT_DETECTED_VALUE = 100

    # Required files
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
        self._num_aps = None  # Will be determined from data

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
        return 'Tampere'

    @property
    def num_aps(self) -> int:
        if self._num_aps is None:
            return 0
        return self._num_aps

    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        filename = self.FILE_MAPPING.get(self.split)
        if filename is None:
            return False
        return (self.data_root / filename).exists()

    def _download(self) -> None:
        """Download Tampere dataset from Zenodo."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading Tampere dataset from Zenodo...")

        try:
            # Download specific files from Zenodo
            filenames = list(self.FILE_MAPPING.values())
            download_from_zenodo(
                record_id=self.ZENODO_RECORD_ID,
                root=self.data_root,
                filenames=filenames,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download Tampere dataset: {e}\n"
                f"Please download manually from: https://zenodo.org/record/{self.ZENODO_RECORD_ID}"
            )

    def _load_data(self) -> None:
        """Load Tampere dataset from CSV file."""
        filename = self.FILE_MAPPING[self.split]
        filepath = self.data_root / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        # Load CSV file
        # Expected format: timestamp,x,y,floor,building,WAP1,WAP2,...
        try:
            import pandas as pd
            df = pd.read_csv(filepath)
        except ImportError:
            raise ImportError(
                "pandas is required to load Tampere dataset.\n"
                "Install with: pip install pandas"
            )

        # Identify coordinate columns
        coord_cols = ['x', 'y', 'floor', 'building']
        if 'timestamp' in df.columns:
            coord_cols.insert(0, 'timestamp')

        # Check which columns exist
        existing_coord_cols = [col for col in coord_cols if col in df.columns]

        # Remaining columns are WAP RSSI values
        wap_cols = [col for col in df.columns if col not in existing_coord_cols]

        if len(wap_cols) == 0:
            raise ValueError("No WAP columns found in dataset")

        # Store number of APs
        self._num_aps = len(wap_cols)

        # Extract data
        rssi_data = df[wap_cols].values.astype(np.float32)
        x_vals = df['x'].values if 'x' in df.columns else np.zeros(len(df))
        y_vals = df['y'].values if 'y' in df.columns else np.zeros(len(df))
        floors = df['floor'].values.astype(int) if 'floor' in df.columns else np.zeros(len(df), dtype=int)
        buildings = df['building'].values.astype(str) if 'building' in df.columns else ['0'] * len(df)

        # Process each sample
        for i in range(len(df)):
            # Create WiFi signal
            signal = WiFiSignal(rssi_values=rssi_data[i])

            # Create location
            location = Location(
                coordinate=Coordinate(x=float(x_vals[i]), y=float(y_vals[i])),
                floor=int(floors[i]),
                building_id=str(buildings[i])
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Loaded {len(self._signals)} samples from Tampere dataset")


# Alias for convenience
Tampere = TampereDataset
