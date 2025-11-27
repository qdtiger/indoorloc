"""
UWB Indoor Localization Dataset Implementation

Ultra-Wideband indoor positioning dataset with high-precision ranging
measurements from multiple UWB anchors.

Reference:
    UWB Indoor Localization Dataset with Time-of-Flight Measurements.
    Zenodo Repository.
    https://zenodo.org/record/5789876

Dataset URL: https://zenodo.org/record/5789876
"""
from pathlib import Path
from typing import Optional, Any, Dict, List
import numpy as np

from .base import UWBDataset
from ..signals.uwb import UWBSignal, UWBAnchor
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS
from ..utils.download import download_from_zenodo


@DATASETS.register_module()
class UWBIndoorDataset(UWBDataset):
    """UWB Indoor Localization Dataset.

    High-precision indoor positioning dataset using Ultra-Wideband (UWB)
    technology with Time-of-Flight (ToF) ranging measurements.

    Args:
        data_root: Root directory containing the dataset files. If None,
            uses the default cache directory (~/.cache/indoorloc/datasets/uwb_indoor).
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        normalize_method: Normalization method ('minmax', 'positive', 'standard').

    Example:
        >>> import indoorloc as iloc
        >>> # Download UWB indoor dataset
        >>> dataset = iloc.UWBIndoor(download=True, split='train')
        >>> # Access UWB signal with anchor distances
        >>> signal = dataset[0]
        >>> for anchor in signal.anchors:
        ...     print(f"Anchor {anchor.anchor_id}: {anchor.distance}m")

    Dataset structure:
        data_root/
        ├── train_data.csv
        ├── test_data.csv
        └── anchor_positions.csv

    CSV format (train_data.csv, test_data.csv):
        - Columns: timestamp, x, y, z, floor, anchor1_id, anchor1_distance,
                   anchor2_id, anchor2_distance, ..., anchorN_id, anchorN_distance
        - Distances in meters
        - 3D coordinates (x, y, z)

    CSV format (anchor_positions.csv):
        - Columns: anchor_id, x, y, z
        - Anchor positions in meters
    """

    # Zenodo record ID
    ZENODO_RECORD_ID = '5789876'

    # Dataset constants
    MAX_DISTANCE = 100.0  # Maximum reasonable UWB range in meters

    # File mapping
    FILE_MAPPING = {
        'train': 'train_data.csv',
        'test': 'test_data.csv',
    }

    ANCHOR_FILE = 'anchor_positions.csv'

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        transform: Optional[Any] = None,
        normalize: bool = False,
        normalize_method: str = 'minmax',
        **kwargs
    ):
        self._num_anchors = None
        self._anchor_positions = {}

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
        return 'UWBIndoor'

    @property
    def num_anchors(self) -> int:
        if self._num_anchors is None:
            return 0
        return self._num_anchors

    @property
    def anchor_positions(self) -> Dict[str, np.ndarray]:
        """Get anchor positions as {anchor_id: [x, y, z]} dict."""
        return self._anchor_positions

    def get_anchor_positions(self) -> Dict[str, tuple]:
        """Get 3D positions of all UWB anchors.

        Returns:
            Dictionary mapping anchor IDs to (x, y, z) positions in meters.
        """
        return {
            anchor_id: tuple(pos)
            for anchor_id, pos in self._anchor_positions.items()
        }

    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        filename = self.FILE_MAPPING.get(self.split)
        if filename is None:
            return False
        anchor_file = self.data_root / self.ANCHOR_FILE
        data_file = self.data_root / filename
        return data_file.exists() and anchor_file.exists()

    def _download(self) -> None:
        """Download UWB indoor dataset from Zenodo."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading UWB indoor dataset from Zenodo...")

        try:
            # Download all required files
            all_files = list(self.FILE_MAPPING.values()) + [self.ANCHOR_FILE]
            download_from_zenodo(
                record_id=self.ZENODO_RECORD_ID,
                root=self.data_root,
                filenames=all_files,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download UWB indoor dataset: {e}\n"
                f"Please download manually from: https://zenodo.org/record/{self.ZENODO_RECORD_ID}"
            )

    def _load_data(self) -> None:
        """Load UWB indoor dataset from CSV files."""
        # Load anchor positions first
        anchor_file = self.data_root / self.ANCHOR_FILE
        if not anchor_file.exists():
            raise FileNotFoundError(f"Anchor file not found: {anchor_file}")

        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required to load UWB indoor dataset.\n"
                "Install with: pip install pandas"
            )

        # Load anchor positions
        anchor_df = pd.read_csv(anchor_file)
        for _, row in anchor_df.iterrows():
            anchor_id = str(row['anchor_id'])
            x = float(row.get('x', 0.0))
            y = float(row.get('y', 0.0))
            z = float(row.get('z', 0.0))
            self._anchor_positions[anchor_id] = np.array([x, y, z])

        self._num_anchors = len(self._anchor_positions)

        # Load measurement data
        filename = self.FILE_MAPPING[self.split]
        filepath = self.data_root / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_csv(filepath)

        # Identify anchor distance columns (anchor{N}_id, anchor{N}_distance pairs)
        anchor_id_cols = [col for col in df.columns if 'anchor' in col.lower() and '_id' in col.lower()]

        # Process each sample
        for idx, row in df.iterrows():
            # Extract location
            x = float(row.get('x', 0.0))
            y = float(row.get('y', 0.0))
            z = float(row.get('z', 0.0))
            floor = int(row.get('floor', 0))
            timestamp = row.get('timestamp', idx)

            # Create UWB anchors
            anchors = []
            for anchor_id_col in anchor_id_cols:
                # Extract anchor number from column name
                anchor_num = anchor_id_col.replace('anchor', '').replace('_id', '')
                distance_col = f'anchor{anchor_num}_distance'

                if anchor_id_col in row and distance_col in row:
                    anchor_id = str(row[anchor_id_col])
                    distance = row[distance_col]

                    if pd.notna(anchor_id) and pd.notna(distance):
                        distance = float(distance)

                        # Get anchor position if available
                        anchor_pos = self._anchor_positions.get(anchor_id)

                        anchor = UWBAnchor(
                            anchor_id=anchor_id,
                            distance=distance,
                            position=anchor_pos,
                            timestamp=timestamp
                        )
                        anchors.append(anchor)

            # Create UWB signal
            if anchors:
                signal = UWBSignal(anchors=anchors)

                # Create location
                location = Location(
                    coordinate=Coordinate(x=x, y=y, z=z),
                    floor=floor,
                    building_id='0'
                )

                self._signals.append(signal)
                self._locations.append(location)

        print(f"Loaded {len(self._signals)} samples from UWB indoor dataset ({self.split} split)")
        print(f"Total UWB anchors: {self._num_anchors}")



def UWBIndoor(data_root=None, split=None, download=False, **kwargs):
    """
    Convenience function for loading UWBIndoor dataset.

    Args:
        data_root: Root directory for dataset storage
        split: Dataset split ('train', 'test', 'all', or None for tuple)
        download: Whether to download if not found
        **kwargs: Additional arguments passed to UWBIndoorDataset

    Returns:
        - If split is 'train' or 'test': Returns single dataset
        - If split is 'all': Returns merged train+test dataset  
        - If split is None: Returns tuple (train_dataset, test_dataset)

    Examples:
        >>> # Load train and test separately (tuple unpacking)
        >>> train, test = UWBIndoor(download=True)

        >>> # Load entire dataset (train + test merged)
        >>> dataset = UWBIndoor(split='all', download=True)

        >>> # Load only training set
        >>> train = UWBIndoor(split='train', download=True)
    """
    if split is None:
        # Return both train and test as tuple
        train_dataset = UWBIndoorDataset(
            data_root=data_root,
            split='train',
            download=download,
            **kwargs
        )
        test_dataset = UWBIndoorDataset(
            data_root=data_root,
            split='test',
            download=download,
            **kwargs
        )
        return train_dataset, test_dataset
    elif split == 'all':
        # Return merged train + test dataset
        from torch.utils.data import ConcatDataset
        train_dataset = UWBIndoorDataset(
            data_root=data_root,
            split='train',
            download=download,
            **kwargs
        )
        test_dataset = UWBIndoorDataset(
            data_root=data_root,
            split='test',
            download=download,
            **kwargs
        )
        return ConcatDataset([train_dataset, test_dataset])
    else:
        # Return single split
        return UWBIndoorDataset(
            data_root=data_root,
            split=split,
            download=download,
            **kwargs
        )

