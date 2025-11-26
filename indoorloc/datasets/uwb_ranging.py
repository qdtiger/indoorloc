"""
UWB Ranging Dataset Implementation

Ultra-Wideband ranging dataset with Two-Way Ranging (TWR) measurements
for indoor localization research.

Reference:
    UWB Ranging Dataset for Indoor Positioning.
    GitHub Repository.
    https://github.com/uwb-positioning/ranging-dataset

Dataset URL: https://raw.githubusercontent.com/UWB-Positioning/RangingDataset/main
"""
from pathlib import Path
from typing import Optional, Any, Dict, List
import numpy as np

from .base import UWBDataset
from ..signals.uwb import UWBSignal, UWBAnchor
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS
from ..utils.download import download_url


@DATASETS.register_module()
class UWBRangingDataset(UWBDataset):
    """UWB Ranging Dataset.

    UWB ranging dataset with Two-Way Ranging (TWR) measurements from
    multiple anchors for indoor positioning applications.

    Args:
        data_root: Root directory containing the dataset files. If None,
            uses the default cache directory (~/.cache/indoorloc/datasets/uwb_ranging).
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        normalize_method: Normalization method ('minmax', 'positive', 'standard').
        train_ratio: Ratio for train/test split (default: 0.7).

    Example:
        >>> import indoorloc as iloc
        >>> # Download UWB ranging dataset
        >>> dataset = iloc.UWBRanging(download=True, split='train')
        >>> # Access UWB signal with Two-Way Ranging measurements
        >>> signal = dataset[0]
        >>> for anchor in signal.anchors:
        ...     print(f"Anchor {anchor.anchor_id}: distance={anchor.distance}m, "
        ...           f"rssi={anchor.rssi}dBm")

    Dataset structure:
        data_root/
        ├── ranging_measurements.csv
        └── anchor_config.csv

    CSV format (ranging_measurements.csv):
        - Columns: measurement_id, x, y, z, anchor_id, distance, rssi, fp_power
        - Multiple rows per location (one per anchor)
        - Distances in meters, RSSI in dBm

    CSV format (anchor_config.csv):
        - Columns: anchor_id, x, y, z, channel
        - Fixed anchor positions and channel configuration
    """

    # GitHub raw content base URL
    BASE_URL = 'https://raw.githubusercontent.com/UWB-Positioning/RangingDataset/main'

    # Dataset constants
    MAX_DISTANCE = 100.0  # Maximum UWB range in meters
    NOT_DETECTED_RSSI = -100.0

    # Required files
    REQUIRED_FILES = ['ranging_measurements.csv', 'anchor_config.csv']

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
        return 'UWBRanging'

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
        return all(
            (self.data_root / f).exists()
            for f in self.REQUIRED_FILES
        )

    def _download(self) -> None:
        """Download UWB ranging dataset from GitHub."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading UWB ranging dataset from GitHub...")

        for filename in self.REQUIRED_FILES:
            url = f"{self.BASE_URL}/{filename}"
            try:
                download_url(
                    url=url,
                    root=self.data_root,
                    filename=filename,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download {filename}: {e}\n"
                    f"Please download manually from: {self.BASE_URL}"
                )

    def _load_data(self) -> None:
        """Load UWB ranging dataset from CSV files."""
        # Load anchor configuration first
        anchor_file = self.data_root / 'anchor_config.csv'
        if not anchor_file.exists():
            raise FileNotFoundError(f"Anchor config file not found: {anchor_file}")

        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required to load UWB ranging dataset.\n"
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

        # Load ranging measurements
        filepath = self.data_root / 'ranging_measurements.csv'
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_csv(filepath)

        # Group by measurement_id (each measurement can have multiple anchor readings)
        if 'measurement_id' in df.columns:
            grouped = df.groupby('measurement_id')
        else:
            # Fallback: group by position
            df['_group'] = df.groupby(['x', 'y', 'z']).ngroup()
            grouped = df.groupby('_group')

        # Split data
        num_groups = len(grouped)
        num_train = int(num_groups * self.train_ratio)

        group_indices = list(grouped.groups.keys())
        if self.split == 'train':
            selected_groups = group_indices[:num_train]
        else:  # test
            selected_groups = group_indices[num_train:]

        # Process selected groups
        for group_id in selected_groups:
            group = grouped.get_group(group_id)

            # Get location from first row (all rows in group should have same location)
            first_row = group.iloc[0]
            x = float(first_row.get('x', 0.0))
            y = float(first_row.get('y', 0.0))
            z = float(first_row.get('z', 0.0))

            # Create UWB anchors from all rows in group
            anchors = []
            for _, row in group.iterrows():
                anchor_id = str(row['anchor_id'])
                distance = float(row.get('distance', 0.0))
                rssi = float(row.get('rssi', self.NOT_DETECTED_RSSI))
                fp_power = float(row.get('fp_power', 0.0)) if 'fp_power' in row else None

                # Get anchor position
                anchor_pos = self._anchor_positions.get(anchor_id)

                anchor = UWBAnchor(
                    anchor_id=anchor_id,
                    distance=distance,
                    rssi=rssi,
                    position=anchor_pos,
                    fp_power=fp_power
                )
                anchors.append(anchor)

            # Create UWB signal
            if anchors:
                signal = UWBSignal(anchors=anchors)

                # Create location
                location = Location(
                    coordinate=Coordinate(x=x, y=y, z=z),
                    floor=0,
                    building_id='0'
                )

                self._signals.append(signal)
                self._locations.append(location)

        print(f"Loaded {len(self._signals)} samples from UWB ranging dataset ({self.split} split)")
        print(f"Total UWB anchors: {self._num_anchors}")


# Alias for convenience
UWBRanging = UWBRangingDataset
