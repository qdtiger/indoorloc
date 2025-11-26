"""
VLC (Visible Light Communication) Indoor Localization Dataset Implementation

Visible light communication-based indoor positioning dataset using LED
transmitters and photodetectors for location determination.

Reference:
    VLC Indoor Positioning System Dataset.
    GitHub Repository.
    https://github.com/vlc-positioning/indoor-dataset

Dataset URL: https://raw.githubusercontent.com/VLC-Positioning/IndoorDataset/main
"""
from pathlib import Path
from typing import Optional, Any, Dict, List
import numpy as np

from .base import BaseDataset
from ..signals.vlc import VLCSignal, LEDTransmitter
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS
from ..utils.download import download_url


@DATASETS.register_module()
class VLCIndoorDataset(BaseDataset):
    """VLC Indoor Localization Dataset.

    Indoor positioning dataset using Visible Light Communication (VLC)
    with LED transmitters and photodetector receivers.

    Args:
        data_root: Root directory containing the dataset files. If None,
            uses the default cache directory (~/.cache/indoorloc/datasets/vlc_indoor).
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        normalize_method: Normalization method ('minmax', 'positive', 'standard').
        train_ratio: Ratio for train/test split (default: 0.7).

    Example:
        >>> import indoorloc as iloc
        >>> # Download VLC indoor dataset
        >>> dataset = iloc.VLCIndoor(download=True, split='train')
        >>> # Access VLC signal with LED transmitter measurements
        >>> signal = dataset[0]
        >>> for tx in signal.transmitters:
        ...     print(f"LED {tx.transmitter_id}: RSSI={tx.rssi}dBm")

    Dataset structure:
        data_root/
        ├── vlc_measurements.csv
        └── led_positions.csv

    CSV format (vlc_measurements.csv):
        - Columns: measurement_id, x, y, z, led_id, rssi, snr, modulation
        - Multiple rows per location (one per visible LED)
        - RSSI in dBm, SNR in dB

    CSV format (led_positions.csv):
        - Columns: led_id, x, y, z, wavelength
        - LED transmitter positions and wavelength (nm)
    """

    # GitHub raw content base URL
    BASE_URL = 'https://raw.githubusercontent.com/VLC-Positioning/IndoorDataset/main'

    # Dataset constants
    NOT_DETECTED_RSSI = -100.0
    MIN_SNR = -20.0  # dB
    MAX_SNR = 40.0   # dB

    # Required files
    REQUIRED_FILES = ['vlc_measurements.csv', 'led_positions.csv']

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
        self._num_transmitters = None
        self._led_positions = {}

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
        return 'VLCIndoor'

    @property
    def signal_type(self) -> str:
        return 'vlc'

    @property
    def num_transmitters(self) -> int:
        if self._num_transmitters is None:
            return 0
        return self._num_transmitters

    @property
    def led_positions(self) -> Dict[str, np.ndarray]:
        """Get LED transmitter positions as {led_id: [x, y, z]} dict."""
        return self._led_positions

    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        return all(
            (self.data_root / f).exists()
            for f in self.REQUIRED_FILES
        )

    def _download(self) -> None:
        """Download VLC indoor dataset from GitHub."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading VLC indoor dataset from GitHub...")

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
        """Load VLC indoor dataset from CSV files."""
        # Load LED positions first
        led_file = self.data_root / 'led_positions.csv'
        if not led_file.exists():
            raise FileNotFoundError(f"LED positions file not found: {led_file}")

        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required to load VLC indoor dataset.\n"
                "Install with: pip install pandas"
            )

        # Load LED positions
        led_df = pd.read_csv(led_file)
        for _, row in led_df.iterrows():
            led_id = str(row['led_id'])
            x = float(row.get('x', 0.0))
            y = float(row.get('y', 0.0))
            z = float(row.get('z', 0.0))
            self._led_positions[led_id] = np.array([x, y, z])

        self._num_transmitters = len(self._led_positions)

        # Load VLC measurements
        filepath = self.data_root / 'vlc_measurements.csv'
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_csv(filepath)

        # Group by measurement_id (each measurement can have multiple LED readings)
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

            # Get location from first row
            first_row = group.iloc[0]
            x = float(first_row.get('x', 0.0))
            y = float(first_row.get('y', 0.0))
            z = float(first_row.get('z', 0.0))

            # Create VLC transmitters from all rows in group
            transmitters = []
            for _, row in group.iterrows():
                led_id = str(row['led_id'])
                rssi = float(row.get('rssi', self.NOT_DETECTED_RSSI))
                snr = float(row.get('snr', 0.0))
                modulation = str(row.get('modulation', 'OOK'))

                # Get LED position
                led_pos = self._led_positions.get(led_id)

                transmitter = LEDTransmitter(
                    transmitter_id=led_id,
                    rssi=rssi,
                    snr=snr,
                    position=led_pos,
                    modulation=modulation
                )
                transmitters.append(transmitter)

            # Create VLC signal
            if transmitters:
                signal = VLCSignal(transmitters=transmitters)

                # Create location
                location = Location(
                    coordinate=Coordinate(x=x, y=y, z=z),
                    floor=0,
                    building_id='0'
                )

                self._signals.append(signal)
                self._locations.append(location)

        print(f"Loaded {len(self._signals)} samples from VLC indoor dataset ({self.split} split)")
        print(f"Total LED transmitters: {self._num_transmitters}")


# Alias for convenience
VLCIndoor = VLCIndoorDataset
