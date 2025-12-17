"""
Wireless Indoor Localization Dataset (UCI) Implementation

WiFi-based indoor localization dataset with RSSI measurements from 7 access
points for room classification.

Reference:
    Rajen Bhatt (2017). Wireless Indoor Localization. UCI Machine Learning
    Repository. DOI: 10.24432/C5PC8M

Dataset URL: https://archive.ics.uci.edu/dataset/422/wireless+indoor+localization
"""
import zipfile
from pathlib import Path
from typing import Optional, Any, List, Union
import numpy as np

from .base import WiFiDataset
from ..signals.wifi import WiFiSignal
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS
from ..utils.download import download_url


@DATASETS.register_module()
class WLANRSSIDataset(WiFiDataset):
    """Wireless Indoor Localization Dataset (UCI).

    WiFi fingerprinting dataset with RSSI measurements from 7 access
    points for room-level classification (4 rooms).

    Args:
        data_root: Root directory containing the dataset files. If None,
            uses the default cache directory (~/.cache/indoorloc/datasets/wlanrssi).
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize RSSI values.
        normalize_method: Normalization method ('minmax', 'positive', 'standard').
        train_ratio: Ratio for train/test split (default: 0.7).

    Example:
        >>> import indoorloc as iloc
        >>> train, test = iloc.WLANRSSI(download=True)

    Dataset structure:
        data_root/
        └── wifi_localization.txt

    File format:
        Tab-separated: rssi1 rssi2 rssi3 rssi4 rssi5 rssi6 rssi7 room
        - 7 WiFi signal strengths in dBm
        - room: 1-4 (room classification label)
        - 2000 total instances (500 per room)
    """

    # UCI download URL
    UCI_URL = "https://archive.ics.uci.edu/static/public/422/wireless+indoor+localization.zip"
    ZIP_FILENAME = "wireless_indoor_localization.zip"

    # Dataset constants
    NOT_DETECTED_VALUE = -110  # Typical WiFi noise floor
    NUM_WAPS = 7  # This dataset has 7 access points
    NUM_ROOMS = 4  # 4 room classes

    # Required files
    REQUIRED_FILES = ['wifi_localization.txt']

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        room: Union[int, List[int], str] = 'all',
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        train_ratio: float = 0.7,
        **kwargs
    ):
        self.train_ratio = train_ratio
        self._room_param = room

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
        return 'wlanrssi'

    @property
    def num_aps(self) -> int:
        return self.NUM_WAPS

    @classmethod
    def list_rooms(cls) -> List[int]:
        """List all available rooms (1-4)."""
        return [1, 2, 3, 4]

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

        self.data_root.mkdir(parents=True, exist_ok=True)
        zip_path = self.data_root / self.ZIP_FILENAME

        # Download zip file
        if not zip_path.exists():
            print("Downloading Wireless Indoor Localization dataset from UCI...")
            try:
                download_url(
                    url=self.UCI_URL,
                    root=self.data_root,
                    filename=self.ZIP_FILENAME,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download WLAN RSSI dataset: {e}\n"
                    f"Please download manually from: "
                    f"https://archive.ics.uci.edu/dataset/422/"
                )

        # Extract zip file
        print("Extracting dataset files...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for member in zf.namelist():
                    if member.endswith('.txt'):
                        zf.extract(member, self.data_root)
                        print(f"  Extracted: {member}")
        except Exception as e:
            raise RuntimeError(f"Failed to extract dataset: {e}")

    def _load_data(self) -> None:
        """Load WLAN RSSI dataset from file."""
        filepath = self.data_root / 'wifi_localization.txt'

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        # Room to coordinate mapping (2x2 grid layout)
        # Room 1: (0,0), Room 2: (1,0), Room 3: (0,1), Room 4: (1,1)
        room_to_coord = {
            1: (0.0, 0.0),
            2: (1.0, 0.0),
            3: (0.0, 1.0),
            4: (1.0, 1.0),
        }

        # Load data: tab-separated rssi1-7 and room label
        all_samples = []
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 8:
                    continue
                try:
                    rssi_values = np.array([float(parts[i]) for i in range(7)], dtype=np.float32)
                    room = int(parts[7])
                    x, y = room_to_coord.get(room, (0.0, 0.0))

                    all_samples.append({
                        'rssi': rssi_values,
                        'room': room,
                        'x': x,
                        'y': y,
                    })
                except (ValueError, IndexError):
                    continue

        if len(all_samples) == 0:
            raise RuntimeError("No valid samples could be parsed")

        # Filter by room
        if self._room_param != 'all':
            if isinstance(self._room_param, int):
                selected_rooms = [self._room_param]
            else:
                selected_rooms = list(self._room_param)
            all_samples = [s for s in all_samples if s['room'] in selected_rooms]

        if len(all_samples) == 0:
            raise ValueError(f"No data found for room={self._room_param}")

        # Shuffle before splitting (use fixed seed for reproducibility)
        np.random.seed(42)
        np.random.shuffle(all_samples)

        # Split into train/test
        num_train = int(len(all_samples) * self.train_ratio)

        if self.split == 'train':
            samples = all_samples[:num_train]
        else:  # test
            samples = all_samples[num_train:]

        # Process samples
        for sample in samples:
            signal = WiFiSignal(rssi_values=sample['rssi'])
            location = Location(
                coordinate=Coordinate(x=sample['x'], y=sample['y']),
                floor=sample['room'],  # Use room as floor for compatibility
                building_id='0'
            )

            self._signals.append(signal)
            self._locations.append(location)

        room_info = f" (room: {self._room_param})" if self._room_param != 'all' else ""
        print(f"Loaded {len(self._signals)} samples from WLAN RSSI dataset ({self.split} split){room_info}")



def WLANRSSI(data_root=None, split=None, download=False, room='all', **kwargs):
    """
    Convenience function for loading WLANRSSI dataset.

    Args:
        data_root: Root directory for dataset storage
        split: Dataset split ('train', 'test', 'all', or None for tuple)
        download: Whether to download if not found
        room: Room(s) to load. Can be:
            - 'all': Load all rooms (default)
            - Single room: 1, 2, 3, 4
            - List of rooms: [1, 2]
        **kwargs: Additional arguments passed to WLANRSSIDataset

    Returns:
        - If split is 'train' or 'test': Returns single dataset
        - If split is 'all': Returns merged train+test dataset
        - If split is None: Returns tuple (train_dataset, test_dataset)

    Examples:
        >>> train, test = WLANRSSI(download=True)
        >>> WLANRSSI.list_rooms()
    """
    if split is None:
        train_dataset = WLANRSSIDataset(
            data_root=data_root,
            split='train',
            download=download,
            room=room,
            **kwargs
        )
        test_dataset = WLANRSSIDataset(
            data_root=data_root,
            split='test',
            download=download,
            room=room,
            **kwargs
        )
        return train_dataset, test_dataset
    elif split == 'all':
        from torch.utils.data import ConcatDataset
        train_dataset = WLANRSSIDataset(
            data_root=data_root,
            split='train',
            download=download,
            room=room,
            **kwargs
        )
        test_dataset = WLANRSSIDataset(
            data_root=data_root,
            split='test',
            download=download,
            room=room,
            **kwargs
        )
        return ConcatDataset([train_dataset, test_dataset])
    else:
        return WLANRSSIDataset(
            data_root=data_root,
            split=split,
            download=download,
            room=room,
            **kwargs
        )


WLANRSSI.list_rooms = WLANRSSIDataset.list_rooms

