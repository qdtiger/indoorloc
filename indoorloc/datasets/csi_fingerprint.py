"""
CSI-dataset-for-indoor-localization Dataset Implementation

WiFi CSI fingerprint dataset using Intel 5300 with Linux 802.11n CSI Tool.
Contains data from multiple indoor areas with NLOS and LOS conditions.

Reference:
    CSI-dataset-for-indoor-localization
    https://github.com/qiang5love1314/CSI-dataset

Dataset URL: https://github.com/qiang5love1314/CSI-dataset

Data format:
    Each .mat file contains 'myData' with shape (3, 30, 1500):
    - 3 antennas (Intel 5300)
    - 30 subcarriers
    - 1500 time samples (CSI packets)
"""
import re
from pathlib import Path
from typing import Optional, Any, List, Union, Dict
import numpy as np

from .base import WiFiDataset
from ..signals.wifi import WiFiSignal
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS


@DATASETS.register_module()
class CSIFingerprintDataset(WiFiDataset):
    """CSI-dataset-for-indoor-localization Dataset.

    WiFi CSI fingerprint dataset collected using Intel 5300 NIC with Linux
    802.11n CSI Tool. Contains measurements from multiple indoor areas:
    - lab: 13.5×11m laboratory (NLOS, 317 points)
    - meeting: 7×10m meeting room (LOS, 176 points)

    Args:
        data_root: Root directory containing the dataset files.
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        area: Area to load ('lab', 'meeting', or 'all').
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        train_ratio: Ratio for train/test split (default: 0.7).
        aggregation: How to aggregate time samples ('mean', 'median', 'first', 'last').
        seed: Random seed for train/test split.

    Example:
        >>> import indoorloc as iloc
        >>> train, test = iloc.CSIFingerprint(download=True, area='lab')
        >>> signal, location = train[0]
    """

    GITHUB_RAW_BASE = 'https://raw.githubusercontent.com/qiang5love1314/CSI-dataset/master'

    # Area mapping: internal name -> GitHub folder name
    AREA_FOLDERS = {
        'lab': 'Lab Dataset',
        'meeting': 'Meeting Room Dataset',
    }

    # Area dimensions (meters)
    AREA_DIMENSIONS = {
        'lab': (13.5, 11.0),      # 13.5×11m, 317 points
        'meeting': (7.0, 10.0),   # 7×10m, 176 points
    }

    # Coordinate subfolders for each area
    COORDINATE_FOLDERS = {
        'lab': ['coordinate 1-100', 'coordinate 101-200', 'coordinate 201-300', 'coordinate 301-317'],
        'meeting': ['coordinate 1-100', 'coordinate 101-176'],
    }

    NOT_DETECTED_VALUE = 0.0
    NUM_ANTENNAS = 3
    NUM_SUBCARRIERS = 30

    AVAILABLE_AREAS = ['lab', 'meeting']

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        area: Union[str, List[str]] = 'all',
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        train_ratio: float = 0.7,
        aggregation: str = 'mean',
        seed: int = 42,
        **kwargs
    ):
        self.train_ratio = train_ratio
        self.aggregation = aggregation
        self.seed = seed

        # Parse area parameter
        if area == 'all':
            self._areas = self.AVAILABLE_AREAS.copy()
        elif isinstance(area, list):
            self._areas = [a.lower() for a in area]
        else:
            self._areas = [area.lower()]

        # Validate areas
        for a in self._areas:
            if a not in self.AVAILABLE_AREAS:
                raise ValueError(f"Unknown area '{a}'. Available: {self.AVAILABLE_AREAS}")

        self.area = self._areas[0] if len(self._areas) == 1 else None

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
        return 'CSIFingerprint'

    @property
    def num_aps(self) -> int:
        return self.NUM_ANTENNAS * self.NUM_SUBCARRIERS  # 90 features

    @classmethod
    def list_areas(cls) -> List[str]:
        """List all available areas."""
        return cls.AVAILABLE_AREAS.copy()

    def _check_exists(self) -> bool:
        """Check if dataset files exist."""
        for area in self._areas:
            area_dir = self.data_root / area
            if not area_dir.exists():
                return False
            mat_files = list(area_dir.glob('*.mat'))
            if len(mat_files) == 0:
                return False
        return True

    def _download(self) -> None:
        """Download CSI fingerprint dataset from GitHub."""
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print("Downloading CSI fingerprint dataset from GitHub...")
        self.data_root.mkdir(parents=True, exist_ok=True)

        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        # Setup session with retry
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('https://', adapter)

        for area in self._areas:
            area_dir = self.data_root / area
            area_dir.mkdir(parents=True, exist_ok=True)

            github_folder = self.AREA_FOLDERS[area]
            coord_folders = self.COORDINATE_FOLDERS[area]

            print(f"  Downloading {area} ({github_folder})...")

            downloaded = 0
            failed = 0
            total = sum(
                int(re.search(r'(\d+)-(\d+)', cf).group(2)) - int(re.search(r'(\d+)-(\d+)', cf).group(1)) + 1
                for cf in coord_folders if re.search(r'(\d+)-(\d+)', cf)
            )

            for coord_folder in coord_folders:
                match = re.search(r'(\d+)-(\d+)', coord_folder)
                if not match:
                    continue
                start, end = int(match.group(1)), int(match.group(2))

                for coord_num in range(start, end + 1):
                    filename = f"coordinate{coord_num}.mat"
                    url = f"{self.GITHUB_RAW_BASE}/{github_folder.replace(' ', '%20')}/{coord_folder.replace(' ', '%20')}/{filename}"

                    dst_path = area_dir / filename
                    if dst_path.exists() and dst_path.stat().st_size > 0:
                        downloaded += 1
                        continue

                    try:
                        resp = session.get(url, timeout=60)
                        if resp.status_code == 200 and len(resp.content) > 1000:
                            dst_path.write_bytes(resp.content)
                            downloaded += 1
                            if downloaded % 50 == 0:
                                print(f"    Progress: {downloaded}/{total} files...")
                        else:
                            failed += 1
                    except Exception:
                        failed += 1

            print(f"    Downloaded {downloaded} files for {area} ({failed} failed)")

        if not self._check_exists():
            print("Warning: Download incomplete. Some files may be missing.")
            print(f"You can manually download from: https://github.com/qiang5love1314/CSI-dataset")

    def _parse_coordinate(self, filename: str, area: str) -> tuple:
        """Parse x, y coordinates from filename.

        According to README: coordinate 715 represents position [7, 15]
        The coordinate number encodes the grid position directly:
        - First digit(s): row (y coordinate in grid units)
        - Last two digits: column (x coordinate in grid units)

        Grid spacing is approximately 0.6m based on area dimensions.
        """
        match = re.search(r'coordinate(\d+)', filename)
        if not match:
            return 0.0, 0.0

        coord_str = match.group(1)

        # Parse: coordinate715 -> row=7, col=15
        if len(coord_str) == 3:
            row = int(coord_str[0])
            col = int(coord_str[1:])
        elif len(coord_str) == 4:
            row = int(coord_str[:2])
            col = int(coord_str[2:])
        else:
            row = 0
            col = int(coord_str)

        # Grid spacing (approximately 0.6m per grid unit based on area dimensions)
        # Lab: 13.5m / 22 cols ≈ 0.6m, 11m / 18 rows ≈ 0.6m
        grid_spacing = 0.6

        x = col * grid_spacing
        y = row * grid_spacing

        return float(x), float(y)

    def _aggregate_time_samples(self, data: np.ndarray) -> np.ndarray:
        """Aggregate time samples (axis=2) into a single feature vector.

        Args:
            data: CSI data with shape (3, 30, T) where T is time samples

        Returns:
            Aggregated features with shape (90,) = 3 antennas × 30 subcarriers
        """
        # Replace inf with nan for proper aggregation
        data = np.where(np.isinf(data), np.nan, data)

        if self.aggregation == 'mean':
            aggregated = np.nanmean(data, axis=2)
        elif self.aggregation == 'median':
            aggregated = np.nanmedian(data, axis=2)
        elif self.aggregation == 'first':
            aggregated = data[:, :, 0]
        elif self.aggregation == 'last':
            aggregated = data[:, :, -1]
        else:
            aggregated = np.nanmean(data, axis=2)

        # Replace any remaining nan with 0
        aggregated = np.nan_to_num(aggregated, nan=0.0)

        return aggregated.flatten().astype(np.float32)

    def _load_data(self) -> None:
        """Load CSI fingerprint dataset from .mat files."""
        try:
            from scipy.io import loadmat
        except ImportError:
            raise ImportError("scipy is required. Install with: pip install scipy")

        all_samples: List[Dict] = []

        for area in self._areas:
            area_dir = self.data_root / area
            if not area_dir.exists():
                print(f"Warning: Area directory not found: {area_dir}")
                continue

            mat_files = sorted(area_dir.glob('*.mat'))
            if not mat_files:
                print(f"Warning: No .mat files found in {area_dir}")
                continue

            for mat_file in mat_files:
                try:
                    data = loadmat(str(mat_file))
                    if 'myData' not in data:
                        continue

                    csi_data = data['myData']  # shape: (3, 30, 1500)
                    if csi_data.ndim != 3:
                        continue

                    # Aggregate time samples
                    features = self._aggregate_time_samples(csi_data)

                    # Parse coordinates
                    x, y = self._parse_coordinate(mat_file.name, area)

                    all_samples.append({
                        'features': features,
                        'x': x,
                        'y': y,
                        'area': area,
                        'filename': mat_file.name,
                    })

                except Exception as e:
                    print(f"Warning: Failed to load {mat_file.name}: {e}")

        if not all_samples:
            raise RuntimeError(f"No valid samples found. Check data at {self.data_root}")

        # Shuffle with seed and split
        np.random.seed(self.seed)
        indices = np.random.permutation(len(all_samples))
        num_train = int(len(all_samples) * self.train_ratio)

        if self.split == 'train':
            selected_indices = indices[:num_train]
        else:
            selected_indices = indices[num_train:]

        for idx in selected_indices:
            sample = all_samples[idx]

            # Use WiFiSignal for aggregated CSI amplitude features
            signal = WiFiSignal(rssi_values=sample['features'])

            location = Location(
                coordinate=Coordinate(x=sample['x'], y=sample['y']),
                floor=0,
                building_id=sample['area'],
            )

            self._signals.append(signal)
            self._locations.append(location)
            self._metadata.append({
                'area': sample['area'],
                'filename': sample['filename'],
                'num_antennas': self.NUM_ANTENNAS,
                'num_subcarriers': self.NUM_SUBCARRIERS,
            })

        areas_info = ','.join(self._areas)
        print(f"Loaded {len(self._signals)} samples from CSI-Fingerprint "
              f"(areas={areas_info}, split={self.split})")


def CSIFingerprint(data_root=None, split=None, download=False, area='all', **kwargs):
    """
    Convenience function for loading CSIFingerprint dataset.

    Args:
        data_root: Root directory for dataset storage
        split: Dataset split ('train', 'test', 'all', or None for tuple)
        download: Whether to download if not found
        area: Area(s) to load ('lab', 'meeting', 'all', or list)
        **kwargs: Additional arguments passed to CSIFingerprintDataset

    Returns:
        - If split is 'train' or 'test': Returns single dataset
        - If split is 'all': Returns merged train+test dataset
        - If split is None: Returns tuple (train_dataset, test_dataset)

    Examples:
        >>> train, test = CSIFingerprint(download=True)
        >>> train = CSIFingerprint(area='lab', split='train')
        >>> CSIFingerprint.list_areas()
    """
    if split is None:
        train_dataset = CSIFingerprintDataset(
            data_root=data_root, split='train', download=download, area=area, **kwargs
        )
        test_dataset = CSIFingerprintDataset(
            data_root=data_root, split='test', download=download, area=area, **kwargs
        )
        return train_dataset, test_dataset
    elif split == 'all':
        from torch.utils.data import ConcatDataset
        train_dataset = CSIFingerprintDataset(
            data_root=data_root, split='train', download=download, area=area, **kwargs
        )
        test_dataset = CSIFingerprintDataset(
            data_root=data_root, split='test', download=download, area=area, **kwargs
        )
        return ConcatDataset([train_dataset, test_dataset])
    else:
        return CSIFingerprintDataset(
            data_root=data_root, split=split, download=download, area=area, **kwargs
        )


# Attach class method to convenience function
CSIFingerprint.list_areas = CSIFingerprintDataset.list_areas
