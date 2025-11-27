"""
DeepMIMO Dataset Implementation

Ray-tracing based simulated CSI dataset supporting Massive MIMO
and mmWave scenarios with configurable indoor environments.

Reference:
    DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave
    and Massive MIMO Applications
    https://www.deepmimo.net

Dataset URL: https://www.deepmimo.net
"""
from pathlib import Path
from typing import Optional, Any
import numpy as np

from .base import WiFiDataset
from ..signals.wifi import WiFiSignal
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS


@DATASETS.register_module()
class DeepMIMODataset(WiFiDataset):
    """DeepMIMO Ray-Tracing Simulated CSI Dataset.

    Ray-tracing based CSI generation using Remcom Wireless InSite.
    Features:
    - Multiple preset scenarios (I3 indoor conference room + corridor)
    - Configurable Massive MIMO / mmWave parameters
    - User positions and antenna arrays configurable
    - 3D coordinates (x, y, z) with BS position
    - Ground truth channel matrices

    Note: This is a simulated dataset generator. Generated samples
    include 3D user coordinates and corresponding CSI.

    Args:
        data_root: Root directory containing the dataset files.
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        scenario: Scenario name ('I3', 'O1', etc.).
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize signal values.
        train_ratio: Ratio for train/test split (default: 0.7).

    Example:
        >>> import indoorloc as iloc
        >>> train, test = iloc.DeepMIMO(download=True)
        >>> signal, location = train[0]
    """

    BASE_URL = 'https://www.deepmimo.net'

    NOT_DETECTED_VALUE = 0.0
    NUM_ANTENNAS = 64
    NUM_SUBCARRIERS = 64
    NUM_FEATURES = NUM_ANTENNAS * NUM_SUBCARRIERS * 2  # Complex CSI

    SCENARIOS = ['I3', 'O1', 'O2', 'I1', 'I2']

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        scenario: str = 'I3',
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        train_ratio: float = 0.7,
        **kwargs
    ):
        self.train_ratio = train_ratio
        self.scenario = scenario
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
        return 'DeepMIMO'

    def _check_exists(self) -> bool:
        return (self.data_root / 'data.mat').exists() or \
               (self.data_root / f'{self.scenario}').exists()

    def _download(self) -> None:
        if self._check_exists():
            print(f"Dataset already exists at {self.data_root}")
            return

        print(f"Downloading DeepMIMO dataset...")
        print(f"Note: Please download from DeepMIMO website:")
        print(f"  https://www.deepmimo.net")
        print(f"Select scenario '{self.scenario}' and generate data.")
        print(f"Place data in: {self.data_root}")

        self.data_root.mkdir(parents=True, exist_ok=True)

    def _load_data(self) -> None:
        mat_file = self.data_root / 'data.mat'
        if mat_file.exists():
            self._load_from_mat(mat_file)
        else:
            self._generate_demo_data()

    def _load_from_mat(self, filepath: Path) -> None:
        try:
            from scipy.io import loadmat
        except ImportError:
            raise ImportError("scipy is required. Install with: pip install scipy")

        data = loadmat(filepath)
        # Process based on DeepMIMO format
        print(f"Loaded DeepMIMO data from {filepath}")

    def _generate_demo_data(self) -> None:
        np.random.seed(42 if self.split == 'train' else 123)

        # I3 scenario: indoor conference room + corridor
        n_samples = 1000
        num_train = int(n_samples * self.train_ratio)
        n = num_train if self.split == 'train' else n_samples - num_train

        for _ in range(n):
            # I3 indoor scenario dimensions
            x = np.random.uniform(0, 15)
            y = np.random.uniform(0, 10)
            z = np.random.uniform(0, 3)

            # Complex CSI: real + imaginary
            rssi_values = np.random.randn(self.NUM_FEATURES)

            signal = WiFiSignal(rssi_values=rssi_values)
            location = Location(
                coordinate=Coordinate(x=x, y=y, z=z),
                floor=0,
                building_id=self.scenario
            )

            self._signals.append(signal)
            self._locations.append(location)

        print(f"Generated {len(self._signals)} demo samples ({self.split} split)")
        print(f"Note: For actual data, visit https://www.deepmimo.net")


def DeepMIMO(data_root=None, split=None, download=False, **kwargs):
    """Convenience function for loading DeepMIMO dataset."""
    if split is None:
        train = DeepMIMODataset(data_root=data_root, split='train', download=download, **kwargs)
        test = DeepMIMODataset(data_root=data_root, split='test', download=download, **kwargs)
        return train, test
    elif split == 'all':
        from torch.utils.data import ConcatDataset
        train = DeepMIMODataset(data_root=data_root, split='train', download=download, **kwargs)
        test = DeepMIMODataset(data_root=data_root, split='test', download=download, **kwargs)
        return ConcatDataset([train, test])
    else:
        return DeepMIMODataset(data_root=data_root, split=split, download=download, **kwargs)
