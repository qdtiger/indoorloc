"""
RFID Indoor Localization Dataset Implementation

RFID-based indoor positioning dataset with tag readings from
multiple RFID readers.

Reference:
    RFID Indoor Localization Dataset.
    UCI Machine Learning Repository.

Dataset URL: https://archive.ics.uci.edu/dataset/640/rfid+indoor+localization
"""
from pathlib import Path
from typing import Optional, Any
import numpy as np

from .base import BaseDataset
from ..signals.base import BaseSignal
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS
from ..utils.download import download_from_uci


@DATASETS.register_module()
class RFIDIndoorDataset(BaseDataset):
    """RFID Indoor Localization Dataset.

    RFID tag-based indoor positioning with reader signal strength.

    Args:
        data_root: Root directory containing the dataset files.
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        train_ratio: Ratio for train/test split (default: 0.7).

    Example:
        >>> import indoorloc as iloc
        >>> dataset = iloc.RFIDIndoor(download=True, split='train')
    """

    UCI_DATASET_NAME = 'rfid-indoor-localization'
    REQUIRED_FILES = ['rfid_data.csv']

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        train_ratio: float = 0.7,
        **kwargs
    ):
        self.train_ratio = train_ratio
        super().__init__(
            data_root=data_root,
            split=split,
            download=download,
            **kwargs
        )

    @property
    def dataset_name(self) -> str:
        return 'RFIDIndoor'

    @property
    def signal_type(self) -> str:
        return 'rfid'

    def _check_exists(self) -> bool:
        return all((self.data_root / f).exists() for f in self.REQUIRED_FILES)

    def _download(self) -> None:
        if self._check_exists():
            return
        download_from_uci(
            dataset_name=self.UCI_DATASET_NAME,
            root=self.data_root,
            filenames=self.REQUIRED_FILES,
        )

    def _load_data(self) -> None:
        import pandas as pd
        df = pd.read_csv(self.data_root / 'rfid_data.csv')

        num_train = int(len(df) * self.train_ratio)
        df_split = df.iloc[:num_train] if self.split == 'train' else df.iloc[num_train:]

        for _, row in df_split.iterrows():
            signal = BaseSignal()  # Simplified for RFID
            location = Location(
                coordinate=Coordinate(x=float(row['x']), y=float(row['y'])),
                floor=int(row.get('floor', 0)),
                building_id='0'
            )
            self._signals.append(signal)
            self._locations.append(location)

        print(f"Loaded {len(self._signals)} RFID samples ({self.split})")


RFIDIndoor = RFIDIndoorDataset
