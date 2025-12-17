"""
UJI BLE RSS Database (UJI_BLE_DB) Dataset Implementation

The UJI BLE RSS database is distributed on Zenodo as a single zip file
(UJI_BLE_DB.zip, record 1618692).

After extracting the zip, it contains:
    rss/, dep/, obs/

Each folder contains (per zone: lib, geo):
- {zone}_rss.csv: RSS matrix (rows=fingerprints, cols=beacons), non-detection=100
- {zone}_crd.csv: 2D coordinates + facing direction
- {zone}_tms.csv: timestamps
- {zone}_ids.csv: beacon identifiers (column order for RSS matrix)

This loader maps 100 -> -100 to match BLESignal.NOT_DETECTED_VALUE.

Reference:
    Mendoza-Silva, G.M.; Matey-Sanz, M.; Torres-Sospedra, J.; Huerta, J.
    BLE RSS Measurements Dataset for Research on Accurate Indoor Positioning.
    Data 2019, 4, 12. https://doi.org/10.3390/data4010012

Dataset URL: https://zenodo.org/record/1618692
"""
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Any, Dict, List, Union
import numpy as np

from .base import BLEDataset
from ..signals.ble import BLESignal
from ..locations.location import Location
from ..locations.coordinate import Coordinate
from ..registry import DATASETS
from ..utils.download import download_from_zenodo


@DATASETS.register_module()
class iBeaconRSSIDataset(BLEDataset):
    """UJI BLE RSS database (UJI_BLE_DB.zip).

    Args:
        data_root: Root directory containing the dataset files. If None,
            uses the default cache directory (~/.cache/indoorloc/datasets/ibeacon_rssi).
        split: Dataset split ('train' or 'test').
        download: Whether to download the dataset if not found.
        zone: Which zone(s) to load: 'lib', 'geo', or 'all' (default).
        db_type: Which folder inside the extracted zip to load from:
            'rss', 'dep', or 'obs' (default: 'rss').
        train_ratio: Train ratio for the shuffled 70/30 split (default: 0.7).
        seed: RNG seed used when shuffle=True (default: 42).
        shuffle: Whether to shuffle before splitting (default: True).
        transform: Optional transform to apply to signals.
        normalize: Whether to normalize RSSI values.
        normalize_method: Normalization method ('minmax', 'positive', 'standard').

    Notes:
        Raw RSS matrices use 100 for non-detection; this loader maps 100 -> -100
        to match BLESignal.NOT_DETECTED_VALUE for dense BLESignal normalization.
    """

    ZENODO_RECORD_ID = '1618692'
    ZIP_FILENAME = 'UJI_BLE_DB.zip'

    AVAILABLE_ZONES = ('lib', 'geo')
    AVAILABLE_DB_TYPES = ('rss', 'dep', 'obs')

    RAW_NOT_DETECTED_VALUE = 100
    NOT_DETECTED_VALUE = -100.0

    def __init__(
        self,
        data_root: Optional[str] = None,
        split: str = 'train',
        download: bool = False,
        zone: Union[str, List[str]] = 'all',
        db_type: str = 'rss',
        train_ratio: float = 0.7,
        seed: int = 42,
        shuffle: bool = True,
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_method: str = 'minmax',
        **kwargs
    ):
        db_type = str(db_type).lower().strip()
        if db_type not in self.AVAILABLE_DB_TYPES:
            raise ValueError(
                f"Invalid db_type='{db_type}'. Expected one of {list(self.AVAILABLE_DB_TYPES)}"
            )
        if not (0.0 < float(train_ratio) < 1.0):
            raise ValueError("train_ratio must be in (0, 1)")

        self.db_type = db_type
        self.train_ratio = float(train_ratio)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)

        self._zones = self._normalize_zones(zone)
        self._num_beacons = None

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
        return 'ibeacon_rssi'

    @property
    def num_beacons(self) -> int:
        if self._num_beacons is None:
            return 0
        return self._num_beacons

    @classmethod
    def list_zones(cls) -> List[str]:
        """List supported deployment zones."""
        return list(cls.AVAILABLE_ZONES)

    @classmethod
    def list_db_types(cls) -> List[str]:
        """List available folder types inside the extracted zip."""
        return list(cls.AVAILABLE_DB_TYPES)

    @classmethod
    def _normalize_zones(cls, zone: Union[str, List[str]]) -> List[str]:
        if isinstance(zone, str):
            zone = zone.strip().lower()
            if zone == 'all':
                zones = list(cls.AVAILABLE_ZONES)
            else:
                zones = [zone]
        else:
            zones = [str(z).strip().lower() for z in zone]

        invalid = [z for z in zones if z not in cls.AVAILABLE_ZONES]
        if invalid:
            raise ValueError(
                f"Invalid zone(s) {invalid}. Expected one of {list(cls.AVAILABLE_ZONES)} or 'all'"
            )
        return zones

    def _zone_files(self, base_dir: Path, zone: str) -> Dict[str, Path]:
        folder = base_dir / self.db_type
        return {
            'rss': folder / f"{zone}_rss.csv",
            'crd': folder / f"{zone}_crd.csv",
            'tms': folder / f"{zone}_tms.csv",
            'ids': folder / f"{zone}_ids.csv",
        }

    def _find_extracted_root(self) -> Optional[Path]:
        """Find the extracted dataset root (handles extra top-level folder in zip).

        The UJI BLE RSS database extracts to a structure like:
            data_root/data/rss/{zone}_rss.csv
        We need to return the directory containing the {db_type}/ folder.
        """
        if not self.data_root.exists():
            return None

        candidates: List[Path] = [self.data_root]
        try:
            # Add immediate subdirectories
            for p in self.data_root.iterdir():
                if p.is_dir():
                    candidates.append(p)
                    # Also add one level deeper (e.g., data_root/data/)
                    for sub in p.iterdir():
                        if sub.is_dir():
                            candidates.append(sub)
        except Exception:
            return None

        for cand in candidates:
            if (cand / self.db_type).is_dir():
                return cand

        return None

    def _has_extracted_files(self) -> bool:
        base_dir = self._find_extracted_root()
        if base_dir is None:
            return False
        for zone in self._zones:
            files = self._zone_files(base_dir, zone)
            if not all(p.exists() for p in files.values()):
                return False
        return True

    @staticmethod
    def _extract_zip(zip_path: Path, dst_root: Path) -> None:
        """Safely extract zip into dst_root (prevents path traversal)."""
        dst_root = Path(dst_root)
        dst_root.mkdir(parents=True, exist_ok=True)
        root_resolved = dst_root.resolve()

        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                member_path = Path(member.filename)
                target_path = (dst_root / member_path).resolve()

                if root_resolved != target_path and root_resolved not in target_path.parents:
                    raise RuntimeError(f"Unsafe path in zip: {member.filename}")

                target_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member, 'r') as src, open(target_path, 'wb') as dst:
                    shutil.copyfileobj(src, dst)

    def _ensure_extracted(self) -> Path:
        base_dir = self._find_extracted_root()
        if base_dir is not None:
            return base_dir

        zip_path = self.data_root / self.ZIP_FILENAME
        if zip_path.exists():
            self._extract_zip(zip_path, self.data_root)
            base_dir = self._find_extracted_root()
            if base_dir is not None:
                return base_dir

        raise FileNotFoundError(
            f"UJI BLE RSS database not found under {self.data_root}.\n"
            f"Use download=True or place {self.ZIP_FILENAME} in that directory."
        )

    @staticmethod
    def _read_numeric_csv(path: Path) -> np.ndarray:
        """Read a numeric CSV that may or may not contain a header."""
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required to load UJI BLE RSS database CSV files.\n"
                "Install with: pip install pandas"
            ) from e

        df = pd.read_csv(path, header=None)
        first_row = pd.to_numeric(df.iloc[0], errors='coerce')
        if first_row.isna().any():
            df = pd.read_csv(path, header=0)

        return df.to_numpy(dtype=np.float32)

    @staticmethod
    def _dedupe_ids(ids: List[str]) -> List[str]:
        seen: Dict[str, int] = {}
        out: List[str] = []
        for bid in ids:
            if bid in seen:
                seen[bid] += 1
                out.append(f"{bid}#{seen[bid]}")
            else:
                seen[bid] = 0
                out.append(bid)
        return out

    @classmethod
    def _read_beacon_ids(cls, path: Path, expected_len: int) -> List[str]:
        """Parse beacon identifiers from {zone}_ids.csv."""
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "pandas is required to load UJI BLE RSS database CSV files.\n"
                "Install with: pip install pandas"
            ) from e

        def clean(v: object) -> Optional[str]:
            if v is None:
                return None
            s = str(v).strip()
            if not s or s.lower() == 'nan':
                return None
            return s

        df = pd.read_csv(path, header=None, dtype=str)

        ids: List[str] = []
        if df.shape[0] == 1:
            ids = [clean(v) for v in df.iloc[0].tolist()]
            ids = [v for v in ids if v]
        elif df.shape[1] == 1:
            ids = [clean(v) for v in df.iloc[:, 0].tolist()]
            ids = [v for v in ids if v]
        else:
            for _, row in df.iterrows():
                parts = [clean(v) for v in row.tolist()]
                parts = [p for p in parts if p]
                if parts:
                    ids.append(':'.join(parts))

        if len(ids) != expected_len:
            flat = [clean(v) for v in df.values.ravel().tolist()]
            flat = [v for v in flat if v]
            if len(flat) == expected_len:
                ids = flat

        if len(ids) != expected_len:
            ids = [f"beacon_{i}" for i in range(expected_len)]

        return cls._dedupe_ids(ids)

    def _load_zone(self, base_dir: Path, zone: str) -> Dict[str, Any]:
        files = self._zone_files(base_dir, zone)
        for key, path in files.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing {key} file for zone='{zone}': {path}")

        rss = self._read_numeric_csv(files['rss']).astype(np.float32)
        crd = self._read_numeric_csv(files['crd']).astype(np.float32)
        # tms has same shape as rss (one timestamp per beacon per sample)
        # Use max timestamp per row as the sample timestamp (latest beacon detection)
        tms_raw = self._read_numeric_csv(files['tms']).astype(np.float64)
        tms = np.max(tms_raw, axis=1)  # shape: (n_samples,)

        if rss.ndim != 2:
            raise ValueError(f"Expected a 2D RSS matrix in {files['rss']}, got shape {rss.shape}")
        if crd.ndim != 2 or crd.shape[1] < 2:
            raise ValueError(f"Expected coordinates with at least 2 columns in {files['crd']}, got shape {crd.shape}")
        if rss.shape[0] != crd.shape[0]:
            raise ValueError(f"Row mismatch for zone='{zone}': rss has {rss.shape[0]} rows but crd has {crd.shape[0]}")
        if tms.shape[0] != rss.shape[0]:
            raise ValueError(f"Row mismatch for zone='{zone}': rss has {rss.shape[0]} rows but tms has {tms.shape[0]}")

        beacon_ids = self._read_beacon_ids(files['ids'], expected_len=rss.shape[1])
        coords_xy = crd[:, :2]
        direction = crd[:, 2] if crd.shape[1] >= 3 else None

        return {
            'zone': zone,
            'rss': rss,
            'coords_xy': coords_xy,
            'direction': direction,
            'timestamps': tms,
            'beacon_ids': beacon_ids,
        }

    def _check_exists(self) -> bool:
        """Check if either extracted files exist or the zip is present."""
        if self._has_extracted_files():
            return True
        return (self.data_root / self.ZIP_FILENAME).exists()

    def _download(self) -> None:
        """Download UJI_BLE_DB.zip from Zenodo and extract it."""
        if self._has_extracted_files():
            print(f"Dataset already exists at {self.data_root}")
            return

        self.data_root.mkdir(parents=True, exist_ok=True)
        zip_path = self.data_root / self.ZIP_FILENAME

        if not zip_path.exists():
            print("Downloading UJI BLE RSS database from Zenodo...")
            try:
                download_from_zenodo(
                    record_id=self.ZENODO_RECORD_ID,
                    root=self.data_root,
                    filenames=[self.ZIP_FILENAME],
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download UJI BLE RSS database: {e}\n"
                    f"Please download manually from: https://zenodo.org/record/{self.ZENODO_RECORD_ID}"
                )

        print("Extracting UJI_BLE_DB.zip...")
        self._extract_zip(zip_path, self.data_root)
        if not self._has_extracted_files():
            raise RuntimeError(
                "Downloaded/extracted the zip, but expected CSV files were not found. "
                "Please verify the archive layout under data_root."
            )

    def _load_data(self) -> None:
        base_dir = self._ensure_extracted()
        zones_data = [self._load_zone(base_dir, z) for z in self._zones]

        beacon_ids: List[str] = []
        seen = set()
        for zd in zones_data:
            for bid in zd['beacon_ids']:
                if bid not in seen:
                    seen.add(bid)
                    beacon_ids.append(bid)
        beacon_pos = {bid: i for i, bid in enumerate(beacon_ids)}

        rssi_blocks = []
        coords_blocks = []
        dir_blocks = []
        ts_blocks = []
        zone_blocks = []

        for zd in zones_data:
            rss = zd['rss'].astype(np.float32)
            # Handle NaN and raw non-detection value
            rss = np.where(np.isnan(rss), self.NOT_DETECTED_VALUE, rss)
            rss[rss == self.RAW_NOT_DETECTED_VALUE] = self.NOT_DETECTED_VALUE
            # Clip to valid RSSI range
            rss = np.clip(rss, self.NOT_DETECTED_VALUE, 0.0)

            if zd['beacon_ids'] != beacon_ids:
                aligned = np.full(
                    (rss.shape[0], len(beacon_ids)),
                    self.NOT_DETECTED_VALUE,
                    dtype=np.float32,
                )
                for j, bid in enumerate(zd['beacon_ids']):
                    aligned[:, beacon_pos[bid]] = rss[:, j]
                rss = aligned

            rssi_blocks.append(rss)
            coords_blocks.append(zd['coords_xy'])
            ts_blocks.append(zd['timestamps'])
            zone_blocks.append(np.array([zd['zone']] * rss.shape[0], dtype=object))
            if zd['direction'] is None:
                dir_blocks.append(np.full(rss.shape[0], np.nan, dtype=np.float32))
            else:
                dir_blocks.append(zd['direction'].astype(np.float32))

        rssi_all = np.vstack(rssi_blocks)
        coords_all = np.vstack(coords_blocks)
        ts_all = np.concatenate(ts_blocks)
        zones_all = np.concatenate(zone_blocks)
        dir_all = np.concatenate(dir_blocks)

        self._num_beacons = len(beacon_ids)

        num_samples = int(rssi_all.shape[0])
        indices = np.arange(num_samples)
        if self.shuffle:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(indices)

        num_train = int(num_samples * self.train_ratio)
        if num_train <= 0 or num_train >= num_samples:
            raise ValueError(
                f"Invalid split: train_ratio={self.train_ratio} yields num_train={num_train} for num_samples={num_samples}"
            )

        if self.split == 'train':
            selected = indices[:num_train]
        elif self.split == 'test':
            selected = indices[num_train:]
        else:
            raise ValueError(f"Unsupported split: {self.split!r} (expected 'train' or 'test')")

        # Map zones to numeric building IDs for compatibility
        zone_to_building = {'lib': '0', 'geo': '1'}

        for idx in selected:
            x, y = coords_all[idx]
            ts = float(ts_all[idx])
            direction = dir_all[idx]
            zone = str(zones_all[idx])

            signal = BLESignal(
                rssi_values=rssi_all[idx].copy(),  # Use copy to avoid aliasing
                beacon_ids=beacon_ids,
                timestamp=ts,
            )

            # Store zone and direction in metadata
            meta: Dict[str, Any] = {'zone': zone, 'db_type': self.db_type, 'timestamp': ts}
            if not np.isnan(direction):
                meta['direction'] = float(direction)

            location = Location(
                coordinate=Coordinate(x=float(x), y=float(y)),
                floor=0,
                building_id=zone_to_building.get(zone, '0'),
            )

            self._signals.append(signal)
            self._locations.append(location)
            self._metadata.append(meta)

        zones_info = ','.join(self._zones)
        print(
            f"Loaded {len(self._signals)} samples from UJI BLE RSS DB "
            f"(zones={zones_info}, db_type={self.db_type}, split={self.split})"
        )
        print(f"Total beacons: {self._num_beacons}")



def iBeaconRSSI(
    data_root=None,
    split=None,
    download=False,
    zone='all',
    db_type='rss',
    train_ratio=0.7,
    seed=42,
    shuffle=True,
    **kwargs,
):
    """
    Convenience function for loading the UJI BLE RSS database.

    Args:
        data_root: Root directory for dataset storage
        split: Dataset split ('train', 'test', 'all', or None for tuple)
        download: Whether to download if not found
        zone: 'lib', 'geo', or 'all'
        db_type: 'rss', 'dep', or 'obs'
        train_ratio: Train ratio for a shuffled split
        seed: RNG seed (used when shuffle=True)
        shuffle: Whether to shuffle before splitting
        **kwargs: Additional arguments passed to iBeaconRSSIDataset

    Returns:
        - If split is 'train' or 'test': Returns single dataset
        - If split is 'all': Returns merged train+test dataset
        - If split is None: Returns tuple (train_dataset, test_dataset)
    """
    if split is None:
        train_dataset = iBeaconRSSIDataset(
            data_root=data_root,
            split='train',
            download=download,
            zone=zone,
            db_type=db_type,
            train_ratio=train_ratio,
            seed=seed,
            shuffle=shuffle,
            **kwargs
        )
        test_dataset = iBeaconRSSIDataset(
            data_root=data_root,
            split='test',
            download=download,
            zone=zone,
            db_type=db_type,
            train_ratio=train_ratio,
            seed=seed,
            shuffle=shuffle,
            **kwargs
        )
        return train_dataset, test_dataset
    elif split == 'all':
        from torch.utils.data import ConcatDataset
        train_dataset = iBeaconRSSIDataset(
            data_root=data_root,
            split='train',
            download=download,
            zone=zone,
            db_type=db_type,
            train_ratio=train_ratio,
            seed=seed,
            shuffle=shuffle,
            **kwargs
        )
        test_dataset = iBeaconRSSIDataset(
            data_root=data_root,
            split='test',
            download=download,
            zone=zone,
            db_type=db_type,
            train_ratio=train_ratio,
            seed=seed,
            shuffle=shuffle,
            **kwargs
        )
        return ConcatDataset([train_dataset, test_dataset])
    else:
        return iBeaconRSSIDataset(
            data_root=data_root,
            split=split,
            download=download,
            zone=zone,
            db_type=db_type,
            train_ratio=train_ratio,
            seed=seed,
            shuffle=shuffle,
            **kwargs
        )


# Attach class methods to convenience function
iBeaconRSSI.list_zones = iBeaconRSSIDataset.list_zones
iBeaconRSSI.list_db_types = iBeaconRSSIDataset.list_db_types
