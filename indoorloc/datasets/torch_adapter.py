"""Optional PyTorch bridge: the only module in L1 that imports torch (at module level, on purpose).

``DataLoader`` fetches whole batches through ``__getitems__`` (one fancy-index per
batch, not batch_size item lookups). Each batch is a dict: numeric columns become
tensors (dtypes kept: X float32, pos float64, labels int64); ``ids`` and non-numeric
group columns (e.g. "source" = "sim"/"real") stay numpy arrays.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..core import SampleTable


def _leaf(a: np.ndarray):
    return torch.from_numpy(np.ascontiguousarray(a)) if a.dtype.kind in "biufc" else a


def _as_is(batch: dict) -> dict:
    return batch  # __getitems__ already returns a collated batch


class TorchDataset(Dataset):
    def __init__(self, table: SampleTable):
        self.table = table

    def __len__(self) -> int:
        return len(self.table)

    def __getitems__(self, indices) -> dict:
        rows, t = np.asarray(indices, dtype=np.intp), self.table
        batch = {"X": t.X, "pos": t.pos, "floor": t.floor, "building": t.building, "ids": t.ids,
                 **{f"groups.{k}": v for k, v in t.groups.items()}}
        return {k: _leaf(v[rows]) for k, v in batch.items() if v is not None}  # fancy index: fresh, writable

    def __getitem__(self, i: int) -> dict:
        return {k: v[0] for k, v in self.__getitems__([i]).items()}


def make_dataloader(table: SampleTable, batch_size: int = 256, shuffle: bool = False, seed: int = 0,
                    **kwargs) -> DataLoader:
    """A DataLoader of dict batches; shuffling is seeded by default (reproducible epochs)."""
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(TorchDataset(table), batch_size=batch_size, shuffle=shuffle, generator=generator,
                      collate_fn=_as_is, **kwargs)
