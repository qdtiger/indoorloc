"""Column-store containers that travel between layers.

Two types only: ``SampleTable`` (L1 exit, L2/L3 entry) and ``Prediction``
(L3 exit, L4/L5 entry). Both hold plain numpy arrays as read-only views; a row is
never an object.
Rule for every array in the library: the first axis is samples.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ._optional import requires


def as_labels(values, n: int, name: str) -> np.ndarray | None:
    """Integer label column. None = no labels; NaN or fractional labels are an error."""
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.dtype.kind == "f":
        if not np.all(np.isfinite(arr)) or np.any(arr != np.round(arr)):
            raise ValueError(f"{name} must hold integer labels (use None for 'no labels'), got {arr.dtype} "
                             "with NaN or fractions")
    arr = arr.astype(np.int64)
    if arr.shape != (n,):
        raise ValueError(f"{name} has shape {arr.shape}, expected ({n},)")
    return arr


def _column(values, n: int, name: str) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values)
    if len(arr) != n:
        raise ValueError(f"{name} has {len(arr)} rows, expected {n}")
    return arr


def _positions(values) -> np.ndarray:
    pos = np.asarray(values, dtype=np.float64)  # float64: float32 is 0.5 m coarse at 4.8e6 m
    return pos[:, None] if pos.ndim == 1 else pos


def _read_only(arr: np.ndarray | None) -> np.ndarray | None:
    """A read-only view: the container cannot be changed in place; the caller's array stays writable."""
    if arr is None:
        return None
    view = arr.view()
    view.flags.writeable = False
    return view


@dataclass(frozen=True, eq=False)
class SampleTable:
    """N samples stored as parallel arrays.

    X         (N, ...) observations in physical units (RSSI dBm, complex CSI, ...);
              NaN marks a missing reading. No sentinel values, no normalization.
    pos       (N, D) float64 coordinates in the frame named by ``meta["crs"]``.
    floor     (N,) int64 or None (None = unlabelled; negative floors are real floors).
    building  (N,) int64 or None.
    groups    name -> (N,) array (user, device, time, source, ...) for grouped splits.
    ids       (N,) stable sample ids; defaults to the row number.
    meta      dataset-level facts: name, split, crs, units, feature_names, sha256, ...
    """

    X: np.ndarray
    pos: np.ndarray
    floor: np.ndarray | None = None
    building: np.ndarray | None = None
    groups: Mapping[str, np.ndarray] = field(default_factory=dict)
    ids: np.ndarray | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        X = np.asarray(self.X)
        if X.ndim < 2:
            raise ValueError(f"X must be (N, ...) with N samples first, got shape {X.shape}")
        n = len(X)
        pos = _positions(self.pos)
        if pos.ndim != 2 or len(pos) != n:
            raise ValueError(f"pos must be (N, D) with N={n}, got shape {pos.shape}")
        set_ = object.__setattr__
        set_(self, "X", _read_only(X))
        set_(self, "pos", _read_only(pos))
        set_(self, "floor", _read_only(as_labels(self.floor, n, "floor")))
        set_(self, "building", _read_only(as_labels(self.building, n, "building")))
        set_(self, "groups", {k: _read_only(_column(v, n, f"groups[{k!r}]")) for k, v in self.groups.items()})
        set_(self, "ids", _read_only(_column(np.arange(n) if self.ids is None else self.ids, n, "ids")))
        set_(self, "meta", dict(self.meta))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, rows) -> SampleTable:
        """Row subset by integer array, boolean mask, slice or int; always a table."""
        if isinstance(rows, (int, np.integer)):
            rows = [rows]
        take = lambda a: None if a is None else a[rows]  # noqa: E731
        return SampleTable(self.X[rows], self.pos[rows], take(self.floor), take(self.building),
                           {k: v[rows] for k, v in self.groups.items()}, self.ids[rows], self.meta)

    def __iter__(self):
        # Row iteration is never what is meant: most often ``train, test = load_dataset(name)`` on a
        # dataset whose only table is "all", which would otherwise fail with "too many values to unpack".
        name = self.meta.get("name") or "this table"
        raise TypeError(f"a SampleTable is not iterable ({name}, {len(self)} rows, split "
                        f"{self.meta.get('split')!r}). Unpacking train, test = load_dataset(...) needs a dataset with an "
                        "official train/test split; otherwise build one with indoorloc.get_protocol(...).folds(table) "
                        "or indoorloc.evaluation.random_split. Rows: table[i], table.X, table.pos.")

    def __repr__(self) -> str:
        meta = self.meta
        facts = ", ".join(f"{k}={meta[k]!r}" for k in ("name", "split", "modality", "units", "crs") if meta.get(k))
        labels = [k for k in ("floor", "building") if getattr(self, k) is not None]
        return (f"SampleTable(n={len(self)}, X={self.X.shape} {self.X.dtype}, pos={self.pos.shape}"
                f"{', ' + '/'.join(labels) if labels else ''}, groups={sorted(self.groups)}"
                f"{', ' + facts if facts else ''}, meta keys={len(meta)})")

    def replace(self, **changes) -> SampleTable:
        """Copy with some fields replaced (L2 transforms return a new table)."""
        return dataclasses.replace(self, **changes)

    @classmethod
    def concat(cls, tables) -> SampleTable:
        """Stack tables row-wise (e.g. simulated + measured). Meta comes from the first table;
        ids must stay unique (give each source its own ids first)."""
        tables = list(tables)
        cat = lambda get: None if get(tables[0]) is None else np.concatenate([get(t) for t in tables])  # noqa: E731
        keys = set(tables[0].groups)
        if any(set(t.groups) != keys for t in tables):
            raise ValueError("tables must have the same group names to be concatenated")
        ids = cat(lambda t: t.ids)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("duplicate ids across the tables; give each source its own ids first, "
                             "e.g. sim.replace(ids=np.char.add('sim-', sim.ids.astype(str)))")
        return cls(cat(lambda t: t.X), cat(lambda t: t.pos), cat(lambda t: t.floor), cat(lambda t: t.building),
                   {k: np.concatenate([t.groups[k] for t in tables]) for k in keys}, ids, tables[0].meta)

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """``(X, pos)``: the arrays every sklearn-style estimator takes (read-only views)."""
        return self.X, self.pos

    def to_dataframe(self):
        """One row per sample; features, coordinates, labels and groups as columns (needs pandas)."""
        pd = requires("pandas", "pandas")
        X = self.X.reshape(len(self), -1)
        names = self.meta.get("feature_names") or [f"f{j}" for j in range(X.shape[1])]
        axes = self.meta.get("pos_names") or [f"pos{d}" for d in range(self.pos.shape[1])]
        cols = {**dict(zip(names, X.T)), **dict(zip(axes, self.pos.T))}
        cols.update({k: v for k, v in (("floor", self.floor), ("building", self.building)) if v is not None})
        cols.update(self.groups)
        return pd.DataFrame(cols, index=pd.Index(self.ids, name="id"))

    def to_torch(self):
        """``(X, pos)`` as tensors (needs torch). For batches with labels use datasets.torch_adapter."""
        torch = requires("torch", "torch")
        return torch.as_tensor(self.X), torch.as_tensor(self.pos)


@dataclass(frozen=True, eq=False)
class Prediction:
    """What a localizer returns.

    pos       (N, D) float64 estimated coordinates.
    floor     (N,) int64 or None;  building likewise.
    ids       (N,) sample ids copied from the input table, if any.
    spread    (N,) float64 or None: a method's own scale of positional uncertainty in
              coordinate units (k-NN: weighted RMS distance of the neighbours from the
              estimate). A heuristic for L5 measurement noise, not a calibrated std.
    """

    pos: np.ndarray
    floor: np.ndarray | None = None
    building: np.ndarray | None = None
    ids: np.ndarray | None = None
    spread: np.ndarray | None = None

    def __post_init__(self):
        pos = _positions(self.pos)
        n = len(pos)
        spread = None if self.spread is None else _column(np.asarray(self.spread, np.float64), n, "spread")
        set_ = object.__setattr__
        set_(self, "pos", _read_only(pos))
        set_(self, "floor", _read_only(as_labels(self.floor, n, "floor")))
        set_(self, "building", _read_only(as_labels(self.building, n, "building")))
        set_(self, "ids", _read_only(_column(self.ids, n, "ids")))
        set_(self, "spread", _read_only(spread))

    def __len__(self) -> int:
        return len(self.pos)

    def __repr__(self) -> str:
        # compact (Jupyter shows repr): the size and which optional columns are present, never the arrays
        present = [k for k in ("floor", "building", "spread", "ids") if getattr(self, k) is not None]
        unplaced = int(np.count_nonzero(~np.isfinite(self.pos).all(axis=tuple(range(1, self.pos.ndim)))))
        return (f"Prediction(n={len(self)}, pos={self.pos.shape}{', ' + '/'.join(present) if present else ''}"
                f"{f', {unplaced} not placed' if unplaced else ''})")

    def __getitem__(self, rows) -> Prediction:
        if isinstance(rows, (int, np.integer)):
            rows = [rows]
        take = lambda a: None if a is None else a[rows]  # noqa: E731
        return Prediction(self.pos[rows], take(self.floor), take(self.building), take(self.ids), take(self.spread))
