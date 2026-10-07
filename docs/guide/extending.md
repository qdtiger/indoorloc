# Extending IndoorLoc

A new dataset, transform, method or protocol plugs into a registry and follows the rules of
[CONTRACTS.md](../architecture/CONTRACTS.md). This page shows a minimal working example of each;
the contracts define the complete rules that code review and the tests enforce.

[Guide home](index.md) · [中文](../zh/extending.md)

## Where things go

| You add | File | Base class | Registered in | Contract |
| --- | --- | --- | --- | --- |
| a dataset | `indoorloc/datasets/<name>.py` | `datasets._base.Dataset` | `DATASETS` in `datasets/__init__.py` (outside the package: `register_dataset`) | [§3](../architecture/CONTRACTS.md#3-adding-a-dataset-l1) |
| a transform | `signals/transforms.py` or `signals/<modality>.py` | `signals.transforms.Transform` | exported from `signals/__init__.py` | [§4](../architecture/CONTRACTS.md#4-adding-a-transform-l2) |
| a method | `indoorloc/methods/<family>.py` | `methods.base.BaseLocalizer` | `METHODS` in `methods/__init__.py` | [§5](../architecture/CONTRACTS.md#5-adding-a-method-l3) |
| a metric, protocol or bound | `indoorloc/evaluation/...` | functions of arrays | `PROTOCOLS` for named protocols | [§6](../architecture/CONTRACTS.md#6-adding-evaluation-code-l4) |
| an application | `indoorloc/apps/<topic>.py` | `core.Estimator` | exported from `apps/__init__.py` | [§7](../architecture/CONTRACTS.md#7-applications-l5) |

Registries hold `"module:Class"` strings, so listing names imports nothing. Code outside the
package does not need to register at all: `load_dataset("mypkg.data:MyDataset")`,
`create_model("mypkg.models:MyLocalizer")`, `indoorloc benchmark --dataset mypkg.data:MyDataset
--method "mypkg.models:MyLocalizer(alpha=0.1)"` and `--protocol mypkg.splits:MY_PROTOCOL` resolve
a module path directly. To use a short name instead, register it: `register_dataset`,
`register_model` and `register_protocol` (the first two also work as decorators).

## A dataset

A dataset is described by class attributes (files with their sha256, download URLs, meta) and
one `_parse` method that returns a `SampleTable` in physical units, with NaN for missing readings.
The file is parsed by column name, and the source's frame and units are kept.

```python
import csv
import numpy as np
import indoorloc as iloc
from indoorloc.core import SampleTable
from indoorloc.datasets import Dataset, sha256sum

# a small file standing in for a downloaded dataset
with open("scans.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["AP1", "AP2", "AP3", "X", "Y", "FLOOR", "PHONE"])
    w.writerows([[-45, -70, 100, 0.0, 0.0, 0, "A"], [-60, -52, -80, 4.0, 0.0, 0, "B"],
                 [100, -48, -61, 4.0, 3.0, 1, "A"], [-72, 100, -50, 0.0, 3.0, 1, "B"]])


class TinyOffice(Dataset):
    """Tiny office: four WiFi scans on two floors (example for the guide).

    References
    ----------
    Nobody, "An example dataset", Nowhere, 2026. https://example.org
    """

    name = "tiny_office"
    urls = ()                                              # the file is placed by hand here
    files = {"all": ("scans.csv", sha256sum("scans.csv"))}  # normally a literal digest
    meta = {"modality": "wifi_rssi", "units": "dBm", "crs": "local", "pos_names": ("x", "y"),
            "pos_units": "m", "license": "CC0-1.0", "citation": "Nobody, An example dataset, 2026",
            "raw_missing_value": 100}

    def _parse(self, path, split):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        aps = [c for c in rows[0] if c.startswith("AP")]                   # columns by name
        X = np.array([[float(r[a]) for a in aps] for r in rows], dtype=np.float32)
        X[X == 100] = np.nan                                                # sentinel -> NaN
        pos = np.array([[float(r["X"]), float(r["Y"])] for r in rows])
        return SampleTable(X, pos, floor=[int(r["FLOOR"]) for r in rows],
                           groups={"device": np.array([r["PHONE"] for r in rows])},
                           ids=[f"all-{i:05d}" for i in range(len(rows))], meta={"feature_names": tuple(aps)})


table = TinyOffice(root=".").load("all")
print(table.X[0], table.meta["sha256"][:12] == sha256sum("scans.csv")[:12], sorted(table.groups))
# [-45. -70.  nan] True ['device']
```

Outside the package, `register_dataset` gives the class a name for `load_dataset` and
`dataset_info` (like `register_model`, it also works as a decorator, `@register_dataset("tiny_office")`).
Without registering, `load_dataset` takes the class itself or its `"package.module:Class"` path
(`load_dataset("mypkg.data:TinyOffice")`):

```python
iloc.register_dataset("tiny_office", TinyOffice)     # also indoorloc.datasets.register_dataset
table = iloc.load_dataset("tiny_office", root=".")    # split=None: the single "all" table
print(len(table), table.meta["name"], "tiny_office" in iloc.list_datasets(),
      len(iloc.load_dataset(TinyOffice, root=".")))
# 4 tiny_office True 4
```

A dataset without `urls`, like this one, is never downloaded: the error for a missing file says
where to place it by hand instead of suggesting `download=True`.

Inside the package, the class is registered as a string in `datasets/__init__.py`
(`"tiny_office": "indoorloc.datasets.tiny_office:TinyOffice"`), after which `load_dataset`,
`indoorloc info` and `indoorloc benchmark` accept its name, and the catalog in
[datasets.md](datasets.md) lists it once `python docs/catalog.py` has run. Before merging,
record the sha256 of the official download, state the license and the recommended protocol in
the docstring, and add a test in `tests/datasets/` that reads a real file when it is present
(`@pytest.mark.skipif(not path.is_file(), ...)`).

## A transform

`__init__` only stores its arguments, statistics learned in `fit` end with `_`, and
`_transform` works on the last axis of one scan or a batch. The base class handles
`SampleTable`s, scan views and `fit_transform`.

```python
from indoorloc.signals import FillMissing
from indoorloc.signals.transforms import Transform


class ClipRSSI(Transform):
    """Clip readings to the range seen in training (NaN stays NaN)."""

    _requires_fit = True

    def __init__(self, margin_db=0.0):
        self.margin_db = margin_db

    def fit(self, X, y=None):
        super().fit(X)                      # records n_features_in_
        x = X.X if isinstance(X, SampleTable) else np.asarray(X)
        self.lo_, self.hi_ = np.nanmin(x) - self.margin_db, np.nanmax(x) + self.margin_db
        return self

    def _transform(self, x):
        return np.clip(x, self.lo_, self.hi_)


clip = ClipRSSI(margin_db=1.0).fit(table)
print(clip.lo_, clip.hi_, clip.transform(np.array([-99.0, -20.0, np.nan])))
# -81.0 -44.0 [-81. -44.  nan]
```

## A method

A method implements `_fit(X, pos, floor, building)` and `_localize(X) -> Prediction`. The base
class supplies `fit`, `predict`, `localize`, `score`, `evaluate`, `save`, `clone` and parameter
handling, and it validates the input. `_allow_nan = True` declares that `X` may contain missing
readings. A sample that the method cannot place gets a NaN position, which `evaluate` counts in
`n_failed`.

```python
from indoorloc.core import Prediction
from indoorloc.methods import BaseLocalizer, create_model, register_model


class StrongestAPLocalizer(BaseLocalizer):
    """Place a scan at the mean training position of the scans with the same strongest AP.

    A deliberately simple proximity baseline: the position follows from which transmitter is
    heard best, not from a distance model. Readings below ``min_dbm`` count as not heard; a scan
    whose strongest transmitter was never the strongest in training is not placed (NaN).

    References
    ----------
    N. Bulusu, J. Heidemann, D. Estrin, "GPS-less low-cost outdoor localization for very small
    devices", IEEE Personal Communications 7(5), 2000. DOI 10.1109/98.878533. (Proximity
    localization from the transmitters that are heard; the strongest-transmitter rule is a
    simplification made for this example.)
    """

    _allow_nan = True

    def __init__(self, min_dbm=-100.0):
        self.min_dbm = min_dbm

    def _strongest(self, X):
        x = np.where(np.isnan(X) | (X < self.min_dbm), -np.inf, X)   # NaN or too weak: not heard
        return np.argmax(x, axis=1), np.isfinite(x).any(axis=1)     # argmax: first index on ties

    def _fit(self, X, pos, floor, building):
        top, heard = self._strongest(X)
        self.cell_pos_ = np.full((X.shape[1], pos.shape[1]), np.nan)
        for j in np.unique(top[heard]):
            self.cell_pos_[j] = pos[heard & (top == j)].mean(axis=0)

    def _localize(self, X):
        top, heard = self._strongest(X)
        return Prediction(np.where(heard[:, None], self.cell_pos_[top], np.nan))


register_model("strongest_ap", StrongestAPLocalizer)
train, test = iloc.load_dataset("synthetic_office")
print(create_model("strongest_ap").fit(train).evaluate(test))
# mean 4.3641  median 4.2789  P90 6.9086  floor n/a  building n/a  (n=200)
```

This example predicts positions only. A method that ships with IndoorLoc also predicts the
floor and building when the training data has them (for example by a vote of the matched
training scans), and it needs a test against a known result (a closed-form case,
a textbook example or a published toy example), not only a smoke test, and a class docstring
with the method in two or three sentences, its parameters and a `References` section
([CONTRACTS.md §8](../architecture/CONTRACTS.md#8-documentation-and-citations)). The catalog in
[methods.md](methods.md) takes the first line and the first reference from that docstring.

## A protocol

A named protocol turns a table into labelled folds of row indices. It is available to
`get_protocol` and `indoorloc benchmark --protocol` once registered:

```python
from indoorloc.evaluation import Fold, Protocol, get_protocol, leave_one_group_out, register_protocol


def _by_room(table, seed):
    rooms = table.groups["room"]
    return [Fold(f"room={r}", tr, te) for r, (tr, te) in zip(np.unique(rooms), leave_one_group_out(rooms))]


register_protocol("leave-one-room-out", Protocol("leave-one-room-out", "each room tests once", _by_room,
                                                 ("groups['room']",)))
folds = get_protocol("leave-one-room-out").folds(train)
print(len(folds), folds[0].name, len(folds[0].test))
# 21 room=0 40
```

## Published numbers

A published result goes into `indoorloc/evaluation/literature/<dataset>.json` with the method,
the values keyed by metric (each defined at file level with its unit), the source (authors, title,
venue, year, DOI), the table or section it comes from, the protocol, and a `check` status
(`verified` only after comparing with the paper's full text). `indoorloc literature <dataset>`
and `literature.compare` show these numbers next to reproduced ones and never mix the two.

## Before you open a pull request

```bash
python -m pytest -q tests                  # every test file runs in seconds on synthetic data
lint-imports                               # the three import contracts in pyproject.toml
python docs/catalog.py                     # regenerate the catalog tables of this guide
python docs/catalog.py --snippets          # run the guide's code blocks
```
