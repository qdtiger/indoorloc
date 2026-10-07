# IndoorLoc 0.2 user guide

IndoorLoc is a Python library for wireless indoor localization. It loads public datasets,
preprocesses signals, runs localization methods, scores them under named protocols and
builds tracking and navigation on top, in five layers that exchange plain numpy arrays.
Each layer can be used without the others.

[Chinese / 中文](../zh/index.md) · [Installation](../installation.md) ·
[Migrating from 0.1](../../MIGRATION.md) · [Changelog](../../CHANGELOG.md) ·
[Benchmarks](../benchmarks.md) · [Developer contracts](../architecture/CONTRACTS.md)

## The five layers

| Layer | Package | Takes | Returns | Page |
| --- | --- | --- | --- | --- |
| L1 data | `indoorloc.datasets` | a dataset name and options | `SampleTable` | [datasets](datasets.md) |
| L2 signals | `indoorloc.signals` | a `SampleTable`, an array or one scan | the same kind, transformed | [signals](signals.md) |
| L3 methods | `indoorloc.methods` | `X` (and `y` to fit) | `Prediction` | [methods](methods.md) |
| L4 evaluation | `indoorloc.evaluation` | truth and `Prediction` (or arrays) | `EvaluationResults`, folds, bounds | [evaluation](evaluation.md) |
| L5 applications | `indoorloc.apps` | fixes, IMU samples, scan streams, floor plans | `Prediction`, routes | [apps](apps.md) |
| command line | `indoorloc.cli` | dataset and method specs | a self-describing JSON file | [cli](cli.md) |

The import rules are enforced by import-linter contracts. `core` depends only on numpy and the
standard library. L1, L2 and L4 are independent of each other. L3 sits above them, and L5 on top.
No layer imports L5, and L3 and L5 never import L1: data arrives as arrays or a `SampleTable`.
`import indoorloc` itself loads no numpy, and every layer imports without torch, scikit-learn,
scipy or pandas. Heavier packages are opt-in extras, imported inside the function that needs
them. [Extending IndoorLoc](extending.md) explains how to add a dataset, a transform or a method.

## How data flows

```text
L1  load_dataset("ujiindoorloc")          -> SampleTable(X, pos, floor, building, groups, ids, meta)
L2  FillMissing(-104).fit_transform(t)    -> SampleTable (same rows, new X)
L3  create_model("wknn").fit(t_train)     -> fitted estimator
    model.localize(t_test)                -> Prediction(pos, floor, building, ids, spread)
L4  evaluate(t_test, prediction)          -> EvaluationResults(mean_error, ..., n_failed)
L5  KalmanTracker().smooth(prediction, t) -> Prediction (scored by L4 like any method)
```

The two data types live in `indoorloc.core`. Both are frozen and hold read-only numpy arrays,
with samples on the first axis:

* `SampleTable(X, pos, floor=None, building=None, groups={}, ids=None, meta={})`. `X` holds
  observations in physical units (dBm, complex CSI, metres, radians), and **a missing reading is
  NaN**, never a sentinel value. `pos` is float64 `(N, D)` in the frame named by `meta["crs"]`.
  `floor`/`building` are int64 or `None`. `groups` holds columns for grouped splits (`user`,
  `device`, `time`, `trajectory`, ...). `meta` holds dataset facts (units, license, sha256, ...).
* `Prediction(pos, floor=None, building=None, ids=None, spread=None)`. `spread` is a method's own
  positional uncertainty scale, which the L5 trackers use as measurement noise.

The full rules, including the `X` layout of each modality, are in
[CONTRACTS.md](../architecture/CONTRACTS.md).

## Install

Until 0.2.0 is published on PyPI, install from a checkout of the repository:

```bash
pip install -e .                  # numpy only: every layer, k-NN, model-based methods, trackers
pip install -e ".[sklearn]"       # + SVM, random forest, extra trees, gradient boosting
pip install -e ".[deep]"          # + MLP / CNN1D / timm-backbone localizers (torch, timm)
pip install -e ".[full]"          # everything above plus pandas, matplotlib, scipy, h5py
```

The [installation page](../installation.md) lists every extra and what needs it.

## Five-minute tour

The tour starts on the built-in simulated office, which needs no download, and then repeats
the same steps on a real dataset.

**1. Load a dataset (L1).** Without `split`, `load_dataset` returns the official `(train, test)`
pair; a dataset with a single table returns that table.

```python
import indoorloc as iloc

train, test = iloc.load_dataset("synthetic_office")   # simulated, generated from seed 0
print(train.X.shape, train.meta["modality"], train.meta["units"])
# (830, 8) wifi_rssi dBm
print(sorted(train.groups), train.meta["crs"])
# ['point', 'room', 'source'] local
```

**2. Choose preprocessing and a method (L2 + L3).** A missing reading is NaN, and k-NN needs
complete vectors, so `FillMissing(-104)` replaces NaN with -104 dBm. Passing it as `preprocess=`
makes it part of the model, so it is fitted on the training data only and applied to every
later input.

```python
model = iloc.create_model("wknn", k=5, preprocess=iloc.FillMissing(-104))
model.fit(train)                      # a SampleTable supplies X, pos, floor and building
pred = model.localize(test)           # Prediction: pos (200, 2), floor, spread
print(pred.pos.shape, pred.floor[:3], pred.spread[:2].round(2))
# (200, 2) [0 0 0] [0.99 1.79]
```

**3. Score it (L4).**

```python
res = iloc.evaluate(test, pred)       # the same as model.evaluate(test)
print(res)
# mean 1.8055  median 1.3816  P90 3.4247  floor 100.00 %  building n/a  (n=200)
```

**4. Track a walk (L5).** The simulator also generates walks that are scanned once per second.
A Kalman smoother over the WKNN fixes uses their `spread` as measurement noise:

```python
walks = iloc.load_dataset("synthetic_office", split="trajectory")
walk = walks[walks.groups["trajectory"] == 0]
fixes = model.localize(walk)
smoothed = iloc.KalmanTracker().smooth(fixes, t=walk.groups["time"])
print(round(iloc.evaluate(walk, fixes).mean_error, 2), round(iloc.evaluate(walk, smoothed).mean_error, 2))
# 2.97 2.11
```

These numbers describe a simulated office; they say nothing about a real building.

**5. The same on real data.** UJIIndoorLoc is downloaded on first use (the UCI zip archive;
its two CSV files are each sha256-checked):

```python
# data: ujiindoorloc
train, test = iloc.load_dataset("ujiindoorloc")
model = iloc.create_model("wknn", k=5, preprocess=iloc.FillMissing(-104)).fit(train)
print(model.evaluate(test))
# mean 8.7937  median 5.3546  P90 19.1537  floor 90.46 %  building 99.73 %  (n=1111)
```

UJIIndoorLoc positions are Web Mercator (EPSG:3857) metres, the unit in which results on this
dataset are usually published. `model.evaluate(test, scale=test.meta["ground_scale"])` reports
ground metres instead (scale 0.7661). The value 8.7937 matches the benchmark tables in
[docs/benchmarks.md](../benchmarks.md), where every cell was run through the command line:

```bash
indoorloc benchmark --dataset ujiindoorloc --method knn --method wknn --out uji.json
indoorloc report uji.json --format markdown
```

## One layer at a time

Every layer takes plain arrays, so it works with data and code from elsewhere.

```python
import numpy as np
from indoorloc.evaluation import evaluate
from indoorloc.methods import create_model

rng = np.random.default_rng(0)
X = rng.normal(-70, 8, size=(300, 6))          # your own RSSI matrix, dBm
y = rng.uniform(0, 20, size=(300, 2))          # your own positions, metres
model = create_model("knn", k=3).fit(X[:250], y[:250])     # L3 on arrays
print(evaluate(y[250:], model.predict(X[250:])).n)         # L4 on arrays
# 50
```

A `SampleTable` also exports to numpy (`table.to_numpy()`), to pandas (`table.to_dataframe()`)
and to torch (`table.to_torch()`, or `indoorloc.datasets.torch_adapter.make_dataloader`).
The [datasets page](datasets.md#using-the-data-elsewhere) shows each of these.

## Where next

* [Datasets](datasets.md): the catalog, options, coordinate frames, exports and plots.
* [Signals](signals.md): transforms and signal functions for RSSI, CSI, ranging, IMU, magnetic and VLC.
* [Methods](methods.md): the method catalog, the estimator contract, saving and loading.
* [Evaluation](evaluation.md): metrics, protocols, competition scores, bounds and published numbers.
* [Applications](apps.md): tracking, PDR, particle filters, streaming and navigation.
* [Command line](cli.md): `indoorloc list | info | benchmark | literature | evaluate | report`.
* [Extending](extending.md): add a dataset, a transform, a method or a protocol.

Complete scripts in [`examples/`](../../examples) run each layer end to end; each file's docstring
says what it measures and which data it downloads:

| Script | What it shows | Data |
| --- | --- | --- |
| [`quickstart.py`](../../examples/quickstart.py) | load, fit a pipeline, evaluate, save and reload | UJIIndoorLoc (`--dataset synthetic_office`: no download) |
| [`01_fingerprinting_benchmark.py`](../../examples/01_fingerprinting_benchmark.py) | seven classic fingerprinting methods and an MLP under one protocol, error CDFs | UJIIndoorLoc |
| [`02_model_based_vs_crlb.py`](../../examples/02_model_based_vs_crlb.py) | model-based estimators against the Cramér-Rao bound (Monte Carlo) | simulated |
| [`03_csi_pipeline.py`](../../examples/03_csi_pipeline.py) | raw and sanitized CSI phase, CSI fingerprints with WKNN | HALOC |
| [`04_tracking_and_fusion.py`](../../examples/04_tracking_and_fusion.py) | WiFi fixes, Kalman smoothing, PDR and particle-filter fusion on phone traces | ILC 2020 sample (site1/F1) |
| [`05_domain_adaptation.py`](../../examples/05_domain_adaptation.py) | cross-device fingerprinting with and without domain adaptation | UJIIndoorLoc |
| [`06_navigation.py`](../../examples/06_navigation.py) | A* routes over three storeys, stairs versus elevator, instructions | simulated floor plan |
| [`dataset_distribution.py`](../../examples/dataset_distribution.py) | sample distributions, density maps, error CDF and error map | simulated or any dataset |

The code blocks in this guide are executed by `python docs/catalog.py --snippets`. Blocks that
begin with `# data: <name>` run only when that dataset is already on disk, and the printed
values in the comments come from those runs. The catalog tables are regenerated from the
registries by `python docs/catalog.py`.
