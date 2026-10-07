# Migrating from IndoorLoc 0.1 to 0.2

IndoorLoc 0.2 rebuilds the library as five layers that exchange numpy arrays (see the
[user guide](docs/guide/index.md)). Most 0.1 code needs changes. This page lists what was
renamed, what was removed, which 0.1 names still work for the 0.2.x releases, and which results
change and why.

## At a glance

| 0.1 | 0.2 |
| --- | --- |
| `import indoorloc` loaded torch, pandas and scikit-learn | `import indoorloc` loads no numpy; each layer needs numpy only |
| `load_dataset("ujindoorloc")` → normalized dataset objects in [0, 1] | `load_dataset("ujiindoorloc")` → `SampleTable`s in dBm, NaN for missing readings |
| a list of `WiFiSignal` objects and a list of `Location` objects | `X` `(N, F)` and `pos` `(N, D)` arrays, or one `SampleTable` |
| `model.fit(signals, locations)`, `model.predict(signal)` → `LocalizationResult` | `model.fit(X, y, floor=...)` or `model.fit(table)`; `predict(X)` → `(N, D)` array, `localize(X)` → `Prediction` |
| `model.predict_batch(signals)` | `model.predict(X)` / `model.localize(X)` |
| `EvaluationResults.from_predictions(preds, truths)` | `evaluate(y_true, y_pred, floor_true=..., floor_pred=...)` or `model.evaluate(test)` |
| YAML configs with `_base_` inheritance, `indoorloc-train`/`-test`/`-benchmark` | Python parameters; `indoorloc benchmark`, `indoorloc evaluate`, `indoorloc report` |
| `@LOCALIZERS.register_module()` | `register_model("name", Cls)` |
| `model.save(path)` with joblib (pickle) | `model.save(path)` → `config.json` + `arrays.npz`; `iloc.load_model(path)` |
| `iloc.list_datasets()` → class names | `iloc.list_datasets()` → registry names (`"ujiindoorloc"`, ...) |

## Installation

0.2 requires Python 3.10 or later, and its only required dependency is numpy (0.1 required torch,
pandas, scikit-learn, pyyaml and tqdm). Everything else is an extra: `[sklearn]`, `[deep]`,
`[pandas]`, `[torch]`, `[plot]`, `[datasets]`, `[transfer]`, `[sim]`, `[full]` and `[dev]`
(see [docs/installation.md](docs/installation.md)). The 0.1 extras `vision`, `deepmimo` and
`docs` were removed; DeepMIMO is now `[sim]`.

## 0.1 names that still work in 0.2.x

These names emit a `FutureWarning` where their behaviour changed, and they will be removed in
0.3. They live in `indoorloc/_legacy.py`, which no library layer imports.

| 0.1 code | What happens in 0.2 | Replace with |
| --- | --- | --- |
| `iloc.WiFiSignal(rssi_values=row)` | works; the 0.1 value 100 becomes NaN. Warning: `WiFiSignal(rssi_values=...) is deprecated and will be removed in 0.3; use WiFiSignal.from_raw(row, missing=100)` | `WiFiSignal.from_raw(row, missing=100)` or `WiFiSignal(rssi=array_in_dBm)` |
| `iloc.Location`, `iloc.Coordinate` | the 0.1 value types, with a warning on every access | position arrays and `Prediction` |
| `iloc.LocalizationResult` | the 0.1 result type, with a warning | `Prediction` (`pos`, `floor`, `building`, `ids`, `spread`) |
| `iloc.list_available_datasets()` | returns the registry names, with a warning | `iloc.list_datasets()` |
| `load_dataset("ujindoorloc")` | the 0.1 object: features min-max scaled to [0, 1], `ds[i] -> (WiFiSignal, Location)`, `to_tensors()`; with a warning | `load_dataset("ujiindoorloc")` and `create_model(..., preprocess=FillMissing(-104))` |
| `model.fit(signals, list_of_Location)` | accepted: positions, floors and buildings are read from the `Location`s | `model.fit(X, y, floor=..., building=...)` |
| `model.predict(wifi_signal)` | returns a `LocalizationResult` for a 0.1 `WiFiSignal` | `model.localize(x[None, :])` |
| `EvaluationResults.from_predictions(preds, truths)` | available once a 0.1 name has been used (it is added by `_legacy`), with a warning | `evaluate(...)` |
| `indoorloc.version.__version__` | works | `indoorloc.__version__` |

The 0.1 README flow therefore still runs, with warnings:

```python
# data: ujiindoorloc
import indoorloc as iloc

train, test = iloc.load_dataset("ujindoorloc")        # FutureWarning: 0.1 normalized objects
model = iloc.create_model("wknn", k=5)
print(model.fit(train).evaluate(test))                 # a 0.2 EvaluationResults
# mean 8.7907  median 5.3364  P90 19.1537  floor 90.46 %  building 99.73 %  (n=1111)
```

It computes on the 0.1 min-max features with the 0.2 neighbour search, so it gives the
float32-features row of the [table below](#why-the-ujiindoorloc-numbers-changed), not the value
recorded with 0.1.

The 0.2 form of the same experiment:

```python
# data: ujiindoorloc
import indoorloc as iloc

train, test = iloc.load_dataset("ujiindoorloc")       # dBm, NaN = not heard, EPSG:3857 metres
model = iloc.create_model("wknn", k=5, preprocess=iloc.FillMissing(-104))
print(model.fit(train).evaluate(test))
# mean 8.7937  median 5.3546  P90 19.1537  floor 90.46 %  building 99.73 %  (n=1111)
```

## Removed without a replacement shim

| 0.1 | 0.2 |
| --- | --- |
| `indoorloc/configs/` (YAML files), `Config`, `load_config`, `merge_configs`, `get_default_config`, `print_config_help`, `explain_model`, `explain_dataset`, `explain_config` | parameters are ordinary Python arguments. Every benchmark result file records the fully resolved parameters, and `get_params()` returns them |
| `build_model(cfg)`, `create_model(config=...)`, `create_model("auto", dataset=train)` | `create_model(name, **params)` or `create_model("package.module:Class", ...)` |
| `create_model("resnet18", dataset=train)` (a timm name as the model name) | `create_model("deep", backbone="resnet18")` (needs timm); `"mlp"` and `"cnn1d"` are registered names |
| `indoorloc-train`, `indoorloc-test`, `indoorloc-benchmark` scripts | `indoorloc benchmark`, `indoorloc evaluate`, `indoorloc report`, `indoorloc literature`, `indoorloc info`, `indoorloc list` |
| `indoorloc.registry` and its `SIGNALS`, `DATASETS`, `TRANSFORMS`, `LOCALIZERS`, `FUSIONS`, `METRICS`, `BACKBONES`, `HEADS`, `TRAINERS`, `VISUALIZERS` | `indoorloc.datasets.DATASETS`, `indoorloc.methods.METHODS`, `indoorloc.evaluation.PROTOCOLS` (lazy `"module:Class"` registries); `register_model`, `register_protocol` |
| signal classes `BaseSignal`, `SignalMetadata`, `APInfo`, `BLEBeacon`, `IMUSignal`, `IMUReading` and the `uwb`, `ultrasound`, `magnetometer`, `hybrid` signal modules | modality layouts of `SampleTable.X` ([CONTRACTS.md §2](docs/architecture/CONTRACTS.md)); functions in `signals.ranging`, `signals.imu`, `signals.magnetic`, `signals.vlc`; `WiFiSignal`/`BLESignal` remain as one-scan views |
| `BaseDataset`, `WiFiDataset`, `BLEDataset`, `UWBDataset`, `HybridDataset`, `MagneticDataset` and the per-dataset classes (`UJIndoorLocDataset`, `TampereDataset`, ...) | `indoorloc.datasets.Dataset` and registry names (`load_dataset("tampere")`) |
| `ds.to_tensors()`, `ds.to_torch_tensors()` | `table.to_numpy()`, `table.to_torch()`, `datasets.torch_adapter.make_dataloader(table)` |
| `get_data_home()` | `indoorloc.datasets.default_root()` (`$INDOORLOC_DATA`, default `~/.cache/indoorloc/datasets`, as in 0.1) |
| `TraditionalLocalizer`, `TransferLocalizer(method="coral"/"tca"/...)` | `BaseLocalizer`; `LocalizerPipeline(CORAL(), create_model("knn"))` fitted with `preprocess__target=X_target`, likewise `TCA` and `SkadaAdapter` |
| `DeepLocalizer(backbone={...}, head={...})`, `BaseBackbone`, `InputAdapter`, `RegressionHead`, `HybridHead`, ... | `DeepLocalizer(backbone="mlp" / "cnn1d" / timm name, hidden=..., epochs=...)`, `MLPLocalizer`, `CNN1DLocalizer`; building blocks in `indoorloc.methods.deep` (`MLPBackbone`, `CNN1DBackbone`, `TimmBackbone`, `MultiTaskHead`, `LocalizationNet`, `train_model`) |
| `model.predict_batch(signals)`, `model.load(path)` | `model.predict(X)`, `iloc.load_model(path)` |
| `EvaluationResults.compare_benchmarks()` and `evaluation.benchmark_data` | `indoorloc.evaluation.literature` (`table`, `compare`), with sources and check status; `indoorloc literature <dataset>` |
| `indoorloc.visualization` | `indoorloc.datasets.plot` (`plot_distribution`, `plot_floor_plan`, `distribution_html`) and `indoorloc.evaluation.plot` |
| `iloc.APFilter(threshold=-90)`, `iloc.RSSINormalize(method="minmax")` | `APFilter(threshold_dbm=-90)`; `Compose([FillMissing(-104), RSSINormalize()])` (default range -104 to 0 dBm), or `PositiveRepresentation`, `ExponentialRepresentation`, `PowedRepresentation` |
| `__version_info__` | `indoorloc.__version__` |

Model files written by 0.1 (joblib) cannot be loaded by 0.2. Fit the model again with 0.2 (the
k-NN fit on UJIIndoorLoc takes well under a second) and save it with `model.save(path)`.

## Behaviour changes

1. **Missing readings are NaN.** 0.1 kept the file's "not detected" value (100 in UJIIndoorLoc)
   inside `WiFiSignal` and replaced it during normalization. In 0.2 every loader converts its
   file's sentinel to NaN and records it in `meta["raw_missing_value"]`. A method that cannot
   use NaN raises an error that names the fix (`preprocess=FillMissing(-104)`), where 0.1 would
   have computed distances on the sentinel.
2. **Datasets are not normalized.** 0.1 `load_dataset` min-max scaled RSSI to [0, 1] by default
   (`normalize=True`, missing = -104 dBm, range [-104, 0]). 0.2 returns physical units, and
   preprocessing is chosen explicitly and fitted on training data only, usually as
   `create_model(..., preprocess=...)`.
3. **Coordinates are float64** everywhere (0.1 used float32 labels in `to_tensors` and in its
   scikit-learn models; float32 is 0.5 m coarse at UJIIndoorLoc's 4.8e6 m northings). The
   coordinates stay in the dataset's own frame, stated in `meta["crs"]`: UJIIndoorLoc is Web
   Mercator (EPSG:3857) metres, and `meta["ground_scale"]` converts them to ground metres.
4. **k-NN ties are broken by training index.** Neighbours are ranked by distances summed in a
   fixed order (exact integers in squared form for whole-dBm readings), and equal distances keep
   training-index order, so a result does not depend on BLAS
   threads, batch size or chunking.
5. **`predict` returns an array.** `predict(X)` returns `(N, D)` positions (scikit-learn style),
   and `localize(X)` returns a `Prediction` with floor, building, ids and `spread`.
6. **Accuracies of missing labels are `None`.** In 0.1 two missing floor labels compared as equal,
   so data without floors scored 100 % floor accuracy. 0.2 returns `None`.
7. **Errors use every coordinate axis.** 0.1 computed 2-D distances. 0.2 uses all axes of `pos`,
   so Tampere errors are 3-D, as in the dataset's own benchmark software.
8. **Samples a method cannot place** (NaN prediction) are counted in `n_failed` and left out of
   the error statistics; always report `n_failed` with them.
9. **`load_dataset(name)`** returns `(train, test)` when the dataset has both splits and a single
   table otherwise (0.1 returned a `(train, test)` pair whenever `split` was omitted). Dataset options are keyword arguments
   (`load_dataset("sodindoorloc", building="HCXY")`).
10. **Published numbers are kept separate.** The 0.1 comparison tables listed published numbers
    together with entries that have no traceable source, values that differ from the papers, and
    numbers IndoorLoc 0.1 produced itself. In 0.2 every entry records its source, where in the
    paper it was checked, the protocol and a check status (`indoorloc literature <dataset> --all`
    shows them all), and published numbers never share a table with reproduced results.

## Why the UJIIndoorLoc numbers changed

A run of the 0.1 README flow (k-NN and WKNN, k = 5, official split, 1,111 validation scans),
recorded with IndoorLoc 0.1 on a machine with 2 CPU threads for the 0.1 README figure, gave k-NN
8.8891 m and WKNN 8.8592 m. That record is
[`examples/readme_case/v0.1_results.json`](examples/readme_case/v0.1_results.json) (mean errors
8.889091885956587 and 8.859248088918953 EPSG:3857 m, `cpu_threads: 2`), kept next to the 0.2 run
of the same case in `examples/readme_case/results.json`; the first row of the table below
reproduces it. 0.2 gives 8.8084 m and 8.7937 m. The data and the evaluation are the same. The
difference comes from which training scans become the five nearest neighbours when several of
them are at exactly the same distance. With RSSI in whole dBm, all distances between scans are
exact integers in squared form, and 19 of the 1,111 validation scans have a tie between their
5th and 6th nearest training scans.

All rows below were measured in one run on the sha256-verified files (Python 3.14.4, numpy 2.4.5,
scikit-learn 1.8.0, OpenBLAS 0.3.31), mean errors in EPSG:3857 metres:

| Setting | k-NN | WKNN |
| --- | ---: | ---: |
| 0.1 recipe: scikit-learn brute-force search, features `(x + 104) / 104` in float32, float32 positions, 2 BLAS threads (the setting of the recorded 0.1 run) | 8.8891 | 8.8592 |
| the same recipe with 1 / 4 / 8 BLAS threads | 8.8665 / 8.8470 / 8.8567 | 8.8376 / 8.8184 / 8.8289 |
| scikit-learn brute-force search on float64 dBm (fill -104), 1 / 2 / 4 / 8 threads | 8.8467 / 8.8725 / 8.8269 / 8.8314 | 8.8311 / 8.8559 / 8.8119 / 8.8166 |
| 0.2 search (fixed-order distances, ties by training index) on the 0.1 float32 features | 8.8055 | 8.7907 |
| **0.2 default**: `FillMissing(-104)` on dBm, 0.2 search | **8.8084** | **8.7937** |

Three things follow from these runs:

* The 0.1 values are reproduced exactly with two BLAS threads (the two means of
  `v0.1_results.json` bit for bit), but the same code gives different numbers with 1, 4 or 8
  threads. scikit-learn's brute-force search does not define the order of equal distances, so the
  choice among tied neighbours followed the thread count.
* With the 0.2 tie rule the result no longer depends on threads. It still differs slightly between
  float32 scaled features (8.8055 m) and integer dBm (8.8084 m), because float32 rounding of
  `(x + 104) / 104` decides some ties that are exact in dBm. 0.2 computes on dBm, where equal
  readings give equal distances.
* The WKNN floor accuracy changed from 90.37 % to 90.46 %, and the k-NN floor accuracy stayed at
  90.28 %.

The 0.2 values are pinned by a real-data test (`tests/cli/test_cli_main.py`) and agree with an
independent plain-numpy recomputation in the benchmark cross-checks
([docs/benchmarks.md](docs/benchmarks.md#cross-checks-against-known-values)).

<details>
<summary>Reproduce the table</summary>

The scikit-learn rows depend on the BLAS library and the machine, which is the point of the table,
so another machine can print other values for them. The 0.2 rows are the same everywhere.

```python
# data: ujiindoorloc
# requires: sklearn, threadpoolctl
import numpy as np
import indoorloc as iloc
from sklearn.neighbors import KNeighborsRegressor
from threadpoolctl import threadpool_limits

train, test = iloc.load_dataset("ujiindoorloc")
fill = iloc.FillMissing(-104)
dbm_train, dbm_test = (fill.transform(t.X).astype(np.float64) for t in (train, test))
old_train, old_test = (((x + 104) / 104).astype(np.float32) for x in (dbm_train, dbm_test))   # 0.1 features


def mean_error(pos):
    return round(float(np.linalg.norm(pos - test.pos, axis=1).mean()), 4)


for name, weights in (("knn", "uniform"), ("wknn", "distance")):
    for threads in (1, 2, 4, 8):
        with threadpool_limits(limits=threads):
            old = KNeighborsRegressor(n_neighbors=5, weights=weights, algorithm="brute")
            old.fit(old_train, train.pos.astype(np.float32))                  # 0.1: float32 positions
            print(name, f"0.1 recipe, {threads} threads:", mean_error(old.predict(old_test)))
            on_dbm = KNeighborsRegressor(n_neighbors=5, weights=weights, algorithm="brute").fit(dbm_train, train.pos)
            print(name, f"scikit-learn on float64 dBm, {threads} threads:", mean_error(on_dbm.predict(dbm_test)))
    on_old = iloc.create_model(name, k=5).fit(old_train, train.pos)
    default = iloc.create_model(name, k=5).fit(dbm_train, train.pos)
    print(name, "0.2 on the 0.1 features:", mean_error(on_old.predict(old_test)),
          "0.2 default:", mean_error(default.predict(dbm_test)))
```

</details>

## A port, step by step

0.1:

```python
# 0.1 code
signals = [iloc.WiFiSignal(rssi_values=row) for row in X_own]
locations = [iloc.Location(coordinate=iloc.Coordinate(x, y), floor=f) for (x, y), f in zip(xy_own, floors)]
model = iloc.create_model("wknn", k=5).fit(signals, locations)
result = model.predict(signals[0])                  # LocalizationResult
print(result.x, result.y, result.floor)
```

0.2 (`X_own` in dBm, with NaN where an access point was not heard):

```python
import numpy as np
import indoorloc as iloc

rng = np.random.default_rng(0)
X_own = rng.integers(-95, -40, size=(50, 6)).astype(np.float32)
X_own[rng.random(X_own.shape) < 0.2] = np.nan                        # not heard
xy_own, floors = rng.uniform(0, 30, size=(50, 2)), rng.integers(0, 2, size=50)

model = iloc.create_model("wknn", k=5, preprocess=iloc.FillMissing(-104))
model.fit(X_own, xy_own, floor=floors)
pred = model.localize(X_own[:1])                                     # Prediction for one scan
print(pred.pos.shape, pred.floor.shape, model.predict(X_own).shape)
# (1, 2) (1,) (50, 2)
```

If your 0.1 arrays still hold the value 100 for "not heard", convert them first with
`np.where(X == 100, np.nan, X)`, or use `FillMissing(value=-104, missing=100)`.
