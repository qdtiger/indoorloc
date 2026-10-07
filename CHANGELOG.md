# Changelog

All notable changes to IndoorLoc. Migration notes for 0.1 users are in [MIGRATION.md](MIGRATION.md).

## 0.2.0 (unreleased)

A rebuild of the library as five layers that exchange numpy arrays: L1 datasets, L2 signals,
L3 methods, L4 evaluation and L5 applications, over a small `core`. Each layer works on its own.
The import rules (L1, L2 and L4 independent; L3 above them; L5 on top; no layer imports L5; L3
and L5 never import L1; numpy the only third-party import at module level) are three
import-linter contracts that CI checks. The [user guide](docs/guide/index.md) is new, in English
and [Chinese](docs/zh/index.md).

### Core

- `SampleTable(X, pos, floor, building, groups, ids, meta)` and `Prediction(pos, floor, building,
  ids, spread)`: frozen containers of read-only numpy arrays, with samples on the first axis.
  Missing readings are NaN, positions are float64 in a stated frame (`meta["crs"]`), and
  `groups` holds the columns for grouped splits.
- `Estimator` with scikit-learn semantics (`get_params`, `set_params`, `clone`), and saving
  without pickle: `save(path)` writes `config.json` + `arrays.npz`, `load_model(path)` checks
  every class path against an allow-list before importing it. A `FloorMap` inside a particle
  filter is stored as arrays.
- `Registry`: lazy `"module:Class"` registries, so listing datasets or methods imports nothing.
  `import indoorloc` loads no numpy.

### L1 datasets

- 15 registry entries. Measured: UJIIndoorLoc, SODIndoorLoc, Tampere, TUJI1, Long-Term WiFi,
  UCI Wireless Indoor Localization, BBIL (`ble_indoor`), UCI BLE RSSI, the UJI iBeacon RSS
  database, the CSI fingerprint rooms of Zhu et al., H-WILD, HALOC and the ILC 2020 sample
  traces (new: WiFi, BLE, IMU and waypoints on 14 floors with GeoJSON floor plans). Simulated:
  `synthetic_office` (new; WiFi/BLE RSSI, UWB ranges, TDoA, AoA, CSI, IMU, VLC and magnetic data
  from one seeded office model) and `deepmimo` (DeepMIMO v4 scenarios).
- Every file of a measured dataset has a sha256 in its loader, and each table records its digests. Loaders parse
  columns by name, keep the source's coordinate frame, and convert each file's sentinel to NaN.
- `load_dataset(name, split=None, **options)` returns the official `(train, test)` pair or the
  single table. Dataset options are keyword arguments. `dataset_info` describes a dataset without
  loading it.
- Exports: `to_numpy`, `to_dataframe` (pandas), `to_torch` and `torch_adapter.make_dataloader`.
  Plots: `datasets.plot.plot_distribution`, `plot_floor_plan`, `distribution_html` (plotly).

### L2 signals

- Transforms with the scikit-learn transformer contract that accept a scan, a batch, a table or a
  scan view: `FillMissing`, `RSSINormalize`, `APFilter`, `APSelect`, the positive, exponential and
  powed representations of Torres-Sospedra et al. (2015), `HampelFilter`, `DeviceCalibration`,
  the augmentations `GaussianNoise` and `APDropout` (seeded), `CSIAmplitude`, `CSIPhaseSanitize`,
  `SubcarrierSelect`, `MagneticFeatures`, `MagnetometerCalibration` and `Compose`.
- Signal functions for RSSI, CSI (amplitude, phase sanitization, conjugate multiplication, CSI
  ratio), ranging (ToA, RTT, single- and double-sided two-way ranging, ultrasound, path-loss
  ranging, range bias, NLOS flags), IMU, the magnetometer (tilt-compensated field components and
  heading, ellipsoid calibration) and visible light (Lambertian channel, power-distance conversion,
  receiver noise).

### L3 methods

- 21 registered methods behind one contract: `fit(X, y, floor=, building=)` or `fit(table)`,
  `localize(X) -> Prediction`, `predict(X) -> array`, `evaluate`, `score`, `save`/`load_model`.
  All are scikit-learn estimators, and `create_model(name, preprocess=...)` builds a
  `LocalizerPipeline`.
  - Fingerprinting: k-NN, WKNN, Horus, a Gaussian-process radio map, SVM, random forest, extra
    trees, gradient boosting, ensembles, stacking and a hierarchical building/floor/position model.
  - Model-based: trilateration (robust Gauss-Newton), TDoA (Chan-Ho with refinement), AoA (MUSIC and
    bearing fusion), log-distance path loss, weighted centroid and visible-light positioning.
  - Sequence matching: magnetic subsequence DTW.
  - Deep learning: MLP, CNN1D and timm backbones with multi-task heads (torch).
- Domain adaptation: `CORAL`, `TCA` (numpy) and `SkadaAdapter`, as transforms fitted on
  unlabelled target scans.
- Deterministic k-NN: candidate distances re-scored in a fixed summation order and equal distances
  broken by training index, independent of BLAS threads and batching. `Prediction.spread` reports each method's own uncertainty scale.

### L4 evaluation

- `evaluate` on arrays or on a table and a `Prediction`. Samples a method could not place count
  in `n_failed` and are left out of the error statistics.
- Protocols as index arrays computed from `groups`: `random_split` (stratified), `kfold` (grouped),
  `group_split`, `leave_one_group_out`, `cross_device_split`, `cross_time_split`, and named
  protocols for the official split, random and k-fold splits, and splits grouped by device, time,
  building, user, reference point, trajectory and month (`indoorloc list protocols`).
  `split_summary` records the sha256 of every index array.
- IPIN and EvAAL scores, CEP, percentiles, success rates and bootstrap confidence intervals.
  Cramér-Rao bounds for ToA, RSS, AoA and TDoA, and dilution of precision.
- Published numbers with provenance: each entry has a source, a location in the paper, a protocol
  and a check status, and entries are always reported apart from reproduced results.
- Reports (Markdown and text) and plots: error CDF, error map, trajectories and bound maps.

### L5 applications (new)

- `KalmanTracker` (constant velocity or acceleration, noise adapted from `Prediction.spread`,
  outlier gate, RTS smoother), `ExtendedKalmanTracker` on raw ranges, and `ParticleFilter` with
  wall constraints and augmented-MCL recovery.
- `StepDetector`, `PDR` (Weinberg and Kim step length; gyro, compass or fused heading) and
  `PDRFusion` of PDR steps with fixes from any localizer on a `FloorMap`.
- `OnlineLocalizer` for streams of `(t, scan)` with latency statistics, and `Navigator` (A* across
  floors, line-of-sight smoothing, turn-by-turn instructions).

### Command line (new)

- `indoorloc list | info | benchmark | literature | evaluate | report`. `benchmark` writes a
  self-describing JSON file with the per-fold and pooled metrics, the resolved parameters, the
  digests of the data and of every split, the environment and the source digest.

### Benchmarks (new)

- `benchmarks/`: a matrix of 25 result tables on 12 real datasets, run cell by cell through the
  command line and rendered as [docs/benchmarks.md](docs/benchmarks.md) and
  [docs/benchmarks_zh.md](docs/benchmarks_zh.md). Of the 349 cells, 347 finished and 2 stopped at
  the 2,500 MB memory limit. 124 cells were run twice in separate processes, and 82 were run again
  with a later version of the code; every metric was identical. On every RSSI table Horus runs both
  after the -104 dBm fill and on the raw readings (NaN = not heard), the input its detection model
  is made for; on the raw readings it has the lowest mean error of four tables (SODIndoorLoc: all
  buildings, CETC331, HCXY; TUJI1). Five reference values (UJIIndoorLoc
  k-NN 8.8084 and WKNN 8.7937, the TUJI1 paper's 1-NN 3.34, HALOC WKNN 3.6685, BBIL office WKNN
  3.5189) agree with an independent plain-numpy recomputation.

### Changed

- Missing readings are NaN in every layer. Loaders no longer normalize, and preprocessing is
  explicit and fitted on training data only.
- Coordinates are float64 in the dataset's own frame. UJIIndoorLoc results are in EPSG:3857
  metres, with `meta["ground_scale"]` for ground metres.
- `predict` returns arrays and `localize` returns a `Prediction`. Errors use every coordinate axis.
  Label accuracies are `None` without labels.
- UJIIndoorLoc k-NN / WKNN (k = 5, official split): 8.8084 / 8.7937 m, where the run recorded
  with 0.1 for the 0.1 README figure had 8.8891 / 8.8592 m (the 0.1 record is
  `examples/readme_case/v0.1_results.json`, next to the 0.2 run of the same case in
  `examples/readme_case/results.json`). [MIGRATION.md](MIGRATION.md#why-the-ujiindoorloc-numbers-changed) gives the
  measured decomposition: the 0.1 values depended on the number of BLAS threads through
  scikit-learn's order of equal distances.
- Only numpy is required, and everything else is an extra. Python 3.10 or later.

### Removed

- The YAML configuration system (`indoorloc/configs`, `Config`, `load_config`, ...), the
  `indoorloc-train`/`-test`/`-benchmark` scripts, `build_model`, the 0.1 registries, the signal
  object classes, `TransferLocalizer`, the backbone/head dictionaries of `DeepLocalizer`,
  `predict_batch`, joblib model files, `indoorloc.visualization` and `__version_info__`. See
  [MIGRATION.md](MIGRATION.md#removed-without-a-replacement-shim) for the replacement of each.

### Deprecated (removed in 0.3)

- `WiFiSignal(rssi_values=...)`, `iloc.Location`, `iloc.Coordinate`, `iloc.LocalizationResult`,
  `iloc.list_available_datasets`, the dataset id `"ujindoorloc"`,
  `EvaluationResults.from_predictions`, fitting on a list of 0.1 `Location`s, and `predict` on a
  0.1 `WiFiSignal`. Each emits a `FutureWarning` where its behaviour changed.
