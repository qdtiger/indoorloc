# IndoorLoc developer contracts (0.2)

The rules every module follows. Code review checks them; `tests/` and the import-linter
contracts in `pyproject.toml` enforce most of them automatically.

## 1. Layers and allowed imports

```
L5  apps        tracking, PDR, fusion, streaming, navigation        may import core, signals, methods, evaluation
L3  methods     fingerprinting, model-based, deep, transfer          may import core, signals; evaluation only inside functions
L1  datasets    loaders + simulators       ┐
L2  signals     transforms + functional    ├ independent of each other; import core only
L4  evaluation  metrics, protocols, bounds ┘
    core        SampleTable, Prediction, Estimator, Registry, persistence (numpy + stdlib only)
```

* No layer imports `apps`, `_legacy` or `cli`. L3 and L5 never import L1: data arrives as arrays or a `SampleTable`.
* numpy is the only third-party import at module level. Any other package (torch, sklearn, scipy,
  pandas, matplotlib, h5py, timm, skada, deepmimo, ...) is imported **inside the function** that needs
  it via `indoorloc.core.requires(module, extra)`, which names the pip extra in its error. The only
  exceptions are the declared bridge modules listed in `pyproject.toml` (`light-layers` contract).
* `import indoorloc` must stay free of numpy; each layer must import without torch/sklearn/scipy/pandas.

## 2. The two data types (`indoorloc.core`)

`SampleTable(X, pos, floor=None, building=None, groups={}, ids=None, meta={})` — the exit of L1 and
the entry of L2/L3. `Prediction(pos, floor=None, building=None, ids=None, spread=None)` — the exit of
L3 and the entry of L4/L5. Both are frozen and hold read-only numpy arrays. The first axis is samples.

| Field | Rule |
|---|---|
| `X` | physical units (dBm, metres, radians, complex CSI). **Missing = NaN**, never a sentinel. No normalization in L1. |
| `pos` | float64 `(N, D)`, coordinates in the frame named by `meta["crs"]` (`"local"` if none). |
| `floor`, `building` | int64 `(N,)` or `None` (= unlabelled). Negative floors are real floors. |
| `groups` | name -> `(N,)` array for grouped splits. Use these names when they apply: `user`, `device`, `time` (unix seconds or a sortable index), `session`, `trajectory`, `source` (`measured`/`simulated`), `month`, `environment`, `room`, `point` (reference point id), `step` (IMU: completed steps so far), `rx_set`. Unknown values: leave the column out and list it in `meta["unknown_groups"]`. |
| `ids` | stable, unique sample ids (e.g. `"train-00042"`). |
| `meta` | dataset-level facts, see §3. |

### X layout by modality (`meta["modality"]`)

| modality | X shape / dtype | notes |
|---|---|---|
| `wifi_rssi`, `ble_rssi`, `rssi` | `(N, n_ap)` float32 dBm | `meta["feature_names"]` = AP/beacon ids in column order; `rssi` = received power of any radio (e.g. DeepMIMO) |
| `csi` | `(N, n_rx, n_tx, n_sub)` complex64 | `meta["subcarriers"]` = OFDM subcarrier indices of the last axis; several APs: rx rows stacked AP by AP, `meta["antenna_anchor"]` maps each row to its AP, `meta["subcarrier_offsets_hz"]` gives the grid |
| `csi_amp`, `csi_phase` | float32 `(N, ..., n_sub)`, linear amplitude (`units` `"dB"` after `CSIAmplitude(db=True)`) or radians | amplitude-only sources load as `csi_amp`; `CSIPhaseSanitize(output="phase")` produces `csi_phase` |
| `ranges` | `(N, n_anchor)` float64 metres (ToA/RTT/UWB) | `meta["anchors"]` `(n_anchor, D)` float64, same frame as `pos` |
| `tdoa` | `(N, n_anchor - 1)` float64 metres, `r_i - r_0` | `meta["anchors"]` as above, `meta["reference_anchor"] = 0` |
| `aoa` | `(N, n_anchor)` float64 radians in `[-pi, pi)`, counter-clockwise from each array's boresight | `meta["anchors"]`, `meta["anchor_orientations"]` (boresight bearing; global bearing = orientation + angle; ULA axis = boresight + 90 deg) |
| `imu` | `(N, C)` float32, one row per time step | `groups["trajectory"]`, `groups["time"]`; `meta["channels"]` named `acc_x/y/z` (m/s^2), `gyr_x/y/z` (rad/s), `mag_x/y/z` (uT), optional `grav_x/y/z` (m/s^2, gravity as the phone reports it, pointing up) and `rv_x/y/z/w`; device frame, z up; `meta["rate_hz"]` |
| `magnetic` | `(N, 3)` float64 uT: total `B`, horizontal `B_h`, vertical `B_v` (positive up) | `meta["feature_names"] = ("B", "B_h", "B_v")`; sequences carry `groups["trajectory"]`, `groups["time"]`, `meta["rate_hz"]` (`signals.MagneticFeatures`, `SyntheticOffice(modality="magnetic")`) |
| `vlc` | `(N, n_led)` float64 received optical power in W; NaN = LED not seen (never 0) | `meta["anchors"]`, `meta["led_positions"]` `(A, 3)`, `meta["anchor_normals"]` (unit emission axes), `lambertian_order`, `tx_power_w`, `receiver_area_m2`, `receiver_fov_rad`, `filter_gain`, `concentrator_gain`, `device_height` |

Maps travel in `meta["floor_plan"]` as arrays: `walls` `(W, 4)` (x0, y0, x1, y1), `wall_floor`, `wall_type`,
`materials`, `bounds`, `n_floors`, `floor_height`, `rooms`, `room_floor`, `room_kind`, `connectors`,
`connector_kind` (only the keys a dataset knows). L5 (`apps.maps.FloorMap`) reads `walls` and `wall_floor`.

Time series (trajectories, streams) are tables with one row per time step plus `groups["trajectory"]`
and `groups["time"]`; they are never nested objects.

## 3. Adding a dataset (L1)

One file per dataset in `indoorloc/datasets/`, one class deriving from `datasets._base.Dataset`,
registered as a string in `datasets/__init__.py`. Code outside the package calls
`register_dataset(name, cls)` (also a decorator) or loads `load_dataset("pkg.module:Class")`
without registering.

```python
class MyDataset(Dataset):
    """One line: what it is (authors, venue, year).  Longer description: rooms, devices, collection."""
    name = "mydataset"                                   # registry id and cache folder
    urls = ("https://.../archive.zip",)                  # mirrors of one archive, or {relpath: url}
    files = {"train": ("train.csv", "<sha256>"),         # or a tuple of (relpath, sha256) pairs
             "test": ("test.csv", "<sha256>")}
    split_aliases = {"validation": "test"}
    meta = {"modality": "wifi_rssi", "units": "dBm", "crs": "local", "pos_names": ("x", "y"),
            "pos_units": "m", "floors": (...), "buildings": (...), "license": "...", "doi": "...",
            "citation": "Authors, Title, Venue, Year", "url": "https://...", "raw_missing_value": 100}

    def _parse(self, path, split):                      # a list of paths when the split has several files
        ...                                              # read with numpy / csv / zipfile (stdlib first)
        return SampleTable(X, pos, floor, building, groups, ids, meta={"feature_names": names})
```

* Every file has a sha256 (compute it from the official download). Parse columns **by name**.
* Keep the original coordinate frame and state it (`crs`, `pos_units`); never silently rescale.
* A dataset without an official split: define `files` for `"all"` and document the recommended
  protocol in the docstring (L4 builds the split from `groups`).
* Constructor options (building subset, month, environment, ...) are keyword arguments of `__init__`
  after `root, *, download, verify`; `load_dataset(name, **options)` forwards them.
* Readers of `.mat`/`.h5` use `requires("scipy.io", "datasets")` / `requires("h5py", "datasets")` inside `_parse`.

## 4. Adding a transform (L2)

Pure functions go in `signals/functional.py` (or `signals/<modality>.py` for CSI/ranging/IMU helpers);
the fit/transform class goes in `signals/transforms.py` (or a modality module) and derives from
`signals.transforms.Transform`.

* `__init__` only stores its arguments. Learned statistics are set in `fit` and end with `_`.
* `_transform(x)` works on the last axis for one sample `(F,)` or a batch `(N, F)`; `Transform.transform`
  already handles SampleTable and WiFiSignal. Transforms that change the feature layout (CSI -> real
  features) override `transform` to keep `meta["feature_names"]` consistent.
* Anything that uses extra data at fit time declares it (`fit(X, y=None, *, target=None)`).
* Augmentations take `random_state=None` and use `np.random.default_rng(random_state)`; never the global RNG.

## 5. Adding a method (L3)

A class deriving from `methods.base.BaseLocalizer`, registered as a string in `methods/__init__.py`.

```python
class MyLocalizer(BaseLocalizer):
    """One line.  Description.  References: Authors, Title, Venue, Year, DOI."""
    _allow_nan = False        # True if X may hold NaN (missing readings)
    _allow_complex = False    # True for raw CSI

    def __init__(self, k=5, random_state=None):   # parameters only, no validation, no work
        self.k = k
        self.random_state = random_state

    def _fit(self, X, pos, floor, building, **fit_params):   # pos is float64 (N, D); floor/building int64 or None
        ...                                                   # learned attributes end with "_"

    def _localize(self, X) -> Prediction:
        return Prediction(pos, floor=..., building=..., spread=...)   # pos float64 (N, D)
```

* `fit`, `predict`, `localize`, `score`, `evaluate`, `save`/`load_model`, `clone`, `get_params`/
  `set_params` come from the base classes: do not override them. Methods are sklearn-compatible.
* Deterministic by construction: ties broken by index, randomness only through `random_state`.
* Predict floor/building when the training data has them (majority vote, classifier, ...).
* `Prediction.spread` (optional): the method's own positional uncertainty scale, in coordinate units.
* **Model-based methods** (ranging, TDoA, AoA, path loss) take the anchor geometry as a constructor
  parameter (`anchors=` `(A, D)` array) and `X` as the measurements. `fit` may only record
  `n_features_in_` or calibrate parameters (bias, path-loss exponent) from labelled data. A method
  that learns nothing from labels sets `_labels_optional` (True, or a property such as
  `not self.calibrate`): `fit(X)` then works without `y` and `_fit` receives `pos=None`.
* Heavy frameworks: module-level import only in declared bridge modules (`methods/sklearn_wrap.py`,
  `methods/deep/`); everything else imports inside functions via `requires`. Non-array state (a torch
  module) overrides `_get_state`/`_set_state` so `save` writes arrays only (no pickle).
* A non-estimator value object that must be saved as a parameter (e.g. `apps.maps.FloorMap`) sets
  `_save_via_dict = True` and implements `to_dict()` / `from_dict()`; persistence stores it as an
  `__object__` node and loads it through the same allow-list as estimators.

## 6. Adding evaluation code (L4)

Functions of plain arrays (the `sklearn.metrics` style); objects only for results. Protocols return
index arrays (`train_idx, test_idx`) computed from `groups`, never copies of data. Literature numbers
live in `evaluation/literature/*.json` with full provenance (paper, DOI, table/figure, metric
definition) and are always reported separately from numbers reproduced by this library. Every entry
states how it was checked: `verified` (compared with the paper's full text), `corrected` (the 0.1 value
was wrong; the paper's value is stored), `unchecked` (source found, text not accessible),
`unidentified` (no traceable source), `missing` (a cited source without a number), `not-literature`
(a number produced by IndoorLoc 0.1, never published). The vocabulary is
`indoorloc.evaluation.literature.CHECKS`.
`evaluate` leaves samples a method could not place (NaN prediction) out of the error statistics and
counts them in `n_failed`; always report it next to the errors.

## 7. Applications (L5)

Classes with parameters derive from `core.Estimator`. Streaming APIs consume iterables of
`(t, measurement)` and yield results; no threads, no global state. Everything runs on numpy.

* Heading: radians, counter-clockwise from +x. Time: seconds.
* Trackers share one protocol: `update(t, z, spread=None)`, `predict_to(t)`, `estimate()`,
  `reset(t)`; any L3 `Prediction.spread` can be used as the measurement noise. Every `t` is an
  **absolute** time on the stream's clock, never an increment. The older `predict` differs by
  class and is kept for compatibility only: `KalmanTracker.predict(t)` takes an absolute time,
  `ParticleFilter.predict(dt)` an increment (and leaves the clock alone). New code and the docs
  use `predict_to`.
* Offline trackers (`KalmanTracker.filter`/`smooth`, `ExtendedKalmanTracker.filter`/`smooth`,
  `ParticleFilter.filter`) return a `core.Prediction` with one row per input row (NaN before the
  track starts) whose `spread` is `sqrt(trace P_pos)` (for particles, of the weighted cloud), so L4
  evaluates them like any L3 method. Methods whose rows are not the input rows say so in their
  return type: `PDR.run` returns a `StepTrack` (one entry per detected step; `position_at(t)`
  interpolates), `PDRFusion.run` returns `(t, Prediction)` with one row per event (steps and
  fixes merged), and `streaming.stack_estimates` returns `(t, Prediction)` with one row per stream
  item (`OnlineLocalizer.run` itself yields `StreamEstimate`s); evaluate those against the truth
  at `t`.

## 8. Documentation and citations

Module docstring: what the module covers. Class docstring: first line summary, then the method in two
or three sentences, parameters, and a `References` section with the original paper(s):
`Authors, "Title", Venue, Year. DOI or URL`. If the implementation deviates from the paper, say how.

## 9. Tests

* `tests/<layer>/test_<module>.py`; each file runs in a few seconds on synthetic data.
* No network access in tests. Real-data checks are marked
  `@pytest.mark.skipif(not path.is_file(), reason=...)` and read from `$INDOORLOC_DATA` (default
  `~/.cache/indoorloc/datasets/<name>`).
* A reproduced method gets at least one test against a known result (a closed-form case, a textbook
  example or a published toy example), not only a smoke test.
* Run: `python -m pytest -q tests` and `lint-imports`.

## 10. Parameter names

One idea has one name in every layer. New parameters use this vocabulary; the exceptions below
are documented and keep their names (renaming them would break user code and saved models).

| Name | Meaning | Used by |
|---|---|---|
| `random_state` | int seed, `np.random.Generator` or None; all randomness goes through `np.random.default_rng(random_state)` | augmentations, `Protocol.folds`, `random_split`, `kfold`, `ParticleFilter`, `PDRFusion` |
| `sigma` | measurement-noise std of a model-based method, in the units of `X` (range noise in metres for ranging and TDoA, radians for AoA, dB for path loss, W for VLC); None = estimated | `TrilaterationLocalizer`, `TDOALocalizer`, `AoALocalizer`, `PathLossLocalizer`, `LambertianLocalizer` |
| `value`, `fill_value` | the number written in place of a missing reading: `value` when it is the class's only number, `fill_value` otherwise; `missing` says what counts as missing (NaN or a raw sentinel such as 100) | `FillMissing(value, missing)`, `APSelect(fill_value=)` |
| `k` | a count: nearest neighbours, or items kept | `KNNLocalizer`, `WKNNLocalizer`, `APSelect`, `RadioMapInterpolator`, `WeightedCentroidLocalizer` |
| `anchors` | `(A, D)` anchor coordinates in the frame of `pos` (as `meta["anchors"]`) | model-based L3 methods, `ExtendedKalmanTracker`, `ParticleFilter` |
| `orientations` | `(A,)` boresight bearings of the anchors, radians (as `meta["anchor_orientations"]`) | `AoALocalizer` |

Documented exceptions:

* `ExtendedKalmanTracker(range_std=)` and `ParticleFilter(range_std=)` are the range-noise std that
  L3 calls `sigma`; L5 names every noise level `<quantity>_std` (`meas_std`, `motion_std`,
  `step_length_std`, `heading_std`).
* `Protocol.split(table, seed)`: the second, positional argument of a protocol's split function is
  the seed; the public call is `Protocol.folds(table, random_state=...)`.
* `SyntheticOffice(seed=)` (`load_dataset("synthetic_office", seed=1)`) takes an int seed of the
  simulator; so do `datasets.torch_adapter.make_dataloader(seed=)` and the CLI's `--seed`.
* `PDR(k=)` (and `step_lengths`, `weinberg_step_length`, `kim_step_length`): `k` is the step-length
  constant of the Weinberg or Kim model, not a count.
