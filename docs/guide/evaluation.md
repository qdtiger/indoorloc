# L4 evaluation

`indoorloc.evaluation` holds metrics, protocols, competition scores, Cramér-Rao bounds and
published numbers, all written as functions of plain arrays in the `sklearn.metrics` style.
It imports numpy only; the plots need matplotlib.

[Guide home](index.md) · [中文](../zh/evaluation.md)

## Metrics

`evaluate(y_true, y_pred, *, floor_true=None, floor_pred=None, building_true=None,
building_pred=None, scale=1.0)` scores positions and, when given, floor and building labels.
`y_true` may be a `SampleTable` and `y_pred` a `Prediction`, and then their label columns are
used (and their ids must match row for row). Errors are Euclidean over every coordinate axis,
in the table's units times `scale`. A sample with a NaN prediction is **not placed**: it is left
out of the error statistics and counted in `n_failed`, which is reported next to them. A NaN in
the ground truth is refused, since an unlabelled row is not a failure of the method: select the
labelled rows first (`test[np.isfinite(test.pos).all(axis=1)]`). The truth comes first and the
estimate second; swapped arguments, or the `(t, Prediction)` tuple of `PDRFusion.run`, raise a
`TypeError` that says so.

```python
import numpy as np
from indoorloc.evaluation import evaluate

y_true = np.array([[0.0, 0.0], [3.0, 4.0], [10.0, 0.0], [5.0, 5.0]])
y_pred = np.array([[0.0, 1.0], [0.0, 0.0], [np.nan, np.nan], [5.0, 5.0]])    # sample 2 not placed
res = evaluate(y_true, y_pred, floor_true=[0, 1, 1, 2], floor_pred=[0, 1, 2, 2])
print(res)
# mean 2.0000  median 1.0000  P90 4.2000  floor 75.00 %  building n/a  (n=4, 1 not placed)
print(res.errors, res.n_failed)
# [ 1.  5. nan  0.] 1
```

`EvaluationResults` fields:
<!-- catalog:results-fields -->
`errors`, `n`, `mean_error`, `median_error`, `p75_error`, `p90_error`, `p95_error`, `rmse`, `max_error`, `floor_accuracy`, `building_accuracy`, `n_failed`
<!-- /catalog:results-fields -->
It also has `cdf(thresholds)`, `to_dict()` and `summary()`. A floor or building accuracy is
`None` when either side has no labels; it is never computed from missing labels.

## Protocols

A protocol decides which rows train and which rows test. Every protocol function returns
sorted `int64` index arrays `(train_idx, test_idx)` into a table, never copies of the data, so a
split is cheap and can be audited: `split_summary` records the sha256 of each index array, and
the command line stores these digests in its result files. Randomness goes only through
`np.random.default_rng(random_state)`.

Named protocols (used by `indoorloc benchmark --protocol`):

<!-- catalog:protocols -->
| Protocol | Summary | Needs |
| --- | --- | --- |
| `cross-device` | leave one device out: each device tests once, trained on all other devices | `groups['device']` |
| `cross-time` | train on the earliest 80 % of rows by time, test on the latest 20 % (no timestamp split) | `groups['time']` |
| `kfold-5` | 5-fold cross-validation over shuffled rows (seeded) | — |
| `leave-one-building-out` | each building tests once, trained on the other buildings (transfer) | `building` |
| `leave-one-trajectory-out` | each recorded trajectory tests once, trained on all others (no within-walk leakage) | `groups['trajectory']` |
| `leave-one-user-out` | each user tests once, trained on the other users (EvAAL: test users unseen in training) | `groups['user']` |
| `official` | the dataset's own train/test files (the setting of most published numbers) | `groups['split']` |
| `point-kfold-5` | 5 folds of whole reference points (seeded): a tested point is never seen in training | `groups['point']` |
| `random-80-20` | uniformly random 80 % train / 20 % test rows (seeded; optimistic for fingerprinting) | — |
| `trajectory-kfold-5` | 5 folds of whole trajectories (seeded): no walk is split between train and test | `groups['trajectory']` |
| `within-month` | for every month, train on its training sets and test on its test sets (LongTermWiFi) | `groups['split']`, `groups['month']` |
<!-- /catalog:protocols -->

```python
import indoorloc as iloc
from indoorloc.evaluation import get_protocol, kfold, random_split, split_summary

train, test = iloc.load_dataset("synthetic_office")
folds = kfold(len(train), 5, groups=train.groups["point"], random_state=0)   # a point never on both sides
print(len(folds), [len(test_idx) for _, test_idx in folds][:3])
# 5 [170, 165, 165]
tr, te = random_split(len(train), 0.2, stratify=train.groups["room"], random_state=0)
print(split_summary(tr, te)["n_test"], split_summary(tr, te)["test_sha256"][:12])
# 166 afe6d017f770
fold = get_protocol("kfold-5").folds(train, random_state=0)[0]
print(fold.name, len(fold.train), len(fold.test))
# fold=0 664 166
```

A protocol splits one table. `load_dataset` returns a dataset's official `(train, test)` pair, so
pool it first with `pool_splits`; `groups["split"]` then records where each row came from (the
`official` protocol reads it). Passing the tuple itself is a `TypeError` that names `pool_splits`.

```python
# data: tuji1
from indoorloc.evaluation import pool_splits

tuji_train, tuji_test = iloc.load_dataset("tuji1")
pooled = pool_splits({"train": tuji_train, "test": tuji_test})
folds = get_protocol("cross-device").folds(pooled)                  # leave one phone out
print(len(pooled), [fold.name for fold in folds])
# 8899 ['device=A12', 'device=POCO', 'device=S20', 'device=S7', 'device=tabS7']
```

Which protocol to use depends on the claim. Random row splits put scans from the same reference
point (or the same walk) on both sides, which is optimistic for fingerprinting. Grouped protocols
keep each reference point, walk or person on one side (`point-kfold-5`, `trajectory-kfold-5`,
`leave-one-trajectory-out`, `leave-one-user-out`), and cross-device, cross-time and
leave-one-building-out splits measure generalization to new phones, later dates and other
buildings. A protocol of your own is a `Protocol(name, summary, split)` passed to
`register_protocol`, or a `module:attribute` spec on the command line (the recorded benchmark
matrix used `benchmarks.protocols:LEAVE_ONE_USER_OUT` this way).

## Competition scores and error statistics

`indoorloc.evaluation.scoring` implements the floor-aware ("penalized") errors of the IPIN and
EvAAL competitions, the circular error probable, percentiles, success rates and bootstrap
confidence intervals:

```python
from indoorloc.evaluation import bootstrap_ci, cep, evaal_etri_score, ipin_score, success_rate

model = iloc.create_model("wknn", preprocess=iloc.FillMissing(-104)).fit(train)
pred = model.localize(test)
print(round(ipin_score(test, pred), 3), round(evaal_etri_score(test, pred), 3))   # P75 + 15 m/floor; mean + 4 m/floor
# 2.39 1.805
errors = evaluate(test, pred).errors
print(round(cep(errors), 3), success_rate(errors, 2.0), [round(v, 3) for v in bootstrap_ci(errors, "mean", random_state=0)])
# 1.382 67.5 [1.614, 2.014]
```

## Bounds

`indoorloc.evaluation.bounds` gives Cramér-Rao lower bounds for ToA, RSS, AoA and TDoA, and the
dilution of precision of an anchor geometry. A bound shows how well any unbiased estimator
could do with that geometry and that noise:

```python
from indoorloc.evaluation import gdop, toa_crlb

r_train, r_test = iloc.load_dataset("synthetic_office", modality="ranges")
anchors = r_test.meta["anchors"]
bound = toa_crlb(anchors, r_test.pos, sigma=0.1)                  # 0.1 m ranging noise
achieved = iloc.create_model("trilateration", anchors=anchors).fit(r_train).evaluate(r_test)
print(round(float(np.median(bound)), 4), round(float(np.median(gdop(anchors, r_test.pos))), 3), round(achieved.rmse, 3))
# 0.1143 1.143 0.743
```

The trilateration RMSE (0.743 m) is far above the median bound (0.114 m). The bound assumes
unbiased Gaussian ranges with a standard deviation of 0.1 m, the simulator's noise level, while
the simulator also adds a positive bias to links whose direct path crosses a wall; the bound does
not model that bias. `evaluation.plot.plot_bound_map` draws a bound over a grid.

## Published numbers

`indoorloc.evaluation.literature` stores numbers published by other authors, with their source
(DOI), the table or section they come from, the protocol and a check status: `verified`
(compared with the paper's full text), `corrected` (0.1 had it wrong and the paper's value is
stored), `unchecked` (source found, number not compared), `unidentified` (no traceable
publication), `missing` (a source without a number) and `not-literature` (numbers produced
by IndoorLoc 0.1 itself). By default only the first three are shown. Published numbers are
always kept separate from results produced by this library: they come from other code, other
preprocessing and often other protocols, so they give context and are not a ranking.

<!-- catalog:literature -->
| Dataset | Entries | `verified` | `corrected` | `unchecked` | `unidentified` | `missing` | `not-literature` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ble_indoor` | 2 | 0 | 0 | 0 | 0 | 2 | 0 |
| `ble_rssi_uci` | 14 | 11 | 0 | 1 | 0 | 0 | 2 |
| `csi2taoa` | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| `csi_fingerprint` | 2 | 0 | 0 | 0 | 0 | 2 | 0 |
| `csiindoor` | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| `ibeacon_rssi` | 2 | 0 | 0 | 0 | 0 | 2 | 0 |
| `longtermwifi` | 5 | 0 | 0 | 3 | 0 | 0 | 2 |
| `magneticindoor` | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| `sodindoorloc` | 7 | 0 | 0 | 7 | 0 | 0 | 0 |
| `tampere` | 8 | 0 | 0 | 0 | 8 | 0 | 0 |
| `tuji1` | 4 | 0 | 2 | 0 | 0 | 0 | 2 |
| `ujiindoorloc` | 20 | 10 | 1 | 7 | 2 | 0 | 0 |
| `wificsid2d` | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| `wildv2` | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| `wlanrssi` | 4 | 0 | 0 | 0 | 1 | 0 | 3 |
<!-- /catalog:literature -->

```python
# data: ujiindoorloc
from indoorloc.evaluation import literature

uji_train, uji_test = iloc.load_dataset("ujiindoorloc")
uji = iloc.create_model("wknn", preprocess=iloc.FillMissing(-104)).fit(uji_train).evaluate(uji_test)
comparison = literature.compare({"wknn": uji}, "ujiindoorloc")          # two separate sections
print(comparison.render("markdown").splitlines()[0])
# ### Reproduced with indoorloc (protocol: official)
```

`indoorloc literature <dataset>` prints the same tables on the command line.

## Reports and plots

```python
from indoorloc.evaluation.report import results_table

knn = iloc.create_model("knn", preprocess=iloc.FillMissing(-104)).fit(train)
table = results_table({"knn": knn.evaluate(test), "wknn": model.evaluate(test)}, style="markdown")
print(table.splitlines()[2])
# | knn | 1.823 | 1.381 | 2.395 | 3.462 | 2.348 | 100.000 |
```

When a method could not place some samples, `results_table` and `benchmark_report` add an
`n_failed` column and print a note under the table.

```python
# requires: matplotlib
import matplotlib.pyplot as plt
from indoorloc.evaluation.plot import plot_cdf, plot_error_map

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
plot_cdf({"knn": knn.evaluate(test).errors, "wknn": errors}, ax=ax1)
plot_error_map(test.pos, pred.pos, ax=ax2, floor_plan=test.meta["floor_plan"], floor=0)
fig.savefig("office_errors.png", dpi=120)
```

## Catalog of functions

<!-- catalog:eval-functions -->
**`indoorloc.evaluation.functional`**: L4: metrics as functions of plain arrays (the sklearn.metrics style).

| Function | What it does |
| --- | --- |
| `error_cdf` | Empirical CDF. No `thresholds`: `(sorted errors, P(E <= e_i))`; else `P(E <= t)` per t. |
| `evaluate` | Score predicted positions and, if given, floor/building labels. |
| `label_accuracy` | Percentage of matching labels (every label counts, including negative floors); None if absent. |
| `position_errors` | Per-sample Euclidean error over every coordinate axis (1-D, 2-D or 3-D), float64. |

**`indoorloc.evaluation.protocols`**: L4 evaluation protocols: which rows train a model and which rows test it.

| Function | What it does |
| --- | --- |
| `cross_device_split` | Train on some devices, test on others (device heterogeneity). |
| `cross_time_split` | Train on the past, test on the future (signal drift, AP changes, furniture). |
| `get_protocol` | The registered `Protocol` called `name` (case-insensitive). |
| `group_split` | Rows whose group is in `test_groups` test; the rest (or `train_groups`) train. |
| `kfold` | K folds; each row (or each group, if `groups` is given) is tested exactly once. |
| `leave_one_group_out` | One fold per distinct group value: fold `i` tests on `np.unique(groups)[i]`. |
| `list_protocols` | Names of the registered evaluation protocols (built-in and `register_protocol`), sorted. |
| `pool_splits` | Stack a dataset's split tables into one table and record where each row came from. |
| `random_split` | Uniformly random hold-out split (`sklearn.model_selection.train_test_split` rules). |
| `register_protocol` | Add a named protocol (e.g. a fixed device pair) for the CLI and `get_protocol`. |
| `split_summary` | Sizes, a disjointness check and sha256 digests of a split (for result files). |

**`indoorloc.evaluation.scoring`**: L4 competition scores and error statistics: IPIN / EvAAL rules, CEP, percentiles, bootstrap CIs.

| Function | What it does |
| --- | --- |
| `bootstrap_ci` | Percentile-bootstrap confidence interval of an error statistic. |
| `cep` | Circular error probable: the radius holding `p` % of the horizontal errors. |
| `evaal_etri_score` | EvAAL-ETRI 2015 score: mean of (x-y error + 4 m per floor + 50 m for a wrong building). |
| `ipin_score` | IPIN / EvAAL accuracy score: the 75th percentile of the floor-aware error (metres). |
| `penalized_errors` | Per-sample floor-aware error: horizontal distance + floor and building penalties. |
| `percentile_error` | The `q`-th percentile (0-100) of an error sample, `np.percentile` semantics. |
| `success_rate` | Percentage of estimates with error `<= threshold` (a point of the empirical CDF). |

**`indoorloc.evaluation.bounds`**: L4 performance bounds: Cramér-Rao lower bounds (CRLB) and dilution of precision (DOP).

| Function | What it does |
| --- | --- |
| `aoa_crlb` | 2-D AoA position RMSE bound; see `aoa_fim`. |
| `aoa_fim` | Fisher information of 2-D bearing (angle-of-arrival) measurements, `sigma` in radians. |
| `crlb_covariance` | The covariance bound `J^-1` itself, (D, D) or (P, D, D); singular `J` gives `inf` entries. |
| `crlb_rmse` | Position RMSE bound `sqrt(trace(J^-1))` from a (D, D) or (P, D, D) Fisher information. |
| `dop` | Dilution of precision of range measurements: `{"gdop", "pdop", "hdop"[, "vdop"][, "tdop"]}`. |
| `gdop` | Geometric dilution of precision `sqrt(trace((H^T H)^-1))`; see `dop`. |
| `rss_crlb` | RSS (log-distance path loss) position RMSE bound; see `rss_fim`. |
| `rss_fim` | Fisher information of log-distance RSS measurements (Patwari et al. 2003). |
| `tdoa_crlb` | TDoA position RMSE bound; see `tdoa_fim`. |
| `tdoa_fim` | Fisher information of range differences `d_i - d_ref` built from ToA noise `sigma`. |
| `toa_crlb` | ToA/ranging position RMSE bound (metres); see `toa_fim`. |
| `toa_fim` | Fisher information of range (ToA / RTT / UWB) measurements, `sigma` in metres. |

**`indoorloc.evaluation.report`**: L4 reports: Markdown / plain-text tables of results, with their provenance.

| Function | What it does |
| --- | --- |
| `benchmark_report` | Render a benchmark JSON (`indoorloc benchmark --out`): results, folds, provenance, literature. |
| `literature_table` | The literature half of `Comparison.to_dict()` (as stored in a benchmark JSON) as a table. |
| `provenance` | The environment, data and code facts of a benchmark JSON as a key/value list. |
| `results_table` | One row per entry of `{name: EvaluationResults or metrics dict}`. |

**`indoorloc.evaluation.plot`**: L4 figures: error CDFs, error maps, trajectories and bound maps (matplotlib, imported inside each function).

| Function | What it does |
| --- | --- |
| `plot_bound_map` | A 2-D map of a positioning bound over a grid, with the anchors marked. |
| `plot_cdf` | Empirical CDF of positioning errors as a step curve, one per method. |
| `plot_error_map` | True positions coloured by their error; optional arrows to the estimates. |
| `plot_trajectories` | Ground-truth and estimated tracks in plan view, over the walls of a floor plan. |
<!-- /catalog:eval-functions -->
