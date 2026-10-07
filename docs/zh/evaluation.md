# L4 评测

`indoorloc.evaluation` 提供指标、协议、竞赛评分、Cramér-Rao 界和已发表结果，全部写成 `sklearn.metrics` 风格的数组函数。
它只导入 numpy；绘图需要 matplotlib。

[指南首页](index.md) · [English](../guide/evaluation.md)

## 指标

`evaluate(y_true, y_pred, *, floor_true=None, floor_pred=None, building_true=None, building_pred=None, scale=1.0)`
评测位置，以及（如果给出）楼层与建筑标签。`y_true` 可以是 `SampleTable`，`y_pred` 可以是 `Prediction`，此时使用它们的标签列
（两者的 id 必须逐行一致）。误差是所有坐标轴上的欧氏距离，单位为表的单位乘以 `scale`。预测为 NaN 的样本记为**未定位**：
不计入误差统计，而是计入 `n_failed`，并与误差统计一起报告。真值中的 NaN 会被拒绝，因为没有标注的行并不是方法的失败：
请先选出有标注的行（`test[np.isfinite(test.pos).all(axis=1)]`）。真值在前、估计在后；参数顺序颠倒，或传入 `PDRFusion.run`
返回的 `(t, Prediction)` 元组，都会抛出说明参数顺序的 `TypeError`。

```python
import numpy as np
from indoorloc.evaluation import evaluate

y_true = np.array([[0.0, 0.0], [3.0, 4.0], [10.0, 0.0], [5.0, 5.0]])
y_pred = np.array([[0.0, 1.0], [0.0, 0.0], [np.nan, np.nan], [5.0, 5.0]])    # 第 2 个样本未定位
res = evaluate(y_true, y_pred, floor_true=[0, 1, 1, 2], floor_pred=[0, 1, 2, 2])
print(res)
# mean 2.0000  median 1.0000  P90 4.2000  floor 75.00 %  building n/a  (n=4, 1 not placed)
print(res.errors, res.n_failed)
# [ 1.  5. nan  0.] 1
```

`EvaluationResults` 的字段：
<!-- catalog:results-fields -->
`errors`, `n`, `mean_error`, `median_error`, `p75_error`, `p90_error`, `p95_error`, `rmse`, `max_error`, `floor_accuracy`, `building_accuracy`, `n_failed`
<!-- /catalog:results-fields -->
另有 `cdf(thresholds)`、`to_dict()` 和 `summary()`。任一侧没有标签时，楼层或建筑准确率为 `None`，从不根据缺失的标签计算。

## 协议

协议决定哪些行用于训练、哪些行用于测试。每个协议函数都返回指向表中行的有序 `int64` 下标数组 `(train_idx, test_idx)`，
而不是数据副本，因此划分代价很低，也便于审计：`split_summary` 记录每个下标数组的 sha256，命令行把这些摘要写入结果文件。
随机性只通过 `np.random.default_rng(random_state)` 产生。

命名协议（供 `indoorloc benchmark --protocol` 使用）：

<!-- catalog:protocols -->
| 协议 | 说明 | 需要的列 |
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
folds = kfold(len(train), 5, groups=train.groups["point"], random_state=0)   # 同一参考点不会跨越两侧
print(len(folds), [len(test_idx) for _, test_idx in folds][:3])
# 5 [170, 165, 165]
tr, te = random_split(len(train), 0.2, stratify=train.groups["room"], random_state=0)
print(split_summary(tr, te)["n_test"], split_summary(tr, te)["test_sha256"][:12])
# 166 afe6d017f770
fold = get_protocol("kfold-5").folds(train, random_state=0)[0]
print(fold.name, len(fold.train), len(fold.test))
# fold=0 664 166
```

协议只划分一张表。`load_dataset` 返回数据集官方的 `(train, test)` 二元组，因此要先用 `pool_splits` 合并；
合并后 `groups["split"]` 记录每一行的来源（`official` 协议读取这一列）。直接传入元组会抛出指明 `pool_splits` 的 `TypeError`。

```python
# data: tuji1
from indoorloc.evaluation import pool_splits

tuji_train, tuji_test = iloc.load_dataset("tuji1")
pooled = pool_splits({"train": tuji_train, "test": tuji_test})
folds = get_protocol("cross-device").folds(pooled)                  # 每次留出一部手机
print(len(pooled), [fold.name for fold in folds])
# 8899 ['device=A12', 'device=POCO', 'device=S20', 'device=S7', 'device=tabS7']
```

选用哪种协议取决于要支持的结论。按行随机划分会把同一参考点（或同一条轨迹）的扫描放到两侧，对指纹方法偏乐观。
分组协议让每个参考点、每条轨迹或每个人只出现在一侧（`point-kfold-5`、`trajectory-kfold-5`、`leave-one-trajectory-out`、
`leave-one-user-out`）；跨设备、跨时间和留一建筑划分衡量对新手机、更晚日期和其他建筑的泛化能力。
自定义协议可以是传给 `register_protocol` 的 `Protocol(name, summary, split)`，也可以在命令行中写成 `module:attribute`
（已记录的基准矩阵就是这样使用 `benchmarks.protocols:LEAVE_ONE_USER_OUT` 的）。

## 竞赛评分与误差统计

`indoorloc.evaluation.scoring` 实现 IPIN 与 EvAAL 竞赛中考虑楼层的（“带惩罚的”）误差、圆概率误差、百分位数、成功率和 bootstrap 置信区间：

```python
from indoorloc.evaluation import bootstrap_ci, cep, evaal_etri_score, ipin_score, success_rate

model = iloc.create_model("wknn", preprocess=iloc.FillMissing(-104)).fit(train)
pred = model.localize(test)
print(round(ipin_score(test, pred), 3), round(evaal_etri_score(test, pred), 3))   # P75 + 每层 15 m；均值 + 每层 4 m
# 2.39 1.805
errors = evaluate(test, pred).errors
print(round(cep(errors), 3), success_rate(errors, 2.0), [round(v, 3) for v in bootstrap_ci(errors, "mean", random_state=0)])
# 1.382 67.5 [1.614, 2.014]
```

## 性能界

`indoorloc.evaluation.bounds` 给出 ToA、RSS、AoA、TDoA 的 Cramér-Rao 下界以及锚点几何的精度衰减因子（DOP）。
性能界说明在这种几何与噪声下，任何无偏估计器最好能做到什么程度：

```python
from indoorloc.evaluation import gdop, toa_crlb

r_train, r_test = iloc.load_dataset("synthetic_office", modality="ranges")
anchors = r_test.meta["anchors"]
bound = toa_crlb(anchors, r_test.pos, sigma=0.1)                  # 测距噪声 0.1 m
achieved = iloc.create_model("trilateration", anchors=anchors).fit(r_train).evaluate(r_test)
print(round(float(np.median(bound)), 4), round(float(np.median(gdop(anchors, r_test.pos))), 3), round(achieved.rmse, 3))
# 0.1143 1.143 0.743
```

三边测量的 RMSE（0.743 m）远高于性能界的中位数（0.114 m）。性能界假设测距为无偏高斯噪声、标准差 0.1 m（仿真器的噪声水平），
而仿真器还会给直达路径穿墙的链路加上正偏差，性能界没有对这种偏差建模。`evaluation.plot.plot_bound_map` 可以在网格上绘制性能界。

## 已发表结果

`indoorloc.evaluation.literature` 保存其他作者发表的数字，附带来源（DOI）、所在的表格或章节、协议以及核查状态：
`verified`（已与论文全文核对）、`corrected`（0.1 中记错，现存论文中的值）、`unchecked`（找到来源，数字未核对）、
`unidentified`（找不到可追溯的出版物）、`missing`（有来源但没有数字）和 `not-literature`（IndoorLoc 0.1 自己产生的数字）。
默认只显示前三种。已发表结果始终与本库复现的结果分开：它们来自不同的代码、不同的预处理，往往还是不同的协议，只作为参照，不构成排名。

<!-- catalog:literature -->
| 数据集 | 条目 | `verified` | `corrected` | `unchecked` | `unidentified` | `missing` | `not-literature` |
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
comparison = literature.compare({"wknn": uji}, "ujiindoorloc")          # 两个独立的部分
print(comparison.render("markdown").splitlines()[0])
# ### Reproduced with indoorloc (protocol: official)
```

`indoorloc literature <dataset>` 在命令行输出同样的表。

## 报告与绘图

```python
from indoorloc.evaluation.report import results_table

knn = iloc.create_model("knn", preprocess=iloc.FillMissing(-104)).fit(train)
table = results_table({"knn": knn.evaluate(test), "wknn": model.evaluate(test)}, style="markdown")
print(table.splitlines()[2])
# | knn | 1.823 | 1.381 | 2.395 | 3.462 | 2.348 | 100.000 |
```

当某个方法有未定位的样本时，`results_table` 和 `benchmark_report` 会增加 `n_failed` 列，并在表下方给出说明。

```python
# requires: matplotlib
import matplotlib.pyplot as plt
from indoorloc.evaluation.plot import plot_cdf, plot_error_map

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
plot_cdf({"knn": knn.evaluate(test).errors, "wknn": errors}, ax=ax1)
plot_error_map(test.pos, pred.pos, ax=ax2, floor_plan=test.meta["floor_plan"], floor=0)
fig.savefig("office_errors.png", dpi=120)
```

## 函数目录

<!-- catalog:eval-functions -->
**`indoorloc.evaluation.functional`**: L4: metrics as functions of plain arrays (the sklearn.metrics style).

| 函数 | 作用（取自 docstring） |
| --- | --- |
| `error_cdf` | Empirical CDF. No `thresholds`: `(sorted errors, P(E <= e_i))`; else `P(E <= t)` per t. |
| `evaluate` | Score predicted positions and, if given, floor/building labels. |
| `label_accuracy` | Percentage of matching labels (every label counts, including negative floors); None if absent. |
| `position_errors` | Per-sample Euclidean error over every coordinate axis (1-D, 2-D or 3-D), float64. |

**`indoorloc.evaluation.protocols`**: L4 evaluation protocols: which rows train a model and which rows test it.

| 函数 | 作用（取自 docstring） |
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

| 函数 | 作用（取自 docstring） |
| --- | --- |
| `bootstrap_ci` | Percentile-bootstrap confidence interval of an error statistic. |
| `cep` | Circular error probable: the radius holding `p` % of the horizontal errors. |
| `evaal_etri_score` | EvAAL-ETRI 2015 score: mean of (x-y error + 4 m per floor + 50 m for a wrong building). |
| `ipin_score` | IPIN / EvAAL accuracy score: the 75th percentile of the floor-aware error (metres). |
| `penalized_errors` | Per-sample floor-aware error: horizontal distance + floor and building penalties. |
| `percentile_error` | The `q`-th percentile (0-100) of an error sample, `np.percentile` semantics. |
| `success_rate` | Percentage of estimates with error `<= threshold` (a point of the empirical CDF). |

**`indoorloc.evaluation.bounds`**: L4 performance bounds: Cramér-Rao lower bounds (CRLB) and dilution of precision (DOP).

| 函数 | 作用（取自 docstring） |
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

| 函数 | 作用（取自 docstring） |
| --- | --- |
| `benchmark_report` | Render a benchmark JSON (`indoorloc benchmark --out`): results, folds, provenance, literature. |
| `literature_table` | The literature half of `Comparison.to_dict()` (as stored in a benchmark JSON) as a table. |
| `provenance` | The environment, data and code facts of a benchmark JSON as a key/value list. |
| `results_table` | One row per entry of `{name: EvaluationResults or metrics dict}`. |

**`indoorloc.evaluation.plot`**: L4 figures: error CDFs, error maps, trajectories and bound maps (matplotlib, imported inside each function).

| 函数 | 作用（取自 docstring） |
| --- | --- |
| `plot_bound_map` | A 2-D map of a positioning bound over a grid, with the anchors marked. |
| `plot_cdf` | Empirical CDF of positioning errors as a step curve, one per method. |
| `plot_error_map` | True positions coloured by their error; optional arrows to the estimates. |
| `plot_trajectories` | Ground-truth and estimated tracks in plan view, over the walls of a floor plan. |
<!-- /catalog:eval-functions -->
