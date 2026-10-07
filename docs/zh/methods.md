# L3 方法

`indoorloc.methods` 把各种定位方法统一到一个 scikit-learn 风格的契约之下。模块级只导入 numpy；
依赖 scikit-learn、torch 和 skada 的方法在拟合或加载时才导入这些包。

[指南首页](index.md) · [English](../guide/methods.md)

## 估计器契约

| 调用 | 含义 |
| --- | --- |
| `create_model(name, *, preprocess=None, **params)` | 按注册名（或 `"package.module:Class"`）创建方法；`preprocess=` 用 `LocalizerPipeline` 把它与一个 L2 变换组合 |
| `fit(X, y=None, *, floor=None, building=None, **fit_params)` | `X` 为 `(N, ...)`，`y` 为 `(N, D)` 位置；或 `X` 为 `SampleTable`，由它提供 `y`、`floor`、`building` |
| `localize(X) -> Prediction` | 位置以及楼层、建筑、id 和 `spread` |
| `predict(X) -> ndarray` | 只有位置，与 scikit-learn 一致 |
| `evaluate(table)` 或 `evaluate(X, y, floor=...)` | L4 的 `EvaluationResults`；方法无法定位的样本不计入误差，而是计入 `n_failed` |
| `score(X, y)` | 平均位置误差的相反数，供 `GridSearchCV` 最大化；只要有样本无法定位就为 NaN（`evaluate(...).n_failed` 统计这些样本） |
| `save(path)`、`iloc.load_model(path)` | `config.json` + `arrays.npz`，不使用 pickle |
| `get_params`、`set_params`、`iloc.clone` | 与 scikit-learn 语义相同；`sklearn.base.clone` 同样可用 |

所有方法遵守以下规则：

* `__init__` 只保存参数。学到的属性以 `_` 结尾；`fit` 失败时模型保持未拟合状态。
* 缺失读数为 NaN。不能处理 NaN 的方法会拒绝输入并给出解决办法（`preprocess=FillMissing(-104)`）；
  目录中的“接受 NaN”一栏说明哪些方法可以直接处理 NaN。
* 结果是确定的：平局按训练样本下标打破，随机性只来自 `random_state`。k-NN 用固定求和顺序重新计算候选近邻的距离
  （读数为整数 dBm 时距离平方是精确整数），距离相等时按训练下标排序，因此结果与 BLAS 线程数和批大小无关。
  这就是 UJIIndoorLoc 的数字在 0.2 与 0.1 之间不同的原因（见 [MIGRATION.md](../../MIGRATION.md#why-the-ujiindoorloc-numbers-changed)）。
* 训练数据带有楼层和建筑标签时，方法会预测它们（投票、分类器或网络输出头）。
* 基于模型的方法（三边测量、TDoA、AoA、路径损耗、加权质心、VLC）通过 `anchors=` `(A, D)` 接收锚点几何；
  `fit` 只记录输入维度，或标定偏差与路径损耗参数。除非 `calibrate=True`（路径损耗：还有参数需要学习），
  `localize` 在 `fit` 之前就能使用，见[基于模型](#基于模型)。

```python
import numpy as np
import indoorloc as iloc
from indoorloc.signals import FillMissing

train, test = iloc.load_dataset("synthetic_office")
model = iloc.create_model("wknn", k=5, preprocess=FillMissing(-104)).fit(train)
pred = model.localize(test)
print(pred.pos.shape, pred.floor[:3], model.predict(test.X[:2]).round(2).tolist())
# (200, 2) [0 0 0] [[13.0, 14.15], [1.78, 17.3]]
model.save("wknn_model")                         # 一个目录：config.json + arrays.npz
again = iloc.load_model("wknn_model")
print(np.array_equal(again.predict(test.X), model.predict(test.X)))
# True
```

### 与 scikit-learn 互操作

模型就是 scikit-learn 估计器，`localizer__k` 这样的嵌套参数可以直接用于网格搜索。按参考点分组的折保证同一参考点采集的扫描不会同时出现在划分的两侧：

```python
# requires: sklearn
from sklearn.model_selection import GridSearchCV, GroupKFold

X, y = train.to_numpy()
search = GridSearchCV(iloc.create_model("knn", preprocess=FillMissing(-104)),
                      {"localizer__k": [1, 3, 5, 9], "localizer__weights": ["uniform", "distance"]},
                      cv=GroupKFold(5))
search.fit(X, y, groups=train.groups["point"])
print(search.best_params_, round(-search.best_score_, 3))
# {'localizer__k': 9, 'localizer__weights': 'distance'} 2.628
```

### 注册自己的方法

`BaseLocalizer` 的子类实现 `_fit` 和 `_localize`；`register_model` 让它可以被 `create_model` 和命令行使用。
完整示例见 [扩展](extending.md#方法)。

## 目录

由 `python docs/catalog.py` 从 `indoorloc.methods.METHODS` 和类的 docstring 生成（“作用”与参考文献保持英文原文）。
extra 一栏是除 numpy 之外需要安装的 pip extra。

<!-- catalog:methods -->
**指纹**

| 名称 | 作用（取自 docstring） | extra | 接受 NaN | 参考文献 |
| --- | --- | --- | --- | --- |
| [`ensemble`](../../indoorloc/methods/ensemble.py)<br>`EnsembleLocalizer` | Averaging ensemble: combine the positions of several localizers, vote their labels. | 仅 numpy | 取决于成员模型 | Dietterich, T. G., "Ensemble methods in machine learning", Multiple Classifier Systems (MCS) 2000, LNCS 1857. [doi:10.1007/3-540-45014-9_1](https://doi.org/10.1007/3-540-45014-9_1) |
| [`extratrees`](../../indoorloc/methods/sklearn_wrap.py)<br>`ExtraTreesLocalizer` | Extremely randomized trees: RandomForestLocalizer with random split thresholds. | `[sklearn]` | 否（先填补） | Geurts, P., Ernst, D., Wehenkel, L., "Extremely randomized trees", Machine Learning 63(1), 2006. [doi:10.1007/s10994-006-6226-1](https://doi.org/10.1007/s10994-006-6226-1) |
| [`gbdt`](../../indoorloc/methods/sklearn_wrap.py)<br>`GradientBoostingLocalizer` | Histogram gradient-boosted trees: one booster per coordinate axis, boosted classifiers for labels. | `[sklearn]` | 是 | Friedman, J. H., "Greedy function approximation: A gradient boosting machine", Annals of Statistics 29(5), 2001. [doi:10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451) (docstring 中另有 1 篇) |
| [`gp_radiomap`](../../indoorloc/methods/gaussian_process.py)<br>`GPRadioMapLocalizer` | Gaussian-process radio map with maximum-likelihood localization (Ferris et al., 2006). | 仅 numpy | 否（先填补） | Ferris, B., Hähnel, D., Fox, D., "Gaussian processes for signal strength-based location estimation", Robotics: Science and Systems II, 2006. [doi:10.15607/RSS.2006.II.039](https://doi.org/10.15607/RSS.2006.II.039) (docstring 中另有 1 篇) |
| [`hierarchical`](../../indoorloc/methods/hierarchical.py)<br>`HierarchicalLocalizer` | Coarse-to-fine localization: building -> floor -> position, one sub-model per group. | 仅 numpy | 取决于成员模型 | Marques, N., Meneses, F., Moreira, A., "Combining similarity functions and majority rules for multi-building, multi-floor, WiFi positioning", IPIN 2012. [doi:10.1109/IPIN.2012.6418937](https://doi.org/10.1109/IPIN.2012.6418937) (docstring 中另有 1 篇) |
| [`horus`](../../indoorloc/methods/probabilistic.py)<br>`HorusLocalizer` | Horus: maximum-likelihood fingerprinting with per-location, per-AP Gaussian RSSI models. | 仅 numpy | 是 | Youssef, M., Agrawala, A., "The Horus WLAN location determination system", MobiSys 2005. [doi:10.1145/1067170.1067193](https://doi.org/10.1145/1067170.1067193) (docstring 中另有 1 篇) |
| [`knn`](../../indoorloc/methods/neighbors.py)<br>`KNNLocalizer` | k-NN fingerprinting: average the positions of the `k` nearest reference scans. | 仅 numpy | 否（先填补） | P. Bahl, V. N. Padmanabhan, "RADAR: an in-building RF-based user location and tracking system", IEEE INFOCOM 2000. [doi:10.1109/INFCOM.2000.832252](https://doi.org/10.1109/INFCOM.2000.832252). (docstring 中另有 1 篇) |
| [`rf`](../../indoorloc/methods/sklearn_wrap.py)<br>别名: `random_forest`<br>`RandomForestLocalizer` | Random-forest fingerprinting: one multi-output forest for the position, forests for labels. | `[sklearn]` | 否（先填补） | Breiman, L., "Random forests", Machine Learning 45(1), 2001. [doi:10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324) (docstring 中另有 1 篇) |
| [`stacking`](../../indoorloc/methods/ensemble.py)<br>`StackingLocalizer` | Stacked generalization: a meta-learner fitted on out-of-fold member predictions. | 仅 numpy | 取决于成员模型 | Wolpert, D. H., "Stacked generalization", Neural Networks 5(2), 1992. [doi:10.1016/S0893-6080(05)80023-1](https://doi.org/10.1016/S0893-6080(05)80023-1) (docstring 中另有 1 篇) |
| [`svm`](../../indoorloc/methods/sklearn_wrap.py)<br>`SVMLocalizer` | Support-vector fingerprinting: one epsilon-SVR per coordinate axis, SVC for labels. | `[sklearn]` | 否（先填补） | Smola, A. J., Schölkopf, B., "A tutorial on support vector regression", Statistics and Computing 14(3), 2004. [doi:10.1023/B:STCO.0000035301.49549.88](https://doi.org/10.1023/B:STCO.0000035301.49549.88) (docstring 中另有 2 篇) |
| [`wknn`](../../indoorloc/methods/neighbors.py)<br>别名: `weighted_knn`<br>`WKNNLocalizer` | Weighted k-NN: KNNLocalizer with inverse-distance weights by default. | 仅 numpy | 否（先填补） | — |

**基于模型**

| 名称 | 作用（取自 docstring） | extra | 接受 NaN | 参考文献 |
| --- | --- | --- | --- | --- |
| [`aoa`](../../indoorloc/methods/aoa.py)<br>`AoALocalizer` | 2-D position from angles of arrival at anchors with known positions and orientations. | 仅 numpy | 是 | R. O. Schmidt, "Multiple emitter location and signal parameter estimation", IEEE Transactions on Antennas and Propagation 34(3), 1986. [doi:10.1109/TAP.1986.1143830](https://doi.org/10.1109/TAP.1986.1143830). (docstring 中另有 4 篇) |
| [`centroid`](../../indoorloc/methods/geometric.py)<br>`WeightedCentroidLocalizer` | Weighted centroid of the anchors that hear the target. | 仅 numpy | 是 | J. Blumenthal, R. Grossmann, F. Golatowski, D. Timmermann, "Weighted centroid localization in Zigbee-based sensor networks", IEEE International Symposium on Intelligent Signal Processing (WISP), 2007. [doi:10.1109/WISP.2007.4447528](https://doi.org/10.1109/WISP.2007.4447528). (docstring 中另有 1 篇) |
| [`pathloss`](../../indoorloc/methods/pathloss.py)<br>`PathLossLocalizer` | Maximum-likelihood localization with a log-distance path-loss model per anchor. | 仅 numpy | 是 | T. S. Rappaport, "Wireless Communications: Principles and Practice", 2nd ed., Prentice Hall, 2002. ISBN 0-13-042232-0. (docstring 中另有 4 篇) |
| [`tdoa`](../../indoorloc/methods/geometric.py)<br>`TDOALocalizer` | Position from range differences (TDoA) with Chan and Ho's closed form. | 仅 numpy | 是 | Y. T. Chan, K. C. Ho, "A simple and efficient estimator for hyperbolic location", IEEE Transactions on Signal Processing 42(8), 1994. [doi:10.1109/78.301830](https://doi.org/10.1109/78.301830). (docstring 中另有 1 篇) |
| [`trilateration`](../../indoorloc/methods/geometric.py)<br>别名: `multilateration`<br>`TrilaterationLocalizer` | Position from ranges to anchors at known positions (ToA, two-way RTT, UWB). | 仅 numpy | 是 | W. H. Foy, "Position-location solutions by Taylor-series estimation", IEEE Transactions on Aerospace and Electronic Systems AES-12(2), 1976. [doi:10.1109/TAES.1976.308294](https://doi.org/10.1109/TAES.1976.308294). (docstring 中另有 4 篇) |
| [`vlc`](../../indoorloc/methods/vlc.py)<br>`LambertianLocalizer` | Visible light positioning by received-power ranging or model fitting (Lambertian LOS). | 仅 numpy | 是 | Y. Zhuang, L. Hua, L. Qi, J. Yang, P. Cao, Y. Cao, Y. Wu, J. Thompson, H. Haas, "A survey of positioning systems using visible LED lights", IEEE Communications Surveys & Tutorials 20(3):1963-1988, 2018. [doi:10.1109/COMST.2018.2806558](https://doi.org/10.1109/COMST.2018.2806558). (docstring 中另有 3 篇) |

**序列匹配**

| 名称 | 作用（取自 docstring） | extra | 接受 NaN | 参考文献 |
| --- | --- | --- | --- | --- |
| [`magnetic_dtw`](../../indoorloc/methods/magnetic.py)<br>`MagneticDTWLocalizer` | Localization by subsequence DTW of the recent magnetic sequence against reference walks. | 仅 numpy | 否 | K. P. Subbu, B. Gozick, R. Dantu, "LocateMe: magnetic-fields-based indoor localization using smartphones", ACM Transactions on Intelligent Systems and Technology 4(4), 2013. [doi:10.1145/2508037.2508054](https://doi.org/10.1145/2508037.2508054). (docstring 中另有 3 篇) |

**深度学习**

| 名称 | 作用（取自 docstring） | extra | 接受 NaN | 参考文献 |
| --- | --- | --- | --- | --- |
| [`cnn1d`](../../indoorloc/methods/deep/localizer.py)<br>`CNN1DLocalizer` | 1-D convolutional fingerprinting: `DeepLocalizer` with the `"cnn1d"` backbone. | `[deep]` | 否（先填补） | X. Song, X. Fan, C. Xiang, Q. Ye, L. Liu, Z. Wang, X. He, N. Yang and G. Fang, "A Novel Convolutional Neural Network Based Indoor Localization Framework With WiFi Fingerprinting", IEEE Access 7:110698-110709, 2019. <https://doi.org/10.1109/ACCESS.2019.2933921> |
| [`deep`](../../indoorloc/methods/deep/localizer.py)<br>`DeepLocalizer` | Deep fingerprinting: a backbone network with multi-task heads, trained end to end. | `[deep]` | 否（先填补） | K. S. Kim, S. Lee and K. Huang, "A scalable deep neural network architecture for multi-building and multi-floor indoor localization based on Wi-Fi fingerprinting", Big Data Analytics 3:4, 2018. <https://doi.org/10.1186/s41044-018-0031-2> (docstring 中另有 2 篇) |
| [`mlp`](../../indoorloc/methods/deep/localizer.py)<br>`MLPLocalizer` | Multi-layer perceptron fingerprinting: `DeepLocalizer` with the `"mlp"` backbone. | `[deep]` | 否（先填补） | K. S. Kim, S. Lee and K. Huang, "A scalable deep neural network architecture for multi-building and multi-floor indoor localization based on Wi-Fi fingerprinting", Big Data Analytics 3:4, 2018. <https://doi.org/10.1186/s41044-018-0031-2> |
<!-- /catalog:methods -->

域适应（`indoorloc.methods.transfer`）是一组 L2 风格的变换，同时从有标签的源域扫描和无标签的目标域扫描中学习：

<!-- catalog:transfer -->
| 名称 | 类型 | 作用（取自 docstring） | extra | 参考文献 |
| --- | --- | --- | --- | --- |
| [`CORAL`](../../indoorloc/methods/transfer.py) | 类 | CORrelation ALignment: re-colour the source features with the target covariance. | 仅 numpy | B. Sun, J. Feng and K. Saenko, "Return of Frustratingly Easy Domain Adaptation", Proceedings of the AAAI Conference on Artificial Intelligence 30(1), 2016. <https://doi.org/10.1609/aaai.v30i1.10306> |
| [`TCA`](../../indoorloc/methods/transfer.py) | 类 | Transfer Component Analysis: a shared low-dimensional embedding in which the domains match. | 仅 numpy | S. J. Pan, I. W. Tsang, J. T. Kwok and Q. Yang, "Domain Adaptation via Transfer Component Analysis", IEEE Transactions on Neural Networks 22(2):199-210, 2011. <https://doi.org/10.1109/TNN.2010.2091281> (its experiments include cross-domain WiFi localization) |
| [`SkadaAdapter`](../../indoorloc/methods/transfer.py) | 类 | Any feature-level adapter of the skada library as an L2 transform (`fit(X, target=...)`). | `[transfer]` | skada, "Scikit-learn-compatible domain adaptation", <https://github.com/scikit-adaptation/skada>; (docstring 中另有 1 篇) |
| [`mmd`](../../indoorloc/methods/transfer.py) | 函数 | Squared maximum mean discrepancy between two samples, `MMD^2(X, Y)`. | 仅 numpy | A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schoelkopf and A. Smola, "A Kernel Two-Sample Test", Journal of Machine Learning Research 13:723-773, 2012. <https://jmlr.org/papers/v13/gretton12a.html> |
<!-- /catalog:transfer -->

## 按类别的示例

以下数字全部来自仿真办公楼，只用于演示调用方式，不代表各方法在真实建筑中的比较。真实数据的结果见 [docs/benchmarks_zh.md](../benchmarks_zh.md)。

### 指纹

```python
for name, params in [("horus", {}), ("gp_radiomap", {}),
                     ("ensemble", {"localizers": ["wknn", "horus", "gp_radiomap"], "combine": "median"})]:
    m = iloc.create_model(name, preprocess=FillMissing(-104), **params).fit(train)
    print(name, round(m.evaluate(test).mean_error, 3))
# horus 1.705
# gp_radiomap 1.989
# ensemble 1.704
```

`svm`、`rf`、`extratrees`、`gbdt` 需要 scikit-learn（`[sklearn]`）；`mlp`、`cnn1d`、`deep` 需要 torch
（`[deep]`；`deep` 使用 timm 骨干网络时还需要 timm）：

```python
# requires: torch
mlp = iloc.create_model("mlp", hidden=(128, 64), epochs=50, random_state=0,
                        preprocess=FillMissing(-104)).fit(train)
print(round(mlp.evaluate(test).mean_error, 3))
# 2.41
```

给定 `random_state` 和 torch 线程数时，MLP 的结果是确定的。

### 基于模型

锚点几何取自表的 `meta["anchors"]`（仿真器）或部署图纸。`from_meta` 从表中读取几何；对 AoA，还会读取
`meta["anchor_orientations"]` 中的阵列朝向：

```python
for modality, cls in [("ranges", iloc.TrilaterationLocalizer), ("tdoa", iloc.TDOALocalizer),
                      ("aoa", iloc.AoALocalizer)]:
    tr, te = iloc.load_dataset("synthetic_office", modality=modality)
    m = cls.from_meta(te.meta)                   # 不需要 fit：几何就是模型
    print(modality, te.X.shape, round(m.evaluate(te).mean_error, 3))
# ranges (200, 4) 0.603
# tdoa (200, 4) 0.48
# aoa (200, 4) 3.545
```

这些模型只需要几何信息，所以 `localize` 在 `fit` 之前就能使用；不带位置的 `fit(X)` 只记录输入维度。
例外是 `calibrate=True`：此时 `fit(X, positions)`（或 `fit(table)`）从有标签的测量中学习每个锚点的偏差和噪声尺度，
必须提供位置。路径损耗模型总要从有标签的扫描中学习噪声尺度（以及未给定的参数），所以它的 `fit` 始终需要位置。

AoA 角度是在各阵列自身的坐标系中测量的。如果手动创建时漏掉朝向（`create_model("aoa", anchors=te.meta["anchors"])`），
同一测试集的平均误差是 74.9 m，而不是 3.5 m。`fit` 拿到位置时，如果某个阵列的角度与这些位置的偏差中位数超过 45 度，会发出警告。

锚点在 3-D 中（近似）共面时（例如装在天花板上的 UWB 锚点），无法区分平面下方的标签和它在平面上方的镜像。
3-D 仿真办公楼的四个测距锚点恰好共面，所以 3-D 求解一个样本也定位不了。已知标签高度时，
`height=` 只求解 `(x, y)`，返回 `(x, y, height)`：

```python
tr3, te3 = iloc.load_dataset("synthetic_office", modality="ranges", dim=3)
for height in (None, 1.2):
    print(height, iloc.TrilaterationLocalizer.from_meta(te3.meta, height=height).evaluate(te3))
# None no sample placed  floor n/a  building n/a  (n=200, 200 not placed)
# 1.2 mean 0.6062  median 0.4958  P90 1.2541  floor n/a  building n/a  (n=200)
```

路径损耗和加权质心方法根据 RSSI 和已知的接入点位置定位。`anchors=None` 时，路径损耗模型还会从有标签的训练扫描中估计发射机位置。

```python
anchors = train.meta["anchors"]
for name in ("pathloss", "centroid"):
    m = iloc.create_model(name, anchors=anchors).fit(train)
    print(name, round(m.evaluate(test).mean_error, 3))
# pathloss 3.068
# centroid 3.645
```

可见光定位从表中读取 LED 布局和光学常数：

```python
from indoorloc.methods.vlc import LambertianLocalizer

vlc_train, vlc_test = iloc.load_dataset("synthetic_office", modality="vlc")
res = LambertianLocalizer.from_meta(vlc_test.meta).fit(vlc_train).evaluate(vlc_test)
print(vlc_test.X.shape, round(res.mean_error, 4), res.n_failed)
# (200, 176) 0.0207 3
```

`n_failed` 统计方法无法定位的样本（这里是可见 LED 存在镜像歧义的位置）。这些样本不计入误差统计，因此报告误差时要同时给出这个数量。

### 序列匹配

`magnetic_dtw` 用子序列动态时间规整，把最近一段地磁特征与参考轨迹匹配。轨迹边界传给 `fit`，`localize_walks` 保证查询窗口不跨越轨迹：

```python
walks = iloc.load_dataset("synthetic_office", split="trajectory", modality="magnetic", n_trajectories=12)
traj = walks.groups["trajectory"]
reference, query = walks[traj < 10], walks[traj >= 10]
dtw = iloc.create_model("magnetic_dtw", window=30).fit(reference, trajectory=reference.groups["trajectory"])
print(iloc.evaluate(query, dtw.localize_walks(query)))
# mean 6.7558  median 1.2264  P90 17.8236  floor 100.00 %  building n/a  (n=1200, 58 not placed)
```

未定位的 58 行是每条查询轨迹最开始、窗口尚未填满时的样本。在实测数据（ILC 2020 样本）上，单独使用地磁匹配的效果差得多，具体数字见该类的 docstring。

### 域适应

CORAL 用无标签目标域扫描的协方差给源域扫描重新“着色”。这里的“新设备”是在仿真测试集上施加了增益和偏移：

```python
from indoorloc.methods import LocalizerPipeline
from indoorloc.methods.transfer import CORAL

fill = FillMissing(-104)
X_src, X_new = fill.transform(train.X), fill.transform(0.8 * test.X - 12.0)    # 仿真的设备差异
plain = iloc.create_model("wknn").fit(X_src, train.pos)
adapted = LocalizerPipeline(CORAL(align_mean=True), iloc.create_model("wknn"))
adapted.fit(X_src, train.pos, preprocess__target=X_new)
print(round(iloc.evaluate(test.pos, plain.predict(X_new)).mean_error, 3),
      round(iloc.evaluate(test.pos, adapted.predict(X_new)).mean_error, 3))
# 2.441 1.934
```
