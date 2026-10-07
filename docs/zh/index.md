# IndoorLoc 0.2 用户指南

IndoorLoc 是一个无线室内定位的 Python 库。它加载公开数据集、预处理信号、运行定位方法、按命名协议评测，
并在此之上提供跟踪与导航。整个库分为五层，层与层之间只传递 numpy 数组，每一层都可以单独使用。

[English](../guide/index.md) · [安装](../installation_zh.md) ·
[从 0.1 迁移](../../MIGRATION.md) · [更新日志](../../CHANGELOG.md) ·
[基准结果](../benchmarks_zh.md) · [开发者契约](../architecture/CONTRACTS.md)

## 五层结构

| 层 | 包 | 输入 | 输出 | 页面 |
| --- | --- | --- | --- | --- |
| L1 数据 | `indoorloc.datasets` | 数据集名称与选项 | `SampleTable` | [数据集](datasets.md) |
| L2 信号 | `indoorloc.signals` | `SampleTable`、数组或单次扫描 | 同类对象（已变换） | [信号](signals.md) |
| L3 方法 | `indoorloc.methods` | `X`（拟合时另加 `y`） | `Prediction` | [方法](methods.md) |
| L4 评测 | `indoorloc.evaluation` | 真值与 `Prediction`（或数组） | `EvaluationResults`、折、界 | [评测](evaluation.md) |
| L5 应用 | `indoorloc.apps` | 定位结果、IMU 采样、扫描流、平面图 | `Prediction`、路线 | [应用](apps.md) |
| 命令行 | `indoorloc.cli` | 数据集与方法描述 | 自描述的 JSON 文件 | [命令行](cli.md) |

导入规则由 import-linter 契约强制执行。`core` 只依赖 numpy 和标准库。L1、L2、L4 彼此独立，L3 位于它们之上，
L5 在最上层。任何层都不导入 L5；L3 和 L5 从不导入 L1，数据以数组或 `SampleTable` 的形式传入。
`import indoorloc` 本身不加载 numpy；每一层都可以在没有 torch、scikit-learn、scipy、pandas 的环境中导入。
较重的依赖是可选 extra，只在需要它的函数内部导入。如何新增数据集、变换或方法，见 [扩展 IndoorLoc](extending.md)。

## 数据如何流动

```text
L1  load_dataset("ujiindoorloc")          -> SampleTable(X, pos, floor, building, groups, ids, meta)
L2  FillMissing(-104).fit_transform(t)    -> SampleTable（行不变，X 已变换）
L3  create_model("wknn").fit(t_train)     -> 拟合好的估计器
    model.localize(t_test)                -> Prediction(pos, floor, building, ids, spread)
L4  evaluate(t_test, prediction)          -> EvaluationResults(mean_error, ..., n_failed)
L5  KalmanTracker().smooth(prediction, t) -> Prediction（与任何方法一样由 L4 评测）
```

两种数据类型定义在 `indoorloc.core` 中。二者都不可变，保存只读的 numpy 数组，第一维是样本：

* `SampleTable(X, pos, floor=None, building=None, groups={}, ids=None, meta={})`。`X` 是物理单位的观测值
  （dBm、复数 CSI、米、弧度），**缺失读数为 NaN**，从不使用哨兵值。`pos` 是 float64 的 `(N, D)`，坐标系由
  `meta["crs"]` 指明。`floor`/`building` 为 int64 或 `None`。`groups` 保存分组划分用的列（`user`、`device`、
  `time`、`trajectory` 等）。`meta` 保存数据集层面的信息（单位、许可、sha256 等）。
* `Prediction(pos, floor=None, building=None, ids=None, spread=None)`。`spread` 是方法自己给出的位置不确定度尺度，
  L5 的跟踪器把它用作观测噪声。

完整规则（包括每种模态的 `X` 布局）见 [CONTRACTS.md](../architecture/CONTRACTS.md)。

## 安装

在 0.2.0 发布到 PyPI 之前，请从仓库源码安装：

```bash
pip install -e .                  # 仅 numpy：所有层、k-NN、基于模型的方法、跟踪器
pip install -e ".[sklearn]"       # + SVM、随机森林、极端随机树、梯度提升
pip install -e ".[deep]"          # + MLP / CNN1D / timm 骨干网络定位器（torch、timm）
pip install -e ".[full]"          # 以上全部，另加 pandas、matplotlib、scipy、h5py
```

[安装页面](../installation_zh.md) 列出了全部 extra 以及各自的用途。

## 五分钟上手

先在内置的仿真办公楼上走一遍（无需下载），再在真实数据集上重复同样的步骤。

**1. 加载数据集（L1）。** 不指定 `split` 时，`load_dataset` 返回官方的 `(train, test)`；只有一张表的数据集直接返回这张表。

```python
import indoorloc as iloc

train, test = iloc.load_dataset("synthetic_office")   # 仿真数据，由 seed 0 生成
print(train.X.shape, train.meta["modality"], train.meta["units"])
# (830, 8) wifi_rssi dBm
print(sorted(train.groups), train.meta["crs"])
# ['point', 'room', 'source'] local
```

**2. 选择预处理与方法（L2 + L3）。** 缺失读数是 NaN，而 k-NN 需要完整的向量，所以用 `FillMissing(-104)` 把 NaN 替换为
-104 dBm。把它作为 `preprocess=` 传入，它就成为模型的一部分：只在训练数据上拟合，并作用于之后的每一个输入。

```python
model = iloc.create_model("wknn", k=5, preprocess=iloc.FillMissing(-104))
model.fit(train)                      # SampleTable 同时提供 X、pos、floor 和 building
pred = model.localize(test)           # Prediction：pos (200, 2)、floor、spread
print(pred.pos.shape, pred.floor[:3], pred.spread[:2].round(2))
# (200, 2) [0 0 0] [0.99 1.79]
```

**3. 评测（L4）。**

```python
res = iloc.evaluate(test, pred)       # 等价于 model.evaluate(test)
print(res)
# mean 1.8055  median 1.3816  P90 3.4247  floor 100.00 %  building n/a  (n=200)
```

**4. 跟踪一条行走轨迹（L5）。** 仿真器还会生成每秒扫描一次的行走轨迹。对 WKNN 定位结果做 Kalman 平滑时，
每个结果的 `spread` 被用作观测噪声：

```python
walks = iloc.load_dataset("synthetic_office", split="trajectory")
walk = walks[walks.groups["trajectory"] == 0]
fixes = model.localize(walk)
smoothed = iloc.KalmanTracker().smooth(fixes, t=walk.groups["time"])
print(round(iloc.evaluate(walk, fixes).mean_error, 2), round(iloc.evaluate(walk, smoothed).mean_error, 2))
# 2.97 2.11
```

这些数字描述的是一座仿真办公楼，不能说明真实建筑中的表现。

**5. 在真实数据上重复。** UJIIndoorLoc 在首次使用时下载（UCI 的 zip 压缩包，其中两个 CSV 文件均校验 sha256）：

```python
# data: ujiindoorloc
train, test = iloc.load_dataset("ujiindoorloc")
model = iloc.create_model("wknn", k=5, preprocess=iloc.FillMissing(-104)).fit(train)
print(model.evaluate(test))
# mean 8.7937  median 5.3546  P90 19.1537  floor 90.46 %  building 99.73 %  (n=1111)
```

UJIIndoorLoc 的坐标是 Web Mercator（EPSG:3857）米，该数据集的文献通常也用这个单位报告结果。
`model.evaluate(test, scale=test.meta["ground_scale"])` 改为报告地面米（比例 0.7661）。8.7937 与
[docs/benchmarks_zh.md](../benchmarks_zh.md) 中的基准表一致；那里的每个单元格都通过命令行运行：

```bash
indoorloc benchmark --dataset ujiindoorloc --method knn --method wknn --out uji.json
indoorloc report uji.json --format markdown
```

## 单独使用某一层

每一层都接受普通数组，因此可以接入其他来源的数据和代码。

```python
import numpy as np
from indoorloc.evaluation import evaluate
from indoorloc.methods import create_model

rng = np.random.default_rng(0)
X = rng.normal(-70, 8, size=(300, 6))          # 你自己的 RSSI 矩阵，dBm
y = rng.uniform(0, 20, size=(300, 2))          # 你自己的位置，米
model = create_model("knn", k=3).fit(X[:250], y[:250])     # L3 直接用数组
print(evaluate(y[250:], model.predict(X[250:])).n)         # L4 直接用数组
# 50
```

`SampleTable` 还可以导出为 numpy（`table.to_numpy()`）、pandas（`table.to_dataframe()`）和 torch
（`table.to_torch()`，或 `indoorloc.datasets.torch_adapter.make_dataloader`），示例见 [数据集页面](datasets.md#在其他代码中使用数据)。

## 接下来

* [数据集](datasets.md)：目录、选项、坐标系、导出与绘图。
* [信号](signals.md)：RSSI、CSI、测距、IMU、地磁和可见光的变换与信号函数。
* [方法](methods.md)：方法目录、估计器契约、保存与加载。
* [评测](evaluation.md)：指标、协议、竞赛评分、性能界与已发表结果。
* [应用](apps.md)：跟踪、PDR、粒子滤波、流式定位与导航。
* [命令行](cli.md)：`indoorloc list | info | benchmark | literature | evaluate | report`。
* [扩展](extending.md)：新增数据集、变换、方法或协议。

[`examples/`](../../examples) 中的完整脚本端到端地演示各层；每个文件的 docstring 说明它测量什么、需要下载哪些数据：

| 脚本 | 内容 | 数据 |
| --- | --- | --- |
| [`quickstart.py`](../../examples/quickstart.py) | 加载、拟合流水线、评测、保存与重新加载 | UJIIndoorLoc（`--dataset synthetic_office` 无需下载） |
| [`01_fingerprinting_benchmark.py`](../../examples/01_fingerprinting_benchmark.py) | 同一协议下的七种经典指纹方法与一个 MLP，误差 CDF | UJIIndoorLoc |
| [`02_model_based_vs_crlb.py`](../../examples/02_model_based_vs_crlb.py) | 基于模型的估计器与 Cramér-Rao 界的比较（蒙特卡洛） | 仿真 |
| [`03_csi_pipeline.py`](../../examples/03_csi_pipeline.py) | 原始与净化后的 CSI 相位，CSI 指纹 + WKNN | HALOC |
| [`04_tracking_and_fusion.py`](../../examples/04_tracking_and_fusion.py) | 手机轨迹上的 WiFi 定位、Kalman 平滑、PDR 与粒子滤波融合 | ILC 2020 样本（site1/F1） |
| [`05_domain_adaptation.py`](../../examples/05_domain_adaptation.py) | 跨设备指纹定位，有无域自适应的对比 | UJIIndoorLoc |
| [`06_navigation.py`](../../examples/06_navigation.py) | 三层楼平面图上的 A* 路径、楼梯与电梯、转向提示 | 仿真平面图 |
| [`dataset_distribution.py`](../../examples/dataset_distribution.py) | 样本分布、密度图、误差 CDF 与误差分布图 | 仿真或任意数据集 |

本指南中的代码块由 `python docs/catalog.py --snippets` 执行。以 `# data: <name>` 开头的代码块只在该数据集已在本地时运行，
注释中的输出来自这些运行。目录表格由 `python docs/catalog.py` 根据注册表重新生成。
