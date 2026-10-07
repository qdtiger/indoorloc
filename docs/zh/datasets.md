# L1 数据集

`indoorloc.datasets` 把公开数据集和仿真器转换为由 numpy 数组组成的 `SampleTable`。它只导入 numpy 和标准库；
读取 `.mat` 与 `.h5` 文件的加载器在函数内部导入 scipy 或 h5py（extra `[datasets]`）。

[指南首页](index.md) · [English](../guide/datasets.md)

## 加载

```text
load_dataset(name, split=None, *, root=None, download=True, verify=True, **options)
```

* `split=None`：数据集同时有 train 和 test 时返回官方的 `(train, test)`，否则返回其唯一的表（`"all"`）。
  `split="test"` 返回一张表，传入划分名的元组则返回多张表。`"validation"` 等别名会映射到数据集自己的划分名。
* 文件缓存在 `$INDOORLOC_DATA/<name>`，默认 `~/.cache/indoorloc/datasets/<name>`；也可以用 `root=` 指定目录。
  文件缺失时自动下载（`download=False` 禁止下载）。除非 `verify=False`，每个文件都会与加载器中记录的 sha256 核对，
  表的 `meta["sha256"]` 记录这些摘要。
* `options` 是数据集特有的选项（建筑、月份、房间、模态、随机种子），见下方选项表。
* `list_datasets()` 返回注册名，包括用 `register_dataset(name, cls)` 添加的名称（见[扩展](extending.md#数据集)）；
  `load_dataset` 也接受 `"package.module:Class"` 路径或 `Dataset` 子类本身。
  `dataset_info(name)` 在不加载数据的情况下返回类级信息（模态、坐标系、许可、DOI、划分）。
  `indoorloc info <name>` 还会列出文件以及文件是否已在本地（`--json` 另外给出 sha256）。

```python
import indoorloc as iloc
from indoorloc.datasets import dataset_info, list_datasets

print(len(list_datasets()), "ujiindoorloc" in list_datasets())
# 15 True
info = dataset_info("tuji1")
print(info["modality"], info["crs"], info["license"], info["splits"])
# wifi_rssi local CC BY 4.0 ('train', 'test')
```

## 表中有什么

| 字段 | 内容 |
| --- | --- |
| `X` | `(N, ...)` 物理单位的观测：RSSI 为 dBm，CSI 为复数，测距为米，角度为弧度。缺失读数为 **NaN**：加载器把文件中的哨兵值（UJIIndoorLoc 的 100、UCI BLE 文件的 -200）转换为 NaN，并把原值记录在 `meta["raw_missing_value"]`。不做任何归一化。 |
| `pos` | float64 `(N, D)`，数据集自己的坐标系，从不缩放。`meta["crs"]` 指明坐标系，`meta["pos_units"]` 指明单位。 |
| `floor`、`building` | int64 `(N,)`；数据集没有该标签时为 `None`。负楼层是真实楼层（`B1` = -1）。 |
| `groups` | 分组划分用的列：`user`、`device`、`time`、`trajectory`、`month`、`point`、`room` 等。某个划分中取值未知的列会被省略，并列入 `meta["unknown_groups"]`。 |
| `ids` | 稳定的样本 id。 |
| `meta` | `name`、`split`、`sha256`、`source_files`、`modality`、`units`、`crs`、`pos_units`、`feature_names`、`license`、`doi`、`citation`，以及 `ground_scale`、`anchors`、`floor_plan` 等数据集特有信息。 |

每种模态（RSSI、CSI、测距、TDoA、AoA、IMU）的 `X` 布局见 [CONTRACTS.md §2](../architecture/CONTRACTS.md#2-the-two-data-types-indoorloccore)。

## 目录

由 `python docs/catalog.py` 从注册表生成。行数取自基准矩阵加载真实文件时的记录（[docs/benchmarks_zh.md](../benchmarks_zh.md)）；
仿真器的行数来自生成其默认表，ILC 2020 的数量来自其文件清单：`ilc2020` 读取的是组织者随竞赛代码发布的样本轨迹（两座商场），而不是完整的竞赛数据。表中的引用信息保持原文。

<!-- catalog:datasets -->
**WiFi RSSI**

| 名称 | 模态 | 行数 | X 单位 | 坐标 | 许可 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| [`longtermwifi`](../../indoorloc/datasets/longtermwifi.py) | `wifi_rssi` | train 23,040 · test 81,120 | dBm | m | CC BY 4.0 (data, Readme.txt); MIT (scripts) | Mendoza-Silva et al., Long-Term WiFi Fingerprinting Dataset for Research on Robust Indoor Positioning, Data 3(1):3, 2018, doi:10.3390/data3010003. [doi:10.5281/zenodo.3748719](https://doi.org/10.5281/zenodo.3748719) |
| [`sodindoorloc`](../../indoorloc/datasets/sodindoorloc.py)<br>别名: `sod` | `wifi_rssi` | train 21,205 · test 2,720 | dBm | m; crs `local-per-building` | not stated in the repository; cite the paper | Bi et al., Supplementary open dataset for WiFi indoor localization based on received signal strength, Satellite Navigation 3:25, 2022. [doi:10.1186/s43020-022-00086-y](https://doi.org/10.1186/s43020-022-00086-y) |
| [`tampere`](../../indoorloc/datasets/tampere.py) | `wifi_rssi` | train 697 · test 3,951 | dBm | m | CC BY 4.0 (data, FINGERPRINTING_DB/README.txt); MIT (software) | Lohan et al., Wi-Fi Crowdsourced Fingerprinting Dataset for Indoor Positioning, Data 2(4):32, 2017, doi:10.3390/data2040032. [doi:10.5281/zenodo.889798](https://doi.org/10.5281/zenodo.889798) |
| [`tuji1`](../../indoorloc/datasets/tuji1.py) | `wifi_rssi` | train 6,752 · test 2,147 | dBm | m | CC BY 4.0 | Klus et al., TUJI1 Dataset: Multi-device dataset for indoor localization with high measurement density, Data in Brief 54:110356, 2024, doi:10.1016/j.dib.2024.110356. [doi:10.5281/zenodo.7641701](https://doi.org/10.5281/zenodo.7641701) |
| [`ujiindoorloc`](../../indoorloc/datasets/ujiindoorloc.py)<br>别名: `uji` | `wifi_rssi` | train 19,937 · test 1,111 | dBm | m (Web Mercator, not ground metres); crs `EPSG:3857` | CC BY 4.0 | Torres-Sospedra et al., UJIIndoorLoc, IPIN 2014. [doi:10.24432/C5MS59](https://doi.org/10.24432/C5MS59) |
| [`wlanrssi`](../../indoorloc/datasets/wlanrssi.py) | `wifi_rssi` | all 2,000 | dBm | — (room_classification) | CC BY 4.0 | Bhatt, Wireless Indoor Localization, UCI Machine Learning Repository, 2017, doi:10.24432/C51880. [doi:10.24432/C51880](https://doi.org/10.24432/C51880) |

**BLE RSSI**

| 名称 | 模态 | 行数 | X 单位 | 坐标 | 许可 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| [`ble_indoor`](../../indoorloc/datasets/ble_indoor.py) | `ble_rssi` | room=office: train 22,237 · valid 3,598 · test 5,110<br>room=lab: train 13,238 · valid 2,686 · test 3,194 | dBm | m; crs `local (one frame per room; see building)` | MIT | Kennedy, Spachos, Taylor, BLE beacon indoor localization dataset, Scholars Portal Dataverse, 2019. [doi:10.5683/SP2/UTZTFT](https://doi.org/10.5683/SP2/UTZTFT) |
| [`ble_rssi_uci`](../../indoorloc/datasets/ble_rssi_uci.py) | `ble_rssi` | all 1,420 · unlabeled 5,191 | dBm | grid cells of the source map (column letter A=1, row number counted downward); cell size not stated | CC BY 4.0 | Mohammadi, Al-Fuqaha, Guizani, Oh, Semisupervised Deep Reinforcement Learning in Support of IoT and Smart City Services, IEEE Internet of Things Journal 5(2), 2018. [doi:10.24432/C54G80](https://doi.org/10.24432/C54G80) |
| [`ibeacon_rssi`](../../indoorloc/datasets/ibeacon_rssi.py) | `ble_rssi` | train 2,748 · test 1,860 · all 4,752 | dBm | m; crs `local (one frame per zone; see building)` | CC BY 4.0 (data), MIT (scripts) | Mendoza-Silva, Matey-Sanz, Torres-Sospedra, Huerta, BLE RSS Measurements Dataset for Research on Accurate Indoor Positioning, Data 4(1), 12, 2019. [doi:10.5281/zenodo.1618692](https://doi.org/10.5281/zenodo.1618692) |

**WiFi CSI**

| 名称 | 模态 | 行数 | X 单位 | 坐标 | 许可 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| [`csi_fingerprint`](../../indoorloc/datasets/csi_fingerprint.py) | `csi_amp` | area=lab, packets=50: all 15,850<br>area=meeting, packets=50: all 8,800<br>area=conference, packets=50: all 8,000<br>area=minilab, packets=50: all 1,750 | dB | grid steps of the area's reference-point grid (spacing not stated by the source); crs `local grid (one per area; see building)` | MIT | Zhu, Qiu, Qu, Zhou, Atiquzzaman, Wu, BLS-Location: A Wireless Fingerprint Localization Algorithm Based on Broad Learning, IEEE TMC 22(1), 2023. [doi:10.1109/TMC.2021.3073005](https://doi.org/10.1109/TMC.2021.3073005) |
| [`haloc`](../../indoorloc/datasets/haloc.py) | `csi` | train 96,491 · valid 28,111 · test 14,277 · all 138,879 | raw int8 I/Q of ESP-IDF (not calibrated) | m | CC BY 4.0 (Zenodo record; the description asks for non-commercial research use) | Strohmayer, Kampel, WiFi CSI-based Long-Range Person Localization Using Directional Antennas, ICLR 2024 Tiny Papers. [doi:10.5281/zenodo.10715595](https://doi.org/10.5281/zenodo.10715595) |
| [`hwild`](../../indoorloc/datasets/hwild.py) | `csi` | environment=conference: all 22,970<br>environment=laboratory: all 26,833<br>environment=office: all 26,935<br>environment=lounge: all 42,554 | Intel 5300 scaled CSI as stored (not calibrated) | m; crs `local (one frame per room; see building)` | not stated by the repository (cite the RLoc paper) | Zhang, Zhang, Wang, Li, Hu, Sun, Chen, RLoc: Towards Robust Indoor Localization by Quantifying Uncertainty, Proc. ACM IMWUT 7(4), 2023. [doi:10.1145/3631437](https://doi.org/10.1145/3631437) |

**多传感器轨迹**

| 名称 | 模态 | 行数 | X 单位 | 坐标 | 许可 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| [`ilc2020`](../../indoorloc/datasets/ilc2020.py) | **`wifi_rssi`**, `ble_rssi`, `imu`, `waypoints` | 1,095 条轨迹，14 个楼层 | dBm | m | MIT | Hu, Fan, Yin, Qian, Ji, Shu, Han, Xu, Liu, Bahl, The Wisdom of 1,170 Teams: Lessons and Experiences from a Large Indoor Localization Competition, ACM MobiCom 2023. [doi:10.1145/3570361.3592507](https://doi.org/10.1145/3570361.3592507) |

**仿真**

| 名称 | 模态 | 行数 | X 单位 | 坐标 | 许可 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| [`deepmimo`](../../indoorloc/datasets/simulated/deepmimo.py) | **`rssi`**, `ranges`, `tdoa`, `aoa`, `csi` | 随场景而定 | — | m | per scenario (see deepmimo.net) | Alkhateeb, DeepMIMO: A generic deep learning dataset for millimeter wave and massive MIMO applications, ITA 2019. [arXiv:1902.06435](https://arxiv.org/abs/1902.06435) |
| [`synthetic_office`](../../indoorloc/datasets/simulated/office.py) | **`wifi_rssi`**, `ble_rssi`, `ranges`, `tdoa`, `aoa`, `csi`, `imu`, `vlc`, `magnetic` | 默认（seed 0）：train 830 · test 200 · trajectory 240 | dBm | m | CC0-1.0 (generated data) | IndoorLoc SyntheticOffice simulator (indoorloc.datasets.simulated). <https://github.com/qdtiger/indoorloc> |

粗体为默认 `modality`. 行数取自基准运行记录 (`benchmarks/results/*.json`).
<!-- /catalog:datasets -->

划分与构造选项：

<!-- catalog:dataset-options -->
| 名称 | 划分（别名） | 选项（`load_dataset(name, **options)`） |
| --- | --- | --- |
| `ble_indoor` | `train`, `valid`, `test` (`validation`→`valid`, `val`→`valid`) | `room='all'` |
| `ble_rssi_uci` | `all`, `unlabeled` (`labeled`→`all`, `labelled`→`all`, `unlabelled`→`unlabeled`) | — |
| `csi_fingerprint` | `all` | `area='all'`, `packets=None` |
| `deepmimo` | `all` | `scenario='asu_campus_3p5'`, `modality='rssi'`, `tx_sets='all'`, `rx_sets='rx_only'`, `max_paths=25`, `dim=3`, `power_offset_db=0.0`, `sensitivity_dbm=None`, `coherent=False`, `n_antennas=8`, `spacing_wavelengths=0.5`, `orientation=0.0`, `subcarriers=<56 values>`, `subcarrier_spacing_hz=312500.0` |
| `haloc` | `train`, `valid`, `test`, `all` (`validation`→`valid`, `val`→`valid`) | `sequences=None`, `subcarriers='lltf'` |
| `hwild` | `all` | `environment='all'`, `users=None`, `interference=None`, `features='csi'` |
| `ibeacon_rssi` | `all`, `train`, `test` | `zone='all'`, `protocol=None` |
| `ilc2020` | `all` | `site='site1'`, `floor='F1'`, `modality='wifi'`, `outside_waypoints='drop'`, `wifi_max_age=2.0`, `ble_window=None` |
| `longtermwifi` | `train`, `test` | `month=None` |
| `sodindoorloc` | `train`, `test` | `building=None`, `macs='all'`, `averaged=False` |
| `synthetic_office` | `train`, `test`, `trajectory` (`trajectories`→`trajectory`, `walk`→`trajectory`) | `seed=0`, `modality='wifi_rssi'`, `n_floors=1`, `n_aps=None`, `grid_spacing=2.0`, `samples_per_point=5`, `n_test=200`, `n_trajectories=4`, `trajectory_duration=60.0`, `dim=2`, `size=(40.0, 20.0)`, `path_loss='multiwall'`, `noise_std=None`, `shadowing_std_db=None`, `n_antennas=4`, `n_scatterers=0`, `scan_rate_hz=None`, `imu_rate_hz=50.0`, `physics=None` |
| `tampere` | `train`, `test` | — |
| `tuji1` | `train`, `test` | — |
| `ujiindoorloc` | `train`, `test` (`validation`→`test`, `val`→`test`) | — |
| `wlanrssi` | `all` | — |
<!-- /catalog:dataset-options -->

## 坐标系

每个加载器都保留数据源的坐标并注明其坐标系。有四种情况需要注意：

* **UJIIndoorLoc** 保存的是 Web Mercator（EPSG:3857）东向、北向坐标。在该校园，一个 Mercator 米等于
  `meta["ground_scale"]` = 0.7661 地面米。该数据集的结果通常以 Mercator 米发表；`evaluate(..., scale=meta["ground_scale"])`
  换算为地面米。
* **每栋建筑或每个房间一个坐标系**（SODIndoorLoc、BBIL `ble_indoor`、iBeacon RSSI、H-WILD、CSI 指纹的各房间、ILC 2020 的各楼层）。
  不同建筑或房间的坐标不可比较，只有建筑（或房间）判断正确时，位置误差才有意义。
* **网格单位。** `ble_rssi_uci` 的坐标是地图格子，`csi_fingerprint` 的坐标是参考点网格步长，数据源都没有给出格子尺寸。
* **没有坐标。** `wlanrssi` 只有房间标签（`groups["room"]`），`pos` 的形状是 `(N, 0)`，应评测房间标签而不是位置。

```python
# data: ujiindoorloc
import indoorloc as iloc

train, test = iloc.load_dataset("ujiindoorloc")
print(train.X.shape, train.X.dtype, test.meta["crs"], round(test.meta["ground_scale"], 4))
# (19937, 520) float32 EPSG:3857 0.7661
print(sorted(test.groups), test.meta["unknown_groups"])
# ['device', 'time'] ('user', 'space', 'relative_position')
```

## 选项与单表数据集

```python
# data: sodindoorloc
train, test = iloc.load_dataset("sodindoorloc", building="HCXY")    # 三栋建筑之一
print(len(train), len(test), train.X.shape[1], train.meta["crs"])
# 11370 860 347 local-per-building
```

没有官方划分的数据集加载为一张表。推荐的划分协议写在其 docstring 中，用 [L4 协议](evaluation.md#协议) 执行：

```python
# data: ble_rssi_uci
from indoorloc.evaluation import get_protocol

table = iloc.load_dataset("ble_rssi_uci")       # 有标签的文件，划分 "all"
fold = get_protocol("random-80-20").folds(table, random_state=0)[0]
train, test = table[fold.train], table[fold.test]
print(len(table), len(train), len(test), table.meta["pos_units"].split(" (")[0])
# 1420 1136 284 grid cells of the source map
```

ILC 2020 加载器每张表读取一个场地、一个或多个楼层、一种模态（`"wifi"`、`"ble"`、`"imu"` 或 `"waypoints"`），
平面图保存在 `meta["floor_plan"]`：

```python
# data: ilc2020
wifi = iloc.load_dataset("ilc2020", site="site1", floor="F1", modality="wifi")
print(wifi.X.shape, sorted(wifi.groups), sorted(wifi.meta["floor_plan"])[:3])
# (2223, 2330) ['device', 'time', 'trajectory'] ['bounds', 'materials', 'wall_floor']
```

## 仿真办公楼

`synthetic_office` 根据 `seed` 生成一座带墙体的走廊式办公楼、每层的锚点以及其中的行走轨迹（许可 CC0）。
它能生成库中处理的每一种模态：`wifi_rssi`、`ble_rssi`、`ranges`（UWB ToA）、`tdoa`、`aoa`、`csi`、`imu`、`vlc` 和 `magnetic`。
锚点几何在 `meta["anchors"]`，平面图在 `meta["floor_plan"]`。它用于测试、教程和受控实验；其结果描述的是仿真，而不是真实建筑。
`modality="imu"` 只有 `trajectory` 一个划分，`split=None` 加载的也是它。

```python
ranges = iloc.load_dataset("synthetic_office", split="test", modality="ranges", n_floors=2, seed=1)
print(ranges.X.shape, ranges.meta["anchors"].shape, ranges.meta["units"], sorted(set(ranges.floor.tolist())))
# (200, 8) (8, 2) m [0, 1]
walk = iloc.load_dataset("synthetic_office", modality="imu")     # 划分 "trajectory"
print(walk.X.shape, walk.meta["rate_hz"], walk.meta["channels"][:3])
# (12000, 6) 50.0 ('acc_x', 'acc_y', 'acc_z')
```

`deepmimo` 把 DeepMIMO v4 射线追踪场景读成 RSSI、测距、AoA 或 CSI 表，需要 `deepmimo` 包（extra `[sim]`，Python 3.11 及以上）和场景文件。

## 在其他代码中使用数据

```python
import numpy as np

train, test = iloc.load_dataset("synthetic_office")
X, pos = train.to_numpy()                          # 只读视图，不复制
first_floor = train[train.floor == 0]              # 选取行：布尔掩码、下标或切片
filled = train.replace(X=np.nan_to_num(train.X, nan=-104.0))
other = iloc.load_dataset("synthetic_office", split="train", seed=1)
other = other.replace(ids=np.char.add("seed1-", other.ids.astype(str)))   # id 必须保持唯一
both = iloc.SampleTable.concat([train, other])
print(X.shape, len(first_floor), np.isnan(filled.X).any(), len(both))
# (830, 8) 830 False 1665
```

```python
# requires: pandas
df = train.to_dataframe()                          # 特征、坐标、标签和分组列
print(df.shape, list(df.columns[-5:]))
# (830, 14) ['y', 'floor', 'source', 'point', 'room']
```

```python
# requires: torch
from indoorloc.datasets.torch_adapter import make_dataloader

loader = make_dataloader(train, batch_size=256, shuffle=True, seed=0)
batch = next(iter(loader))
print(sorted(batch), tuple(batch["X"].shape), batch["X"].dtype)
# ['X', 'floor', 'groups.point', 'groups.room', 'groups.source', 'ids', 'pos'] (256, 8) torch.float32
```

## 绘图

`indoorloc.datasets.plot` 绘制一张或多张表的样本分布：每层（或每个建筑坐标系）一个子图、楼层叠放的三维视图或密度图，
底层叠加 `meta["floor_plan"]` 中的墙体。需要 matplotlib（extra `[plot]`）。`distribution_html` 用 plotly 生成交互式 HTML 页面，
plotly 同样由 `[plot]` extra 安装。

```python
# requires: matplotlib
from indoorloc.datasets.plot import plot_distribution

fig = plot_distribution({"train": train, "test": test}, color_by="split")
fig.savefig("office_distribution.png", dpi=120)
```

`examples/dataset_distribution.py --dataset ujiindoorloc` 为一个数据集输出一组标准图：二维与三维分布、密度图，
以及 k-NN 和 WKNN 的误差 CDF 与误差分布图。
