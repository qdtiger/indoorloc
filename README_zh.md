<div align="center">

<img src="assets/logo.png" width="600">

**IndoorLoc | 室内定位工具库**

*集数据集、算法与评测于一体的室内无线定位研究框架*

[![PyPI](https://img.shields.io/pypi/v/indoorloc)](https://pypi.org/project/indoorloc/)
[![Downloads](https://static.pepy.tech/badge/indoorloc)](https://pepy.tech/project/indoorloc)
[![CI](https://github.com/qdtiger/indoorloc/actions/workflows/ci.yml/badge.svg)](https://github.com/qdtiger/indoorloc/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://qdtiger.github.io/indoorloc/)
[![Python](https://img.shields.io/pypi/pyversions/indoorloc)](https://pypi.org/project/indoorloc/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Stars](https://img.shields.io/github/stars/qdtiger/indoorloc?style=social)](https://github.com/qdtiger/indoorloc)

[📘 文档](https://qdtiger.github.io/indoorloc/) |
[🧱 架构](#简介) |
[🛠️ 安装](#安装) |
[🚀 快速开始](#快速开始) |
[🗂️ 逐层说明](#逐层说明) |
[🗺️ 路线图](#路线图) |
[🤝 贡献](#贡献)

[English](README.md) | [中文](README_zh.md)

</div>

---

```python
import indoorloc as iloc

train, test = iloc.load_dataset("ujindoorloc")   # 自动下载，统一格式
model = iloc.create_model("wknn", k=5)           # 传统 ML 或深度（timm）模型
results = model.fit(train).evaluate(test)        # 统一评测
print(results)                                   # 平均/中位误差 · 楼层/建筑准确率
```

## 简介

室内定位研究长期存在结果难以复现、方法难以横向比较的问题：多数论文不公开代码；公开数据集大多缺少统一的训练/测试划分；不同文献的评测口径各异，精度数字难以直接对比。社区中已有大量部署系统、采集工具和单篇论文的配套代码，缺少的是一个将数据、算法与评测统一起来的研究框架。

IndoorLoc 的目标正是补上这一层。整个库组织为五层，并且**每一层都可以独立使用**：只取数据集，或用本库模型训练自有数据，或用本库指标评测自有模型。

<p align="center">
  <img src="assets/architecture_zh.png" width="820"><br>
  <b>图</b>：五层架构。实心为已实现，空心为规划中；每一层都提供标准格式的输入与输出接口。
</p>

<details open>
<summary>主要特性</summary>

- **统一数据**：数据集统一注册、自动下载，WiFi / BLE / CSI 采用一致的样本格式
- **统一算法**：传统机器学习与深度模型共用一套 `fit / predict / evaluate` 接口
- **统一评测**：指标只有一份实现（含楼层、建筑准确率），结果可与文献数值直接对照
- **配置化复现**：实验由 YAML 配置驱动，支持 `_base_` 继承，任何已报告的结果都可由单条命令复现
- **按需取用**：数据集可导出为标准格式，算法可直接训练自有数据，评测可作用于自有模型的预测结果

</details>

## 最新进展

- **2026-09-03**：发布五层架构、各层独立使用方式，以及包含复现契约的公开[开发规划](docs/DEVELOPMENT_PLAN.md)。
- **2025-12-31**：[文档站](https://qdtiger.github.io/indoorloc/datasets_zh.html)为全部已集成数据集补齐文献基准表。
- **2025-12-17**：数据集注册表收敛为 12 个已集成数据集，并公布待接入清单。
- **2025-12-16**：`indoorloc` 0.1.3 发布至 [PyPI](https://pypi.org/project/indoorloc/)。
- **2025-11-26**：首个公开版本（0.1.0）。

## 总览

每一项均链接到对应实现。

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td><b>数据集（12）</b></td>
      <td><b>定位器（8）</b></td>
      <td><b>骨干（3）· 预测头（7）</b></td>
      <td><b>评测（9 项指标）</b></td>
    </tr>
    <tr valign="top">
      <td>
        <b>WiFi RSSI</b>
        <ul>
          <li><a href="indoorloc/datasets/ujindoorloc.py">UJIndoorLoc</a></li>
          <li><a href="indoorloc/datasets/sodindoorloc.py">SODIndoorLoc</a></li>
          <li><a href="indoorloc/datasets/longtermwifi.py">LongTermWiFi</a></li>
          <li><a href="indoorloc/datasets/tampere.py">Tampere</a></li>
          <li><a href="indoorloc/datasets/wlanrssi.py">WLANRSSI</a></li>
          <li><a href="indoorloc/datasets/tuji1.py">TUJI1</a></li>
        </ul>
        <b>BLE</b>
        <ul>
          <li><a href="indoorloc/datasets/ble_indoor.py">BLEIndoor</a></li>
          <li><a href="indoorloc/datasets/ibeacon_rssi.py">iBeaconRSSI</a></li>
          <li><a href="indoorloc/datasets/ble_rssi_uci.py">BLE RSSI UCI</a></li>
        </ul>
        <b>CSI</b>
        <ul>
          <li><a href="indoorloc/datasets/csi_fingerprint.py">CSI Fingerprint</a></li>
          <li><a href="indoorloc/datasets/hwild.py">HWILD</a></li>
          <li><a href="indoorloc/datasets/haloc.py">HALOC</a></li>
        </ul>
      </td>
      <td>
        <b>指纹方法</b>
        <ul>
          <li><a href="indoorloc/localizers/fingerprint/knn.py">KNN</a></li>
          <li><a href="indoorloc/localizers/fingerprint/knn.py">WKNN</a></li>
          <li><a href="indoorloc/localizers/fingerprint/traditional.py">SVM</a></li>
          <li><a href="indoorloc/localizers/fingerprint/traditional.py">Random Forest</a></li>
        </ul>
        <b>集成融合</b>
        <ul>
          <li><a href="indoorloc/localizers/fusion.py">Ensemble</a></li>
          <li><a href="indoorloc/localizers/fusion.py">Stacking</a></li>
        </ul>
        <b>迁移学习</b>
        <ul>
          <li><a href="indoorloc/localizers/transfer.py">TransferLocalizer</a>（SKADA：CORAL · TCA · KMM）</li>
        </ul>
        <b>深度模型</b>
        <ul>
          <li><a href="indoorloc/models/localizers/deep_localizer.py">DeepLocalizer</a>（骨干 × 预测头）</li>
        </ul>
      </td>
      <td>
        <b>骨干网络</b>
        <ul>
          <li><a href="indoorloc/models/backbones/mlp.py">MLP</a></li>
          <li><a href="indoorloc/models/backbones/cnn1d.py">CNN1D</a></li>
          <li><a href="indoorloc/models/backbones/timm_wrapper.py">timm 封装</a>（ResNet · EfficientNet · ViT ……）</li>
        </ul>
        <b>预测头</b>
        <ul>
          <li><a href="indoorloc/models/heads/regression.py">坐标回归</a></li>
          <li><a href="indoorloc/models/heads/regression.py">多尺度回归</a></li>
          <li><a href="indoorloc/models/heads/classification.py">通用分类</a></li>
          <li><a href="indoorloc/models/heads/classification.py">楼层</a></li>
          <li><a href="indoorloc/models/heads/classification.py">建筑</a></li>
          <li><a href="indoorloc/models/heads/hybrid.py">混合</a></li>
          <li><a href="indoorloc/models/heads/hybrid.py">层次化</a></li>
        </ul>
      </td>
      <td>
        <b>指标</b>
        <ul>
          <li><a href="indoorloc/evaluation/metrics.py">平均 / 中位 / RMS / 最大定位误差</a></li>
          <li><a href="indoorloc/evaluation/metrics.py">分位误差（P75、P90 ……）</a></li>
          <li><a href="indoorloc/evaluation/metrics.py">楼层 · 建筑 · 楼层+建筑准确率</a></li>
          <li><a href="indoorloc/evaluation/metrics.py">CDF 分析</a></li>
        </ul>
        <b>对照</b>
        <ul>
          <li><a href="indoorloc/evaluation/benchmarks.py">文献结果表</a>（文献报告值，明确标注）</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

## 安装

### GPU（CUDA 11.8）

```bash
conda create -n indoorloc python=3.10 pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda activate indoorloc
pip install "indoorloc[full]"
```

### 仅 CPU

```bash
conda create -n indoorloc python=3.10 pytorch torchvision cpuonly -c pytorch -y
conda activate indoorloc
pip install "indoorloc[full]"
```

### 自检（可选）

```bash
python -c "import indoorloc, torch; print('indoorloc', indoorloc.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
```

更多安装方式见 `docs/installation_zh.md`。

## 快速开始

### Python API（推荐）

```python
import indoorloc as iloc

train, test = iloc.load_dataset("ujindoorloc")            # 任意已集成数据集 ID
model = iloc.create_model("resnet18", dataset=train)      # timm 骨干，自动配置
results = model.fit(train).evaluate(test)
```

### YAML 配置 + 命令行

配置模板在 `indoorloc/configs/` 下。

```bash
indoorloc-train indoorloc/configs/wifi/resnet18_ujindoorloc.yaml

# 覆盖任意参数（布尔值需写 Python 字面量 True/False）
indoorloc-train indoorloc/configs/wifi/resnet18_ujindoorloc.yaml \
  --model.backbone.model_name efficientnet_b0 \
  --train.lr 5e-4 --train.epochs 200
```

```yaml
# indoorloc/configs/wifi/resnet18_ujindoorloc.yaml（节选）
_base_:
  - ../_base_/default.yaml
  - ../_base_/datasets/ujindoorloc.yaml
  - ../_base_/models/resnet.yaml
  - ../_base_/schedules/schedule_1x.yaml

model:
  backbone: {model_name: resnet18, pretrained: true, input_type: '1d'}
  head:     {type: HybridHead, num_coords: 2, num_floors: 5, num_buildings: 3}

train: {epochs: 100, batch_size: 64, lr: 1e-3}
```

## 逐层说明

以下每层按同一结构说明：现有内容、独立使用方式、下一步。

### L1 · 数据

> 数据集目录（Web）：https://qdtiger.github.io/indoorloc/datasets_zh.html · 列出全部 ID：`iloc.list_available_datasets()`

状态：✅ 已验证 · 🧪 已集成，验证中

| 类型 | 数据集 | ID | 样本数 | 状态 |
|------|--------|-----|--------|:----:|
| **WiFi** | [UJIndoorLoc](https://archive.ics.uci.edu/dataset/310/ujiindoorloc) | `ujindoorloc` | 21k | ✅ |
| | [SODIndoorLoc](https://github.com/renwudao24/SODIndoorLoc) | `sodindoorloc` | 24k | 🧪 |
| | [LongTermWiFi](https://zenodo.org/record/1309317) | `longtermwifi` | 104k | 🧪 |
| | [Tampere](https://zenodo.org/record/889798) | `tampere` | 4.6k | 🧪 |
| | [WLANRSSI](https://archive.ics.uci.edu/dataset/422/wireless+indoor+localization) | `wlanrssi` | 2k | 🧪 |
| | [TUJI1](https://zenodo.org/record/7641701) | `tuji1` | 8.9k | 🧪 |
| **BLE** | [BLEIndoor](https://github.com/co60ca/BBIL) | `ble_indoor` | 44k | 🧪 |
| | [iBeaconRSSI](https://zenodo.org/record/1618692) | `ibeacon_rssi` | 4.7k | 🧪 |
| | [BLE RSSI UCI](https://archive.ics.uci.edu/dataset/435/ble+rssi+dataset+for+indoor+localization+and+navigation) | `ble_rssi_uci` | 1.4k | 🧪 |
| **CSI** | [CSI Fingerprint](https://github.com/qiang5love1314/CSI-dataset) | `csi_fingerprint` | 489 | 🧪 |
| | [HWILD](https://github.com/H-WILD/human_held_device_wifi_indoor_localization_dataset) | `hwild` | 409k | 🧪 |
| | [HALOC](https://zenodo.org/records/10715595) | `haloc` | 111k | 🧪 |

> **关于数据集状态**：✅ 表示已完整通过自动下载、训练、评测的端到端验证，验证记录随仓库提供；🧪 表示加载器已实现，验证工作正在逐个进行。进度详见[开发规划](docs/DEVELOPMENT_PLAN.md)。

只取数据：加载器可直接导出为数组，供任意框架使用。

```python
train, test = iloc.load_dataset("ujindoorloc")
X, y = train.to_tensors()            # numpy：X (N, D)，y (N, 4) = [x, y, floor, building]
X_t, y_t = train.to_torch_tensors()  # 或 torch 张量
```

<details>
<summary>待接入数据集（欢迎贡献）</summary>

以下数据集已有公开下载源，加载器尚未实现：

| 数据集 | 来源 | 说明 |
|--------|------|------|
| DeepMIMO | [deepmimo.net](https://www.deepmimo.net) | 射线追踪合成数据，仿真数据层的首个接入目标 |
| DICHASUS | [DaRUS](https://darus.uni-stuttgart.de/dataverse/dichasus) | Massive-MIMO CSI，厘米级真值 |
| MaMIMO CSI | [IEEE DataPort](https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset) | 需注册账号 |
| OpenCSI | [Figshare](https://doi.org/10.6084/m9.figshare.19596379.v1) | 约 2GB，格式待确认 |
| CSUIndoorLoc | [GitHub](https://github.com/EPIC-CSU/csi-rssi-dataset-indoor-nav) | 格式待确认 |
| ESPARGOS | [espargos.net](https://espargos.net/datasets/) | 17–86GB |
| CSI2Pos / CSI2TAoA | [TIB](https://service.tib.eu/ldmservice/) | 需登录 |
| WILDv2 | [Kaggle](https://www.kaggle.com/competitions/wild-v2) | 需 Kaggle API |

> 贡献加载器请参见 `CONTRIBUTING.md`。

</details>

下一步：`to_numpy()` / `to_dataframe()` / CSV 导出，随每个数据集提交版本化标准切分，以 DeepMIMO v4 作为首个仿真数据源。

### L2 · 信号

已定义九类信号（WiFi、BLE、CSI、UWB、IMU、地磁、VLC、超声、混合），当前加载器产出 WiFi 与 BLE 两类。预处理变换的组合方式与 torchvision 一致，可直接作用于由自有数组构造的单条信号：

```python
sig = iloc.WiFiSignal(rssi_values=rssi_row)   # 自有数据中的一条 RSSI 向量
pipeline = iloc.Compose([iloc.APFilter(threshold=-90), iloc.RSSINormalize(method="minmax")])
sig = pipeline(sig)
```

下一步：经过验证的 CSI 预处理管线（相位清洗、幅度 / 角度提取）。

### L3 · 方法

> 算法总览（Web）：https://qdtiger.github.io/indoorloc/algorithms.html · 列出模型：`iloc.list_models()` · 实现清单：[总览](#总览)

用本库模型训练自有数据：

```python
signals   = [iloc.WiFiSignal(rssi_values=row) for row in X_own]
locations = [iloc.Location(coordinate=iloc.Coordinate(x, y)) for x, y in xy_own]
model  = iloc.create_model("wknn", k=5).fit(signals, locations)
result = model.predict(signals[0])              # LocalizationResult
```

裸数组形式的 `fit(X, y)` 与 scikit-learn estimator 兼容已列入路线图，届时模型可直接放入 `Pipeline` 与 `GridSearchCV`。

<details>
<summary>自定义模型注册</summary>

```python
import indoorloc as iloc
from indoorloc.registry import LOCALIZERS
from indoorloc.localizers.base import BaseLocalizer

@LOCALIZERS.register_module()
class MyLocalizer(BaseLocalizer):
    @property
    def localizer_type(self) -> str:
        return "my_localizer"

    def _fit_impl(self, signals, locations, **kwargs):
        ...  # 你的训练逻辑
        self._is_trained = True
        return self

    def predict(self, signal):
        ...  # 返回 LocalizationResult

model = iloc.create_model("MyLocalizer")
```

</details>

<details>
<summary>规划中（尚未实现）</summary>

- **自监督预训练**：SimCLR、MoCo、BYOL、SimSiam、VICReg 等
- **元学习 / 少样本**：MAML、FOMAML、Reptile、ProtoNet、MatchingNet 等
- **深度域适应**：DANN、MDD、DeepCORAL 等
- **模型驱动 / 免模型方法**：几何解算、贝叶斯滤波、加权质心、Channel Charting 等

</details>

下一步：为每一条基准结果发布权重与训练日志，并建立等搜索预算的调参协议，使传统方法与深度方法在同等条件下比较。

### L4 · 评测

评测任意来源的预测结果：

```python
from indoorloc.evaluation import EvaluationResults

truths = [iloc.Location(coordinate=iloc.Coordinate(x, y)) for x, y in y_true]
preds  = [iloc.Location(coordinate=iloc.Coordinate(x, y)) for x, y in y_pred]
results = EvaluationResults.from_predictions(preds, truths)
results.mean_error, results.p75_error         # 指标以属性形式提供
print(results.summary())
```

对模型调用 `evaluate()` 时还会与同一数据集上的文献报告值对照。本仓库实际复现的结果与文献摘录的数值始终分列标注，不作混排。

下一步：版本化标准切分、泛化协议套件（跨设备、跨时间、仿真到实测），以及函数式入口 `iloc.evaluate(y_true, y_pred)`。

### L5 · 应用

规划中：实时推理、跟踪滤波（Kalman / 粒子滤波）、行人航位推算、导航。本层尚无实现。

## 路线图

近期方向，按优先级排列：

1. **数据验证**：所有数据集达到 ✅，附校验和与随仓库提交的标准切分
2. **评测协议**：版本化切分；跨设备 / 跨时间 / 仿真到实测协议
3. **模型库**：为每一条基准结果发布权重与训练日志
4. **仿真数据**：先接入 DeepMIMO v4，再扩展至 Sionna RT 等射线追踪源

<details>
<summary>完整分类树（含规划项）</summary>

图例：✅ 已实现 · 🚧 部分实现 · 📋 规划中

```text
L1  数据层
├── 实测数据                        ✅ 12 个数据集（WiFi 6 · BLE 3 · CSI 3）
│   └── 高精度 Massive-MIMO CSI     📋 DICHASUS、MaMIMO（待接入）
├── 仿真数据                        📋
│   ├── 射线追踪引擎                📋 Sionna RT · Wireless InSite · NVIDIA AODT
│   ├── 统计信道模型                📋 3GPP TR 38.901（InH/InF）· QuaDRiGa
│   ├── 预生成合成数据集            📋 DeepMIMO v4（首选目标）· WAIR-D
│   └── 学习式射频场                📋 NeRF2 · 高斯泼溅 RF
└── 仿真-实测孪生配对               📋 如 DICHASUS ↔ Sionna RT 校准

L2  信号/观测层                     🚧 RSSI + BLE 已激活；CSI、UWB(ToF/TDoA)、
                                       IMU、地磁、VLC、超声 已定义

L3  方法层
├── Model-based（模型驱动）         📋 几何解算 · 贝叶斯滤波 · 地图约束
├── Model-free（免模型）            📋 加权质心 · 空间插值 · 图/流形 · Channel Charting
└── Data-driven（数据驱动）         🚧
    ├── 确定性指纹                  ✅ kNN · WKNN
    ├── 机器学习回归                ✅ SVM · 随机森林
    ├── 深度神经网络                ✅ MLP · CNN1D · timm 骨干 × 7 种预测头
    ├── 集成融合                    ✅ Ensemble · Stacking
    ├── 迁移/域适应                 🚧 浅层 skada（CORAL · TCA · KMM）；深度 DA 规划中
    ├── 自监督预训练                📋 SimCLR · MoCo · BYOL ...
    ├── 元学习/少样本               📋 MAML · ProtoNet ...
    ├── 概率/生成式指纹             📋
    └── 序列跟踪 · 神经无线电地图 · Channel Charting   📋

L4  评测层                          ✅ 9 项指标 · 文献基准对照
    └── CRLB 理论界 · 跨设备/跨时间与 Sim2Real 协议    📋

L5  应用/部署层                     📋 实时推理 · 跟踪滤波（Kalman/粒子滤波）· PDR · 导航
```

</details>

详细的开发计划、里程碑与已知问题清单见 [`docs/DEVELOPMENT_PLAN.md`](docs/DEVELOPMENT_PLAN.md)。

<details>
<summary>项目结构</summary>

```
indoorloc/
├── signals/          # L2 · WiFi、BLE、CSI、IMU 等信号类 + 变换
├── locations/        # 坐标与位置类
├── datasets/         # L1 · 数据集 loader + 注册表
├── localizers/       # L3 · 传统 ML 定位器（指纹 / 融合 / 迁移）
├── models/           # L3 · 深度模型：骨干 × 预测头 + DeepLocalizer
├── evaluation/       # L4 · 指标 + 文献基准表
└── configs/          # YAML 配置（支持 _base_ 继承）
```

</details>

## 贡献

欢迎通过 PR 参与贡献。当前最需要的是新数据集加载器与新算法实现，贡献规范见 `CONTRIBUTING.md`。

## 引用

```bibtex
@software{indoorloc,
  title = {IndoorLoc: A Unified Framework for Indoor Localization},
  year = {2025},
  url = {https://github.com/qdtiger/indoorloc}
}
```

## 许可证

Apache License 2.0

## 致谢

- [OpenMMLab](https://github.com/open-mmlab)：注册表和配置系统的设计来源
- [timm](https://github.com/huggingface/pytorch-image-models)：预训练骨干网络
- [scikit-learn](https://scikit-learn.org/) / [SKADA](https://github.com/scikit-adaptation/skada)：传统机器学习与域适应
