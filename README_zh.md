<div align="center">

<img src="assets/logo.png" width="600">

**IndoorLoc | 室内定位工具库**

*把室内定位研究的数据集、算法和评测收进同一个库*

[![PyPI](https://img.shields.io/pypi/v/indoorloc)](https://pypi.org/project/indoorloc/)
[![CI](https://github.com/qdtiger/indoorloc/actions/workflows/ci.yml/badge.svg)](https://github.com/qdtiger/indoorloc/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://qdtiger.github.io/indoorloc/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Stars](https://img.shields.io/github/stars/qdtiger/indoorloc?style=social)](https://github.com/qdtiger/indoorloc)

[文档](https://qdtiger.github.io/indoorloc/) · [安装](#安装) · [快速开始](#快速开始) · [数据集](#数据集) · [模型](#模型与算法) · [路线图](#全栈分类与路线图) · [贡献](#贡献)

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

## 为什么做 IndoorLoc？

做室内定位研究的人多半都遇到过这几件事：想复现一篇论文，发现没放代码；找到了公开数据集，却没有统一的训练/测试切分；各家论文里的精度数字口径不一，根本没法直接比。这个领域不缺部署方案，不缺采集工具，更不缺单篇论文的配套仓库，缺的是把这些东西串起来的框架。

IndoorLoc 照着 [OpenMMLab](https://github.com/open-mmlab) 的路子来补这一课：

- 数据集统一注册、自动下载，WiFi / BLE / CSI 共用一种样本格式
- 传统机器学习和深度模型走同一套 `fit / predict / evaluate` 接口
- 评测指标只有一份实现（含楼层、建筑准确率），结果可以直接和文献数字对照
- 实验由 YAML 配置驱动，支持 `_base_` 继承，改参数、换模型都是一行命令的事

> **关于数据集状态**：下表中 ✅ 表示从自动下载到训练、评测完整跑通过，记录在仓库里可查；🧪 表示加载器已经写好，还在逐个复核。进度见[开发规划](docs/DEVELOPMENT_PLAN.md)。

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

# 覆盖任意参数（布尔值写 True/False，小写不认）
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

## 数据集

> 数据集目录（Web）：https://qdtiger.github.io/indoorloc/datasets_zh.html · 列出全部 ID：`iloc.list_available_datasets()`

状态：✅ 完整跑通 · 🧪 已集成，复核中

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

<details>
<summary>还没接入的数据集（欢迎认领）</summary>

这些数据集有公开下载源，但加载器还没写：

| 数据集 | 来源 | 说明 |
|--------|------|------|
| DeepMIMO | [deepmimo.net](https://www.deepmimo.net) | 射线追踪合成数据，仿真层打算最先接它 |
| DICHASUS | [DaRUS](https://darus.uni-stuttgart.de/dataverse/dichasus) | Massive-MIMO CSI，厘米级真值 |
| MaMIMO CSI | [IEEE DataPort](https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset) | 需注册账号 |
| OpenCSI | [Figshare](https://doi.org/10.6084/m9.figshare.19596379.v1) | 约 2GB，格式待确认 |
| CSUIndoorLoc | [GitHub](https://github.com/EPIC-CSU/csi-rssi-dataset-indoor-nav) | 格式待确认 |
| ESPARGOS | [espargos.net](https://espargos.net/datasets/) | 17–86GB |
| CSI2Pos / CSI2TAoA | [TIB](https://service.tib.eu/ldmservice/) | 需登录 |
| WILDv2 | [Kaggle](https://www.kaggle.com/competitions/wild-v2) | 需 Kaggle API |

> 想认领一个？见 `CONTRIBUTING.md`。

</details>

## 模型与算法

> 算法总览（Web）：https://qdtiger.github.io/indoorloc/algorithms.html · 列出模型：`iloc.list_models()`

已经实现的：

| 类别 | 方法 |
|------|------|
| 传统机器学习 | kNN · WKNN · SVM · 随机森林 |
| 深度监督 | MLP · CNN1D · [timm](https://github.com/huggingface/pytorch-image-models) 骨干（ResNet · EfficientNet · ViT ……）× 7 种任务头（回归 / 多尺度回归 / 分类 / 楼层 / 建筑 / 混合 / 层次化） |
| 集成融合 | Ensemble · Stacking |
| 迁移学习（浅层） | CORAL · TCA · KMM（基于 [SKADA](https://github.com/scikit-adaptation/skada)） |

<details>
<summary>路线图上的（还没有代码）</summary>

- 自监督预训练：SimCLR、MoCo、BYOL、SimSiam、VICReg 等
- 元学习 / 少样本：MAML、FOMAML、Reptile、ProtoNet、MatchingNet 等
- 深度域适应：DANN、MDD、DeepCORAL 等
- 模型驱动 / 免模型方法：几何解算、贝叶斯滤波、加权质心、Channel Charting 等

</details>

### 注册自己的模型

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

## 评估指标

| 指标 | 说明 |
|------|------|
| 平均 / 中位 / RMS / 75 分位定位误差 | 定位误差（米） |
| 楼层准确率 | 楼层分类 |
| 建筑准确率 | 建筑分类 |
| CDF 分析 | 误差分布 |

`evaluate()` 会把你的结果和同一数据集上文献报告的数字放在一起对照。我们自己复现出来的数字和从文献里摘的数字始终分开标注，不会混在一列。

## 全栈分类与路线图

IndoorLoc 按室内定位综述里常见的分类方式组织成五层，自底向上是数据、信号、方法、评测、应用。

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

更细的开发计划、里程碑和已知问题清单都在 [`docs/DEVELOPMENT_PLAN.md`](docs/DEVELOPMENT_PLAN.md)。

<details>
<summary>项目结构</summary>

```
indoorloc/
├── signals/          # WiFi、BLE、CSI、IMU 等信号类
├── locations/        # 坐标与位置类
├── datasets/         # 数据集 loader + 注册表 + 变换
├── localizers/       # 传统 ML 定位器（指纹 / 融合 / 迁移）
├── models/           # 深度模型：骨干 × 预测头 + DeepLocalizer
├── evaluation/       # 指标 + 文献基准表
└── configs/          # OpenMMLab 风格 YAML 配置
```

</details>

## 贡献

欢迎提 PR。眼下最缺的是新数据集的加载器和新算法实现，具体要求见 `CONTRIBUTING.md`。

## 许可证

Apache License 2.0

## 引用

```bibtex
@software{indoorloc,
  title = {IndoorLoc: A Unified Framework for Indoor Localization},
  year = {2025},
  url = {https://github.com/qdtiger/indoorloc}
}
```

## 致谢

- [OpenMMLab](https://github.com/open-mmlab)：注册表和配置系统的设计来源
- [timm](https://github.com/huggingface/pytorch-image-models)：预训练骨干网络
- [scikit-learn](https://scikit-learn.org/) / [SKADA](https://github.com/scikit-adaptation/skada)：传统机器学习与域适应
