<div align="center">

<img src="assets/logo.png" width="600">

**开源室内定位工具箱**

[![PyPI](https://img.shields.io/pypi/v/indoorloc)](https://pypi.org/project/indoorloc/)
[![Build](https://github.com/qdtiger/indoorloc/actions/workflows/ci.yml/badge.svg)](https://github.com/qdtiger/indoorloc/actions)
[![Docs](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://qdtiger.github.io/indoorloc/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Stars](https://img.shields.io/github/stars/qdtiger/indoorloc?style=social)](https://github.com/qdtiger/indoorloc)

[![Issues](https://img.shields.io/github/issues/qdtiger/indoorloc)](https://github.com/qdtiger/indoorloc/issues)
[![Last Commit](https://img.shields.io/github/last-commit/qdtiger/indoorloc)](https://github.com/qdtiger/indoorloc/commits/main)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/qdtiger/indoorloc/pulls)

[文档](https://qdtiger.github.io/indoorloc/) · [安装](#安装) · [快速开始](#快速开始) · [支持的数据集](#支持的数据集) · [支持的算法](#支持的算法) · [贡献](#贡献) · [引用](#引用)

[English](README.md) | [中文](README_zh.md)

</div>

---

## 简介

**一行代码**加载任意室内定位数据集。**一行代码**完成训练和评估。

```python
import indoorloc as iloc

train, test = iloc.UJIndoorLoc(download=True)  # 就这么简单。自动下载、自动解析、开箱即用。
```

IndoorLoc 为 36+ 室内定位数据集提供统一接口，涵盖 WiFi、BLE、UWB、CSI 等多种信号。

### 给初学者

**一行代码 = 一个数据集。** 专注于学习算法，而不是和数据格式较劲。

```python
import indoorloc as iloc

# 36+ 数据集，统一 API
train, test = iloc.UJIndoorLoc(download=True)    # WiFi RSSI
train, test = iloc.CSIIndoor(download=True)       # WiFi CSI
train, test = iloc.UWBIndoor(download=True)       # UWB 测距
```

### 给专家

**需要时完全可控。** 灵活的数据流水线、可定制的预处理、插件化的算法扩展。

```python
import indoorloc as iloc
from indoorloc.localizers.base import BaseLocalizer
from indoorloc.registry import LOCALIZERS

# 自定义预处理流水线（先过滤弱AP，再归一化）
dataset = iloc.UJIndoorLoc(
    download=True,
    normalize=False,  # 禁用默认归一化以使用自定义流水线
    transform=iloc.Compose([
        iloc.APFilter(threshold=-90),       # 1. 过滤弱信号 (原始 dBm)
        iloc.RSSINormalize(method='minmax') # 2. 归一化到 [0,1]
    ])
)

# 注册你自己的算法
@LOCALIZERS.register_module()
class MyNovelLocalizer(BaseLocalizer):
    ...
```

## 特性

- **多算法支持**：k-NN、SVM、随机森林、深度学习（CNN、LSTM、Transformer）
- **多模态支持**：WiFi、BLE、IMU、视觉、UWB 等多种信号
- **多数据集支持**：UJIndoorLoc、Tampere、Microsoft Indoor 等
- **统一接口**：所有算法采用一致的 API 设计
- **配置驱动**：基于 YAML 的配置系统，支持继承
- **可扩展架构**：基于注册表的插件化设计
- **PyPI 发布**：支持 pip 一键安装

## 安装

### GPU (CUDA 11.8)

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

更多安装选项（legacy、可选 extras、排错）：`docs/installation_zh.md`。

## 快速开始

```python
import indoorloc as iloc
import numpy as np

# 创建 k-NN 定位器
model = iloc.create_model('KNNLocalizer', k=5)

# 准备训练数据
train_signals = [
    iloc.WiFiSignal(rssi_values=np.random.randn(520).astype(np.float32))
    for _ in range(100)
]
train_locations = [
    iloc.Location.from_coordinates(
        x=np.random.uniform(0, 100),
        y=np.random.uniform(0, 100),
        floor=np.random.randint(0, 3)
    )
    for _ in range(100)
]

# 训练模型
model.fit(train_signals, train_locations)

# 进行预测
test_signal = iloc.WiFiSignal(rssi_values=np.random.randn(520).astype(np.float32))
result = model.predict(test_signal)

print(f"预测位置: ({result.x:.2f}, {result.y:.2f})")
print(f"预测楼层: {result.floor}")
```

## 使用配置文件

```yaml
# indoorloc/configs/wifi/knn_ujindoorloc.yaml
model:
  type: KNNLocalizer
  k: 5
  weights: distance
  metric: euclidean
  predict_floor: true
  predict_building: true
```

```python
import indoorloc as iloc

# 从配置文件加载
model = iloc.create_model(config='indoorloc/configs/wifi/knn_ujindoorloc.yaml')
```

## 自定义模型注册

```python
from indoorloc.registry import LOCALIZERS
from indoorloc.localizers.base import BaseLocalizer

@LOCALIZERS.register_module()
class MyCustomLocalizer(BaseLocalizer):
    """自定义定位器"""

    def __init__(self, custom_param=1.0, **kwargs):
        super().__init__(**kwargs)
        self.custom_param = custom_param

    @property
    def localizer_type(self) -> str:
        return 'my_custom'

    def fit(self, signals, locations, **kwargs):
        # 训练逻辑
        self._is_trained = True
        return self

    def predict(self, signal):
        # 预测逻辑
        pass

# 使用自定义模型
model = iloc.create_model('MyCustomLocalizer', custom_param=2.0)
```

## 项目结构

```
indoorloc/
├── indoorloc/
│   ├── signals/          # 信号抽象层（WiFi、BLE、IMU 等）
│   ├── locations/        # 位置和坐标类
│   ├── datasets/         # 数据集实现
│   ├── localizers/       # 定位算法
│   │   ├── fingerprint/  # 传统机器学习（k-NN、SVM、RF）
│   │   ├── deep/         # 深度学习（CNN、LSTM、Transformer）
│   │   └── pdr/          # 惯性导航
│   ├── fusion/           # 多传感器融合
│   ├── evaluation/       # 评估指标
│   ├── engine/           # 训练工具
│   ├── visualization/    # 可视化工具
│   ├── configs/          # 内置配置
│   └── utils/            # 工具函数
├── tools/                # 命令行工具
├── examples/             # 使用示例
├── tests/                # 单元测试
└── docs/                 # 文档
```

## 支持的算法

算法总览（Web）：https://qdtiger.github.io/indoorloc/algorithms.html

- 列出可用模型：`iloc.list_models()`
- 创建模型：`iloc.create_model("KNNLocalizer", k=5)` / `iloc.create_model("resnet18", dataset=train)`

<details>
<summary>算法家族（概览）</summary>

依托 [sklearn](https://scikit-learn.org/) (30+)、[timm](https://github.com/huggingface/pytorch-image-models) (700+)、[lightly](https://github.com/lightly-ai/lightly) (10+)、[learn2learn](https://github.com/learnables/learn2learn) (7+) 与 [SKADA](https://github.com/scikit-adaptation/skada) (20+)。

- **有监督**
  - 传统 ML：k-NN, WKNN, SVM, RF...
  - 深度学习：MLP, CNN1D, ResNet, ViT...
- **自监督**
  - 对比：SimCLR, MoCo, NNCLR
  - 非对比：BYOL, SimSiam, VICReg
- **元学习**
  - 基于梯度：MAML, FOMAML, Reptile
  - 基于度量：ProtoNet, MatchingNet
- **迁移学习**
  - 特征：CORAL, TCA
  - 重加权：KMM, KLIEP
  - 深度：DANN, MDD

</details>

## 支持的数据集

数据集目录（Web）：https://qdtiger.github.io/indoorloc/datasets_zh.html

- 列出可用数据集 ID：`iloc.list_available_datasets()`
- 加载数据集：`train, test = iloc.load_dataset("ujindoorloc")`

已验证数据集 ID（12）：

- WiFi：`ujindoorloc`, `sodindoorloc`, `longtermwifi`, `tampere`, `wlanrssi`, `tuji1`
- BLE：`ble_indoor`, `ibeacon_rssi`, `ble_rssi_uci`
- CSI：`csi_fingerprint`, `hwild`, `haloc`

<details>
<summary>已验证数据集（表格）</summary>

| 类型 | 数据集 | ID | 样本数 |
|------|--------|----|--------|
| **WiFi** | [UJIndoorLoc](https://archive.ics.uci.edu/dataset/310/ujiindoorloc) | `ujindoorloc` | 21k |
| | [SODIndoorLoc](https://github.com/renwudao24/SODIndoorLoc) | `sodindoorloc` | 24k |
| | [LongTermWiFi](https://zenodo.org/record/1309317) | `longtermwifi` | 104k |
| | [Tampere](https://zenodo.org/record/889798) | `tampere` | 4.6k |
| | [WLANRSSI](https://archive.ics.uci.edu/dataset/422/wireless+indoor+localization) | `wlanrssi` | 2k |
| | [TUJI1](https://zenodo.org/record/7641701) | `tuji1` | 8.9k |
| **BLE** | [BLEIndoor](https://github.com/co60ca/BBIL) | `ble_indoor` | 44k |
| | [iBeaconRSSI](https://zenodo.org/record/1618692) | `ibeacon_rssi` | 4.7k |
| | [BLE RSSI UCI](https://archive.ics.uci.edu/dataset/435/ble+rssi+dataset+for+indoor+localization+and+navigation) | `ble_rssi_uci` | 1.4k |
| **CSI** | [CSI Fingerprint](https://github.com/qiang5love1314/CSI-dataset) | `csi_fingerprint` | 489 |
| | [HWILD](https://github.com/H-WILD/human_held_device_wifi_indoor_localization_dataset) | `hwild` | 409k |
| | [HALOC](https://zenodo.org/records/10715595) | `haloc` | 111k |

</details>

<details>
<summary>待集成数据集（欢迎贡献）</summary>

这些数据集有下载来源，但尚未集成：

| 数据集 | 来源 | 备注 |
|--------|------|------|
| OpenCSI | [Figshare](https://doi.org/10.6084/m9.figshare.19596379.v1) | ~2GB，格式未验证 |
| CSUIndoorLoc | [GitHub](https://github.com/EPIC-CSU/csi-rssi-dataset-indoor-nav) | 格式未验证 |
| DICHASUS | [DaRUS](https://darus.uni-stuttgart.de/dataverse/dichasus) | 14 scenarios |
| ESPARGOS | [espargos.net](https://espargos.net/datasets/) | 17-86GB |
| DeepMIMO | [deepmimo.net](https://www.deepmimo.net) | 需要 `pip install DeepMIMO` |
| CSI2Pos | [TIB](https://service.tib.eu/ldmservice/dataset/csi2pos) | 需要登录 |
| CSI2TAoA | [TIB](https://service.tib.eu/ldmservice/dataset/csi2taoa) | 需要登录 |
| MaMIMO CSI | [IEEE DataPort](https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset) | 需要账号 |
| WILDv2 | [Kaggle](https://www.kaggle.com/competitions/wild-v2) | 需要 Kaggle API |

> 贡献：欢迎帮我们验证并接入更多数据集！详见 `CONTRIBUTING.md`。

</details>

<sub>更多数据集、下载方式与许可证说明：https://qdtiger.github.io/indoorloc/datasets_zh.html</sub>

## 评估指标

| 指标 | 描述 |
|-----|------|
| Mean Position Error | 平均定位误差（米） |
| Median Position Error | 中位数定位误差（米） |
| 75th/95th Percentile | 75%/95% 分位数误差 |
| Floor Accuracy | 楼层分类准确率 |
| Building Accuracy | 建筑分类准确率 |
| CDF Analysis | 累积分布函数分析 |

## 许可证

Apache License 2.0

## 引用

```bibtex
@software{indoorloc,
  title = {IndoorLoc: A Unified Framework for Indoor Localization},
  year = {2024},
  url = {https://github.com/qdtiger/indoorloc}
}
```

## 致谢

- [OpenMMLab](https://github.com/open-mmlab) - 注册表和配置系统设计参考
- [UJIndoorLoc](https://archive.ics.uci.edu/dataset/310/ujiindoorloc) - 数据集提供

## 贡献

欢迎提交 Issue 和 Pull Request！
