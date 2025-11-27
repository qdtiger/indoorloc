<div align="center">

<img src="assets/logo.png" width="600">

**开源室内定位工具箱**

[![PyPI](https://img.shields.io/pypi/v/indoorloc)](https://pypi.org/project/indoorloc/)
[![Build](https://github.com/qdtiger/indoorloc/actions/workflows/ci.yml/badge.svg)](https://github.com/qdtiger/indoorloc/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Stars](https://img.shields.io/github/stars/qdtiger/indoorloc?style=social)](https://github.com/qdtiger/indoorloc)

[![Issues](https://img.shields.io/github/issues/qdtiger/indoorloc)](https://github.com/qdtiger/indoorloc/issues)
[![Last Commit](https://img.shields.io/github/last-commit/qdtiger/indoorloc)](https://github.com/qdtiger/indoorloc/commits/main)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/qdtiger/indoorloc/pulls)

[English](README.md) | [中文](README_zh.md)

</div>

---

## 简介

**一行代码**加载任意室内定位数据集。**一行代码**完成训练和评估。

```python
train, test = iloc.UJIndoorLoc(download=True)  # 就这么简单。自动下载、自动解析、开箱即用。
```

IndoorLoc 为 36+ 室内定位数据集提供统一接口，涵盖 WiFi、BLE、UWB、CSI 等多种信号。

### 给初学者

**一行代码 = 一个数据集。** 专注于学习算法，而不是和数据格式较劲。

```python
# 36+ 数据集，统一 API
train, test = iloc.UJIndoorLoc(download=True)    # WiFi RSSI
train, test = iloc.CSIIndoor(download=True)       # WiFi CSI
train, test = iloc.UWBIndoor(download=True)       # UWB 测距
```

### 给专家

**需要时完全可控。** 灵活的数据流水线、可定制的预处理、插件化的算法扩展。

```python
# 自定义预处理流水线
dataset = iloc.UJIndoorLoc(
    download=True,
    transform=iloc.Compose([
        iloc.RSSINormalize(method='minmax'),
        iloc.APFilter(threshold=-90),
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

```bash
# 基础安装
pip install indoorloc

# 包含视觉定位支持
pip install indoorloc[vision]

# 完整安装（所有功能）
pip install indoorloc[full]

# 开发模式安装
git clone https://github.com/qdtiger/indoorloc.git
cd indoorloc
pip install -e ".[dev]"
```

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
# configs/wifi/knn_ujindoorloc.yaml
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
model = iloc.create_model(config='configs/wifi/knn_ujindoorloc.yaml')
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

### 指纹定位
- [x] k-NN（k近邻）
- [x] 加权 k-NN
- [ ] SVM（支持向量机）
- [ ] 随机森林
- [ ] 高斯过程

### 深度学习
- [ ] MLP（多层感知机）
- [ ] CNN（卷积神经网络）
- [ ] LSTM（长短期记忆网络）
- [ ] Transformer

### 融合算法
- [ ] 卡尔曼滤波
- [ ] 扩展卡尔曼滤波
- [ ] 粒子滤波

## 支持的数据集

**36 个数据集**，支持多种信号模态，并提供**自动下载**功能。[📊 交互式浏览器](https://qdtiger.github.io/indoorloc/datasets_zh.html)

<table>
<thead>
<tr>
<th>RSSI</th>
<th>CSI</th>
<th>ToF/TWR</th>
<th>磁场</th>
<th>多模态</th>
<th>其他</th>
</tr>
</thead>
<tbody>
<tr>
<td valign="top">

**WiFi RSSI**
- [UJIndoorLoc](indoorloc/datasets/ujindoorloc.py)
- [SODIndoorLoc](indoorloc/datasets/sodindoorloc.py)
- [LongTermWiFi](indoorloc/datasets/longtermwifi.py)
- [Tampere](indoorloc/datasets/tampere.py)
- [WLANRSSI](indoorloc/datasets/wlanrssi.py)
- [TUJI1](indoorloc/datasets/tuji1.py)
- [RSSBased](indoorloc/datasets/rss_based.py)

**BLE RSSI**
- [iBeaconRSSI](indoorloc/datasets/ibeacon_rssi.py)
- [BLEIndoor](indoorloc/datasets/ble_indoor.py)
- [BLERSSIU_UCI](indoorloc/datasets/ble_rssi_uci.py)

</td>
<td valign="top">

**WiFi CSI**
- [CSIIndoor](indoorloc/datasets/csi_indoor.py)
- [CSIFingerprint](indoorloc/datasets/csi_fingerprint.py)
- [HWILD](indoorloc/datasets/hwild.py)
- [CSUIndoorLoc](indoorloc/datasets/csu_csi_rssi.py)
- [WILDv2](indoorloc/datasets/wild_v2.py)
- [HALOC](indoorloc/datasets/haloc.py)
- [CSIBench](indoorloc/datasets/csi_bench.py)
- [WiFiCSID2D](indoorloc/datasets/wifi_csi_d2d.py)

**LTE CSI**
- [OpenCSI](indoorloc/datasets/opencsi.py)

**Massive MIMO CSI**
- [MaMIMOCSI](indoorloc/datasets/mamimo_csi.py)
- [DICHASUS](indoorloc/datasets/dichasus.py)
- [ESPARGOS](indoorloc/datasets/espargos.py)
- [MaMIMOUAV](indoorloc/datasets/mamimo_uav.py)
- [DeepMIMO](indoorloc/datasets/deepmimo.py)

**RF CSI**
- [CSI2Pos](indoorloc/datasets/csi2pos.py)
- [CSI2TAoA](indoorloc/datasets/csi2taoa.py)

</td>
<td valign="top">

- [UWBIndoor](indoorloc/datasets/uwb_indoor.py)
- [UWBRanging](indoorloc/datasets/uwb_ranging.py)

</td>
<td valign="top">

- [MagneticIndoor](indoorloc/datasets/magnetic_indoor.py)

</td>
<td valign="top">

- [WiFiIMUHybrid](indoorloc/datasets/wifi_imu_hybrid.py)
- [WiFiMagneticHybrid](indoorloc/datasets/wifi_magnetic_hybrid.py)
- [MultiModalIndoor](indoorloc/datasets/multimodal_indoor.py)
- [SensorFusion](indoorloc/datasets/sensor_fusion.py)

</td>
<td valign="top">

- [VLCIndoor](indoorloc/datasets/vlc_indoor.py)
- [UltrasoundIndoor](indoorloc/datasets/ultrasound_indoor.py)
- [RFIDIndoor](indoorloc/datasets/rfid_indoor.py)

</td>
</tr>
</tbody>
</table>

### 自动下载用法

```python
import indoorloc as iloc

# UJIndoorLoc - 自动下载到 ~/.cache/indoorloc/datasets/
dataset = iloc.UJIndoorLoc(download=True)
train = iloc.UJIndoorLoc(download=True, split='train')
test = iloc.UJIndoorLoc(download=True, split='test')

# SODIndoorLoc - 指定建筑
cetc_train = iloc.SODIndoorLoc(building='CETC331', download=True)
hcxy_train = iloc.SODIndoorLoc(building='HCXY', download=True)

# BLE 数据集
ble_dataset = iloc.iBeaconRSSI(download=True)

# UWB 数据集
uwb_dataset = iloc.UWBIndoor(download=True)

# 混合数据集（返回多种信号）
hybrid = iloc.WiFiIMUHybrid(download=True)
signals, location = hybrid[0]
wifi_signal = signals['wifi']
imu_signal = signals['imu']

# 或指定自定义目录
dataset = iloc.UJIndoorLoc(data_root='./data', download=True)
```

### 手动下载

也可以从官方源手动下载数据集：

| 数据集 | 官方来源 | 说明 |
|-------|---------|------|
| UJIndoorLoc | [UCI 机器学习库](https://archive.ics.uci.edu/dataset/310/ujiindoorloc) | 直接 ZIP 下载 |
| SODIndoorLoc | [GitHub 仓库](https://github.com/renwudao24/SODIndoorLoc) | GitHub 上的 CSV 文件 |
| Microsoft Indoor 2.0 | [Microsoft Research](https://aka.ms/location20dataset) + [GitHub](https://github.com/location-competition/indoor-location-competition-20) | 多传感器 (WiFi, BLE, IMU, Magnetometer) |
| TUJI1 | [Tampere 大学](https://trepo.tuni.fi/handle/10024/211225) + [Zenodo](https://zenodo.org/records/1226835) | 多设备采集 |
| WiFi-RSSI | [GitHub 仓库](https://github.com/m-nabati/WiFi-RSSI-Localization-Dataset) | 小规模 (250 点) |
| OWP-IMU | [arXiv](https://arxiv.org/abs/2505.16823) | 光学无线 + IMU 融合 |

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
