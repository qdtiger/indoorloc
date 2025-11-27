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

IndoorLoc 是一个室内定位统一框架，灵感来源于 [MMPretrain](https://github.com/open-mmlab/mmpretrain)。它提供了**从数据集到算法的一站式解决方案**，实现不同定位方法之间的**自动适配**。

### 适用人群

- **初学者**：快速上手室内定位，无需关心底层实现细节
- **科研人员**：使用统一接口复现和对比各类算法，专注于创新研究
- **开发者**：基于生产级代码构建和部署室内定位系统

### 为什么选择 IndoorLoc？

- **零样板代码**：几行代码即可完成数据加载、模型训练和结果评估
- **公平对比**：所有算法使用相同的数据流水线和评估指标
- **轻松复现**：内置配置文件，一键复现论文结果
- **快速原型**：专注于你的创新想法，而非工程细节

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

**36 个数据集**，支持多种信号模态，并提供**自动下载**功能。

<!-- 快速统计 -->
<table>
<tr>
<td align="center"><b>RSSI</b><br/>10</td>
<td align="center"><b>CSI</b><br/>16</td>
<td align="center"><b>ToF</b><br/>2</td>
<td align="center"><b>磁场</b><br/>1</td>
<td align="center"><b>混合</b><br/>4</td>
<td align="center"><b>其他</b><br/>3</td>
</tr>
</table>

<details open>
<summary><b>按信号类型查看</b></summary>

| RSSI (10) | CSI (16) | ToF/TWR (2) | 磁场 (1) | 多模态 (4) | 其他 (3) |
|-----------|----------|-------------|----------|------------|----------|
| **WiFi RSSI** | **WiFi CSI** | [UWBIndoor](indoorloc/datasets/uwb_indoor.py) | [MagneticIndoor](indoorloc/datasets/magnetic_indoor.py) | [WiFiIMUHybrid](indoorloc/datasets/wifi_imu_hybrid.py) | [VLCIndoor](indoorloc/datasets/vlc_indoor.py) |
| [UJIndoorLoc](indoorloc/datasets/ujindoorloc.py) | [CSIIndoor](indoorloc/datasets/csi_indoor.py) | [UWBRanging](indoorloc/datasets/uwb_ranging.py) | | [WiFiMagneticHybrid](indoorloc/datasets/wifi_magnetic_hybrid.py) | [UltrasoundIndoor](indoorloc/datasets/ultrasound_indoor.py) |
| [SODIndoorLoc](indoorloc/datasets/sodindoorloc.py) | [CSIFingerprint](indoorloc/datasets/csi_fingerprint.py) | | | [MultiModalIndoor](indoorloc/datasets/multimodal_indoor.py) | [RFIDIndoor](indoorloc/datasets/rfid_indoor.py) |
| [LongTermWiFi](indoorloc/datasets/longtermwifi.py) | [HWILD](indoorloc/datasets/hwild.py) | | | [SensorFusion](indoorloc/datasets/sensor_fusion.py) | |
| [Tampere](indoorloc/datasets/tampere.py) | [CSUIndoorLoc](indoorloc/datasets/csu_csi_rssi.py) | | | | |
| [WLANRSSI](indoorloc/datasets/wlanrssi.py) | [WILDv2](indoorloc/datasets/wild_v2.py) | | | | |
| [TUJI1](indoorloc/datasets/tuji1.py) | [HALOC](indoorloc/datasets/haloc.py) | | | | |
| [RSSBased](indoorloc/datasets/rss_based.py) | [CSIBench](indoorloc/datasets/csi_bench.py) | | | | |
| **BLE RSSI** | [WiFiCSID2D](indoorloc/datasets/wifi_csi_d2d.py) | | | | |
| [iBeaconRSSI](indoorloc/datasets/ibeacon_rssi.py) | **LTE CSI** | | | | |
| [BLEIndoor](indoorloc/datasets/ble_indoor.py) | [OpenCSI](indoorloc/datasets/opencsi.py) | | | | |
| [BLERSSIU_UCI](indoorloc/datasets/ble_rssi_uci.py) | **Massive MIMO** | | | | |
| | [MaMIMOCSI](indoorloc/datasets/mamimo_csi.py) | | | | |
| | [DICHASUS](indoorloc/datasets/dichasus.py) | | | | |
| | [ESPARGOS](indoorloc/datasets/espargos.py) | | | | |
| | [MaMIMOUAV](indoorloc/datasets/mamimo_uav.py) | | | | |
| | [DeepMIMO](indoorloc/datasets/deepmimo.py) | | | | |
| | **RF CSI** | | | | |
| | [CSI2Pos](indoorloc/datasets/csi2pos.py) | | | | |
| | [CSI2TAoA](indoorloc/datasets/csi2taoa.py) | | | | |

</details>

<details>
<summary><b>按信号源查看</b></summary>

| WiFi (15) | BLE (3) | UWB (2) | Massive MIMO (5) | LTE (1) | 其他 (10) |
|-----------|---------|---------|------------------|---------|-----------|
| [UJIndoorLoc](indoorloc/datasets/ujindoorloc.py) | [iBeaconRSSI](indoorloc/datasets/ibeacon_rssi.py) | [UWBIndoor](indoorloc/datasets/uwb_indoor.py) | [MaMIMOCSI](indoorloc/datasets/mamimo_csi.py) | [OpenCSI](indoorloc/datasets/opencsi.py) | [MagneticIndoor](indoorloc/datasets/magnetic_indoor.py) |
| [SODIndoorLoc](indoorloc/datasets/sodindoorloc.py) | [BLEIndoor](indoorloc/datasets/ble_indoor.py) | [UWBRanging](indoorloc/datasets/uwb_ranging.py) | [DICHASUS](indoorloc/datasets/dichasus.py) | | [WiFiIMUHybrid](indoorloc/datasets/wifi_imu_hybrid.py) |
| [LongTermWiFi](indoorloc/datasets/longtermwifi.py) | [BLERSSIU_UCI](indoorloc/datasets/ble_rssi_uci.py) | | [ESPARGOS](indoorloc/datasets/espargos.py) | | [WiFiMagneticHybrid](indoorloc/datasets/wifi_magnetic_hybrid.py) |
| [Tampere](indoorloc/datasets/tampere.py) | | | [MaMIMOUAV](indoorloc/datasets/mamimo_uav.py) | | [MultiModalIndoor](indoorloc/datasets/multimodal_indoor.py) |
| [WLANRSSI](indoorloc/datasets/wlanrssi.py) | | | [DeepMIMO](indoorloc/datasets/deepmimo.py) | | [SensorFusion](indoorloc/datasets/sensor_fusion.py) |
| [TUJI1](indoorloc/datasets/tuji1.py) | | | | | [CSI2Pos](indoorloc/datasets/csi2pos.py) |
| [RSSBased](indoorloc/datasets/rss_based.py) | | | | | [CSI2TAoA](indoorloc/datasets/csi2taoa.py) |
| [CSIIndoor](indoorloc/datasets/csi_indoor.py) | | | | | [VLCIndoor](indoorloc/datasets/vlc_indoor.py) |
| [CSIFingerprint](indoorloc/datasets/csi_fingerprint.py) | | | | | [UltrasoundIndoor](indoorloc/datasets/ultrasound_indoor.py) |
| [HWILD](indoorloc/datasets/hwild.py) | | | | | [RFIDIndoor](indoorloc/datasets/rfid_indoor.py) |
| [CSUIndoorLoc](indoorloc/datasets/csu_csi_rssi.py) | | | | | |
| [WILDv2](indoorloc/datasets/wild_v2.py) | | | | | |
| [HALOC](indoorloc/datasets/haloc.py) | | | | | |
| [CSIBench](indoorloc/datasets/csi_bench.py) | | | | | |
| [WiFiCSID2D](indoorloc/datasets/wifi_csi_d2d.py) | | | | | |

</details>

<details>
<summary><b>按数据源查看</b></summary>

| UCI (5) | Zenodo (10) | GitHub (8) | Figshare (2) | 其他 (11) |
|---------|-------------|------------|--------------|-----------|
| [UJIndoorLoc](indoorloc/datasets/ujindoorloc.py) | [LongTermWiFi](indoorloc/datasets/longtermwifi.py) | [SODIndoorLoc](indoorloc/datasets/sodindoorloc.py) | [OpenCSI](indoorloc/datasets/opencsi.py) | [WILDv2](indoorloc/datasets/wild_v2.py) *(Kaggle)* |
| [WLANRSSI](indoorloc/datasets/wlanrssi.py) | [Tampere](indoorloc/datasets/tampere.py) | [TUJI1](indoorloc/datasets/tuji1.py) | [WiFiCSID2D](indoorloc/datasets/wifi_csi_d2d.py) | [MaMIMOCSI](indoorloc/datasets/mamimo_csi.py) *(IEEE)* |
| [iBeaconRSSI](indoorloc/datasets/ibeacon_rssi.py) | [RSSBased](indoorloc/datasets/rss_based.py) | [CSIIndoor](indoorloc/datasets/csi_indoor.py) | | [DICHASUS](indoorloc/datasets/dichasus.py) *(DaRUS)* |
| [BLERSSIU_UCI](indoorloc/datasets/ble_rssi_uci.py) | [UWBIndoor](indoorloc/datasets/uwb_indoor.py) | [CSIFingerprint](indoorloc/datasets/csi_fingerprint.py) | | [ESPARGOS](indoorloc/datasets/espargos.py) *(Web)* |
| [WiFiMagneticHybrid](indoorloc/datasets/wifi_magnetic_hybrid.py) | [UWBRanging](indoorloc/datasets/uwb_ranging.py) | [HWILD](indoorloc/datasets/hwild.py) | | [MaMIMOUAV](indoorloc/datasets/mamimo_uav.py) *(DOI)* |
| | [MagneticIndoor](indoorloc/datasets/magnetic_indoor.py) | [CSUIndoorLoc](indoorloc/datasets/csu_csi_rssi.py) | | [DeepMIMO](indoorloc/datasets/deepmimo.py) *(Web)* |
| | [WiFiIMUHybrid](indoorloc/datasets/wifi_imu_hybrid.py) | [BLEIndoor](indoorloc/datasets/ble_indoor.py) | | [CSI2Pos](indoorloc/datasets/csi2pos.py) *(TIB)* |
| | [MultiModalIndoor](indoorloc/datasets/multimodal_indoor.py) | [HALOC](indoorloc/datasets/haloc.py) | | [CSI2TAoA](indoorloc/datasets/csi2taoa.py) *(TIB)* |
| | [SensorFusion](indoorloc/datasets/sensor_fusion.py) | | | [CSIBench](indoorloc/datasets/csi_bench.py) *(Web)* |
| | [VLCIndoor](indoorloc/datasets/vlc_indoor.py) | | | [RFIDIndoor](indoorloc/datasets/rfid_indoor.py) *(Web)* |
| | [UltrasoundIndoor](indoorloc/datasets/ultrasound_indoor.py) | | | |

</details>

<details>
<summary><b>按定位类型查看</b></summary>

| 2D (11) | 2D + 楼层 (21) | 3D (4) |
|---------|----------------|--------|
| [WLANRSSI](indoorloc/datasets/wlanrssi.py) | [UJIndoorLoc](indoorloc/datasets/ujindoorloc.py) | [UWBIndoor](indoorloc/datasets/uwb_indoor.py) |
| [RSSBased](indoorloc/datasets/rss_based.py) | [SODIndoorLoc](indoorloc/datasets/sodindoorloc.py) | [DICHASUS](indoorloc/datasets/dichasus.py) |
| [iBeaconRSSI](indoorloc/datasets/ibeacon_rssi.py) | [LongTermWiFi](indoorloc/datasets/longtermwifi.py) | [MaMIMOUAV](indoorloc/datasets/mamimo_uav.py) |
| [BLERSSIU_UCI](indoorloc/datasets/ble_rssi_uci.py) | [Tampere](indoorloc/datasets/tampere.py) | [DeepMIMO](indoorloc/datasets/deepmimo.py) |
| [CSIFingerprint](indoorloc/datasets/csi_fingerprint.py) | [TUJI1](indoorloc/datasets/tuji1.py) | |
| [CSIBench](indoorloc/datasets/csi_bench.py) | [BLEIndoor](indoorloc/datasets/ble_indoor.py) | |
| [WiFiCSID2D](indoorloc/datasets/wifi_csi_d2d.py) | [CSIIndoor](indoorloc/datasets/csi_indoor.py) | |
| [OpenCSI](indoorloc/datasets/opencsi.py) | [HWILD](indoorloc/datasets/hwild.py) | |
| [MaMIMOCSI](indoorloc/datasets/mamimo_csi.py) | [CSUIndoorLoc](indoorloc/datasets/csu_csi_rssi.py) | |
| [ESPARGOS](indoorloc/datasets/espargos.py) | [WILDv2](indoorloc/datasets/wild_v2.py) | |
| [UWBRanging](indoorloc/datasets/uwb_ranging.py) | [HALOC](indoorloc/datasets/haloc.py) | |
| | [MagneticIndoor](indoorloc/datasets/magnetic_indoor.py) | |
| | [WiFiIMUHybrid](indoorloc/datasets/wifi_imu_hybrid.py) | |
| | [WiFiMagneticHybrid](indoorloc/datasets/wifi_magnetic_hybrid.py) | |
| | [MultiModalIndoor](indoorloc/datasets/multimodal_indoor.py) | |
| | [SensorFusion](indoorloc/datasets/sensor_fusion.py) | |
| | [VLCIndoor](indoorloc/datasets/vlc_indoor.py) | |
| | [UltrasoundIndoor](indoorloc/datasets/ultrasound_indoor.py) | |
| | [RFIDIndoor](indoorloc/datasets/rfid_indoor.py) | |
| | [CSI2Pos](indoorloc/datasets/csi2pos.py) | |
| | [CSI2TAoA](indoorloc/datasets/csi2taoa.py) | |

</details>

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
