<div align="center">

# IndoorLoc

**室内定位统一框架**

[![PyPI version](https://badge.fury.io/py/indoorloc.svg)](https://badge.fury.io/py/indoorloc)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[English](README.md) | [中文](README_zh.md)

</div>

---

## 简介

IndoorLoc 是一个室内定位统一框架，灵感来源于 [MMPretrain](https://github.com/open-mmlab/mmpretrain)。它为室内定位研究和开发提供了一套完整的解决方案。

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

- [x] UJIndoorLoc
- [ ] Tampere
- [ ] Microsoft Indoor Localization
- [ ] 自定义数据集

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
