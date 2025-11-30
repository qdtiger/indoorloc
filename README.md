<div align="center">

<img src="assets/logo.png" width="600">

**Open Source Indoor Localization Toolbox**

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

## Introduction

**3 lines of code** to train and evaluate. **Full control** when you need it.

IndoorLoc provides a unified interface for 36+ indoor localization datasets across WiFi, BLE, UWB, CSI, and more.

### For Beginners: 3 Lines = Complete Workflow

Skip the boilerplate. Focus on algorithms, not data formats.

```python
import indoorloc as iloc

train, test = iloc.load_dataset('ujindoorloc')           # Load any of 36+ datasets
model = iloc.create_model('resnet18', dataset=train)     # Auto-configure model
results = model.fit(train).evaluate(test)                # Train & evaluate
```

✓ Auto-download datasets | ✓ Auto-adapt dimensions | ✓ Auto-configure model

### For Experts: YAML Config + CLI

Full control via OpenMMLab-style configuration system.

```bash
# Train with config file
python tools/train.py configs/wifi/resnet18_ujindoorloc.yaml

# Override any parameter from command line
python tools/train.py configs/wifi/resnet18_ujindoorloc.yaml \
    --model.backbone.model_name efficientnet_b0 \
    --train.lr 5e-4 --train.epochs 200
```

```yaml
# configs/wifi/resnet18_ujindoorloc.yaml
_base_:
  - ../_base_/models/resnet.yaml
  - ../_base_/schedules/schedule_1x.yaml

model:
  backbone:
    model_name: resnet18
    pretrained: true
  head:
    num_floors: 5
    num_buildings: 3

train:
  epochs: 100
  lr: 0.001
  fp16: true
```

### Key Contributions

| Feature | Description |
|---------|-------------|
| **Unified Dataset API** | 36+ datasets with identical `load_dataset()` interface |
| **Auto-Configuration** | Model dimensions auto-adapt to any dataset |
| **Dual Interface** | Code API for beginners, YAML configs for experts |
| **Registry System** | OpenMMLab-style extensible architecture |

## Features

- **Multi-Algorithm Support**: k-NN, SVM, Random Forest, Deep Learning (CNN, LSTM, Transformer)
- **Multi-Modal**: WiFi, BLE, IMU, Visual, UWB signal support
- **Multi-Dataset**: UJIndoorLoc, Tampere, Microsoft Indoor, and more
- **Unified Interface**: Consistent API across all algorithms
- **Easy Configuration**: YAML-based configuration with inheritance
- **Extensible**: Registry-based plugin architecture
- **PyPI Ready**: Easy installation via pip

## Installation

```bash
# Basic installation
pip install indoorloc

# With vision support
pip install indoorloc[vision]

# Full installation (all features)
pip install indoorloc[full]

# Development installation
git clone https://github.com/qdtiger/indoorloc.git
cd indoorloc
pip install -e ".[dev]"
```

## Quick Start

```python
import indoorloc as iloc
import numpy as np

# Create a k-NN localizer
model = iloc.create_model('KNNLocalizer', k=5)

# Prepare training data
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

# Train the model
model.fit(train_signals, train_locations)

# Make predictions
test_signal = iloc.WiFiSignal(rssi_values=np.random.randn(520).astype(np.float32))
result = model.predict(test_signal)

print(f"Predicted position: ({result.x:.2f}, {result.y:.2f})")
print(f"Predicted floor: {result.floor}")
```

## Using Configuration Files

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

# Load from config
model = iloc.create_model(config='configs/wifi/knn_ujindoorloc.yaml')
```

## Custom Model Registration

```python
from indoorloc.registry import LOCALIZERS
from indoorloc.localizers.base import BaseLocalizer

@LOCALIZERS.register_module()
class MyCustomLocalizer(BaseLocalizer):
    def __init__(self, custom_param=1.0, **kwargs):
        super().__init__(**kwargs)
        self.custom_param = custom_param

    @property
    def localizer_type(self) -> str:
        return 'my_custom'

    def fit(self, signals, locations, **kwargs):
        # Training logic
        self._is_trained = True
        return self

    def predict(self, signal):
        # Prediction logic
        pass

# Use the custom model
model = iloc.create_model('MyCustomLocalizer', custom_param=2.0)
```

## Project Structure

```
indoorloc/
├── indoorloc/
│   ├── signals/          # Signal abstractions (WiFi, BLE, IMU, etc.)
│   ├── locations/        # Location and coordinate classes
│   ├── datasets/         # Dataset implementations
│   ├── localizers/       # Localization algorithms
│   │   ├── fingerprint/  # Traditional ML (k-NN, SVM, RF)
│   │   ├── deep/         # Deep learning (CNN, LSTM, Transformer)
│   │   └── pdr/          # Inertial navigation
│   ├── fusion/           # Multi-sensor fusion
│   ├── evaluation/       # Metrics and evaluation
│   ├── engine/           # Training utilities
│   ├── visualization/    # Plotting tools
│   ├── configs/          # Built-in configurations
│   └── utils/            # Utility functions
├── tools/                # CLI tools
├── examples/             # Usage examples
├── tests/                # Unit tests
└── docs/                 # Documentation
```

## Supported Datasets

**36 datasets** across multiple signal modalities with **auto-download** support. [View Details →](https://qdtiger.github.io/indoorloc/datasets.html)

<table>
<tr>
<th align="center" bgcolor="#f6f8fa">RSSI</th>
<th align="center" bgcolor="#f6f8fa">CSI</th>
<th align="center" bgcolor="#f6f8fa">Other</th>
</tr>
<tr>
<td valign="top">

**WiFi**
- [UJIndoorLoc](https://archive.ics.uci.edu/dataset/310/ujiindoorloc)
- [SODIndoorLoc](https://github.com/renwudao24/SODIndoorLoc)
- [LongTermWiFi](https://zenodo.org/record/889798)
- [Tampere](https://zenodo.org/record/1066041)
- [WLANRSSI](https://archive.ics.uci.edu/dataset/422/localization+data+for+person+activity)
- [TUJI1](https://github.com/IndoorLocation/IPIN2021-Competition-Track3-Dataset)
- [RSSBased](https://zenodo.org/record/5678901)

**BLE**
- [iBeaconRSSI](https://zenodo.org/record/1066044)
- [BLEIndoor](https://github.com/BLE-Indoor-Positioning/Dataset)
- [BLERSSIU_UCI](https://archive.ics.uci.edu/dataset/519/ble+rssi+dataset+for+indoor+localization)

</td>
<td valign="top">

**WiFi**
- [CSIIndoor](https://github.com/CSI-Positioning/IndoorDataset)
- [CSIFingerprint](https://github.com/qiang5love1314/CSI-dataset)
- [HWILD](https://github.com/H-WILD/human_held_device_wifi_indoor_localization_dataset)
- [CSUIndoorLoc](https://github.com/EPIC-CSU/csi-rssi-dataset-indoor-nav)
- [WILDv2](https://www.kaggle.com/c/wild-v2)
- [HALOC](https://zenodo.org/records/10715595)
- [CSIBench](https://ai-iot-sensing.github.io/projects/project.html)
- [WiFiCSID2D](https://figshare.com/articles/dataset/WiFi_CSI_D2D/20943706)

**LTE**
- [OpenCSI](https://figshare.com/articles/dataset/OpenCSI/19596379)

**Massive MIMO**
- [MaMIMOCSI](https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset)
- [DICHASUS](https://darus.uni-stuttgart.de/dataverse/dichasus)
- [ESPARGOS](https://espargos.net/datasets/)
- [MaMIMOUAV](https://doi.org/10.48804/0IMQDF)
- [DeepMIMO](https://www.deepmimo.net)

**RF**
- [CSI2Pos](https://service.tib.eu/ldmservice/dataset/csi2pos)
- [CSI2TAoA](https://service.tib.eu/ldmservice/dataset/csi2taoa)

</td>
<td valign="top">

**UWB**
- [UWBIndoor](https://zenodo.org/record/5789876)
- [UWBRanging](https://github.com/UWB-Positioning/RangingDataset)

**Magnetic**
- [MagneticIndoor](https://zenodo.org/record/4321098)

**Fusion**
- [WiFiIMUHybrid](https://zenodo.org/record/3932395)
- [WiFiMagneticHybrid](https://archive.ics.uci.edu/dataset/626/wifi+magnetic+indoor+localization)
- [MultiModalIndoor](https://github.com/IndoorPositioning/MultiModalDataset)
- [SensorFusion](https://zenodo.org/record/4567890)

**VLC**
- [VLCIndoor](https://github.com/VLC-Positioning/IndoorDataset)

**Ultrasound**
- [UltrasoundIndoor](https://archive.ics.uci.edu/dataset/632/ultrasound+indoor+localization)

**RFID**
- [RFIDIndoor](https://archive.ics.uci.edu/dataset/640/rfid+indoor+localization)

</td>
</tr>
</table>

<sub>📋 All datasets retain their original licenses. Downloads use official sources/APIs when available. [View license details →](https://qdtiger.github.io/indoorloc/datasets.html#licenses)</sub>

### Auto-Download Usage

```python
import indoorloc as iloc

# UJIndoorLoc - Auto-download to ~/.cache/indoorloc/datasets/
dataset = iloc.UJIndoorLoc(download=True)
train = iloc.UJIndoorLoc(download=True, split='train')
test = iloc.UJIndoorLoc(download=True, split='test')

# SODIndoorLoc - Specify building
cetc_train = iloc.SODIndoorLoc(building='CETC331', download=True)
hcxy_train = iloc.SODIndoorLoc(building='HCXY', download=True)

# Or specify custom directory
dataset = iloc.UJIndoorLoc(data_root='./data', download=True)
```

### Manual Download

You can also download datasets manually from official sources:

| Dataset | Official Source | Notes |
|---------|-----------------|-------|
| UJIndoorLoc | [UCI ML Repository](https://archive.ics.uci.edu/dataset/310/ujiindoorloc) | Direct ZIP download |
| SODIndoorLoc | [GitHub Repository](https://github.com/renwudao24/SODIndoorLoc) | CSV files on GitHub |
| Microsoft Indoor 2.0 | [Microsoft Research](https://aka.ms/location20dataset) + [GitHub](https://github.com/location-competition/indoor-location-competition-20) | Multi-sensor (WiFi, BLE, IMU, Magnetometer) |
| TUJI1 | [Tampere University](https://trepo.tuni.fi/handle/10024/211225) + [Zenodo](https://zenodo.org/records/1226835) | Multi-device collection |
| WiFi-RSSI | [GitHub Repository](https://github.com/m-nabati/WiFi-RSSI-Localization-Dataset) | Small-scale (250 points) |
| OWP-IMU | [arXiv](https://arxiv.org/abs/2505.16823) | Optical wireless + IMU fusion |

## Supported Algorithms

**Traditional + Deep Learning** localization methods with unified interface. Deep learning models powered by [timm](https://github.com/huggingface/pytorch-image-models) (700+ pretrained architectures).

<table>
<tr>
<th align="center" bgcolor="#f6f8fa">Traditional</th>
<th align="center" bgcolor="#f6f8fa">Deep Learning Backbones <sub><i>(via timm)</i></sub></th>
<th align="center" bgcolor="#f6f8fa">Prediction Heads</th>
</tr>
<tr>
<td valign="top">

**Fingerprint-based**
- [x] [k-NN](https://ieeexplore.ieee.org/document/1053964)
- [x] [Weighted k-NN](https://ieeexplore.ieee.org/document/4309523)
- [ ] [SVM](https://link.springer.com/article/10.1007/BF00994018)
- [ ] [Random Forest](https://link.springer.com/article/10.1023/A:1010933404324)
- [ ] [Gaussian Process](http://www.gaussianprocess.org/gpml/)

**Fusion**
- [ ] [Kalman Filter](https://asmedigitalcollection.asme.org/fluidsengineering/article/82/1/35/397706)
- [ ] [Extended Kalman Filter](https://ieeexplore.ieee.org/document/1098671)
- [ ] [Particle Filter](https://ieeexplore.ieee.org/document/210672)

</td>
<td valign="top">

**CNN**
- [ResNet](https://arxiv.org/abs/1512.03385)
- [EfficientNet](https://arxiv.org/abs/1905.11946)
- [ConvNeXt](https://arxiv.org/abs/2201.03545)
- [MobileNetV3](https://arxiv.org/abs/1905.02244)
- [RegNet](https://arxiv.org/abs/2003.13678), [DenseNet](https://arxiv.org/abs/1608.06993), [VGG](https://arxiv.org/abs/1409.1556)...

**Vision Transformer**
- [ViT](https://arxiv.org/abs/2010.11929)
- [Swin](https://arxiv.org/abs/2103.14030)
- [DeiT](https://arxiv.org/abs/2012.12877), [BEiT](https://arxiv.org/abs/2106.08254), [EVA](https://arxiv.org/abs/2211.07636)...

**Hybrid**
- [CoAtNet](https://arxiv.org/abs/2106.04803), [MaxViT](https://arxiv.org/abs/2204.01697)
- [EfficientFormer](https://arxiv.org/abs/2206.01191)...

<sub>📦 `pip install timm torchvision`</sub>

</td>
<td valign="top">

**Regression**
- [x] RegressionHead
- [x] MultiScaleRegressionHead

**Classification**
- [x] ClassificationHead
- [x] FloorHead
- [x] BuildingHead

**Hybrid**
- [x] HybridHead
- [x] HierarchicalHead

</td>
</tr>
</table>

### Deep Learning Usage

```python
import indoorloc as iloc

# End-to-end deep localizer with ResNet backbone
model = iloc.DeepLocalizer(
    backbone=dict(
        type='TimmBackbone',
        model_name='resnet18',    # or 'efficientnet_b0', 'vit_tiny', 'swin_tiny'...
        pretrained=True,
        input_type='1d',          # '1d' for RSSI, '2d' for CSI
    ),
    head=dict(
        type='HybridHead',        # Joint coordinate + floor prediction
        num_coords=2,
        num_floors=4,
    ),
)

# Forward pass
coords, floor_logits = model(wifi_rssi_tensor)
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Mean Position Error | Average localization error (meters) |
| Median Position Error | Median localization error (meters) |
| 75th/95th Percentile | 75%/95% percentile error |
| Floor Accuracy | Floor classification accuracy |
| Building Accuracy | Building classification accuracy |
| CDF Analysis | Cumulative distribution function analysis |

## License

Apache License 2.0

## Citation

```bibtex
@software{indoorloc,
  title = {IndoorLoc: A Unified Framework for Indoor Localization},
  year = {2024},
  url = {https://github.com/qdtiger/indoorloc}
}
```

## Acknowledgements

- [OpenMMLab](https://github.com/open-mmlab) - Registry and config system design reference
- [timm](https://github.com/huggingface/pytorch-image-models) - PyTorch Image Models (700+ pretrained architectures)
- [UJIndoorLoc](https://archive.ics.uci.edu/dataset/310/ujiindoorloc) - Dataset provider

## Contributing

We welcome issues and pull requests!
