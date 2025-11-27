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

**One line of code** to load any indoor localization dataset. **One line of code** to train and evaluate.

```python
train, test = iloc.UJIndoorLoc(download=True)  # That's it. Auto-download, auto-parse, ready to use.
```

IndoorLoc eliminates the painful data preprocessing that every indoor localization researcher faces. No more parsing CSVs, handling missing values, or writing data loaders from scratch.

### For Beginners

**One line = One dataset.** Focus on learning algorithms, not fighting with data formats.

```python
# Load 36+ datasets with identical API
train, test = iloc.UJIndoorLoc(download=True)    # WiFi RSSI
train, test = iloc.CSIIndoor(download=True)       # WiFi CSI
train, test = iloc.UWBIndoor(download=True)       # UWB ranging
```

### For Experts

**Full control when you need it.** Flexible data pipeline, customizable preprocessing, plug-in your own algorithms.

```python
# Custom preprocessing pipeline
dataset = iloc.UJIndoorLoc(
    download=True,
    transform=iloc.Compose([
        iloc.RSSINormalize(method='minmax'),
        iloc.APFilter(threshold=-90),
    ])
)

# Register your own algorithm
@LOCALIZERS.register_module()
class MyNovelLocalizer(BaseLocalizer):
    ...
```

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

## Supported Algorithms

### Fingerprint-based
- [x] k-NN (k-Nearest Neighbors)
- [x] Weighted k-NN
- [ ] SVM (Support Vector Machine)
- [ ] Random Forest
- [ ] Gaussian Process

### Deep Learning
- [ ] MLP (Multi-Layer Perceptron)
- [ ] CNN (Convolutional Neural Network)
- [ ] LSTM (Long Short-Term Memory)
- [ ] Transformer

### Fusion
- [ ] Kalman Filter
- [ ] Extended Kalman Filter
- [ ] Particle Filter

## Supported Datasets

**36 datasets** across multiple signal modalities with **auto-download** support. [📊 Interactive Browser](https://qdtiger.github.io/indoorloc/datasets.html)

<table>
<thead>
<tr>
<th>RSSI</th>
<th>CSI</th>
<th>ToF/TWR</th>
<th>Magnetic</th>
<th>Multi-modal</th>
<th>Other</th>
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
- [UJIndoorLoc](https://archive.ics.uci.edu/dataset/310/ujiindoorloc) - Dataset provider

## Contributing

We welcome issues and pull requests!
