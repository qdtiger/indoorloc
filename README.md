<div align="center">

<img src="assets/logo.png" width="600">

**室内定位工具库 | Multi-dataset, multi-model indoor localization toolkit**

[![PyPI](https://img.shields.io/pypi/v/indoorloc)](https://pypi.org/project/indoorloc/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Stars](https://img.shields.io/github/stars/qdtiger/indoorloc?style=social)](https://github.com/qdtiger/indoorloc)

[English](README.md) | [中文](README_zh.md)

</div>

---

### For Beginners: 3 Lines = Complete Workflow

Skip the boilerplate. Focus on algorithms, not data formats.

```python
import indoorloc as iloc

train, test = iloc.load_dataset('ujindoorloc')           # Load any of 36+ datasets
model = iloc.create_model('resnet18', dataset=train)     # Auto-configure model
results = model.fit(train).evaluate(test)                # Train & evaluate
```

Auto-download datasets · Auto-adapt dimensions · Auto-configure model

### For Experts: YAML Config + CLI

Full control via OpenMMLab-style configuration system.

```bash
python tools/train.py configs/wifi/resnet18_ujindoorloc.yaml

# Override any parameter
python tools/train.py configs/wifi/resnet18_ujindoorloc.yaml \
    --model.backbone.model_name efficientnet_b0 \
    --train.lr 5e-4 --train.epochs 200
```

```yaml
# configs/wifi/resnet18_ujindoorloc.yaml
_base_:
  - ../_base_/models/resnet.yaml

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
```

---

## Supported Datasets

**WiFi, BLE, CSI, UWB, Magnetic, and more** — all with `iloc.load_dataset()`. [View Details →](https://qdtiger.github.io/indoorloc/datasets.html)

<table>
<tr>
<th align="center">RSSI</th>
<th align="center">CSI</th>
<th align="center">Other</th>
</tr>
<tr>
<td valign="top">

**WiFi**
- [UJIndoorLoc](https://archive.ics.uci.edu/dataset/310/ujiindoorloc)
- [SODIndoorLoc](https://github.com/renwudao24/SODIndoorLoc)
- [LongTermWiFi](https://zenodo.org/record/889798)
- [Tampere](https://zenodo.org/record/1066041)
- [TUJI1](https://github.com/IndoorLocation/IPIN2021-Competition-Track3-Dataset)

**BLE**
- [iBeaconRSSI](https://zenodo.org/record/1066044)
- [BLEIndoor](https://github.com/BLE-Indoor-Positioning/Dataset)
- [BLERSSIU_UCI](https://archive.ics.uci.edu/dataset/519/ble+rssi+dataset+for+indoor+localization)

</td>
<td valign="top">

**WiFi CSI**
- [CSIIndoor](https://github.com/CSI-Positioning/IndoorDataset)
- [HWILD](https://github.com/H-WILD/human_held_device_wifi_indoor_localization_dataset)
- [HALOC](https://zenodo.org/records/10715595)
- [CSIBench](https://ai-iot-sensing.github.io/projects/project.html)

**Massive MIMO**
- [MaMIMOCSI](https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset)
- [DICHASUS](https://darus.uni-stuttgart.de/dataverse/dichasus)
- [DeepMIMO](https://www.deepmimo.net)

</td>
<td valign="top">

**UWB**
- [UWBIndoor](https://zenodo.org/record/5789876)
- [UWBRanging](https://github.com/UWB-Positioning/RangingDataset)

**Magnetic**
- [MagneticIndoor](https://zenodo.org/record/4321098)

**Fusion**
- [WiFiIMUHybrid](https://zenodo.org/record/3932395)
- [MultiModalIndoor](https://github.com/IndoorPositioning/MultiModalDataset)

**VLC / RFID / Ultrasound**
- [VLCIndoor](https://github.com/VLC-Positioning/IndoorDataset)
- [RFIDIndoor](https://archive.ics.uci.edu/dataset/640/rfid+indoor+localization)

</td>
</tr>
</table>

---

## Supported Algorithms

**Traditional ML + 700+ Deep Learning backbones** via [timm](https://github.com/huggingface/pytorch-image-models). [View Details →](https://qdtiger.github.io/indoorloc/algorithms.html)

<table>
<tr>
<th align="center">Traditional</th>
<th align="center">Deep Learning Backbones</th>
<th align="center">Prediction Heads</th>
</tr>
<tr>
<td valign="top">

- [x] k-NN
- [x] Weighted k-NN
- [ ] SVM
- [ ] Random Forest
- [ ] Gaussian Process

</td>
<td valign="top">

**CNN**: ResNet, EfficientNet, ConvNeXt, MobileNet, RegNet, DenseNet...

**ViT**: ViT, Swin, DeiT, BEiT, EVA...

**Hybrid**: CoAtNet, MaxViT, EfficientFormer...

</td>
<td valign="top">

- [x] RegressionHead
- [x] ClassificationHead
- [x] FloorHead
- [x] BuildingHead
- [x] HybridHead
- [x] HierarchicalHead

</td>
</tr>
</table>

---

## Installation

```bash
pip install indoorloc
```

<details>
<summary>More options</summary>

```bash
pip install indoorloc[vision]   # With vision support
pip install indoorloc[full]     # All features
pip install -e ".[dev]"         # Development
```

</details>

---

<details>
<summary><b>Advanced Usage</b></summary>

### YAML Configuration

```yaml
# configs/wifi/resnet18_ujindoorloc.yaml
_base_:
  - ../_base_/models/resnet.yaml

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
```

```bash
python tools/train.py configs/wifi/resnet18_ujindoorloc.yaml
```

### Custom Model Registration

```python
from indoorloc.registry import LOCALIZERS
from indoorloc.localizers.base import BaseLocalizer

@LOCALIZERS.register_module()
class MyLocalizer(BaseLocalizer):
    def fit(self, signals, locations, **kwargs):
        self._is_trained = True
        return self

    def predict(self, signal):
        pass

model = iloc.create_model('MyLocalizer')
```

### Project Structure

```
indoorloc/
├── signals/          # WiFi, BLE, IMU, etc.
├── locations/        # Location classes
├── datasets/         # 36+ datasets
├── localizers/       # ML & DL algorithms
├── evaluation/       # Metrics
└── configs/          # YAML configs
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Mean Position Error | Average error (m) |
| Median Position Error | Median error (m) |
| Floor Accuracy | Floor classification |
| Building Accuracy | Building classification |

</details>

---

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

- [OpenMMLab](https://github.com/open-mmlab) — Registry and config system
- [timm](https://github.com/huggingface/pytorch-image-models) — 700+ pretrained models
