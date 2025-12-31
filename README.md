<div align="center">

<img src="assets/logo.png" width="600">

**IndoorLoc | 室内定位工具库**  
Multi-dataset, multi-model indoor localization toolkit

[![PyPI](https://img.shields.io/pypi/v/indoorloc)](https://pypi.org/project/indoorloc/)
[![CI](https://github.com/qdtiger/indoorloc/actions/workflows/ci.yml/badge.svg)](https://github.com/qdtiger/indoorloc/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://qdtiger.github.io/indoorloc/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Stars](https://img.shields.io/github/stars/qdtiger/indoorloc?style=social)](https://github.com/qdtiger/indoorloc)

[Docs](https://qdtiger.github.io/indoorloc/) · [Installation](#installation) · [Quickstart](#quickstart) · [Datasets](#datasets) · [Models](#models--algorithms) · [Contributing](#contributing) · [Citation](#citation)

[English](README.md) | [中文](README_zh.md)

</div>

---

## Highlights

- Unified API for indoor localization across WiFi / BLE / CSI / UWB
- 12 verified datasets (auto-download when available) + extensible dataset registry
- Classic ML (scikit-learn) + deep models (PyTorch, `timm`)
- OpenMMLab-style YAML configs for reproducible experiments

## Installation

### GPU (CUDA 11.8)

```bash
conda create -n indoorloc python=3.10 pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda activate indoorloc
pip install "indoorloc[full]"
```

### CPU-only

```bash
conda create -n indoorloc python=3.10 pytorch torchvision cpuonly -c pytorch -y
conda activate indoorloc
pip install "indoorloc[full]"
```

### Verify (optional)

```bash
python -c "import indoorloc, torch; print('indoorloc', indoorloc.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
```

More install options (legacy, optional extras, troubleshooting): `docs/installation.md`.

---

## Quickstart

### Python API (recommended)

```python
import indoorloc as iloc

train, test = iloc.load_dataset("ujindoorloc")            # 12 verified datasets
model = iloc.create_model("resnet18", dataset=train)      # Auto-configure model
results = model.fit(train).evaluate(test)                # Train & evaluate
```

Auto-download datasets · Auto-adapt dimensions · Auto-configure model

### YAML Config + CLI

Config templates live in `indoorloc/configs/`.

```bash
indoorloc-train indoorloc/configs/wifi/resnet18_ujindoorloc.yaml

# Override any parameter
indoorloc-train indoorloc/configs/wifi/resnet18_ujindoorloc.yaml \
  --model.backbone.model_name efficientnet_b0 \
  --train.lr 5e-4 --train.epochs 200
```

```yaml
# indoorloc/configs/wifi/resnet18_ujindoorloc.yaml
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

## Documentation

- Dataset catalogue: https://qdtiger.github.io/indoorloc/datasets.html
- Algorithm zoo: https://qdtiger.github.io/indoorloc/algorithms.html
- Config reference: `indoorloc/configs/README.md`

## Datasets

- List available dataset IDs: `iloc.list_available_datasets()`
- Load a dataset: `train, test = iloc.load_dataset("ujindoorloc")`

Verified dataset IDs (12):

- WiFi: `ujindoorloc`, `sodindoorloc`, `longtermwifi`, `tampere`, `wlanrssi`, `tuji1`
- BLE: `ble_indoor`, `ibeacon_rssi`, `ble_rssi_uci`
- CSI: `csi_fingerprint`, `hwild`, `haloc`

<details>
<summary>Verified datasets (table)</summary>

| Type | Dataset | ID | Samples |
|------|---------|-----|---------|
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
<summary>Pending datasets (help wanted)</summary>

These datasets have download sources but are **not yet integrated**:

| Dataset | Source | Notes |
|---------|--------|-------|
| OpenCSI | [Figshare](https://doi.org/10.6084/m9.figshare.19596379.v1) | ~2GB, format unverified |
| CSUIndoorLoc | [GitHub](https://github.com/EPIC-CSU/csi-rssi-dataset-indoor-nav) | Format unverified |
| DICHASUS | [DaRUS](https://darus.uni-stuttgart.de/dataverse/dichasus) | 14 scenarios |
| ESPARGOS | [espargos.net](https://espargos.net/datasets/) | 17-86GB |
| DeepMIMO | [deepmimo.net](https://www.deepmimo.net) | Requires `pip install DeepMIMO` |
| CSI2Pos | [TIB](https://service.tib.eu/ldmservice/dataset/csi2pos) | Requires login |
| CSI2TAoA | [TIB](https://service.tib.eu/ldmservice/dataset/csi2taoa) | Requires login |
| MaMIMO CSI | [IEEE DataPort](https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset) | Requires account |
| WILDv2 | [Kaggle](https://www.kaggle.com/competitions/wild-v2) | Requires Kaggle API |

> Contribute: help us verify pending datasets! See `CONTRIBUTING.md`.

</details>

## Models & Algorithms

- List available models: `iloc.list_models()`
- Create a model: `iloc.create_model("KNNLocalizer", k=5)` / `iloc.create_model("resnet18", dataset=train)`

<details>
<summary>Algorithm families</summary>

**Supervised**: [sklearn](https://scikit-learn.org/) (30+) + [timm](https://github.com/huggingface/pytorch-image-models) (700+) | **Self-supervised**: [lightly](https://github.com/lightly-ai/lightly) (10+) | **Meta-learning**: [learn2learn](https://github.com/learnables/learn2learn) (7+) | **Transfer**: [SKADA](https://github.com/scikit-adaptation/skada) (20+).

<table>
<tr>
<th align="center">Supervised</th>
<th align="center">Self-supervised</th>
<th align="center">Meta-learning</th>
<th align="center">Transfer</th>
</tr>
<tr>
<td valign="top">

**Traditional (30+)**
- k-NN, WKNN, SVM, RF...

**Deep (700+)**
- MLP, CNN1D, ResNet, ViT...

</td>
<td valign="top">

**Contrastive**
- SimCLR, MoCo, NNCLR

**Non-contrastive**
- BYOL, SimSiam, VICReg

</td>
<td valign="top">

**Gradient-based**
- MAML, FOMAML, Reptile

**Metric-based**
- ProtoNet, MatchingNet

</td>
<td valign="top">

**Feature**: CORAL, TCA

**Reweight**: KMM, KLIEP

**Deep**: DANN, MDD

</td>
</tr>
</table>

</details>

<details>
<summary><b>Advanced usage</b></summary>

### Custom model registration

```python
import indoorloc as iloc
from indoorloc.registry import LOCALIZERS
from indoorloc.localizers.base import BaseLocalizer

@LOCALIZERS.register_module()
class MyLocalizer(BaseLocalizer):
    def fit(self, signals, locations, **kwargs):
        self._is_trained = True
        return self

    def predict(self, signal):
        raise NotImplementedError

model = iloc.create_model("MyLocalizer")
```

### Project structure

```
indoorloc/
├── signals/          # WiFi, BLE, IMU, etc.
├── locations/        # Location classes
├── datasets/         # Verified + pending
├── localizers/       # ML & DL algorithms
├── evaluation/       # Metrics
└── configs/          # YAML configs
```

### Evaluation metrics

| Metric | Description |
|--------|-------------|
| Mean Position Error | Average error (m) |
| Median Position Error | Median error (m) |
| Floor Accuracy | Floor classification |
| Building Accuracy | Building classification |

</details>

## Contributing

See `CONTRIBUTING.md`.

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
