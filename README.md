<div align="center">

<img src="assets/logo.png" width="600">

**IndoorLoc | 室内定位工具库**

*Datasets · Algorithms · Evaluation — one reproducible stack for indoor wireless localization.*

[![PyPI](https://img.shields.io/pypi/v/indoorloc)](https://pypi.org/project/indoorloc/)
[![CI](https://github.com/qdtiger/indoorloc/actions/workflows/ci.yml/badge.svg)](https://github.com/qdtiger/indoorloc/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://qdtiger.github.io/indoorloc/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Stars](https://img.shields.io/github/stars/qdtiger/indoorloc?style=social)](https://github.com/qdtiger/indoorloc)

[Docs](https://qdtiger.github.io/indoorloc/) · [Installation](#installation) · [Quickstart](#quickstart) · [Datasets](#datasets) · [Models](#models--algorithms) · [Roadmap](#taxonomy--roadmap) · [Contributing](#contributing)

[English](README.md) | [中文](README_zh.md)

</div>

---

```python
import indoorloc as iloc

train, test = iloc.load_dataset("ujindoorloc")   # auto-download, unified format
model = iloc.create_model("wknn", k=5)           # classic ML or deep (timm) models
results = model.fit(train).evaluate(test)        # unified metrics
print(results)                                   # mean/median error · floor & building accuracy
```

## Why IndoorLoc?

Indoor-positioning research has a comparability problem: most papers never release code, few public datasets ship standard train/test splits, and published numbers are rarely reproducible across labs. Deployment services, dataset tools, and hundreds of single-paper repos exist — but no framework unifies the three things a benchmark needs.

IndoorLoc is built to be that missing layer:

- **Unified data** — one registry, auto-download, one sample format across WiFi / BLE / CSI datasets
- **Unified algorithms** — classic ML and deep models behind one `fit / predict / evaluate` API
- **Unified evaluation** — shared metrics (incl. floor/building accuracy) and published-benchmark comparison
- **Config-driven reproducibility** — YAML configs with `_base_` inheritance; a full experiment reruns from one command
- **À la carte** — every layer stands on its own: datasets export to standard formats for any framework, models train on your own arrays, and the evaluation protocols work with your own predictions (interfaces rolling out per the [development plan](docs/DEVELOPMENT_PLAN.md))

> **Dataset status.** In the table below, ✅ means the full auto-download → train → evaluate pipeline has been run end-to-end, with the evidence committed to this repo; 🧪 means the loader is implemented and re-verification is in progress. Details in the [development plan](docs/DEVELOPMENT_PLAN.md).

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

More install options: `docs/installation.md`.

## Quickstart

### Python API (recommended)

```python
import indoorloc as iloc

train, test = iloc.load_dataset("ujindoorloc")            # any integrated dataset ID
model = iloc.create_model("resnet18", dataset=train)      # timm backbone, auto-configured
results = model.fit(train).evaluate(test)
```

### YAML config + CLI

Config templates live in `indoorloc/configs/`.

```bash
indoorloc-train indoorloc/configs/wifi/resnet18_ujindoorloc.yaml

# Override any parameter (Python literals: True/False, not true/false)
indoorloc-train indoorloc/configs/wifi/resnet18_ujindoorloc.yaml \
  --model.backbone.model_name efficientnet_b0 \
  --train.lr 5e-4 --train.epochs 200
```

```yaml
# indoorloc/configs/wifi/resnet18_ujindoorloc.yaml (abridged)
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

## Datasets

> Catalogue (web): https://qdtiger.github.io/indoorloc/datasets.html · List IDs: `iloc.list_available_datasets()`

Status: ✅ verified end-to-end · 🧪 integrated, re-verification in progress

| Type | Dataset | ID | Samples | Status |
|------|---------|-----|---------|:------:|
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
<summary>Pending datasets (help wanted)</summary>

These datasets have download sources but are **not yet integrated**:

| Dataset | Source | Notes |
|---------|--------|-------|
| DeepMIMO | [deepmimo.net](https://www.deepmimo.net) | Ray-tracing synthetic; first target of the simulated-data layer |
| DICHASUS | [DaRUS](https://darus.uni-stuttgart.de/dataverse/dichasus) | Massive-MIMO CSI, cm-accurate ground truth |
| MaMIMO CSI | [IEEE DataPort](https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset) | Requires account |
| OpenCSI | [Figshare](https://doi.org/10.6084/m9.figshare.19596379.v1) | ~2GB, format unverified |
| CSUIndoorLoc | [GitHub](https://github.com/EPIC-CSU/csi-rssi-dataset-indoor-nav) | Format unverified |
| ESPARGOS | [espargos.net](https://espargos.net/datasets/) | 17–86GB |
| CSI2Pos / CSI2TAoA | [TIB](https://service.tib.eu/ldmservice/) | Requires login |
| WILDv2 | [Kaggle](https://www.kaggle.com/competitions/wild-v2) | Requires Kaggle API |

> Contribute a loader: see `CONTRIBUTING.md`.

</details>

## Models & Algorithms

> Zoo (web): https://qdtiger.github.io/indoorloc/algorithms.html · List models: `iloc.list_models()`

**Implemented today:**

| Family | Methods |
|--------|---------|
| Traditional ML | kNN · WKNN · SVM · Random Forest |
| Deep supervised | MLP · CNN1D · [timm](https://github.com/huggingface/pytorch-image-models) backbones (ResNet · EfficientNet · ViT · …) × 7 task heads (regression / multi-scale / classification / floor / building / hybrid / hierarchical) |
| Fusion | Ensemble · Stacking |
| Transfer (shallow) | CORAL · TCA · KMM via [SKADA](https://github.com/scikit-adaptation/skada) |

<details>
<summary>Planned families (roadmap — no code yet)</summary>

- **Self-supervised pretraining**: SimCLR, MoCo, BYOL, SimSiam, VICReg …
- **Meta-learning / few-shot**: MAML, FOMAML, Reptile, ProtoNet, MatchingNet …
- **Deep domain adaptation**: DANN, MDD, DeepCORAL …
- **Model-based / model-free methods**: geometric solvers, Bayesian filtering, weighted centroid, channel charting …

</details>

### Custom model registration

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
        ...  # your training logic
        self._is_trained = True
        return self

    def predict(self, signal):
        ...  # return a LocalizationResult

model = iloc.create_model("MyLocalizer")
```

## Evaluation

| Metric | Description |
|--------|-------------|
| Mean / Median / RMS / P75 Position Error | Localization error (m) |
| Floor Accuracy | Floor classification |
| Building Accuracy | Building classification |
| CDF Analysis | Error distribution |

`evaluate()` can compare your run against literature-reported numbers for the same dataset. Reproduced-by-this-repo numbers and literature-reported numbers are always labeled separately — they never share a column.

## Taxonomy & Roadmap

IndoorLoc is organized as a five-layer stack, following the taxonomy used by recent indoor-localization surveys — from data sources at the bottom to applications on top.

Legend: ✅ available · 🚧 partial · 📋 planned

```text
L1  Data
├── Measured                                  ✅ 12 datasets (6 WiFi · 3 BLE · 3 CSI)
│   └── High-precision massive-MIMO CSI       📋 DICHASUS, MaMIMO (pending)
├── Simulated                                 📋
│   ├── Ray-tracing engines                   📋 Sionna RT · Wireless InSite · NVIDIA AODT
│   ├── Statistical channel models            📋 3GPP TR 38.901 (InH/InF) · QuaDRiGa
│   ├── Pre-generated synthetic datasets      📋 DeepMIMO v4 (first target) · WAIR-D
│   └── Learned radio fields                  📋 NeRF2 · Gaussian-splatting RF
└── Sim2Real digital-twin pairs               📋 e.g. DICHASUS ↔ Sionna RT calibration

L2  Signals / Observables                     🚧 RSSI + BLE active; CSI, UWB (ToF/TDoA),
                                                 IMU, magnetic, VLC, ultrasound defined

L3  Methods
├── Model-based                               📋 geometric solvers · Bayesian filtering ·
│                                                map constraints
├── Model-free                                📋 weighted centroid · interpolation ·
│                                                graph/manifold · channel charting
└── Data-driven                               🚧
    ├── Deterministic fingerprinting          ✅ kNN · WKNN
    ├── ML regression                         ✅ SVM · Random Forest
    ├── Deep neural networks                  ✅ MLP · CNN1D · timm backbones × 7 heads
    ├── Ensembling                            ✅ Ensemble · Stacking
    ├── Transfer / domain adaptation          🚧 shallow skada (CORAL · TCA · KMM);
    │                                            deep DA planned
    ├── Self-supervised pretraining           📋 SimCLR · MoCo · BYOL ...
    ├── Meta-learning / few-shot              📋 MAML · ProtoNet ...
    ├── Probabilistic / generative            📋
    └── Sequential tracking · neural radio    📋
        maps · channel charting

L4  Evaluation                                ✅ 9 metrics · published benchmarks
    └── CRLB bounds · cross-device/time       📋
        & sim2real protocols

L5  Applications / Deployment                 📋 real-time inference · tracking filters
                                                 (Kalman/PF) · PDR · navigation
```

Milestones, decision gates, and the full known-issues list live in [`docs/DEVELOPMENT_PLAN.md`](docs/DEVELOPMENT_PLAN.md).

<details>
<summary>Project structure</summary>

```
indoorloc/
├── signals/          # WiFi, BLE, CSI, IMU, ... signal classes
├── locations/        # Coordinate & Location classes
├── datasets/         # Dataset loaders + registry + transforms
├── localizers/       # Classic ML localizers (fingerprint / fusion / transfer)
├── models/           # Deep models: backbones × heads + DeepLocalizer
├── evaluation/       # Metrics + published-benchmark tables
└── configs/          # YAML configs with _base_ inheritance
```

</details>

## Contributing

PRs welcome. New dataset loaders and new localizer implementations are what we need most — see `CONTRIBUTING.md` for the requirements.

## License

Apache License 2.0

## Citation

```bibtex
@software{indoorloc,
  title = {IndoorLoc: A Unified Framework for Indoor Localization},
  year = {2025},
  url = {https://github.com/qdtiger/indoorloc}
}
```

## Acknowledgements

- [OpenMMLab](https://github.com/open-mmlab) — registry and config system design
- [timm](https://github.com/huggingface/pytorch-image-models) — pretrained backbones
- [scikit-learn](https://scikit-learn.org/) / [SKADA](https://github.com/scikit-adaptation/skada) — classic ML & domain adaptation
