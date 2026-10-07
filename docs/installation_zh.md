# 安装

IndoorLoc 0.2 需要 Python 3.10 及以上版本和 numpy。其他所有包都是可选 extra，只在用到它们的函数内部导入。
缺少某个包时会抛出 `ImportError`，并指出需要安装的 extra，例如
`this feature needs sklearn.ensemble: pip install 'indoorloc[sklearn]'`。

[English](installation.md) · [用户指南](zh/index.md) · [从 0.1 迁移](../MIGRATION.md)

## 安装

本文档对应 0.2 系列（本仓库中为 `0.2.0.dev0`）。在 0.2.0 发布到 PyPI 之前，请从源码安装：

```bash
git clone https://github.com/qdtiger/indoorloc.git
cd indoorloc
pip install -e .                    # 仅 numpy
pip install -e ".[full]"            # 或者安装全部可选依赖
```

0.2.0 发布到 PyPI 后，`pip install "indoorloc[...]"` 使用同样的 extra。

## Extra

由 `python docs/catalog.py` 从 `pyproject.toml` 生成：

<!-- catalog:extras -->
| 安装 | 依赖包 | 用途（pyproject.toml 中的注释） |
| --- | --- | --- |
| `indoorloc` | `numpy>=2.0` | 五层全部只需 numpy |
| `indoorloc[sklearn]` | `scikit-learn>=1.4.2` | SVM/RF/GBDT localizers, Pipeline/GridSearchCV interop |
| `indoorloc[pandas]` | `pandas>=2.2.2` | SampleTable.to_dataframe |
| `indoorloc[torch]` | `torch>=2.3` | datasets.torch_adapter, SampleTable.to_torch |
| `indoorloc[deep]` | `torch>=2.3`, `timm>=0.9` | MLP / CNN1D / timm-backbone localizers |
| `indoorloc[transfer]` | `scikit-learn>=1.5`, `skada>=0.6` | CORAL/TCA are numpy; the skada adapter needs skada |
| `indoorloc[plot]` | `matplotlib>=3.8.4`, `plotly>=5` | evaluation.plot, datasets.plot (plotly: interactive HTML) |
| `indoorloc[datasets]` | `h5py>=3.11`, `scipy>=1.13` | the CSI loaders that read .mat / .h5 files |
| `indoorloc[sim]` | `DeepMIMO>=4.0; python_version >= '3.11'` | DeepMIMO ray-traced scenarios (needs Python 3.11+) |
| `indoorloc[full]` | `scikit-learn>=1.4.2`, `pandas>=2.2.2`, `torch>=2.3`, `timm>=0.9`, `matplotlib>=3.8.4`, `plotly>=5`, `h5py>=3.11`, `scipy>=1.13` | every extra above except transfer and sim |
| `indoorloc[dev]` | `pytest>=8`, `threadpoolctl>=3`, `scikit-learn>=1.6`, `import-linter>=2.1`, `ruff>=0.6` | tests (check_estimator(on_fail=) is new in 1.6), lint |
<!-- /catalog:extras -->

各功能需要的 extra：

* **只需 numpy：** 除 `csi_fingerprint`、`hwild` 和 `deepmimo`（见下）以外的所有数据集加载器、仿真办公楼、所有 L2 变换与信号函数、k-NN/WKNN、Horus、GP 无线电地图、
  集成与分层模型、所有基于模型的方法（三边测量、TDoA、AoA、路径损耗、加权质心、VLC）、地磁 DTW、CORAL 与 TCA、除绘图外的全部 L4、
  全部 L5 以及命令行。
* `[sklearn]`：`svm`、`rf`、`extratrees`、`gbdt`，以及在任意模型外使用 scikit-learn 的 `GridSearchCV`/`Pipeline`。
* `[deep]`：`mlp`、`cnn1d` 和 `deep`（torch；timm 骨干网络如 `"resnet18"` 还需要 timm）。请先按 pytorch.org 的选择器为你的硬件安装 torch，再安装该 extra。
* `[datasets]`：`csi_fingerprint`（用 scipy 读取 `.mat`）和 `hwild`（用 h5py 读取 `.h5`）。
* `[plot]`：`indoorloc.datasets.plot` 和 `indoorloc.evaluation.plot`（matplotlib），以及交互式的 `datasets.plot.distribution_html`（plotly）。
* `[transfer]`：`SkadaAdapter`（任意 skada 特征级适配器）。CORAL 与 TCA 只需 numpy。
* `[sim]`：`deepmimo` 数据集（DeepMIMO v4，Python 3.11 及以上）。
* `[pandas]`、`[torch]`：`SampleTable.to_dataframe()`，以及 `SampleTable.to_torch()` 与 `datasets.torch_adapter`。

## 检查安装

```bash
python -c "import indoorloc; print(indoorloc.__version__)"
indoorloc list methods
indoorloc benchmark --dataset synthetic_office --method knn --method wknn     # 无需下载
```

## 数据集存放位置

数据集在首次使用时下载到 `$INDOORLOC_DATA/<name>`（默认 `~/.cache/indoorloc/datasets/<name>`），每个文件都与加载器中记录的
sha256 核对。要使用已在磁盘上的数据，设置 `INDOORLOC_DATA` 或给 `load_dataset` 传 `root=`。`indoorloc info <name>` 显示应有的文件以及它们是否存在。
下载使用标准库（`urllib`），因此常用的 `https_proxy`/`no_proxy` 变量同样有效。如果某个主机无法访问，可以手动把文件放到 `indoorloc info` 给出的目录中。

## 开发环境

```bash
pip install -e ".[dev,full]"
python -m pytest -q tests                  # 每个测试文件在仿真数据上几秒内跑完
lint-imports                               # import-linter 层级契约
python docs/catalog.py --check             # 指南中生成的表格是否最新
python docs/catalog.py --snippets          # 指南中的代码块能否运行、输出是否与页面一致
python docs/catalog.py --snippets --no-data   # 同上，但不使用任何数据集文件（与 CI 相同）
```

读取真实数据的测试在数据文件不存在时自动跳过。基准框架有自己的测试：`python -m pytest -q benchmarks/tests`。

## 常见问题

* `ImportError: this feature needs ...: pip install 'indoorloc[...]'`：安装它指出的 extra。
* `ValueError: X contains NaN or inf (missing readings?)`：该方法需要完整的向量，请给 `create_model` 加上
  `preprocess=FillMissing(-104)`（见 [信号](zh/signals.md)）。
* `ValueError: checksum mismatch for ...`：磁盘上的文件与发布的文件不同。删除它以重新下载；若确实要使用修改过的副本，传 `verify=False`。
* 在 GPU 机器上 `torch.cuda.is_available()` 为 `False` 时，检查 NVIDIA 驱动是否与 torch 的 CUDA 版本匹配。深度模型默认在 CPU 上运行（`device="cpu"`）。
