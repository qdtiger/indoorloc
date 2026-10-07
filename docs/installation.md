# Installation

IndoorLoc 0.2 needs Python 3.10 or later and numpy. Every other package is an optional extra,
and it is imported only by the functions that use it. A missing package raises an `ImportError`
that names the extra to install, for example
`this feature needs sklearn.ensemble: pip install 'indoorloc[sklearn]'`.

[中文](installation_zh.md) · [User guide](guide/index.md) · [Migrating from 0.1](../MIGRATION.md)

## Install

This guide describes the 0.2 series (`0.2.0.dev0` in this repository). Until 0.2.0 is published on
PyPI, install it from a checkout:

```bash
git clone https://github.com/qdtiger/indoorloc.git
cd indoorloc
pip install -e .                    # numpy only
pip install -e ".[full]"            # or with every optional stack
```

Once 0.2.0 is on PyPI, the same extras apply to `pip install "indoorloc[...]"`.

## Extras

Generated from `pyproject.toml` by `python docs/catalog.py`:

<!-- catalog:extras -->
| Install | Packages | For (comment in pyproject.toml) |
| --- | --- | --- |
| `indoorloc` | `numpy>=2.0` | L1-L5 on numpy alone |
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

What needs which extra:

* **Nothing beyond numpy:** every dataset loader except `csi_fingerprint`, `hwild` and `deepmimo`
  (below), the simulated office, all L2 transforms and signal functions, k-NN/WKNN, Horus, the GP
  radio map, ensembles, hierarchical models, every model-based method (trilateration, TDoA, AoA, path loss, weighted centroid, VLC),
  magnetic DTW, CORAL and TCA, all of L4 except plots, all of L5, and the command line.
* `[sklearn]`: `svm`, `rf`, `extratrees`, `gbdt`, and scikit-learn's `GridSearchCV`/`Pipeline` around
  any model.
* `[deep]`: `mlp`, `cnn1d` and `deep` (torch; timm for timm backbones such as `"resnet18"`).
  Install the torch build for your hardware first (see the selector at pytorch.org), then the extra.
* `[datasets]`: `csi_fingerprint` (scipy reads its `.mat` files) and `hwild` (h5py reads its `.h5`
  files).
* `[plot]`: `indoorloc.datasets.plot` and `indoorloc.evaluation.plot` (matplotlib), and the
  interactive `datasets.plot.distribution_html` (plotly).
* `[transfer]`: `SkadaAdapter` (any skada feature-level adapter). CORAL and TCA need only numpy.
* `[sim]`: the `deepmimo` dataset (DeepMIMO v4, Python 3.11 or later).
* `[pandas]`, `[torch]`: `SampleTable.to_dataframe()`, and `SampleTable.to_torch()` with
  `datasets.torch_adapter`.

## Check the installation

```bash
python -c "import indoorloc; print(indoorloc.__version__)"
indoorloc list methods
indoorloc benchmark --dataset synthetic_office --method knn --method wknn     # no download needed
```

## Where datasets are stored

Datasets are downloaded on first use into `$INDOORLOC_DATA/<name>`, which defaults to
`~/.cache/indoorloc/datasets/<name>`, and every file is checked against the sha256 recorded in the
loader. To use data that is already on disk, set `INDOORLOC_DATA` or pass `root=` to
`load_dataset`. `indoorloc info <name>` shows the expected files and whether they are present.
Downloads use the standard library (`urllib`), so the usual `https_proxy`/`no_proxy` variables
apply. If a host is blocked, fetch the files yourself into the folder that `indoorloc info`
names.

## Development install

```bash
pip install -e ".[dev,full]"
python -m pytest -q tests                  # every test file runs in seconds on synthetic data
lint-imports                               # the import-linter layer contracts
python docs/catalog.py --check             # the guide's generated tables are up to date
python docs/catalog.py --snippets          # the guide's code blocks run and print what the pages show
python docs/catalog.py --snippets --no-data   # the same without dataset files, as on CI
```

Tests that read real data are skipped when the dataset files are absent. The benchmark harness
has its own tests: `python -m pytest -q benchmarks/tests`.

## Troubleshooting

* `ImportError: this feature needs ...: pip install 'indoorloc[...]'`: install the extra it names.
* `ValueError: X contains NaN or inf (missing readings?)`: the method needs complete vectors, so
  add `preprocess=FillMissing(-104)` to `create_model` (see [signals](guide/signals.md)).
* `ValueError: checksum mismatch for ...`: the file on disk differs from the published one. Delete
  it to download it again, or pass `verify=False` if you deliberately use a modified copy.
* On a GPU machine where `torch.cuda.is_available()` is `False`, check that the NVIDIA driver
  matches the CUDA build of torch. The deep models run on the CPU by default (`device="cpu"`).
