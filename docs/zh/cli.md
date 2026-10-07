# 命令行

`indoorloc` 命令（也可以用 `python -m indoorloc`）列出库中的内容、描述数据集、按命名协议运行基准并渲染结果。
每个子命令只导入它用到的层：`indoorloc list methods` 既不加载数据集，也不加载 torch。

[指南首页](index.md) · [English](../guide/cli.md)

## 典型流程

```bash
indoorloc list datasets -v                   # 注册名、模态和一行说明
indoorloc info tuji1                         # 坐标系、单位、许可、DOI、文件（是否在本地）、文献
indoorloc benchmark --dataset tuji1 --method knn --method "wknn(k=3)" --out tuji1.json
indoorloc report tuji1.json --format markdown > tuji1.md
indoorloc literature tuji1 --results tuji1.json          # 已发表结果，放在单独的部分
```

`benchmark` 加载数据集（需要时下载并校验 sha256），按协议（默认 `official`）划分，做预处理，在每一折上拟合每个方法，
然后输出报告。使用 `--out` 时写出一个自描述的 JSON 文件，内容包括：

* 每个方法的合并指标和逐折指标（`mean_error`、百分位数、`rmse`、楼层与建筑准确率、`n_failed`、IPIN 评分、EvAAL 带惩罚均值、
  均值的 bootstrap 95 % 区间以及 CDF 点）；
* 完全展开的预处理与模型参数，以及拟合和预测耗时；
* 每一折训练与测试下标的 sha256、随机种子、数据文件的摘要；
* 库、Python、numpy 的版本，已加载的可选包，BLAS 库，CPU 数，git 提交（以及工作区是否有未提交修改），库源码摘要，以及命令行本身。

`indoorloc report` 重新渲染这种文件；协议一致时，`indoorloc literature --results` 把它与已发表结果并列显示。

### 方法、预处理与协议的写法

* `--method` 接受注册名，可在括号中给参数：`"wknn(k=3)"`、`"ensemble(localizers=[\"wknn\",\"rf\",\"horus\"],combine=median)"`；
  也接受类路径 `"mypkg.models:MyLocalizer(alpha=0.1)"`。参数值先按 JSON 解析，再按 Python 字面量解析，否则保留为字符串。该选项可以重复。
* `--preprocess` 接受 `fill`、`normalize`、`positive`、`exponential`、`powed`、`none` 或任意 `indoorloc.signals` 类名，
  用 `+` 串联，括号中给参数：`"fill(value=-110)+exponential"`、`CSIAmplitude`。RSSI 数据默认 `fill`（-104 dBm），其他数据默认不做预处理。
* `--protocol` 接受 `indoorloc list protocols` 中的名称，或解析为 `Protocol` 的 `module:attribute`
  （例如在仓库根目录下使用 `benchmarks.protocols:LEAVE_ONE_USER_OUT`）。`official` 以外的协议作用于把数据集所有划分合并后的表。
* `--dataset-option KEY=VALUE` 传递构造选项，例如 `--dataset-option building=HCXY`。
* `--units ground` 把误差乘以 `meta["ground_scale"]`（UJIIndoorLoc：EPSG:3857 米换算为地面米）。默认的 `native` 使用数据集自己的单位，与文献一致。

```bash
indoorloc benchmark --dataset synthetic_office --protocol kfold-5 \
    --preprocess "fill(value=-110)+exponential" --method knn --method "wknn(k=3)" \
    --predictions preds.npz --out office_kfold.json -q
```

这次运行把仿真办公楼的 1,270 行合并后分成 5 折，合并后的平均误差为 1.784 m（k-NN）和 1.728 m（WKNN，k=3）。
`preds.npz` 保存每个方法的逐样本预测、折号和 id。

### 保存的模型

`--save-models DIR`（仅限单折协议）把每个拟合好的模型写为 `DIR/<method>/config.json` 和 `arrays.npz`，不使用 pickle。
`indoorloc evaluate` 在某个划分上评测这样的模型；如果该划分参与过训练，会给出警告：

```bash
indoorloc benchmark --dataset synthetic_office --method knn --save-models models -q --out office.json
indoorloc evaluate --model models/knn --dataset synthetic_office --split test
```

### 复现已发布的基准表

[docs/benchmarks_zh.md](../benchmarks_zh.md) 的每张表都给出了其中一个单元格的命令；`python benchmarks/run.py --list` 列出每个单元格的命令，`benchmarks/results/*.json` 中的每个单元格也记录了自己的命令，例如：

```bash
indoorloc benchmark --dataset ujiindoorloc --protocol official --preprocess fill --method wknn --seed 0 --no-download
```

矩阵运行器 `benchmarks/run.py` 在独立进程中运行每个单元格，记录墙钟时间、CPU 时间和峰值内存，并渲染文档（见 `benchmarks/README.md`）。

## 参考

由 `python docs/catalog.py` 从参数解析器生成（选项说明保持英文原文）。

<!-- catalog:cli -->
#### `indoorloc list`

list datasets, methods, protocols or literature tables

```text
indoorloc list [-h] [-v] {datasets,methods,protocols,literature}
```

| 选项 | 说明 |
| --- | --- |
| `what {datasets,methods,protocols,literature}` | — |
| `-v, --verbose` | one-line description of each entry |

#### `indoorloc info`

facts about a dataset: modality, frame, license, files, literature

```text
indoorloc info [-h] [--root ROOT] [--json] dataset
```

| 选项 | 说明 |
| --- | --- |
| `dataset` | — |
| `--root` | dataset folder (default: $INDOORLOC_DATA/<name>) |
| `--json` | print JSON |

#### `indoorloc benchmark`

fit and score methods on a dataset under a named protocol

```text
indoorloc benchmark [-h] --dataset DATASET --method METHOD [--protocol PROTOCOL] [--preprocess PREPROCESS] [--seed SEED] [--units {native,ground}] [--root ROOT] [--dataset-option KEY=VALUE] [--no-download] [--no-verify] [--out OUT] [--report REPORT] [--predictions PREDICTIONS] [--save-models DIR] [-q]
```

| 选项 | 说明 |
| --- | --- |
| `--dataset` | registry name or module:Class |
| `--method` | method spec, repeatable: knn, 'wknn(k=3)', 'pkg.mod:MyLocalizer(alpha=0.1)' |
| `--protocol` | see `indoorloc list protocols -v` (default: official) |
| `--preprocess` | L2 preprocessing: fill, normalize, positive, exponential, powed, none, or a signals class; chain with '+', arguments in parentheses: 'fill(value=-110)+normalize'. Default: fill for RSSI data, none otherwise |
| `--seed` | seed of random protocols and of methods with random_state |
| `--units {native,ground}` | report errors in the dataset's coordinates (default, as in the literature) or ground metres |
| `--root` | dataset folder (default: $INDOORLOC_DATA/<name>) |
| `--dataset-option` | dataset constructor option |
| `--no-download` | fail instead of downloading missing files |
| `--no-verify` | skip sha256 checks of the data files |
| `--out` | write the result JSON here |
| `--report` | also write a Markdown report here |
| `--predictions` | save per-sample predictions (.npz) |
| `--save-models` | save each fitted model to DIR/<method> (no pickle) |
| `-q, --quiet` | no progress or report on the terminal |

#### `indoorloc literature`

published numbers for a dataset, with provenance

```text
indoorloc literature [-h] [--all] [--results RESULTS] [--format {text,markdown}] dataset
```

| 选项 | 说明 |
| --- | --- |
| `dataset` | — |
| `--all` | every ported entry, also those without a traceable source or number and 0.1's own runs |
| `--results` | a benchmark JSON to show next to the literature (kept separate) |
| `--format {text,markdown}` | — |

#### `indoorloc evaluate`

score a saved model (benchmark --save-models) on a dataset split

```text
indoorloc evaluate [-h] --model MODEL --dataset DATASET [--split SPLIT] [--root ROOT] [--no-download] [--units {native,ground}] [--format {text,markdown}]
```

| 选项 | 说明 |
| --- | --- |
| `--model` | folder written by --save-models |
| `--dataset` | — |
| `--split` | split to score (default: the official test split when the dataset has one, else 'all') |
| `--root` | dataset folder (default: $INDOORLOC_DATA/<name>) |
| `--no-download` | — |
| `--units {native,ground}` | — |
| `--format {text,markdown}` | — |

#### `indoorloc report`

render a benchmark JSON as a Markdown or text report

```text
indoorloc report [-h] [--format {markdown,text}] [--no-literature] results
```

| 选项 | 说明 |
| --- | --- |
| `results` | — |
| `--format {markdown,text}` | — |
| `--no-literature` | leave out published numbers |
<!-- /catalog:cli -->

错误以 `indoorloc: error: ...` 报告，退出状态为 1（用法错误为 2）；加 `--traceback` 可查看完整回溯。
