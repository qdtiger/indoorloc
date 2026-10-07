# Command line

The `indoorloc` command (also `python -m indoorloc`) lists what the library has, describes a
dataset, runs benchmarks under named protocols and renders their results. A command imports only
the layers it uses: `indoorloc list methods` loads neither the datasets nor torch.

[Guide home](index.md) · [中文](../zh/cli.md)

## Typical session

```bash
indoorloc list datasets -v                   # registry names with modality and a one-line description
indoorloc info tuji1                         # frame, units, license, DOI, files (present or not), literature
indoorloc benchmark --dataset tuji1 --method knn --method "wknn(k=3)" --out tuji1.json
indoorloc report tuji1.json --format markdown > tuji1.md
indoorloc literature tuji1 --results tuji1.json          # published numbers, kept in a separate section
```

`benchmark` loads the dataset (downloading and sha256-checking it if needed), splits it with the
protocol (`official` by default), preprocesses, fits every method on every fold and prints a
report. With `--out` it writes a self-describing JSON file with the following content:

* the pooled and per-fold metrics of every method (`mean_error`, percentiles, `rmse`, floor and
  building accuracy, `n_failed`, the IPIN score, the EvAAL penalized mean, a bootstrap 95 %
  interval of the mean, and CDF points);
* the fully resolved preprocessing and model parameters, and the fit and predict times;
* the sha256 of every fold's train and test indices, the seed, and the dataset's file digests;
* the library, Python and numpy versions, the loaded optional packages, the BLAS library, the CPU
  count, the git commit (and whether the working tree differs from it), a digest of the library
  source, and the command line.

`indoorloc report` renders such a file again, and `indoorloc literature --results` places it next to
the published numbers when the protocols match.

### Method, preprocessing and protocol specs

* `--method` takes a registry name with optional parameters in parentheses, `"wknn(k=3)"`,
  `"ensemble(localizers=[\"wknn\",\"rf\",\"horus\"],combine=median)"`, or a class path,
  `"mypkg.models:MyLocalizer(alpha=0.1)"`. Values are parsed as JSON, then as Python literals,
  and otherwise kept as strings. The option can be repeated.
* `--preprocess` takes `fill`, `normalize`, `positive`, `exponential`, `powed`, `none` or any
  `indoorloc.signals` class name, chained with `+` and given arguments in parentheses:
  `"fill(value=-110)+exponential"`, `CSIAmplitude`. The default is `fill` (-104 dBm) for RSSI data
  and no preprocessing otherwise.
* `--protocol` takes a name from `indoorloc list protocols` or a `module:attribute` spec that
  resolves to a `Protocol` (for example `benchmarks.protocols:LEAVE_ONE_USER_OUT` from the
  repository root). A protocol other than `official` is applied to all splits of the dataset
  pooled into one table.
* `--dataset-option KEY=VALUE` passes constructor options, e.g. `--dataset-option building=HCXY`.
* `--units ground` multiplies errors by `meta["ground_scale"]` (UJIIndoorLoc: EPSG:3857 metres to
  ground metres). The default `native` reports the dataset's own units, as published results do.

```bash
indoorloc benchmark --dataset synthetic_office --protocol kfold-5 \
    --preprocess "fill(value=-110)+exponential" --method knn --method "wknn(k=3)" \
    --predictions preds.npz --out office_kfold.json -q
```

This run pooled the 1,270 rows of the simulated office into 5 folds. The pooled mean errors were
1.784 m (k-NN) and 1.728 m (WKNN, k=3). `preds.npz` holds the per-sample predictions, fold
numbers and ids of every method.

### Saved models

`--save-models DIR` (single-fold protocols) writes each fitted model as `DIR/<method>/config.json`
and `arrays.npz`, without pickle. `indoorloc evaluate` scores such a model on a split and warns when
that split was used for training:

```bash
indoorloc benchmark --dataset synthetic_office --method knn --save-models models -q --out office.json
indoorloc evaluate --model models/knn --dataset synthetic_office --split test
```

### Reproducing the published benchmark tables

Each table of [docs/benchmarks.md](../benchmarks.md) shows the command of one of its cells,
`python benchmarks/run.py --list` prints the command of every cell, and each cell of
`benchmarks/results/*.json` stores its own command, for example:

```bash
indoorloc benchmark --dataset ujiindoorloc --protocol official --preprocess fill --method wknn --seed 0 --no-download
```

The matrix runner `benchmarks/run.py` runs every cell in its own process, records wall time,
CPU time and peak memory, and renders the documents (see `benchmarks/README.md`).

## Reference

Generated from the argument parser by `python docs/catalog.py`.

<!-- catalog:cli -->
#### `indoorloc list`

list datasets, methods, protocols or literature tables

```text
indoorloc list [-h] [-v] {datasets,methods,protocols,literature}
```

| Option | Help |
| --- | --- |
| `what {datasets,methods,protocols,literature}` | — |
| `-v, --verbose` | one-line description of each entry |

#### `indoorloc info`

facts about a dataset: modality, frame, license, files, literature

```text
indoorloc info [-h] [--root ROOT] [--json] dataset
```

| Option | Help |
| --- | --- |
| `dataset` | — |
| `--root` | dataset folder (default: $INDOORLOC_DATA/<name>) |
| `--json` | print JSON |

#### `indoorloc benchmark`

fit and score methods on a dataset under a named protocol

```text
indoorloc benchmark [-h] --dataset DATASET --method METHOD [--protocol PROTOCOL] [--preprocess PREPROCESS] [--seed SEED] [--units {native,ground}] [--root ROOT] [--dataset-option KEY=VALUE] [--no-download] [--no-verify] [--out OUT] [--report REPORT] [--predictions PREDICTIONS] [--save-models DIR] [-q]
```

| Option | Help |
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

| Option | Help |
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

| Option | Help |
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

| Option | Help |
| --- | --- |
| `results` | — |
| `--format {markdown,text}` | — |
| `--no-literature` | leave out published numbers |
<!-- /catalog:cli -->

Errors are reported as `indoorloc: error: ...` with exit status 1 (2 for usage errors); pass
`--traceback` to see the full traceback.
