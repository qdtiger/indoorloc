# IndoorLoc benchmark matrix

The library's reproducible real-data benchmark: every measured dataset against the methods that
apply to it, each run through the public command line (`indoorloc benchmark`) exactly as a user
would run it. The results are in [`results/`](results/) (one JSON file per dataset) and are
rendered as [`docs/benchmarks.md`](../docs/benchmarks.md) and
[`docs/benchmarks_zh.md`](../docs/benchmarks_zh.md).

| File | Role |
|---|---|
| `matrix.py` | what runs: one `Table` per result table (dataset, options, protocol, preprocessing and method specs); every method not run is a `Skip` with its reason |
| `run.py` | runs the cells, one process each, records wall time and peak memory, writes `results/<dataset>.json`, renders the docs |
| `labels.py` | the label-accuracy runner for `wlanrssi` (room labels, no coordinates), same provenance as the CLI |
| `protocols.py` | protocols the library does not ship: leave-one-user-out, 5 folds over reference points, within-month, stratified k-fold |
| `baselines.py` | `TrainingCentroid`, the constant answer every table includes |
| `crosscheck.py` | recomputes the cross-checked k-NN cells outside the library (plain numpy, exact distances where the readings allow it) and writes `results/crosscheck.json` |
| `render.py` | `results/*.json` to the two Markdown documents |
| `tests/` | tests of the harness on synthetic data (`python -m pytest -q benchmarks/tests`) |

## Requirements

* `pip install "indoorloc[full]"` (scikit-learn for the forests and the SVM, PyTorch for the MLP,
  scipy/h5py for the CSI loaders), or run from a checkout with those packages installed.
* The datasets under `$INDOORLOC_DATA` (default `~/.cache/indoorloc/datasets/<name>`). `run.py`
  passes `--no-download` so a missing file stops a cell instead of starting a download; fetch a
  dataset once with `python -c "import indoorloc as il; il.load_dataset('tuji1')"` (every file is
  sha256-checked on every load). `indoorloc info <dataset>` lists the files and their digests.

## Reproducing

From the repository root:

```bash
python benchmarks/run.py --list                                # every cell and its command, nothing runs
python benchmarks/run.py --dataset tuji1                       # one dataset (all its tables), then the docs
python benchmarks/run.py --dataset sodindoorloc --table official-HCXY    # one table
python benchmarks/run.py --dataset ujiindoorloc --method wknn  # the cells whose method spec contains "wknn"
python benchmarks/run.py                                       # the whole matrix, then the docs
python benchmarks/run.py --resume                              # only the cells without a finished result
python -m benchmarks.crosscheck                                # recompute the cross-checked cells outside the library
python benchmarks/render.py                                    # only regenerate the docs from results/
```

A single cell is an ordinary command; the docs print one per table and the result files keep
each cell's `command`:

```bash
indoorloc benchmark --dataset tuji1 --protocol official --preprocess positive --method 'knn(k=1)'
indoorloc benchmark --dataset hwild --dataset-option environment=office \
    --protocol benchmarks.protocols:LEAVE_ONE_USER_OUT --preprocess CSIAmplitude --method wknn
python -m benchmarks.labels --dataset wlanrssi --label room --preprocess fill --method wknn --out wknn.json
```

(`benchmarks.protocols:...` and `benchmarks.baselines:TrainingCentroid` resolve from the
repository root, like any `module:attribute` spec.)

The whole matrix takes hours, one cell at a time with 8 BLAS/OpenMP threads; the *Setup* section
of the docs gives the measured wall and CPU time of the published run. While it runs, the working
tree may change; `--freeze DIR` copies `indoorloc/` and `benchmarks/` to `DIR` first, runs that copy
and records the commit and the source digest it was taken from (`DIR/SNAPSHOT.json`):

```bash
python benchmarks/run.py --freeze /tmp/indoorloc-bench-snapshot
```

Re-running a subset replaces only those cells in the result files; the others are kept. A cell
that already had a finished result records how the new run compares with it (`rerun_check`: the
largest difference over all pooled and per-fold metrics), so every rerun is also a reproducibility
check. `--resume` continues an interrupted matrix: it runs only the cells whose stored status is
not `ok` (never run, or stopped at a limit).

### Checking determinism

```bash
python benchmarks/run.py --dataset tuji1 --verify                         # rerun, compare, overwrite nothing
python benchmarks/run.py --dataset tuji1 --verify --record-verification   # and store the comparison
```

`--verify` compares every pooled and per-fold metric of the stored cells with a fresh run and
prints the largest difference; it fails (exit status 1) on any difference and on a stored cell
that no longer finishes. `--record-verification` writes the comparison, with the source digest
of the code that ran it, to `results/verification.json`, which the docs quote.

## What a cell records

`results/<dataset>.json` (`format: "indoorloc-benchmark-matrix"`):

* `harness`: hardware (CPU, logical CPUs, RAM, platform), the memory limits of the Linux control
  group the cells ran in, thread settings, the per-cell limits and the code of the latest run
  (`code`: git commit and branch, whether the tree had uncommitted changes, whether it was a
  `--freeze` copy, and its source digest). When a subset is rerun with other code, the code
  blocks of the runs whose cells are kept move to `code_history`, so every cell's
  `source_sha256` can be traced to the code it ran.
* `environment`: indoorloc, Python, numpy and BLAS versions, and the versions of scikit-learn,
  PyTorch, scipy ... that any cell loaded.
* `tables[]`: the table's dataset facts (file sha256 per split, row counts, options, units, CRS,
  license, DOI), the protocol with the sha256 of every fold's train and test indices, the skips
  with their reasons (English and Chinese), the notes, and `cells[]`.
* `cells[]`: preprocessing and method spec, the exact command, `status` (`ok`, `timeout`,
  `memory`, `failed`) and its reason, wall and CPU time and peak resident memory of the process,
  the load average before and after and the memory-pressure stall of the cgroup during the cell
  (`memory_stall_s`, Linux PSI), the limits it ran under, fit and predict seconds, the pooled metrics (mean, median, P75, P90, P95, RMSE, max, floor/building
  accuracy, `n_failed`, IPIN score, EvAAL-ETRI penalised errors, bootstrap 95 % interval of the
  mean, error CDF) and the per-fold metrics, the fully resolved model and preprocessing
  parameters, the library source digest and a digest of the folds.
* `literature`: the published numbers of `indoorloc.evaluation.literature` for the dataset with
  their check status, plus the count of hidden entries; never merged with the measured cells.

## Rules the matrix follows

* Numbers in the docs come only from `results/`; `results/` comes only from `run.py` (and
  `results/crosscheck.json` from `crosscheck.py`).
* Published numbers come only from `indoorloc.evaluation.literature` and are shown in separate
  tables with their check status (`verified`, `corrected`, `unchecked`); entries without a
  traceable source are counted, not shown.
* A method that is not run on a table is listed with its reason; a cell that hits the time or
  memory limit (30 minutes, 2,500 MB by default) is reported as such, never dropped.
* Errors are in the units the dataset defines (EPSG:3857 metres for UJIIndoorLoc, grid cells for
  the UCI BLE data and the CSI fingerprint dataset, metres elsewhere); room-only data report room
  accuracy.
* Seed 0 everywhere; one process per cell, one cell at a time, 8 BLAS/OpenMP threads.
* Times and memory are those of one shared workstation; they rank methods, they do not predict
  another machine. The published run shared a memory-limited control group with other processes:
  above its `memory.high` the kernel slows allocations down, so the docs mark (†) the times of
  cells during which some task of the group waited for memory for more than 10 % of the cell's
  wall time (PSI `some`, the stricter of the two totals). Stalls change times, never results:
  the cells are deterministic.
* A method left out of a table needs a reason that holds: a size argument (e.g. an n x n kernel
  matrix that does not fit) or a measurement. Where the reason is an extrapolation from a
  measured cell, the skip says "estimated, not run".

## Adding to the matrix

Add or extend a `Table` in `matrix.py`: give every new method spec a display name in `DISPLAY`
(and every preprocessing spec one in `PREPROCESS`), and list any applicable method you leave out
as a `Skip` with an English and a Chinese reason. Then run the table (`--dataset ... --table ...`);
`tests/test_benchmarks.py` checks that every cell resolves to a registered dataset, protocol,
preprocessing and method.
