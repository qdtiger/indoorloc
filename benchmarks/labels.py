"""Label-accuracy benchmark for datasets labelled only by a class (e.g. the room of ``wlanrssi``).

``indoorloc benchmark`` scores positions and refuses a dataset without coordinates. This runner
scores a label instead, with the same public pieces and the same provenance as the command line
(``indoorloc.cli.benchmark``: ``build_preprocess``, ``parse_spec``, ``describe``, ``environment``)::

    python -m benchmarks.labels --dataset wlanrssi --label room --method wknn --preprocess fill \\
        --out wknn.json

Protocol: stratified k-fold over the label (``benchmarks.protocols.stratified_kfold``, seeded);
every row is tested once and the pooled accuracy is the fraction of rows whose label is right.

How a localizer predicts a label: the label is passed as the ``floor`` label, which every
localizer predicts (k-NN by a vote of the neighbours, Horus by the most likely location, forests,
SVM and the MLP by their classifier). The dataset has no coordinates, so the position target is a
single constant axis (``pos = 0``): it carries no information, is never scored, and lets the
methods whose position regressor needs at least one axis (forests, SVM) run unchanged.

References
----------
R. Kohavi, "A study of cross-validation and bootstrap for accuracy estimation and model
selection", IJCAI 1995. URL: https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FORMAT = "indoorloc-benchmark-labels"
FORMAT_VERSION = 1


def _jsonable(value):
    """Plain JSON types: numpy scalars and arrays become Python numbers and lists."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _accuracy(true, pred) -> float:
    return float(np.mean(np.asarray(true) == np.asarray(pred)) * 100.0)


def run_labels(dataset: str, method: str, *, label: str = "room", preprocess: str | None = None, n_splits: int = 5,
               seed: int = 0, root=None, download: bool = False, dataset_options: dict | None = None,
               command=None) -> dict:
    """Fit and score one method spec on every stratified fold; return a JSON-ready result document."""
    from indoorloc.cli.benchmark import RSSI_MODALITIES, build_preprocess, describe, environment, parse_spec
    from indoorloc.datasets import DATASETS
    from indoorloc.evaluation import split_summary
    from indoorloc.methods import METHODS, create_model

    from .protocols import stratified_kfold

    start = time.perf_counter()
    cls = DATASETS.get(dataset)
    pre_spec = preprocess if preprocess is not None else (
        "fill" if cls.meta.get("modality") in RSSI_MODALITIES else "none")
    build_preprocess(pre_spec)  # fail before loading data
    name, params = parse_spec(method)
    if "random_state" in inspect.signature(METHODS.get(name).__init__).parameters and "random_state" not in params:
        params["random_state"] = seed
    source = cls(root, download=download, **(dataset_options or {}))
    if tuple(source.files) != ("all",):
        raise ValueError(f"{dataset}: the label runner expects a single 'all' split, got {tuple(source.files)}")
    table = source.load("all")
    if label not in table.groups:
        raise ValueError(f"{dataset} has no groups[{label!r}]; groups: {sorted(table.groups)}")
    y = np.asarray(table.groups[label])
    placeholder = np.zeros((len(table), 1))  # no coordinates: one constant axis, never scored
    folds = stratified_kfold(y, n_splits, random_state=seed)
    rows, true_all, pred_all = [], [], []
    model = None
    for f, (tr, te) in enumerate(folds):
        model = create_model(name, preprocess=build_preprocess(pre_spec), **params)
        t0 = time.perf_counter()
        model.fit(table.X[tr], placeholder[tr], floor=y[tr])
        t1 = time.perf_counter()
        pred = model.localize(table.X[te])
        t2 = time.perf_counter()
        if pred.floor is None:
            raise RuntimeError(f"{method} predicted no label")
        true_all.append(y[te])
        pred_all.append(pred.floor)
        rows.append({"fold": f"fold={f}", "fit_s": t1 - t0, "predict_s": t2 - t1,
                     "metrics": {"n": int(len(te)), "accuracy": _accuracy(y[te], pred.floor)}})
    true_all, pred_all = np.concatenate(true_all), np.concatenate(pred_all)
    classes = np.unique(y)
    confusion = [[int(np.sum((true_all == a) & (pred_all == b))) for b in classes] for a in classes]
    pooled = {"n": int(len(true_all)), "accuracy": _accuracy(true_all, pred_all),
              "per_class_accuracy": {str(c): _accuracy(true_all[true_all == c], pred_all[true_all == c])
                                     for c in classes},
              "classes": classes.tolist(), "confusion": confusion}
    payload = {
        "format": FORMAT, "format_version": FORMAT_VERSION,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": list(command) if command else None,
        "dataset": {"name": cls.name, "class": f"{cls.__module__}:{cls.__qualname__}",
                    "options": _jsonable(dataset_options or {}), "splits": ["all"],
                    "n_samples": {"all": len(table)}, "n_pooled": len(table),
                    "sha256": {"all": _jsonable(table.meta.get("sha256"))},
                    **{k: _jsonable(table.meta.get(k)) for k in ("modality", "task", "citation", "doi", "license",
                                                                 "url") if table.meta.get(k) is not None}},
        "task": {"label": label, "position_target": "constant placeholder (one axis of zeros), not scored",
                 "label_passed_as": "floor"},
        "protocol": {"name": f"stratified-kfold-{n_splits}",
                     "summary": f"{n_splits} folds stratified by groups[{label!r}] (seeded); every row tested once",
                     "folds": [{"name": f"fold={i}", **split_summary(tr, te, len(table))}
                               for i, (tr, te) in enumerate(folds)]},
        "preprocess": {"spec": pre_spec, "default": preprocess is None,
                       "resolved": describe(build_preprocess(pre_spec))},
        "seed": seed, "units": "% of rows with the right label", "scale": 1.0,
        "methods": [{"label": method, "method": name, "params": _jsonable(params), "model": repr(model),
                     "model_params": describe(model), "folds": rows, "pooled": pooled}],
        "environment": environment(),
    }
    payload["wall_time_s"] = time.perf_counter() - start
    return _jsonable(payload)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(prog="python -m benchmarks.labels", description=__doc__.split("\n")[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--label", default="room", help="groups column holding the class (default: room)")
    p.add_argument("--method", required=True, help="one method spec, e.g. 'wknn(k=3)'")
    p.add_argument("--preprocess", default=None, help="as for `indoorloc benchmark` (default: fill for RSSI)")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root")
    p.add_argument("--dataset-option", action="append", metavar="KEY=VALUE", default=[])
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args(argv)
    from indoorloc.cli.benchmark import parse_spec

    options = parse_spec(f"options({', '.join(args.dataset_option)})")[1]  # key=value, values as in method specs
    payload = run_labels(args.dataset, args.method, label=args.label, preprocess=args.preprocess,
                         n_splits=args.n_splits, seed=args.seed, root=args.root, download=not args.no_download,
                         dataset_options=options, command=["python", "-m", "benchmarks.labels", *argv])
    Path(args.out).write_text(json.dumps(payload, indent=1, ensure_ascii=False, allow_nan=False) + "\n",
                              encoding="utf-8")
    if not args.quiet:
        m = payload["methods"][0]
        print(f"{m['label']}: {m['pooled']['accuracy']:.2f} % of {m['pooled']['n']} rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
