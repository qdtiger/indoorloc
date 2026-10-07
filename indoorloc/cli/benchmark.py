"""``indoorloc benchmark``: load a dataset, split it by a named protocol, fit and score methods.

The result file is self-describing: besides the numbers it records what is needed to trust
and repeat them — library, python and numpy versions, loaded optional packages, BLAS, CPU
count, the git commit (and whether the package differs from it), a digest of the library
source, the dataset's file digests, the sha256 of every fold's train/test indices, the seed,
the fully resolved preprocessing and model parameters, timings and the command line.
Published numbers, when the protocol matches the literature's, are stored in a separate
``literature`` block (see :func:`indoorloc.evaluation.literature.compare`).
"""
from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FORMAT = "indoorloc-benchmark"
FORMAT_VERSION = 1
RSSI_MODALITIES = ("wifi_rssi", "ble_rssi")
CDF_THRESHOLDS = (1.0, 2.0, 5.0, 10.0, 20.0)
# --preprocess names -> indoorloc.signals attribute; any other signals class name also works
PREPROCESS = {"fill": "FillMissing", "normalize": "RSSINormalize", "positive": "PositiveRepresentation",
              "exponential": "ExponentialRepresentation", "powed": "PowedRepresentation"}


# --------------------------------------------------------------------------- specs
def _split_top(text: str, sep: str) -> list[str]:
    """Split on ``sep`` outside brackets/parentheses/quotes."""
    parts, depth, quote, start = [], 0, None, 0
    for i, ch in enumerate(text):
        if quote:
            quote = None if ch == quote else quote
        elif ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts]


def _value(text: str):
    """A parameter value: JSON (``3``, ``true``, ``[1, 2]``), else a Python literal (``True``,
    ``None``, ``(1, 2)``), else the bare string."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text.strip("\"'")


def parse_spec(spec: str) -> tuple[str, dict]:
    """``"wknn(k=3, weights=distance)"`` -> ``("wknn", {"k": 3, "weights": "distance"})``.

    Values are JSON where possible (numbers, true/false/null, lists), else strings. The name
    may be a registry name or a ``"package.module:Class"`` path.
    """
    spec = spec.strip()
    if "(" not in spec:
        if not spec:
            raise ValueError("empty spec")
        return spec, {}
    if not spec.endswith(")"):
        raise ValueError(f"cannot parse {spec!r}: expected name(key=value, ...)")
    name, _, body = spec[:-1].partition("(")
    params = {}
    for item in filter(None, _split_top(body, ",")):
        key, eq, raw = item.partition("=")
        if not eq or not key.strip().isidentifier():
            raise ValueError(f"cannot parse {item!r} in {spec!r}: expected key=value")
        params[key.strip()] = _value(raw.strip())
    return name.strip(), params


def build_preprocess(spec: str | None):
    """``"fill"``, ``"fill(value=-110)+normalize"`` or ``"none"`` -> an L2 transform (or None)."""
    if spec is None or spec.strip().lower() in ("", "none"):
        return None
    from .. import signals

    steps = []
    for part in _split_top(spec, "+"):
        name, params = parse_spec(part)
        attr = PREPROCESS.get(name.lower(), name)
        cls = getattr(signals, attr, None)
        if cls is None:
            raise ValueError(f"unknown preprocessing {name!r}: use {', '.join(PREPROCESS)}, 'none', or a class of "
                             f"indoorloc.signals ({attr} is not available in this version)")
        steps.append(cls(**params))
    return steps[0] if len(steps) == 1 else signals.Compose(steps)


# --------------------------------------------------------------------------- provenance
def _jsonable(value):
    """Plain JSON types (numpy scalars/arrays, tuples, Paths, estimators by repr)."""
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
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def describe(obj):
    """An estimator as ``{"class": "module:Class", "params": {...}}``, recursively, with every
    constructor parameter (defaults included); non-finite floats are kept as strings ("nan")."""
    if obj is None:
        return None
    if hasattr(obj, "get_params"):
        cls = type(obj)
        return {"class": f"{cls.__module__}:{cls.__qualname__}",
                "params": {k: describe(v) for k, v in obj.get_params(deep=False).items()}}
    if isinstance(obj, (list, tuple)):
        return [describe(v) for v in obj]
    if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
        return repr(float(obj))
    return _jsonable(obj)


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]  # .../indoorloc


@functools.lru_cache(maxsize=1)
def git_info() -> dict | None:
    """Commit of the checkout that holds this package, and whether the package differs from it.

    None unless the package directory is the ``indoorloc/`` folder at the top of a git
    work tree (an installed wheel inside some other repository is not attributed to it).
    ``dirty`` counts modified and untracked files under ``indoorloc/``.
    """
    pkg = _package_root()

    def git(*args) -> str | None:
        try:
            out = subprocess.run(["git", "-C", str(pkg.parent), *args], capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    top = git("rev-parse", "--show-toplevel")
    if not top or Path(top).resolve() != pkg.parent:
        return None
    commit = git("rev-parse", "HEAD")
    if not commit:
        return None
    status = git("status", "--porcelain", "--", pkg.name) or ""
    return {"commit": commit, "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status), "changed_files": len(status.splitlines())}


@functools.lru_cache(maxsize=1)
def source_digest() -> str:
    """sha256 over the library's .py and .json files (relative path + bytes, sorted): identifies
    the exact code that ran, with or without git. Computed once per process, like ``git_info``."""
    pkg = _package_root()
    digest = hashlib.sha256()
    for path in sorted(p for p in pkg.rglob("*") if p.suffix in (".py", ".json") and "__pycache__" not in p.parts):
        digest.update(path.relative_to(pkg).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _blas() -> dict | None:
    try:
        deps = np.show_config(mode="dicts")["Build Dependencies"]
        return {k: {kk: v.get(kk) for kk in ("name", "version")} for k, v in deps.items() if k in ("blas", "lapack")}
    except Exception:  # numpy < 1.26 has no mode="dicts"; the facts are optional
        return None


def environment() -> dict:
    """Versions and machine facts for a result file (loads nothing new)."""
    from .._version import __version__

    heavy = ("sklearn", "torch", "scipy", "pandas", "timm", "matplotlib")
    packages = {m: getattr(sys.modules[m], "__version__", "?") for m in heavy if m in sys.modules}
    threads = {k: os.environ[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
               if k in os.environ}
    return {"indoorloc": __version__, "python": platform.python_version(),
            "implementation": platform.python_implementation(), "numpy": np.__version__,
            "platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
            "blas": _blas(), "thread_env": threads, "packages": packages, "git": git_info(),
            "source_sha256": source_digest()}


# --------------------------------------------------------------------------- scoring
def _score(y_true, y_pred, floor_t, floor_p, bldg_t, bldg_p, scale: float, seed) -> dict:
    """EvaluationResults plus competition scores, CDF points and a bootstrap CI of the mean."""
    from ..evaluation import evaluate
    from ..evaluation.scoring import (BUILDING_PENALTY, EVAAL_ETRI_FLOOR_PENALTY, bootstrap_ci, ipin_score,
                                      penalized_errors)

    res = evaluate(y_true, y_pred, floor_true=floor_t, floor_pred=floor_p, building_true=bldg_t,
                   building_pred=bldg_p, scale=scale)
    out = res.to_dict()
    placed = np.isfinite(res.errors)
    if not placed.all():  # samples the method could not place: statistics over the others (n_failed says how many)
        pick = lambda a: None if a is None else np.asarray(a)[placed]  # noqa: E731
        y_true, y_pred = np.asarray(y_true)[placed], np.asarray(y_pred)[placed]
        floor_t, floor_p, bldg_t, bldg_p = pick(floor_t), pick(floor_p), pick(bldg_t), pick(bldg_p)
        out["failed_note"] = (f"{res.n_failed} of {res.n} samples were not placed (NaN estimate): error statistics, "
                              "scores and the CI cover the placed ones; the CDF counts unplaced samples as misses")
    labels = dict(floor_true=floor_t, floor_pred=floor_p, building_true=bldg_t, building_pred=bldg_p)
    out["ipin_score"] = out["mean_penalized_error"] = out["p75_penalized_error"] = out["mean_error_ci95"] = None
    if not placed.any():
        out["scores_note"] = "no sample was placed: no error statistic, score or CI exists"
    else:
        try:
            out["ipin_score"] = ipin_score(y_true, y_pred, **labels, scale=scale)
            pen = penalized_errors(y_true, y_pred, **labels, floor_penalty=EVAAL_ETRI_FLOOR_PENALTY,
                                   building_penalty=BUILDING_PENALTY, scale=scale)
            out["mean_penalized_error"] = float(pen.mean())
            out["p75_penalized_error"] = float(np.percentile(pen, 75))
        except ValueError as err:  # e.g. the dataset has floors but the method does not predict them
            out["ipin_score"] = out["mean_penalized_error"] = out["p75_penalized_error"] = None
            out["scores_note"] = str(err)
        out["mean_error_ci95"] = list(bootstrap_ci(res.errors[placed], "mean", confidence=0.95, n_resamples=1000,
                                                   random_state=seed))
    # percent of ALL samples within t (success_rate's formula; an unplaced sample never is within t)
    out["cdf"] = {f"<={t:g}": float(np.count_nonzero(res.errors[placed] <= t) * 100.0 / res.n) for t in CDF_THRESHOLDS}
    return out


def _labels_of(parts: list, key: str):
    cols = [p[key] for p in parts]
    return None if any(c is None for c in cols) else np.concatenate(cols)


# --------------------------------------------------------------------------- run
def run_benchmark(dataset: str, methods, *, protocol: str = "official", preprocess: str | None = None,
                  seed: int = 0, root=None, download: bool = True, verify: bool = True, units: str = "native",
                  dataset_options: dict | None = None, command=None, predictions: str | None = None,
                  save_models: str | None = None, log=None) -> dict:
    """Run every method under one protocol and return the result document (JSON-ready dict).

    dataset      registry name or ``"module:Class"`` of an L1 dataset.
    methods      method specs, e.g. ``["knn", "wknn(k=3)"]`` (see :func:`parse_spec`).
    protocol     a name from ``indoorloc.evaluation.PROTOCOLS``.
    preprocess   L2 preprocessing spec (see :func:`build_preprocess`); None = ``"fill"`` for RSSI
                 data (missing readings -> -104 dBm, the library default) and ``"none"`` otherwise.
    units        ``"native"`` (the dataset's coordinates, as in the literature) or ``"ground"``
                 (times ``meta["ground_scale"]``, e.g. EPSG:3857 -> ground metres).
    predictions  optional .npz path for per-sample predictions of every method and fold.
    save_models  optional folder: each fitted model is saved to ``<folder>/<label>`` with
                 :mod:`indoorloc.core.persistence` (config.json + arrays.npz, no pickle), its
                 ``info`` holding the dataset digests, protocol, fold digests and seed.
                 Single-fold protocols only.

    Every split of the dataset is loaded and pooled with
    :func:`~indoorloc.evaluation.protocols.pool_splits` before the protocol picks its rows:
    rows without a position label (an ``"unlabeled"`` split) are left out, a sample listed in
    two splits (an ``"all"`` split that repeats ``"train"`` and ``"test"``) is kept once, and
    both counts are recorded under ``dataset`` in the result. Datasets without coordinates
    (room labels only) are refused. Samples a method cannot place (NaN estimates, e.g. a scan that
    hears fewer than three anchors) are counted in ``n_failed`` and left out of the error statistics.
    """
    from ..datasets import DATASETS
    from ..evaluation import get_protocol, literature, pool_splits, split_summary
    from ..methods import METHODS, create_model

    log = log or (lambda msg: None)
    start = time.perf_counter()
    # resolve every name before loading data, so a typo fails in milliseconds
    cls = DATASETS.get(dataset)
    rule = get_protocol(protocol)
    pre_spec = preprocess if preprocess is not None else (
        "fill" if cls.meta.get("modality") in RSSI_MODALITIES else "none")
    build_preprocess(pre_spec)
    specs = []
    for spec in methods:
        name, params = parse_spec(spec)
        method_cls = METHODS.get(name)
        if "random_state" in inspect.signature(method_cls.__init__).parameters and "random_state" not in params:
            params["random_state"] = seed  # the run's seed reaches every seeded method
        specs.append((spec, name, params))
    if units not in ("native", "ground"):
        raise ValueError(f"units must be 'native' or 'ground', got {units!r}")

    source = cls(root, download=download, verify=verify, **(dataset_options or {}))
    # "all" (the whole dataset, by convention) goes last: pool_splits keeps a sample listed in two
    # splits under the first one, so rows of "all" that are also in "train"/"test" keep those labels
    splits = tuple(sorted(source.files, key=lambda s: s == "all"))
    log(f"loading {cls.name or dataset} splits {', '.join(splits)}")
    tables, n_samples, unlabelled = {}, {}, {}
    for s in splits:
        t = source.load(s)
        n_samples[s] = len(t)
        if t.pos.shape[1] == 0:
            raise ValueError(f"{cls.name or dataset} has no coordinates (pos is {t.pos.shape}); the benchmark scores "
                             "positions, so a dataset labelled only by room or zone cannot be benchmarked here")
        located = np.all(np.isfinite(t.pos), axis=1)
        if not located.all():  # e.g. an "unlabeled" split: nothing to train on or score against
            unlabelled[s] = int(np.count_nonzero(~located))
            log(f"{s}: {unlabelled[s]} of {len(t)} rows have no position label and are left out")
            t = t[located] if located.any() else None
        if t is not None:
            tables[s] = t
    table = pool_splits(tables)
    del tables  # the pooled table holds the data; the per-split copies would double the memory
    meta = table.meta
    if meta.get("pooled_duplicates"):
        log(f"rows listed in two splits were kept once: {meta['pooled_duplicates']} duplicates dropped")
    folds = rule.folds(table, random_state=seed)
    if save_models and len(folds) != 1:
        raise ValueError(f"--save-models needs a single-fold protocol; {protocol!r} has {len(folds)} folds")
    if units == "native":
        scale, unit = 1.0, str(meta.get("pos_units") or "m")
    else:
        if "ground_scale" not in meta:
            raise ValueError(f"{cls.name}: no meta['ground_scale']; its coordinates are already ground units")
        scale, unit = float(meta["ground_scale"]), "m (ground)"
    saved = {}
    results = []
    labels_seen = {}
    for spec, name, params in specs:
        label = spec if spec not in labels_seen else f"{spec}#{labels_seen[spec] + 1}"
        labels_seen[spec] = labels_seen.get(spec, 0) + 1
        fold_rows, parts = [], []
        model = None
        for fold in folds:
            train, test = table[fold.train], table[fold.test]
            model = create_model(name, preprocess=build_preprocess(pre_spec), **params)
            t0 = time.perf_counter()
            model.fit(train)
            t1 = time.perf_counter()
            pred = model.localize(test)
            t2 = time.perf_counter()
            if pred.ids is not None and not np.array_equal(pred.ids, test.ids):
                raise RuntimeError(f"{label} returned predictions for other rows than it was given")
            if pred.pos.shape != test.pos.shape:
                raise RuntimeError(f"{label} returned positions of shape {pred.pos.shape} on fold {fold.name} "
                                   f"for {test.pos.shape} targets")
            part = {"pos": test.pos, "pred": pred.pos, "floor_t": test.floor, "floor_p": pred.floor,
                    "bldg_t": test.building, "bldg_p": pred.building, "ids": test.ids}
            parts.append(part)
            metrics = _score(part["pos"], part["pred"], part["floor_t"], part["floor_p"], part["bldg_t"],
                             part["bldg_p"], scale, seed)
            fold_rows.append({"fold": fold.name, "fit_s": t1 - t0, "predict_s": t2 - t1, "metrics": metrics})
            log(f"{label} | {fold.name}: mean {metrics['mean_error']:.4f} {unit}  "
                f"({len(fold.train)} train / {len(fold.test)} test, fit {t1 - t0:.2f} s, predict {t2 - t1:.2f} s)")
            if predictions:
                for key in ("pred", "floor_p", "bldg_p"):
                    if part[key] is not None:
                        saved.setdefault(f"{label}/{key}", []).append(np.asarray(part[key]))
                saved.setdefault(f"{label}/fold", []).append(np.full(len(fold.test), len(fold_rows) - 1))
                saved.setdefault(f"{label}/ids", []).append(np.asarray(part["ids"]).astype(str))
        pooled = _score(np.concatenate([p["pos"] for p in parts]), np.concatenate([p["pred"] for p in parts]),
                        _labels_of(parts, "floor_t"), _labels_of(parts, "floor_p"), _labels_of(parts, "bldg_t"),
                        _labels_of(parts, "bldg_p"), scale, seed)
        results.append({"label": label, "method": name, "params": _jsonable(params), "model": repr(model),
                        "model_params": describe(model),
                        "folds": fold_rows, "pooled": pooled})
        if save_models:
            target = Path(save_models) / "".join(c if c.isalnum() or c in "._=-" else "_" for c in label)
            model.save(target, info={"dataset": cls.name or dataset, "dataset_sha256": meta.get("sha256"),
                                     "protocol": protocol, "fold": folds[0].name, "seed": seed,
                                     "preprocess": pre_spec, "method": spec,
                                     "train_splits": sorted(set(table.groups["split"][folds[0].train].tolist())),
                                     **split_summary(folds[0].train, folds[0].test)})
            results[-1]["saved_to"] = str(target)
            log(f"saved {label} to {target}")
    if predictions:
        np.savez_compressed(predictions, **{k: np.concatenate(v) for k, v in saved.items()})
    payload = {
        "format": FORMAT, "format_version": FORMAT_VERSION,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": list(command) if command else None,
        "dataset": {"name": cls.name or dataset, "class": f"{cls.__module__}:{cls.__qualname__}",
                    "options": _jsonable(dataset_options or {}), "splits": list(splits),
                    "n_samples": n_samples, "n_pooled": len(table),
                    **({"unlabelled_rows_left_out": unlabelled} if unlabelled else {}),
                    **({"duplicate_rows_dropped": meta["pooled_duplicates"]} if meta.get("pooled_duplicates") else {}),
                    "sha256": _jsonable(meta.get("sha256")),
                    **{k: _jsonable(meta.get(k)) for k in ("modality", "crs", "pos_units", "citation", "doi",
                                                           "license", "url") if meta.get(k) is not None}},
        "protocol": {"name": protocol, "summary": rule.summary,
                     "folds": [{"name": f.name, **split_summary(f.train, f.test, len(table))} for f in folds]},
        "preprocess": {"spec": pre_spec, "default": preprocess is None,
                       "resolved": describe(build_preprocess(pre_spec))},
        "seed": seed, "units": unit, "scale": scale,
        "methods": results,
        "environment": environment(),
    }
    lit_name = cls.name or dataset
    if lit_name in literature.list_tables():
        if protocol == "official" and scale == 1.0:
            comparison = literature.compare({r["label"]: r["pooled"] for r in results}, lit_name, protocol=protocol)
            payload["literature"] = comparison.to_dict()
        elif scale != 1.0:  # published numbers are in the dataset's own coordinates
            payload["literature_note"] = (f"published numbers for {lit_name} are in the dataset's native units, "
                                          f"these results in {unit}; run with --units native to see them side by "
                                          f"side, or `indoorloc literature {lit_name}`")
        else:
            payload["literature_note"] = (f"published numbers for {lit_name} exist, but for other protocols; "
                                          f"see `indoorloc literature {lit_name}`")
    payload["wall_time_s"] = time.perf_counter() - start
    return _jsonable(payload)
