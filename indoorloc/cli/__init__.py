"""Command line interface (top layer: may use every layer).

::

    indoorloc list datasets|methods|protocols|literature [-v]
    indoorloc info ujiindoorloc
    indoorloc benchmark --dataset ujiindoorloc --method knn --method "wknn(k=3)" \\
                        [--protocol official] [--preprocess fill] [--seed 0] --out results.json
    indoorloc literature ujiindoorloc [--all] [--results results.json]
    indoorloc report results.json [--format markdown|text]
    indoorloc evaluate --model models/knn --dataset ujiindoorloc [--split test]

``python -m indoorloc`` (or ``python -m indoorloc.cli``) runs the same program. Commands import the layers they use only
when they run, so ``indoorloc list methods`` loads neither the datasets nor torch, and
importing this module loads nothing but the standard library.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["build_parser", "main"]


def _options(pairs) -> dict:
    from .benchmark import _value

    out = {}
    for pair in pairs or ():
        key, eq, raw = pair.partition("=")
        if not eq or not key.isidentifier():
            raise ValueError(f"expected key=value, got {pair!r}")
        out[key] = _value(raw)
    return out


def _print(text: str) -> None:
    sys.stdout.write(text if text.endswith("\n") else text + "\n")


# --------------------------------------------------------------------------- commands
def _cmd_list(args) -> int:
    if args.what == "datasets":
        from ..datasets import DATASETS

        for name in DATASETS.names():
            if not args.verbose:
                _print(name)
                continue
            try:
                cls = DATASETS.get(name)
                doc = (cls.__doc__ or "").strip().split("\n")[0]
                _print(f"{name:<18} {cls.meta.get('modality', '?'):<10} {doc}")
            except (ImportError, AttributeError) as err:
                _print(f"{name:<18} {'?':<10} (not available: {type(err).__name__}: {err})")
    elif args.what == "methods":
        from ..methods import METHODS

        for name in METHODS.names():
            if not args.verbose:
                _print(name)
                continue
            try:
                doc = (METHODS.get(name).__doc__ or "").strip().split("\n")[0]
                _print(f"{name:<14} {doc}")
            except (ImportError, AttributeError) as err:
                _print(f"{name:<14} (not available: {type(err).__name__}: {err})")
    elif args.what == "protocols":
        from ..evaluation import get_protocol, list_protocols

        for name in list_protocols():
            _print(f"{name:<24} {get_protocol(name).summary}" if args.verbose else name)
    else:  # literature
        from ..evaluation import literature

        for name in literature.list_tables():
            if args.verbose:
                doc = literature.load(name)
                counts = {}
                for e in doc["entries"]:
                    counts[e["check"]] = counts.get(e["check"], 0) + 1
                _print(f"{name:<16} {len(doc['entries']):>3} entries  "
                       + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
            else:
                _print(name)
    return 0


def _cmd_info(args) -> int:
    from ..datasets import DATASETS, dataset_info, default_root
    from ..evaluation import literature

    cls = DATASETS.get(args.dataset)
    info = dataset_info(args.dataset)
    root = Path(args.root) if args.root else default_root() / cls.name
    source = cls(root)
    files = [{"split": split, "path": rel, "sha256": sha, "present": (root / rel).is_file()}
             for split in source.files for rel, sha in source._entries(split)]
    lit_name = cls.name if cls.name in literature.list_tables() else None
    info.update({"class": f"{cls.__module__}:{cls.__qualname__}", "root": str(root), "files": files,
                 "split_aliases": dict(cls.split_aliases)})
    if lit_name:
        entries = literature.load(lit_name)["entries"]
        info["literature"] = {"entries": len(entries), "command": f"indoorloc literature {lit_name}"}
    if args.json:
        from .benchmark import _jsonable

        _print(json.dumps(_jsonable(info), indent=1, ensure_ascii=False))
        return 0
    skip = {"files", "doc", "name"}
    _print(f"{info.get('name', args.dataset)}: {info.get('doc', '')}")
    for key, value in info.items():
        if key not in skip and value not in (None, "", (), {}, []):
            if isinstance(value, dict):
                value = ", ".join(f"{k}={v}" for k, v in value.items())
            elif isinstance(value, (list, tuple)):
                value = ", ".join(map(str, value))
            _print(f"  {key:<18} {value}")
    for f in files:
        state = "present" if f["present"] else "missing (downloaded on first use)"
        _print(f"  {'file':<18} [{f['split']}] {f['path']}  {state}")
    return 0


def _cmd_benchmark(args, argv) -> int:
    from ..evaluation.report import benchmark_report
    from .benchmark import run_benchmark

    log = (lambda msg: None) if args.quiet else (lambda msg: print(f"[indoorloc] {msg}", file=sys.stderr))
    payload = run_benchmark(args.dataset, args.method, protocol=args.protocol, preprocess=args.preprocess,
                            seed=args.seed, root=args.root, download=not args.no_download, verify=not args.no_verify,
                            units=args.units, dataset_options=_options(args.dataset_option),
                            command=["indoorloc", *argv], predictions=args.predictions,
                            save_models=args.save_models, log=log)
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=1, ensure_ascii=False, allow_nan=False) + "\n",
                                  encoding="utf-8")
        log(f"wrote {args.out}")
    if args.report:
        Path(args.report).write_text(benchmark_report(payload, style="markdown"), encoding="utf-8")
        log(f"wrote {args.report}")
    if not args.quiet:
        _print(benchmark_report(payload, style="text"))
    return 0


def _cmd_literature(args) -> int:
    from ..evaluation import literature

    include = "all" if args.all else literature.DEFAULT_CHECKS
    kinds = "all" if args.all else ("literature",)
    if args.results:
        payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
        if payload.get("scale", 1.0) != 1.0:
            raise ValueError(f"{args.results} holds errors in {payload.get('units')} (--units ground), but published "
                             "numbers are in the dataset's native units; rerun the benchmark with --units native")
        results = {m["label"]: m["pooled"] for m in payload.get("methods", [])}
        protocol = payload.get("protocol", {}).get("name", "official")
        comparison = literature.compare(results, args.dataset, protocol=protocol, include=include, kinds=kinds)
        _print(comparison.render(args.format))
    else:
        _print(literature.table(args.dataset, include=include, kinds=kinds).render(args.format))
    return 0


def _evaluation_split(name: str, root) -> str | None:
    """The split ``evaluate`` scores when ``--split`` is not given: the dataset's official
    ``test`` split when it has one, else ``"all"`` (wlanrssi, csi_fingerprint, hwild, ilc2020),
    else the dataset's first default split."""
    from ..datasets import DATASETS

    try:
        dataset = DATASETS.get(name)(root)
        splits, aliases = tuple(dataset.splits), dict(dataset.split_aliases)
        default = tuple(dataset.default_splits)
    except (KeyError, ImportError, AttributeError, TypeError, ValueError):
        return "test"  # e.g. a 0.1 id: let load_dataset report what is wrong
    if "test" in splits or aliases.get("test") in splits:
        return "test"
    return "all" if "all" in splits else (default[0] if default else "test")


def _cmd_evaluate(args) -> int:
    """Score a model saved by ``benchmark --save-models`` on one split of a dataset."""
    from ..core import load_model
    from ..datasets import load_dataset
    from ..evaluation import evaluate
    from ..evaluation.report import results_table

    model = load_model(args.model)
    split = args.split if args.split is not None else _evaluation_split(args.dataset, args.root)
    table = load_dataset(args.dataset, split=split, root=args.root, download=not args.no_download)
    info = getattr(model, "saved_info_", {}) or {}
    if info.get("dataset") == table.meta.get("name") and table.meta.get("split") in info.get("train_splits", ()):
        print(f"indoorloc: warning: rows of the {table.meta.get('split')!r} split were used to train this model "
              f"({info.get('protocol')} protocol); the score is not a held-out estimate", file=sys.stderr)
    scale = float(table.meta.get("ground_scale", 1.0)) if args.units == "ground" else 1.0
    res = evaluate(table, model.localize(table), scale=scale)
    _print(f"{args.model} on {table.meta.get('name')} [{table.meta.get('split')}], n={res.n}")
    _print(results_table({info.get("method", "model"): res}, style=args.format))
    return 0


def _cmd_report(args) -> int:
    from ..evaluation.report import benchmark_report

    payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    _print(benchmark_report(payload, style=args.format, literature=not args.no_literature))
    return 0


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    from .._version import __version__

    parser = argparse.ArgumentParser(prog="indoorloc", description="Wireless indoor localization toolkit.")
    parser.add_argument("--version", action="version", version=f"indoorloc {__version__}")
    parser.add_argument("--traceback", action="store_true", help="show the full traceback on errors")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="list datasets, methods, protocols or literature tables")
    p.add_argument("what", choices=("datasets", "methods", "protocols", "literature"))
    p.add_argument("-v", "--verbose", action="store_true", help="one-line description of each entry")

    p = sub.add_parser("info", help="facts about a dataset: modality, frame, license, files, literature")
    p.add_argument("dataset")
    p.add_argument("--root", help="dataset folder (default: $INDOORLOC_DATA/<name>)")
    p.add_argument("--json", action="store_true", help="print JSON")

    p = sub.add_parser("benchmark", help="fit and score methods on a dataset under a named protocol",
                       description="Fit and score methods under a protocol and write a self-describing JSON file.")
    p.add_argument("--dataset", required=True, help="registry name or module:Class")
    p.add_argument("--method", action="append", required=True,
                   help="method spec, repeatable: knn, 'wknn(k=3)', 'pkg.mod:MyLocalizer(alpha=0.1)'")
    p.add_argument("--protocol", default="official", help="see `indoorloc list protocols -v` (default: official)")
    p.add_argument("--preprocess", default=None,
                   help="L2 preprocessing: fill, normalize, positive, exponential, powed, none, or a signals class; "
                        "chain with '+', arguments in parentheses: 'fill(value=-110)+normalize'. "
                        "Default: fill for RSSI data, none otherwise")
    p.add_argument("--seed", type=int, default=0, help="seed of random protocols and of methods with random_state")
    p.add_argument("--units", choices=("native", "ground"), default="native",
                   help="report errors in the dataset's coordinates (default, as in the literature) or ground metres")
    p.add_argument("--root", help="dataset folder (default: $INDOORLOC_DATA/<name>)")
    p.add_argument("--dataset-option", action="append", metavar="KEY=VALUE", help="dataset constructor option")
    p.add_argument("--no-download", action="store_true", help="fail instead of downloading missing files")
    p.add_argument("--no-verify", action="store_true", help="skip sha256 checks of the data files")
    p.add_argument("--out", help="write the result JSON here")
    p.add_argument("--report", help="also write a Markdown report here")
    p.add_argument("--predictions", help="save per-sample predictions (.npz)")
    p.add_argument("--save-models", metavar="DIR", help="save each fitted model to DIR/<method> (no pickle)")
    p.add_argument("-q", "--quiet", action="store_true", help="no progress or report on the terminal")

    p = sub.add_parser("literature", help="published numbers for a dataset, with provenance")
    p.add_argument("dataset")
    p.add_argument("--all", action="store_true",
                   help="every ported entry, also those without a traceable source or number and 0.1's own runs")
    p.add_argument("--results", help="a benchmark JSON to show next to the literature (kept separate)")
    p.add_argument("--format", choices=("text", "markdown"), default="text")

    p = sub.add_parser("evaluate", help="score a saved model (benchmark --save-models) on a dataset split")
    p.add_argument("--model", required=True, help="folder written by --save-models")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default=None,
                   help="split to score (default: the official test split when the dataset has one, else 'all')")
    p.add_argument("--root", help="dataset folder (default: $INDOORLOC_DATA/<name>)")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--units", choices=("native", "ground"), default="native")
    p.add_argument("--format", choices=("text", "markdown"), default="text")

    p = sub.add_parser("report", help="render a benchmark JSON as a Markdown or text report")
    p.add_argument("results")
    p.add_argument("--format", choices=("markdown", "text"), default="markdown")
    p.add_argument("--no-literature", action="store_true", help="leave out published numbers")
    return parser


def main(argv=None) -> int:
    """Entry point of the ``indoorloc`` script; returns the exit status (0 ok, 1 error, 2 usage)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list":
            return _cmd_list(args)
        if args.command == "info":
            return _cmd_info(args)
        if args.command == "benchmark":
            return _cmd_benchmark(args, argv)
        if args.command == "literature":
            return _cmd_literature(args)
        if args.command == "evaluate":
            return _cmd_evaluate(args)
        return _cmd_report(args)
    except (KeyError, ValueError, TypeError, FileNotFoundError, ImportError, RuntimeError) as err:
        if args.traceback:
            raise
        message = err.args[0] if isinstance(err, KeyError) and err.args else err
        print(f"indoorloc: error: {message}", file=sys.stderr)
        return 1
