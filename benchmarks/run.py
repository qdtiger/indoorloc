"""Run the benchmark matrix (``matrix.py``) cell by cell and write ``results/<dataset>.json``.

Every cell (one dataset table x one preprocessing x one method) runs in its own process through
the public command line, exactly as a user would type it::

    python -m indoorloc benchmark --dataset tuji1 --protocol official --preprocess fill \\
        --method wknn --seed 0 --no-download -q --out cell.json

(label-only datasets go through ``python -m benchmarks.labels`` instead). A separate process per
cell gives an honest peak memory (``ru_maxrss`` of that process, data loading included), keeps a
slow or memory-hungry cell from affecting the others, and lets a cell be killed at a time or
memory limit, which is then recorded as the cell's outcome instead of a number.

Usage (from the repository root)::

    python benchmarks/run.py                          # the whole matrix, then render the docs
    python benchmarks/run.py --dataset tuji1          # one dataset (all its tables)
    python benchmarks/run.py --dataset sodindoorloc --table official-HCXY --method wknn
    python benchmarks/run.py --list                   # the cells, without running anything
    python benchmarks/run.py --dataset tuji1 --verify # rerun and compare with the stored numbers
    python benchmarks/run.py --freeze /tmp/il-snap    # copy the code first, run the copy

Re-running a subset replaces only those cells in the result files. ``--freeze`` copies
``indoorloc/`` and ``benchmarks/`` to a directory and runs from there, so that edits made to the
working tree while the matrix runs (it takes hours) cannot mix two versions of the code; the copy
records the git commit and state it was taken from in ``SNAPSHOT.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORMAT = "indoorloc-benchmark-matrix"
FORMAT_VERSION = 1
THREADS = "8"
THREAD_ENV = {"OPENBLAS_NUM_THREADS": THREADS, "OMP_NUM_THREADS": THREADS, "MKL_NUM_THREADS": THREADS,
              "PYTHONHASHSEED": "0"}


def _code_root(bench_dir: Path) -> Path:
    return bench_dir.parent


sys.path.insert(0, str(_code_root(HERE)))  # the indoorloc and benchmarks packages of this checkout (or copy)

from benchmarks import matrix  # noqa: E402


# --------------------------------------------------------------------------- machine facts
def hardware() -> dict:
    cpu = platform.processor() or "?"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    mem = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                mem = round(int(line.split()[1]) / 1024 ** 2, 1)
    except OSError:
        pass
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "memory_gb": mem, "platform": platform.platform(),
            "python": platform.python_version()}


def _cgroup_dir() -> Path | None:
    """The cgroup v2 folder of this process (Linux), or None."""
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                path = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
                return path if (path / "memory.pressure").is_file() else None
    except OSError:
        pass
    return None


def cgroup_limits() -> dict | None:
    """Memory limits of the cgroup the cells run in (MB; None = no limit), or None off Linux.

    The cells inherit this cgroup, and every other process in it shares the same limits: above
    ``memory.high`` the kernel throttles allocations, which stretches wall times without changing
    any result.
    """
    cg = _cgroup_dir()
    if cg is None:
        return None

    def mb(name):
        try:
            text = (cg / name).read_text().strip()
        except OSError:
            return None
        return None if text == "max" else round(int(text) / 1024 ** 2, 1)

    return {"memory_high_mb": mb("memory.high"), "memory_max_mb": mb("memory.max"),
            "swap_max_mb": mb("memory.swap.max")}


def memory_stall_us() -> tuple[int, int] | None:
    """Cumulative (some, full) memory-pressure stall of this cgroup in microseconds (Linux PSI).

    ``some``: time in which at least one task of the cgroup waited for memory (reclaim, swap-in,
    throttling above ``memory.high``); ``full``: time in which all non-idle tasks did at once.
    """
    cg = _cgroup_dir()
    if cg is None:
        return None
    try:
        values = {}
        for line in (cg / "memory.pressure").read_text().splitlines():
            kind, *fields = line.split()
            values[kind] = int(dict(f.split("=") for f in fields)["total"])
        return values["some"], values["full"]
    except (OSError, KeyError, ValueError):
        return None


def git_state(repo: Path) -> dict | None:
    def git(*args):
        try:
            out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if not commit:
        return None
    status = git("status", "--porcelain", "--", "indoorloc") or ""
    return {"commit": commit, "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "dirty": bool(status),
            "changed_files_under_indoorloc": len(status.splitlines())}


# --------------------------------------------------------------------------- freeze
def freeze(target: Path) -> Path:
    """Copy indoorloc/ and benchmarks/ (without results and caches) to ``target``; record the source state."""
    root = _code_root(HERE)
    if target.exists():
        raise SystemExit(f"--freeze: {target} exists; choose a new directory")
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "results")
    shutil.copytree(root / "indoorloc", target / "indoorloc", ignore=ignore)
    shutil.copytree(root / "benchmarks", target / "benchmarks", ignore=ignore)
    env = {**os.environ, "PYTHONPATH": str(target)}
    digest = subprocess.run([sys.executable, "-c", "from indoorloc.cli.benchmark import source_digest; "
                             "print(source_digest())"], cwd=target, env=env, capture_output=True, text=True,
                            check=True).stdout.strip()
    info = {"taken": datetime.now(timezone.utc).isoformat(timespec="seconds"), "from": str(root),
            "git": git_state(root), "source_sha256": digest, "frozen": True}
    (target / "SNAPSHOT.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    return target


def code_state(code_root: Path) -> dict:
    """The code a run uses: the ``SNAPSHOT.json`` of a ``--freeze`` copy, else the checkout's git
    state and the library's source digest when the run starts (each cell also records the digest
    of the code it imported, so a working tree edited during the run is still visible per cell)."""
    snapshot = _load(code_root / "SNAPSHOT.json")
    if snapshot:
        return {"frozen": True, **snapshot}
    from indoorloc.cli.benchmark import source_digest  # the checkout's indoorloc (code_root is on sys.path)

    return {"taken": datetime.now(timezone.utc).isoformat(timespec="seconds"), "from": str(code_root),
            "git": git_state(code_root), "source_sha256": source_digest(), "frozen": False}


# --------------------------------------------------------------------------- cells
def _anchors(table) -> str:
    """meta['anchors'] of the table's dataset selection, as a JSON list (exact float reprs)."""
    from indoorloc.datasets import DATASETS

    source = DATASETS.get(table.dataset)(None, download=False, **table.options)
    split = "test" if "test" in source.files else next(iter(source.files))
    anchors = source.load(split).meta.get("anchors")
    if anchors is None:
        raise ValueError(f"{table.key}: a method spec needs {{anchors}} but the dataset has no meta['anchors']")
    return json.dumps([[float(v) for v in row] for row in anchors], separators=(",", ":"))


def command(table, pre: str, spec: str, out: Path, seed: int = 0) -> list[str]:
    """The argv (after ``python``) of one cell."""
    opts = [a for k, v in table.options.items() for a in ("--dataset-option", f"{k}={v}")]
    if table.runner == "labels":
        return ["-m", "benchmarks.labels", "--dataset", table.dataset, "--label", table.label, "--preprocess", pre,
                "--method", spec, "--seed", str(seed), "--no-download", "-q", "--out", str(out), *opts]
    return ["-m", "indoorloc", "benchmark", "--dataset", table.dataset, "--protocol", table.protocol,
            "--preprocess", pre, "--method", spec, "--seed", str(seed), "--no-download", "-q", "--out", str(out), *opts]


def _shown(argv: list[str]) -> str:
    """The command as a user types it: ``indoorloc benchmark ...`` without the scratch --out path."""
    args = list(argv)
    if "--out" in args:
        i = args.index("--out")
        del args[i:i + 2]
    head = ["indoorloc"] if args[:2] == ["-m", "indoorloc"] else ["python", "-m", args[1]]
    return shlex.join(head + args[2:])


def _rss_mb(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    except (OSError, ValueError):
        pass
    return 0.0


def run_cell(argv: list[str], *, cwd: Path, log: Path, timeout: float, max_rss_mb: float) -> dict:
    """Run one cell; return its status, wall and CPU time, peak memory (MB, ru_maxrss) and how
    busy the machine was (load average, memory-pressure stall of the cgroup during the cell)."""
    env = {**os.environ, **THREAD_ENV, "PYTHONPATH": str(cwd)}
    load_before = os.getloadavg()[0]  # the machine may be shared: record how busy it was
    stall_before = memory_stall_us()
    start = time.perf_counter()
    with open(log, "w", encoding="utf-8") as err:
        proc = subprocess.Popen([sys.executable, *argv], cwd=cwd, env=env, stdout=err, stderr=subprocess.STDOUT)
        status, reason, usage, polled_peak = "ok", None, None, 0.0
        while True:
            pid, code, usage = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                proc.returncode = os.waitstatus_to_exitcode(code)
                break
            rss = _rss_mb(proc.pid)
            polled_peak = max(polled_peak, rss)
            elapsed = time.perf_counter() - start
            if rss > max_rss_mb or elapsed > timeout:
                status = "memory" if rss > max_rss_mb else "timeout"
                reason = (f"stopped at {rss:.0f} MB resident, above the {max_rss_mb:.0f} MB limit per process"
                          if status == "memory" else f"did not finish within the {timeout / 60:.0f}-minute limit")
                proc.send_signal(signal.SIGKILL)
                _, code, usage = os.wait4(proc.pid, 0)
                proc.returncode = os.waitstatus_to_exitcode(code)
                break
            time.sleep(0.2)
    wall = time.perf_counter() - start
    peak = max(usage.ru_maxrss / 1024 if usage else 0.0, polled_peak)  # ru_maxrss is in KiB on Linux
    if status == "ok" and proc.returncode != 0:
        status = "failed"
        tail = [ln for ln in log.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
        reason = tail[-1] if tail else f"exit status {proc.returncode}"
    stall_after = memory_stall_us()
    stall = None
    if stall_before is not None and stall_after is not None:
        stall = {"some": round((stall_after[0] - stall_before[0]) / 1e6, 2),
                 "full": round((stall_after[1] - stall_before[1]) / 1e6, 2)}
    cpu = round(usage.ru_utime + usage.ru_stime, 2) if usage else None
    return {"status": status, "reason": reason, "wall_time_s": round(wall, 3), "cpu_time_s": cpu,
            "peak_rss_mb": round(peak, 1), "load_avg_1min": [round(load_before, 2), round(os.getloadavg()[0], 2)],
            "memory_stall_s": stall}


def _cell_record(table, pre, spec, argv, outcome, payload) -> dict:
    rec = {"preprocess": pre, "method": spec, "display": [matrix.display(spec, 0), matrix.display(spec, 1)],
           "command": _shown(argv), **outcome}
    if payload is None:
        return rec
    m = payload["methods"][0]
    env = payload.get("environment", {})
    rec.update({
        "created": payload.get("created"),
        "fit_s": round(sum(f["fit_s"] for f in m["folds"]), 4),
        "predict_s": round(sum(f["predict_s"] for f in m["folds"]), 4),
        "pooled": m["pooled"],
        "folds": [{"fold": f["fold"], "fit_s": round(f["fit_s"], 4), "predict_s": round(f["predict_s"], 4),
                   "metrics": {k: v for k, v in f["metrics"].items() if k in _FOLD_KEYS}} for f in m["folds"]],
        "params": m.get("params"), "model_params": m.get("model_params"),
        "preprocess_resolved": payload.get("preprocess", {}).get("resolved"),
        "source_sha256": env.get("source_sha256"),
        "packages": env.get("packages"),  # sklearn / torch versions, when the cell loaded them
        "thread_env": env.get("thread_env"),
        "folds_sha256": _folds_digest(payload["protocol"]["folds"]),
    })
    return rec


_FOLD_KEYS = ("n", "mean_error", "median_error", "p75_error", "p90_error", "floor_accuracy", "building_accuracy",
              "ipin_score", "n_failed", "accuracy")


def _folds_digest(folds) -> str:
    text = json.dumps([[f["name"], f["train_sha256"], f["test_sha256"]] for f in folds])
    return hashlib.sha256(text.encode()).hexdigest()


def _table_header(table, payload) -> dict:
    """Facts shared by every cell of a table, taken from one cell's result document."""
    ds = payload["dataset"]
    return {"dataset": {k: ds.get(k) for k in ("name", "class", "options", "splits", "n_samples", "n_pooled",
                                               "unlabelled_rows_left_out", "duplicate_rows_dropped", "sha256",
                                               "modality", "crs", "pos_units", "task", "citation", "doi",
                                               "license", "url") if ds.get(k) is not None},
            "protocol": payload["protocol"], "units": payload.get("units"), "scale": payload.get("scale"),
            "seed": payload.get("seed")}


def literature_block(dataset: str) -> dict:
    """Published numbers for the dataset from indoorloc.evaluation.literature, kept apart from ours."""
    from indoorloc.evaluation import literature

    if dataset not in literature.list_tables():
        return {"available": False, "entries": [], "hidden": 0, "checks_all": {}}
    doc = literature.load(dataset)
    shown = literature.table(dataset)  # verified, corrected, unchecked: a traceable source and a number
    checks = {}
    for e in doc["entries"]:
        key = f"{e['kind']}/{e['check']}"
        checks[key] = checks.get(key, 0) + 1
    return {"available": True, "display_name": shown.display_name, "metrics": shown.metrics,
            "entries": [{"method": e["method"], "values": e["values"], "protocol": e["protocol"],
                         "check": e["check"], "location": e.get("location"), "source": e.get("source")}
                        for e in shown.entries],
            "hidden": shown.excluded, "checks_all": checks,
            "rule": "shown: literature entries checked 'verified', 'corrected' or 'unchecked'; hidden: "
                    "'unidentified', 'missing' and numbers IndoorLoc 0.1 produced itself"}


# --------------------------------------------------------------------------- result files
def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def merge_harness(old: dict | None, new: dict | None, digests) -> dict | None:
    """The harness block of a result file whose cells may come from several runs.

    ``new["code"]`` describes the latest run; the code blocks of earlier runs whose source digest
    some kept cell still records move to ``code_history``, so re-running a subset never erases the
    provenance of the cells it leaves alone. ``new=None`` keeps ``old`` (``--remerge``)."""
    current = new if new is not None else old
    if current is None:
        return None
    out = dict(current)
    seen = {(out.get("code") or {}).get("source_sha256")}
    history = []
    for h in (current, old or {}):
        for code in [h.get("code"), *(h.get("code_history") or [])]:
            digest = (code or {}).get("source_sha256")
            if digest and digest not in seen and digest in digests:
                seen.add(digest)
                history.append(code)
    out.pop("code_history", None)
    if history:
        out["code_history"] = history
    return out


def merge(dataset: str, new_cells: dict, headers: dict, environment: dict | None, results_dir: Path,
          harness: dict | None) -> dict:
    """Fold freshly run cells into results/<dataset>.json, in matrix order; keep the others.

    ``harness=None`` keeps the stored harness block (``--remerge``: only titles, skips, notes and the
    literature block are refreshed from ``matrix.py`` and ``indoorloc.evaluation.literature``)."""
    path = results_dir / f"{dataset}.json"
    old = _load(path) or {}
    old_tables = {t["id"]: t for t in old.get("tables", [])}
    out_tables = []
    for table in matrix.tables([dataset]):
        prev = old_tables.get(table.id, {})
        prev_cells = {(c["preprocess"], c["method"]): c for c in prev.get("cells", [])}
        cells = []
        for pre, spec in table.cells():
            cell = new_cells.get((table.id, pre, spec)) or prev_cells.get((pre, spec))
            cells.append(cell or {"preprocess": pre, "method": spec, "display": [matrix.display(spec, 0),
                                                                                 matrix.display(spec, 1)],
                                  "status": "not-run"})
        header = headers.get(table.id) or {k: prev[k] for k in ("dataset", "protocol", "units", "scale", "seed")
                                           if k in prev}
        digests = {c["folds_sha256"] for c in cells if c.get("folds_sha256")}
        out_tables.append({
            "id": table.id, "title": {"en": table.title[0], "zh": table.title[1]}, "runner": table.runner,
            "protocol_spec": table.protocol, "options": table.options, **header,
            "same_folds_in_every_cell": len(digests) <= 1,
            "skips": [{"method": s.method, "en": s.en, "zh": s.zh} for s in table.skips],
            "notes": [{"en": en, "zh": zh} for en, zh in table.notes],
            "cells": cells})
    digests = sorted({c["source_sha256"] for t in out_tables for c in t["cells"] if c.get("source_sha256")})
    harness = merge_harness(old.get("harness"), harness, set(digests))
    env = dict(environment or old.get("environment") or {})
    packages = {}
    for t in out_tables:
        for c in t["cells"]:
            for name, version in (c.get("packages") or {}).items():
                packages.setdefault(name, set()).add(version)
    env["packages"] = {name: sorted(v)[0] if len(v) == 1 else sorted(v) for name, v in sorted(packages.items())}
    env.pop("git", None)  # the code state is recorded once, under harness["code"]
    env.pop("source_sha256", None)  # per cell, and the set of them below
    doc = {"format": FORMAT, "format_version": FORMAT_VERSION, "dataset": dataset,
           "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "harness": harness, "environment": env,
           "source_sha256": digests, "tables": out_tables, "literature": literature_block(dataset)}
    _write(path, doc)
    return doc


# --------------------------------------------------------------------------- verification
def _numbers(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _numbers(v, f"{prefix}{k}.")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _numbers(v, f"{prefix}{i}.")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix.rstrip("."), float(obj)


def compare_cells(stored: dict, fresh: dict) -> tuple[int, float, str | None]:
    """(number of metrics compared, largest absolute difference, its key) over pooled and fold metrics."""
    a = dict(_numbers({"pooled": stored.get("pooled"), "folds": [f["metrics"] for f in stored.get("folds", [])]}))
    b = dict(_numbers({"pooled": fresh.get("pooled"), "folds": [f["metrics"] for f in fresh.get("folds", [])]}))
    worst, where = 0.0, None
    for key in set(a) | set(b):
        if key not in a or key not in b:
            return len(a), float("inf"), key
        d = abs(a[key] - b[key])
        if d > worst:
            worst, where = d, key
    return len(a), worst, where


def stored_cells(results_dir: Path, datasets) -> dict:
    """``(dataset, table id, preprocess, method) -> cell`` for the stored result files of ``datasets``."""
    out = {}
    for dataset in datasets:
        doc = _load(results_dir / f"{dataset}.json") or {"tables": []}
        for tb in doc["tables"]:
            for c in tb["cells"]:
                out[(dataset, tb["id"], c["preprocess"], c["method"])] = c
    return out


def rerun_check(previous: dict, fresh: dict) -> dict:
    """How a rerun of a cell compares with the result it replaces (same metrics as ``--verify``)."""
    count, worst, where = compare_cells(previous, fresh)
    return {"previous_created": previous.get("created"), "previous_source_sha256": previous.get("source_sha256"),
            "numbers_compared": count, "largest_difference": worst if math.isfinite(worst) else None,
            "where": where}


def verification_report(fresh_cells: dict, stored: dict) -> tuple[list[dict], int]:
    """``--verify``: compare freshly run cells with the stored ones; return (rows, mismatches).

    ``fresh_cells`` maps dataset -> {(table id, preprocess, method): cell}. A mismatch is any
    difference in a pooled or per-fold metric, a metric present in only one of the two, or a cell
    that finished before and does not finish now: a determinism check must not pass because the
    rerun crashed. A cell without a stored finished result is reported as not comparable and is
    not a mismatch. Rows are JSON-ready (``largest_difference`` is None when not comparable).
    """
    report, mismatches = [], 0
    for dataset, cells in fresh_cells.items():
        for (table_id, pre, method), fresh in cells.items():
            old = stored.get((dataset, table_id, pre, method)) or {}
            row = {"dataset": dataset, "table": table_id, "preprocess": pre, "method": method,
                   "stored_source_sha256": old.get("source_sha256"),
                   "rerun_source_sha256": fresh.get("source_sha256")}
            if old.get("status") != "ok":
                row.update(comparable=False, numbers_compared=0, largest_difference=None,
                           where=f"no finished stored result (stored status: {old.get('status')})")
            elif fresh.get("status") != "ok":
                mismatches += 1
                row.update(comparable=False, numbers_compared=0, largest_difference=None,
                           where=f"the rerun did not finish ({fresh.get('status')}: {fresh.get('reason')})")
            else:
                count, worst, where = compare_cells(old, fresh)
                if math.isfinite(worst):
                    row.update(comparable=True, numbers_compared=count, largest_difference=worst, where=where)
                else:
                    row.update(comparable=False, numbers_compared=count, largest_difference=None,
                               where=f"metric {where} is recorded by only one of the two runs")
                mismatches += not math.isfinite(worst) or worst > 0
            report.append(row)
    return report, mismatches


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", action="append", help="dataset(s) to run (default: all)")
    p.add_argument("--table", action="append", help="table id(s) to run, e.g. official-HCXY")
    p.add_argument("--method", action="append", help="only cells whose method spec contains this text")
    p.add_argument("--preprocess", action="append", help="only cells with this preprocessing spec")
    p.add_argument("--list", action="store_true", help="print the cells and their commands, run nothing")
    p.add_argument("--verify", action="store_true", help="rerun and compare with results/, do not overwrite")
    p.add_argument("--record-verification", action="store_true",
                   help="with --verify: store the comparison in results/verification.json (shown in the docs)")
    p.add_argument("--timeout", type=float, default=1800, help="seconds per cell (default 1800)")
    p.add_argument("--max-rss-mb", type=float, default=2500, help="kill a cell above this resident memory")
    p.add_argument("--results-dir", type=Path, default=HERE / "results")
    p.add_argument("--workdir", type=Path, default=None, help="scratch folder for per-cell files (default: temp)")
    p.add_argument("--freeze", type=Path, help="copy the code to this new folder and run the copy")
    p.add_argument("--no-render", action="store_true", help="do not regenerate docs/benchmarks*.md")
    p.add_argument("--remerge", action="store_true",
                   help="run nothing: refresh titles, skips, notes and published numbers of the stored result files")
    p.add_argument("--resume", action="store_true",
                   help="skip the cells whose stored result has status 'ok' (continue an interrupted matrix)")
    p.add_argument("--rerun-legacy", action="store_true",
                   help="run only the finished cells recorded by an older harness (without cpu_time_s); each rerun "
                        "records rerun_check against the result it replaces")
    args = p.parse_args(argv)
    if args.remerge:
        for dataset in sorted({t.dataset for t in matrix.tables(args.dataset, args.table)}):
            if (args.results_dir / f"{dataset}.json").is_file():
                merge(dataset, {}, {}, None, args.results_dir, None)
                print(f"[bench] refreshed {args.results_dir / f'{dataset}.json'}")
        return 0 if args.no_render else render_docs(args.results_dir)

    if args.freeze:
        target = freeze(args.freeze.resolve())
        rest = list(argv if argv is not None else sys.argv[1:])
        for i, a in enumerate(rest):  # drop "--freeze DIR" or "--freeze=DIR": the copy must not freeze again
            if a == "--freeze":
                del rest[i:i + 2]
                break
            if a.startswith("--freeze="):
                del rest[i]
                break
        results = ["--results-dir", str(args.results_dir.resolve())]
        print(f"[bench] frozen code in {target}; running from there", flush=True)
        code = subprocess.call([sys.executable, str(target / "benchmarks" / "run.py"), *rest, *results,
                                "--no-render"])
        if not (args.no_render or args.list or args.verify):  # cells stopped at a limit are results too
            render_docs(args.results_dir.resolve())
        return code

    code_root = _code_root(HERE)
    selected = matrix.tables(args.dataset, args.table)
    if not selected:
        raise SystemExit("no table matches --dataset/--table")
    plan = [(t, pre, spec) for t in selected for pre, spec in t.cells()
            if (not args.method or any(m in spec for m in args.method))
            and (not args.preprocess or pre in args.preprocess)]
    stored = stored_cells(args.results_dir, {t.dataset for t, _, _ in plan})
    if args.resume:
        done = [c for c in plan if (stored.get((c[0].dataset, c[0].id, c[1], c[2])) or {}).get("status") == "ok"]
        plan = [c for c in plan if c not in done]
        print(f"[bench] --resume: {len(done)} cells already have a result, {len(plan)} to run", flush=True)
    if args.rerun_legacy:
        plan = [c for c in plan if (stored.get((c[0].dataset, c[0].id, c[1], c[2])) or {}).get("status") == "ok"
                and "cpu_time_s" not in stored[(c[0].dataset, c[0].id, c[1], c[2])]]
        print(f"[bench] --rerun-legacy: {len(plan)} cells recorded by an older harness", flush=True)
    workdir = Path(args.workdir or tempfile.mkdtemp(prefix="indoorloc-bench-"))
    workdir.mkdir(parents=True, exist_ok=True)
    anchors = {}
    for t, _, spec in plan:
        if "{anchors}" in spec and t.key not in anchors:
            anchors[t.key] = _anchors(t)
    if args.list:
        for t, pre, spec in plan:
            argv_ = command(t, pre, spec.replace("{anchors}", anchors.get(t.key, "{anchors}")), Path("cell.json"))
            print(f"{t.key:36s} {pre:16s} {_shown(argv_)}")
        print(f"{len(plan)} cells")
        return 0

    harness = {"runner": "benchmarks/run.py", "hardware": hardware(), "cgroup": cgroup_limits(),
               "thread_env": THREAD_ENV, "timeout_s": args.timeout, "max_rss_mb": args.max_rss_mb,
               "code": code_state(code_root)}
    by_dataset: dict[str, dict] = {}
    headers: dict[str, dict] = {}
    environment: dict[str, dict] = {}
    failures = 0
    for n, (t, pre, spec) in enumerate(plan, 1):
        resolved = spec.replace("{anchors}", anchors.get(t.key, ""))
        tag = hashlib.sha256(f"{t.key}|{pre}|{spec}".encode()).hexdigest()[:12]
        out, log = workdir / f"{tag}.json", workdir / f"{tag}.log"
        argv_ = command(t, pre, resolved, out)
        print(f"[bench {n}/{len(plan)}] {t.key} | {pre} | {matrix.display(spec)} ...", end=" ", flush=True)
        outcome = run_cell(argv_, cwd=code_root, log=log, timeout=args.timeout, max_rss_mb=args.max_rss_mb)
        payload = _load(out) if outcome["status"] == "ok" else None
        rec = _cell_record(t, pre, spec, argv_, outcome, payload)
        if payload is not None:
            pooled = payload["methods"][0]["pooled"]
            score = pooled.get("mean_error", pooled.get("accuracy"))
            print(f"{score:.4f} ({outcome['wall_time_s']:.1f} s, {outcome['peak_rss_mb']:.0f} MB)", flush=True)
            headers.setdefault(t.dataset, {}).setdefault(t.id, _table_header(t, payload))
            env = dict(payload.get("environment", {}))
            environment.setdefault(t.dataset, env)
        else:
            failures += 1
            print(f"{outcome['status'].upper()}: {outcome['reason']}", flush=True)
        rec["limits"] = {"timeout_s": args.timeout, "max_rss_mb": args.max_rss_mb}
        previous = stored.get((t.dataset, t.id, pre, spec))
        if not args.verify and payload is not None and (previous or {}).get("status") == "ok":
            rec["rerun_check"] = rerun_check(previous, rec)  # a rerun documents its own reproducibility
        by_dataset.setdefault(t.dataset, {})[(t.id, pre, spec)] = rec
        if args.verify:
            continue
        merge(t.dataset, by_dataset[t.dataset], headers.get(t.dataset, {}), environment.get(t.dataset),
              args.results_dir, harness)  # after every cell: an interrupted run keeps what finished

    if args.verify:
        report, mismatches = verification_report(by_dataset, stored)
        for r in report:
            what = f"[verify] {r['dataset']}/{r['table']} | {r['preprocess']} | {matrix.display(r['method'])}"
            if r["largest_difference"] is None:
                print(f"{what}: NOT COMPARABLE ({r['where']})")
            else:
                print(f"{what}: {r['numbers_compared']} numbers, largest difference {r['largest_difference']:.3g}"
                      + (f" ({r['where']})" if r["largest_difference"] else " (identical)"))
        if args.record_verification and report:
            path = args.results_dir / "verification.json"
            prev = (_load(path) or {}).get("cells", [])
            keys = {(r["dataset"], r["table"], r["preprocess"], r["method"]) for r in report}
            kept = [r for r in prev if (r["dataset"], r["table"], r["preprocess"], r["method"]) not in keys]
            _write(path, {"format": "indoorloc-benchmark-verification", "format_version": 1,
                          "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          "harness": harness, "cells": kept + report})
        return 1 if mismatches else 0
    if not args.no_render:
        render_docs(args.results_dir)
    return 1 if failures else 0


def render_docs(results_dir: Path) -> int:
    from benchmarks import render

    render.main(["--results-dir", str(results_dir)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
