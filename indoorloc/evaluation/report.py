"""L4 reports: Markdown / plain-text tables of results, with their provenance.

``results_table`` formats any set of results; ``benchmark_report`` renders the JSON written by
``indoorloc benchmark`` (per-fold and pooled numbers, the environment, dataset digests, the
exact split digests) and, for the official protocol, the published numbers in a separate
section via :func:`indoorloc.evaluation.literature.compare`. Standard library only.
"""
from __future__ import annotations

from typing import Mapping

from ._format import cite, fmt, render

DEFAULT_METRICS = ("mean_error", "median_error", "p75_error", "p90_error", "rmse", "floor_accuracy",
                   "building_accuracy", "ipin_score", "mean_penalized_error")
_UNITS = {"floor_accuracy": "%", "building_accuracy": "%"}
_COUNTS = ("n", "n_failed")  # sample counts: no unit in the header


def _header(metric: str, unit: str) -> str:
    return metric if metric in _COUNTS else f"{metric} [{_UNITS.get(metric, unit)}]"


def _failed(row: Mapping) -> int:
    try:
        return int(row.get("n_failed") or 0)
    except (TypeError, ValueError):
        return 0


def _with_failed(cols: list, rows) -> list:
    """``cols`` plus ``n_failed`` when some row has samples its method could not place (CONTRACTS §6)."""
    return cols + ["n_failed"] if "n_failed" not in cols and any(_failed(r) > 0 for r in rows) else cols


def _failed_notes(results: Mapping) -> list[str]:
    """One line per entry with unplaced samples: its ``failed_note`` (benchmark JSON) or a count."""
    notes = []
    for name, r in results.items():
        if _failed(r) > 0:
            note = r.get("failed_note") or (f"{_failed(r)} of {r.get('n', '?')} samples were not placed (NaN "
                                            "estimate) and are left out of the error statistics")
            notes.append(f"Not placed ({name}): {note}")
    return notes


def _metrics_of(result) -> dict:
    if hasattr(result, "to_dict"):
        return result.to_dict()
    if isinstance(result, Mapping):
        return dict(result)
    raise TypeError(f"expected EvaluationResults or a mapping of metrics, got {type(result).__name__}")


def results_table(results: Mapping, *, metrics=DEFAULT_METRICS, style: str = "markdown", digits: int = 3,
                  unit: str = "m") -> str:
    """One row per entry of ``{name: EvaluationResults or metrics dict}``.

    Metrics no entry reports are dropped; a missing value prints as ``-``. ``unit`` labels the
    error columns (keep it short, e.g. ``"m"``; state the frame in a caption); accuracies are
    percentages. When some entry has samples its method could not place, an ``n_failed``
    column is added and a note per such entry (its ``failed_note``, if any) follows the table.
    """
    rows = {str(k): _metrics_of(v) for k, v in results.items()}
    cols = _with_failed([m for m in metrics if any(r.get(m) is not None for r in rows.values())], rows.values())
    header = ["Method", *[_header(m, unit) for m in cols]]
    table = render(header, [[name, *[fmt(r.get(m), digits) for m in cols]] for name, r in rows.items()], style)
    notes = _failed_notes(rows)
    return table + ("\n\n" + "\n".join(notes) if notes else "")  # a blank line ends a Markdown table


def _heading(text: str, style: str, level: int = 2) -> str:
    return f"{'#' * level} {text}" if style == "markdown" else f"{text}\n{('=' if level <= 2 else '-') * len(text)}"


def _kv(pairs, style: str) -> str:
    pairs = [(k, v) for k, v in pairs if v not in (None, "", {})]
    if style == "markdown":
        return "\n".join(f"- **{k}**: {v}" for k, v in pairs)
    width = max((len(k) for k, _ in pairs), default=0)
    return "\n".join(f"{k.ljust(width)}  {v}" for k, v in pairs)


def provenance(payload: Mapping, style: str = "markdown") -> str:
    """The environment, data and code facts of a benchmark JSON as a key/value list."""
    env = payload.get("environment", {})
    data = payload.get("dataset", {})
    git = env.get("git") or {}
    sha = data.get("sha256")
    if isinstance(sha, Mapping):
        sha = ", ".join(f"{k}: {_digest(v)}" for k, v in sha.items())
    pairs = [
        ("indoorloc", env.get("indoorloc")), ("python", env.get("python")), ("numpy", env.get("numpy")),
        ("platform", env.get("platform")),
        ("other packages", ", ".join(f"{k} {v}" for k, v in (env.get("packages") or {}).items())),
        ("git commit", (git.get("commit", "")[:12] + (" (uncommitted changes)" if git.get("dirty") else ""))
         if git else None),
        ("dataset", f"{data.get('name')} ({', '.join(f'{k}: {v}' for k, v in (data.get('n_samples') or {}).items())})"),
        ("rows pooled", data.get("n_pooled")),
        ("duplicate rows dropped", ", ".join(f"{k}: {v}" for k, v in (data.get("duplicate_rows_dropped") or {}).items())
         or None),
        ("unlabelled rows left out",
         ", ".join(f"{k}: {v}" for k, v in (data.get("unlabelled_rows_left_out") or {}).items()) or None),
        ("dataset sha256", sha), ("citation", data.get("citation")),
        ("protocol", payload.get("protocol", {}).get("name")), ("seed", payload.get("seed")),
        ("units", payload.get("units")), ("created", payload.get("created")),
        ("wall time", f"{payload['wall_time_s']:.2f} s" if "wall_time_s" in payload else None),
        ("command", " ".join(payload.get("command") or []) or None),
    ]
    return _kv(pairs, style)


def _digest(value) -> str:
    if isinstance(value, Mapping):
        return ", ".join(f"{k}={_digest(v)}" for k, v in value.items())
    return str(value)[:12] + "..." if isinstance(value, str) and len(value) > 16 else str(value)


def benchmark_report(payload: Mapping, *, style: str = "markdown", literature: bool = True, digits: int = 3) -> str:
    """Render a benchmark JSON (``indoorloc benchmark --out``): results, folds, provenance, literature."""
    units = payload.get("units") or "m"
    unit = units.split(" ")[0]  # "m (Web Mercator, ...)" -> "m" in headers; the caption keeps the full text
    methods = payload.get("methods", [])
    caption = (f"Errors in {units}; accuracies in %. ipin_score: 75th percentile of the error + 15 m per floor + "
               "50 m per wrong building; mean_penalized_error: mean with 4 m per floor + 50 m per building "
               "(EvAAL-ETRI 2015 rules).")
    out = [_heading(f"Benchmark: {payload.get('dataset', {}).get('name')} / protocol "
                    f"{payload.get('protocol', {}).get('name')}", style, 1 if style == "markdown" else 2), ""]
    out += [_heading("Results (pooled over all test folds)", style), "",
            results_table({m["label"]: m["pooled"] for m in methods}, style=style, digits=digits, unit=unit), "",
            caption, ""]
    ci = [f"{m['label']}: mean {m['pooled']['mean_error']:.{digits}f} {unit}, 95 % bootstrap CI "
          f"[{m['pooled']['mean_error_ci95'][0]:.{digits}f}, {m['pooled']['mean_error_ci95'][1]:.{digits}f}]"
          for m in methods if m.get("pooled", {}).get("mean_error_ci95")]
    if ci:
        out += ["\n".join(ci), ""]
    folds = payload.get("protocol", {}).get("folds", [])
    if len(folds) > 1:
        cols = [k for k in DEFAULT_METRICS if any(f["metrics"].get(k) is not None for m in methods for f in m["folds"])]
        cols = _with_failed(cols, [f["metrics"] for m in methods for f in m["folds"]])
        rows = [[m["label"], f["fold"], *[fmt(f["metrics"].get(k), digits) for k in cols]]
                for m in methods for f in m["folds"]]
        out += [_heading("Per fold", style), "",
                render(["Method", "Fold", *[_header(k, unit) for k in cols]], rows, style), ""]
    split_rows = [[f["name"], str(f["n_train"]), str(f["n_test"]), f["test_sha256"][:12]] for f in folds]
    if split_rows:
        out += [_heading("Splits", style), "",
                render(["Fold", "Train", "Test", "Test rows sha256"], split_rows, style), ""]
    timing = [[m["label"], *[fmt(sum(f[k] for f in m["folds"]), 3) for k in ("fit_s", "predict_s")]] for m in methods]
    if timing:
        out += [render(["Method", "fit [s]", "predict [s]"], timing, style), ""]
    out += [_heading("Provenance", style), "", provenance(payload, style), ""]
    lit = payload.get("literature")
    if literature and lit:
        out += [_heading("Published numbers (as reported; not re-run, kept apart from the results above)", style), "",
                literature_table(lit, style=style, digits=2), ""]
        out += [f"Note: {n}" for n in lit.get("notes", [])]
    elif literature and payload.get("literature_note"):
        out += [f"Literature: {payload['literature_note']}"]
    return "\n".join(out).rstrip() + "\n"


def literature_table(comparison: Mapping, *, style: str = "markdown", digits: int = 2) -> str:
    """The literature half of ``Comparison.to_dict()`` (as stored in a benchmark JSON) as a table."""
    metrics = list(comparison.get("metrics", []))
    protocol = comparison.get("protocol")
    header = ["Method", *metrics, f"Same protocol ({protocol})", "Check", "Source"]
    same = {True: "yes", False: "no"}
    rows = [[e["method"], *[fmt(e["values"].get(m), digits) for m in metrics],
             same.get(e.get("same_protocol"), "unknown"), e.get("check", ""), cite(e.get("source"))]
            for e in comparison.get("literature", [])]
    return render(header, rows, style) if rows else "(no published numbers with a traceable source)"


__all__ = ["DEFAULT_METRICS", "benchmark_report", "literature_table", "provenance", "results_table"]
