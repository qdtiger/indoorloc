"""Published numbers per dataset, with provenance, always kept apart from numbers this library produces.

One JSON file per dataset (``<registry name>.json``, standard library only). Each entry has:

``method``, ``values``   the numbers, keyed by metric; ``metrics`` at file level defines each key and
                         its unit (accuracies in percent, errors in metres as reported).
``source``               authors, title, venue, year, DOI/URL (resolved with Crossref/DataCite).
``location``             table/section of the paper the number comes from, when it was checked.
``protocol``             ``"official"`` (the dataset's own train/test files) or a description of
                         what the paper did instead; comparisons flag any difference.
``kind``                 ``"literature"``, ``"indoorloc-0.1"`` (numbers IndoorLoc 0.1 produced
                         itself) or ``"indoorloc-0.1-demo"`` (0.1 numbers on synthetic demo data).
``check``                what was done to trust the number:

                         ==============  ==================================================
                         verified        matches the paper's full text (``location``)
                         corrected       0.1 had it wrong; ``values`` hold the paper's numbers
                         unchecked       source resolved, number not compared with the paper
                         unidentified    no publication could be identified
                         missing         a source without a number (0.1 TODO)
                         not-literature  produced by IndoorLoc 0.1, not published
                         ==============  ==================================================

``ported``               the 0.1 entry as it was (citation string, values as fractions, notes),
                         so every change made while porting is visible.

:func:`table` lists the entries (by default the literature that names a source and a number);
:func:`compare` puts reproduced results next to them in two separate sections and never ranks
one against the other: a published number was obtained under its own protocol, preprocessing
and code, and is not re-run here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .._format import cite, fmt, render

_DIR = Path(__file__).resolve().parent
CHECKS = ("verified", "corrected", "unchecked", "unidentified", "missing", "not-literature")
DEFAULT_CHECKS = ("verified", "corrected", "unchecked")  # a traceable source and a number
_ALIASES: dict[str, str] | None = None


def _alias_index() -> dict[str, str]:
    global _ALIASES
    if _ALIASES is None:
        index = {}
        for path in sorted(_DIR.glob("*.json")):
            doc = json.loads(path.read_text(encoding="utf-8"))
            for name in (doc["dataset"], *doc.get("aliases", ())):
                index[name.lower()] = doc["dataset"]
        _ALIASES = index
    return _ALIASES


def list_tables() -> list[str]:
    """Dataset names that have a literature file (registry names; 0.1 ids work as aliases)."""
    return sorted(set(_alias_index().values()))


def load(dataset: str) -> dict:
    """The raw JSON document of a dataset (a fresh copy)."""
    name = _alias_index().get(dataset.lower())
    if name is None:
        raise KeyError(f"no literature table for {dataset!r}; available: {', '.join(list_tables())}")
    return json.loads((_DIR / f"{name}.json").read_text(encoding="utf-8"))


def _select(entries, include, kinds) -> list[dict]:
    """``include`` filters the check status of literature entries; ``kinds`` filters entry kinds."""
    if include != "all":
        include = (include,) if isinstance(include, str) else tuple(include)
        unknown = set(include) - set(CHECKS)
        if unknown:
            raise ValueError(f"unknown check status {sorted(unknown)}; use {CHECKS} or 'all'")
    if kinds != "all":
        kinds = (kinds,) if isinstance(kinds, str) else tuple(kinds)
    return [e for e in entries if (kinds == "all" or e["kind"] in kinds)
            and (include == "all" or e["kind"] != "literature" or e["check"] in include)]


def _primary(metrics: Mapping) -> str | None:
    for key in ("mean_error", "mean_error_3d", "accuracy", "mean_penalized_error"):
        if key in metrics:
            return key
    return next(iter(metrics), None)


def _sort_key(metric: str | None):
    def key(entry):
        v = entry["values"].get(metric) if metric else None
        missing = v is None
        return (missing, -v if (not missing and metric.endswith("accuracy")) else (v or 0.0), entry["method"])
    return key


@dataclass(frozen=True)
class LiteratureTable:
    """The selected entries of one dataset's literature file (see the module docstring)."""

    dataset: str
    display_name: str
    metrics: dict
    protocols: dict
    citation: dict
    entries: tuple
    excluded: int = 0  # entries left out by ``include``
    notes: str = ""

    def __len__(self) -> int:
        return len(self.entries)

    def columns(self) -> list[str]:
        """Metrics that at least one selected entry reports, in file order."""
        return [m for m in self.metrics if any(e["values"].get(m) is not None for e in self.entries)]

    def to_rows(self) -> list[dict]:
        return [{"method": e["method"], **e["values"], "protocol": e["protocol"], "check": e["check"],
                 "source": cite(e["source"]), "doi": (e["source"] or {}).get("doi")} for e in self.entries]

    def render(self, style: str = "markdown", digits: int = 2) -> str:
        cols = self.columns()
        header = ["Method", *[_header(m, self.metrics) for m in cols], "Protocol", "Check", "Source"]
        rows = [[e["method"], *[fmt(e["values"].get(m), digits) for m in cols], _short(e["protocol"]), e["check"],
                 cite(e["source"])] for e in self.entries]
        title = f"Literature on {self.display_name} (as reported; not re-run by indoorloc)"
        note = (f"\n{self.excluded} entr{'y' if self.excluded == 1 else 'ies'} hidden: no traceable source or number, "
                "or numbers IndoorLoc 0.1 produced itself (include='all', kinds='all' show them)."
                if self.excluded else "")
        return f"{title}\n\n{render(header, rows, style)}{note}"

    def to_markdown(self) -> str:
        return self.render("markdown")

    def to_text(self) -> str:
        return self.render("text")

    __str__ = to_text


def _header(metric: str, metrics: Mapping) -> str:
    unit = (metrics.get(metric) or {}).get("unit")
    return f"{metric} [{unit}]" if unit else metric


def _short(protocol: str, width: int = 48) -> str:
    return protocol if len(protocol) <= width else protocol[:width - 3] + "..."


def table(dataset: str, *, include=DEFAULT_CHECKS, kinds=("literature",)) -> LiteratureTable:
    """Published numbers for ``dataset``, best first by the primary metric.

    include  check statuses of literature entries to keep (default: verified, corrected,
             unchecked, i.e. a traceable source and a number), or ``"all"``.
    kinds    entry kinds to keep (default: literature only); add ``"indoorloc-0.1"`` for the
             numbers IndoorLoc 0.1 produced itself, or pass ``"all"``.
    """
    doc = load(dataset)
    entries = doc["entries"]
    chosen = _select(entries, include, kinds)
    chosen.sort(key=_sort_key(_primary(doc["metrics"])))
    return LiteratureTable(doc["dataset"], doc["display_name"], doc["metrics"], doc.get("protocols", {}),
                           doc.get("dataset_citation") or {}, tuple(chosen), len(entries) - len(chosen),
                           doc.get("notes", ""))


# --------------------------------------------------------------------------- comparison
def _as_metrics(result) -> dict:
    if hasattr(result, "to_dict"):
        result = result.to_dict()
    if not isinstance(result, Mapping):
        raise TypeError(f"expected EvaluationResults or a mapping of metrics, got {type(result).__name__}")
    return {k: v for k, v in result.items() if isinstance(v, (int, float)) and not isinstance(v, bool) or v is None}


def _named_results(results, label: str) -> dict[str, dict]:
    """One result, a {name: result} mapping, or a metrics mapping -> {name: metrics}."""
    if hasattr(results, "to_dict"):
        return {label: _as_metrics(results)}
    if isinstance(results, Mapping):
        if results and all(hasattr(v, "to_dict") or isinstance(v, Mapping) for v in results.values()):
            return {str(k): _as_metrics(v) for k, v in results.items()}
        return {label: _as_metrics(results)}
    raise TypeError(f"cannot compare {type(results).__name__}; pass EvaluationResults or a mapping")


@dataclass(frozen=True)
class Comparison:
    """Reproduced results and published numbers for one dataset, in two separate sections.

    ``reproduced``  rows ``{"method", **metrics}`` measured by this library (this run).
    ``literature``  the :class:`LiteratureTable` entries, each with ``same_protocol``.
    No ranking or "state of the art" verdict is computed: see the module docstring.
    """

    dataset: str
    protocol: str
    metrics: tuple
    reproduced: tuple
    literature: LiteratureTable
    notes: tuple = field(default_factory=tuple)

    def to_dict(self) -> dict:
        lit = [{"method": e["method"], "values": e["values"], "protocol": e["protocol"],
                "same_protocol": _same(e["protocol"], self.protocol), "check": e["check"], "kind": e["kind"],
                "location": e["location"], "source": e["source"]} for e in self.literature.entries]
        return {"dataset": self.dataset, "protocol": self.protocol, "metrics": list(self.metrics),
                "reproduced": [dict(r) for r in self.reproduced], "literature": lit, "notes": list(self.notes)}

    def render(self, style: str = "markdown", digits: int = 2) -> str:
        head = ["Method", *[_header(m, self.literature.metrics) for m in self.metrics]]
        rep = [[r["method"], *[fmt(r.get(m), digits) for m in self.metrics]] for r in self.reproduced]
        lit_rows = [[e["method"], *[fmt(e["values"].get(m), digits) for m in self.metrics],
                     _yes_no(_same(e["protocol"], self.protocol)), e["check"], cite(e["source"])]
                    for e in self.literature.entries]
        heading = (lambda s: f"### {s}") if style == "markdown" else (lambda s: f"{s}\n{'=' * len(s)}")
        out = [heading(f"Reproduced with indoorloc (protocol: {self.protocol})"), "", render(head, rep, style), "",
               heading(f"Literature on {self.literature.display_name} (as reported; not re-run)"), "",
               render([*head, f"Same protocol ({self.protocol})", "Check", "Source"], lit_rows, style)
               if lit_rows else "(no entries)"]
        if self.literature.excluded:
            out += ["", f"{self.literature.excluded} entr{'y' if self.literature.excluded == 1 else 'ies'} not shown: "
                        "no traceable source or number, or numbers IndoorLoc 0.1 produced itself."]
        out += [""] + [f"Note: {n}" for n in self.notes]
        return "\n".join(out).rstrip() + "\n"

    def to_markdown(self) -> str:
        return self.render("markdown")

    def to_text(self) -> str:
        return self.render("text")

    __str__ = to_text


def _same(entry_protocol: str, protocol: str) -> bool | None:
    """True/False, or None when the paper's protocol is not known ("unspecified").

    An entry's ``protocol`` starts with a registered protocol name (``"official"``,
    ``"official (inferred: ...)"``) or describes what the paper did instead.
    """
    first = entry_protocol.split()[0].rstrip(",;:") if entry_protocol.strip() else "unspecified"
    return None if first == "unspecified" else first == protocol


def _yes_no(same: bool | None) -> str:
    return "unknown" if same is None else "yes" if same else "no"


def compare(results, dataset: str, *, protocol: str = "official", metrics=None, include=DEFAULT_CHECKS,
            kinds=("literature",), label: str = "this library") -> Comparison:
    """Reproduced ``results`` next to the published numbers for ``dataset``.

    results   an EvaluationResults, a metrics mapping (``EvaluationResults.to_dict()`` or a
              benchmark JSON's ``pooled`` block), or ``{method name: either}``.
    protocol  the protocol the results were obtained with; literature rows are marked
              ``same protocol: yes/no`` against it.
    metrics   metric names to show; default: the literature metrics the results also report,
              plus, for any shown entry that reports none of them, its first metric (so a
              paper that gives only a 3-D error still shows its number, in its own column).
    include   literature check statuses to show; ``kinds`` the entry kinds (see :func:`table`).
    """
    lit = table(dataset, include=include, kinds=kinds)
    named = _named_results(results, label)
    if metrics is None:
        reported = {k for r in named.values() for k, v in r.items() if v is not None}
        chosen = {m for m in lit.metrics if m in reported}
        for e in lit.entries:
            given = [m for m in lit.metrics if e["values"].get(m) is not None]
            if given and not chosen.intersection(given):
                chosen.add(given[0])
        metrics = [m for m in lit.metrics if m in chosen]
    metrics = tuple(metrics)
    notes = []
    if any(_same(e["protocol"], protocol) is False for e in lit.entries):
        notes.append(f"literature rows marked 'no' were not obtained with the {protocol!r} protocol; their numbers "
                     "are not directly comparable.")
    if any(_same(e["protocol"], protocol) is None for e in lit.entries):
        notes.append("'unknown': the entry does not record the paper's protocol (not checked against the paper).")
    if dataset.lower() in ("ujiindoorloc", "uji", "ujindoorloc"):
        notes.append("UJIIndoorLoc errors are in EPSG:3857 metres (as in the literature); multiply by "
                     "meta['ground_scale'] (0.766) for ground metres.")
    reproduced = tuple({"method": name, **{m: r.get(m) for m in metrics}} for name, r in named.items())
    return Comparison(lit.dataset, protocol, metrics, reproduced, lit, tuple(notes))


__all__ = ["CHECKS", "Comparison", "DEFAULT_CHECKS", "LiteratureTable", "compare", "list_tables", "load", "table"]
