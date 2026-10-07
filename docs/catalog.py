#!/usr/bin/env python3
"""Catalog tables of the user guide, generated from the registries and docstrings.

    python docs/catalog.py              rewrite the generated blocks of docs/guide/*.md, docs/zh/*.md and
                                        docs/installation*.md
    python docs/catalog.py --check      exit 1 if a generated block is out of date (for CI)
    python docs/catalog.py --snippets   run every ```python block of those pages, page by page
    python docs/catalog.py --snippets --no-data   the same as on CI: every ``# data:`` block skipped
    python -m doctest docs/catalog.py   the parsers (references, expected output) on known inputs

A generated block sits between ``<!-- catalog:NAME -->`` and ``<!-- /catalog:NAME -->``; the text
outside the markers is written by hand and never touched. Every source is local; nothing is
downloaded:

* datasets    ``indoorloc.datasets.DATASETS``: class attributes, ``meta`` and the file manifest.
              Row counts come from the recorded benchmark runs (``benchmarks/results/*.json``,
              measured on the sha256-verified files); the simulator's from generating its default
              tables (deterministic, well under a second); otherwise from the file manifest.
* methods     ``indoorloc.methods.METHODS``: class docstrings, the families listed in the package
              docstring, the pip extra named by the ``requires(module, extra)`` calls of the module.
* signals, evaluation, apps   each package's ``__all__`` and docstrings, ``PROTOCOLS`` and the
              literature files.
* cli         the argparse parser of ``indoorloc.cli``.

``--snippets`` executes the Python blocks of each page in order in one namespace per page, inside
a temporary directory. A block that starts with a comment line ``# data: name[, name]`` runs only
when those datasets are already on disk (``$INDOORLOC_DATA``), so the check never downloads; a
starting line ``# requires: module[, module]`` skips the block when an optional package is missing,
and a first line ``# 0.1 ...`` marks 0.1 code shown for comparison (never run). Comment lines directly after a
``print(...)`` call are the output the page shows: the block fails when it prints anything else.
Each page runs in its own interpreter, so registrations made by one page never reach another.
A block without a ``# data:`` line must not depend on one that has it (CI has no datasets);
``--no-data`` checks exactly that.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import inspect
import io
import json
import os
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")  # snippets that plot must not open windows

PAGES = {"en": ROOT / "docs" / "guide", "zh": ROOT / "docs" / "zh"}
EXTRA_PAGES = {"en": [ROOT / "docs" / "installation.md", ROOT / "MIGRATION.md"],  # checked like the guide pages
               "zh": [ROOT / "docs" / "installation_zh.md"]}
SRC = "../../"  # from docs/guide/*.md and docs/zh/*.md to the repository root

T = {  # table headers and fixed words, per language
    "en": {
        "name": "Name", "modality": "Modality", "rows": "Rows", "xunits": "X units", "pos": "Positions",
        "license": "License", "source": "Source", "what": "What it does", "extra": "Extra", "nan": "NaN input",
        "ref": "Reference", "class": "Class", "module": "Module", "fit": "Learns in `fit`", "function": "Function",
        "protocol": "Protocol", "needs": "Needs", "summary": "Summary", "dataset": "Dataset",
        "entries": "Entries", "option": "Option", "help": "Help", "kind": "Kind",
        "splits": "Splits (aliases)", "options": "Options (`load_dataset(name, **options)`)",
        "install": "Install", "packages": "Packages", "purpose": "For (comment in pyproject.toml)",
        "yes": "yes", "no": "no", "as_members": "as its members", "depends": "depends on parameters",
        "fill_first": "no (fill first)", "more": "+{n} more in the docstring", "none": "—",
        "default_mod": "bold = the default `modality`", "numpy": "numpy only",
        "recorded": "rows as recorded by the benchmark runs", "manifest": "{traces:,} traces on {floors} floors",
        "manifest_files": "{n:,} files", "generated": "default (seed 0): {counts}", "per_scenario": "per scenario",
        "cls": "class", "fn": "function",
        "groups": {"wifi": "WiFi RSSI", "ble": "BLE RSSI", "csi": "WiFi CSI", "multi": "Multi-sensor traces",
                   "sim": "Simulated", "other": "Other"},
        "families": {"fingerprinting": "Fingerprinting", "model-based": "Model-based", "sequence": "Sequence matching",
                     "deep": "Deep learning", "other": "Other"},
    },
    "zh": {
        "name": "名称", "modality": "模态", "rows": "行数", "xunits": "X 单位", "pos": "坐标",
        "license": "许可", "source": "出处", "what": "作用（取自 docstring）", "extra": "extra", "nan": "接受 NaN",
        "ref": "参考文献", "class": "类", "module": "模块", "fit": "`fit` 学习统计量", "function": "函数",
        "protocol": "协议", "needs": "需要的列", "summary": "说明", "dataset": "数据集",
        "entries": "条目", "option": "选项", "help": "说明", "kind": "类型",
        "splits": "划分（别名）", "options": "选项（`load_dataset(name, **options)`）",
        "install": "安装", "packages": "依赖包", "purpose": "用途（pyproject.toml 中的注释）",
        "yes": "是", "no": "否", "as_members": "取决于成员模型", "depends": "取决于参数",
        "fill_first": "否（先填补）", "more": "docstring 中另有 {n} 篇", "none": "—",
        "default_mod": "粗体为默认 `modality`", "numpy": "仅 numpy",
        "recorded": "行数取自基准运行记录", "manifest": "{traces:,} 条轨迹，{floors} 个楼层",
        "manifest_files": "{n:,} 个文件", "generated": "默认（seed 0）：{counts}", "per_scenario": "随场景而定",
        "cls": "类", "fn": "函数",
        "groups": {"wifi": "WiFi RSSI", "ble": "BLE RSSI", "csi": "WiFi CSI", "multi": "多传感器轨迹",
                   "sim": "仿真", "other": "其他"},
        "families": {"fingerprinting": "指纹", "model-based": "基于模型", "sequence": "序列匹配",
                     "deep": "深度学习", "other": "其他"},
    },
}


# --------------------------------------------------------------------------- text helpers
def rst(text: str) -> str:
    """Docstring markup to Markdown: ``x`` -> `x`, :func:`x` -> `x`, whitespace collapsed."""
    text = re.sub(r":\w+:`~?([^`]+)`", r"`\1`", text)
    text = text.replace("``", "`")
    return re.sub(r"\s+", " ", text).strip()


def cell(text) -> str:
    """A Markdown table cell: pipes escaped, never empty."""
    text = "" if text is None else str(text)
    return text.replace("|", "\\|").replace("\n", " ").strip() or "—"


def first_line(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    return rst(doc.split("\n\n")[0].split("\n")[0]) if doc else ""


def table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
    out += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


_ID = re.compile(r"(DOI|doi\.org|https?://|ISBN|arXiv|URL)", re.IGNORECASE)
# The start of an author list: "T. S. Rappaport, ...", "Rappaport, T. S., ...", "Y. T. Chan and ...".
_AUTHOR = re.compile(r"^(?:[A-Z]\.(?:[ -]?[A-Z]\.)*\s+[A-Z][\w'’-]+|[A-Z][\w'’-]+,\s+[A-Z]\.)")


def references(obj) -> list[str]:
    """The entries of the ``References`` section of a docstring.

    Two layouts occur: numpydoc (``References`` underlined with dashes, entries flush left, the
    section ending at the next underlined header) and an indented block under a bare
    ``References`` line (the section ending at the next line that is not indented). Entries are
    separated by blank lines, by indentation (continuation lines indented), or, in a flush block,
    by a line that starts with an author name after an entry that has its DOI/URL or that ended
    with a full stop (so a book reference without a DOI does not swallow the next entry).

    >>> class Flush:
    ...     '''Summary.
    ...
    ...     References
    ...         T. S. Rappaport, "Wireless Communications", 2nd ed., Prentice Hall,
    ...         2002, ch. 4.
    ...         S. Y. Seidel, T. S. Rappaport, "914 MHz path loss prediction models",
    ...         IEEE Trans. Antennas Propag. 40(2), 1992. https://doi.org/10.1109/8.127405
    ...     '''
    >>> [r[:14] for r in references(Flush)]
    ['T. S. Rappapor', 'S. Y. Seidel, ']
    >>> class Numpydoc:
    ...     '''Summary.
    ...
    ...     References
    ...     ----------
    ...     F. Evennou, F. Marx, "Advanced integration of WiFi and inertial navigation systems for
    ...         indoor mobile positioning", EURASIP JASP, 2006. DOI 10.1155/ASP/2006/86706.
    ...     O. Woodman, R. Harle, "Pedestrian localisation for indoor environments", UbiComp 2008.
    ...     '''
    >>> [r.split(",")[0] for r in references(Numpydoc)]
    ['F. Evennou', 'O. Woodman']
    """
    doc = inspect.getdoc(obj) or ""
    match = re.search(r"^References\n-{3,}\n(.*?)(?=^\S[^\n]*\n-{3,}\n|\Z)", doc, re.DOTALL | re.MULTILINE)
    if match:
        body = match.group(1)
    else:
        match = re.search(r"^References:?\n((?:[ \t]+\S[^\n]*\n?|[ \t]*\n)+)", doc, re.MULTILINE)
        if not match:
            return []
        body = "\n".join(line[4:] if line.startswith("    ") else line.strip()
                         for line in match.group(1).splitlines())
    entries, current = [], []
    for line in body.splitlines():
        if not line.strip():
            if current:
                entries.append(current)
                current = []
            continue
        starts_new = current and not line.startswith(" ") and (
            _ID.search(" ".join(current)) or (current[-1].endswith(".") and _AUTHOR.match(line)))
        if starts_new:
            entries.append(current)
            current = []
        current.append(line.strip())
    if current:
        entries.append(current)
    return [rst(" ".join(e)) for e in entries]


def linkify(ref: str) -> str:
    """Turn a trailing DOI or URL of a reference into a Markdown link.

    >>> linkify("Venue, 1992. DOI 10.1109/8.127405.")
    'Venue, 1992. [doi:10.1109/8.127405](https://doi.org/10.1109/8.127405).'
    >>> linkify("Venue, 2015. https://doi.org/10.1016/j.eswa.2015.08.013")
    'Venue, 2015. <https://doi.org/10.1016/j.eswa.2015.08.013>'
    """
    ref = re.sub(r"(?:DOI:?|doi:)\s*(10\.\d{4,9}/[^\s,;]+?)(?=[.,;]?(\s|$))",
                 lambda m: f"[doi:{m.group(1)}](https://doi.org/{m.group(1)})", ref)
    ref = re.sub(r"(?<![(<\[])(https?://[^\s,;)]+?)(?=[.,;]?(\s|$))", r"<\1>", ref)
    return ref


def ref_cell(obj, lang: str, fallback: dict | None = None, key: str | None = None) -> str:
    refs = references(obj)
    if not refs and fallback and key in fallback:
        refs = [fallback[key]]
    if not refs:
        return T[lang]["none"]
    more = f" ({T[lang]['more'].format(n=len(refs) - 1)})" if len(refs) > 1 else ""
    return linkify(refs[0]) + more


def source_link(obj, label: str) -> str:
    path = Path(inspect.getsourcefile(obj)).resolve().relative_to(ROOT)
    return f"[{label}]({SRC}{path.as_posix()})"


def extras_of(module) -> list[str]:
    """pip extras named by ``requires(module, extra)`` calls in a module (and its package, for packages)."""
    files = [Path(inspect.getsourcefile(module))]
    if files[0].name == "__init__.py":
        files = sorted(files[0].parent.glob("*.py"))
    found = set()
    for f in files:
        found |= set(re.findall(r"requires\(\s*\"[^\"]+\",\s*\"(\w+)\"\s*\)", f.read_text(encoding="utf-8")))
    return sorted(found)


def class_flag(cls, attr: str, lang: str, *, members: bool = False) -> str:
    """A boolean class attribute; properties (meta-estimators, parameter-dependent) are described."""
    raw = inspect.getattr_static(cls, attr, None)
    if isinstance(raw, property):
        return T[lang]["as_members"] if members else T[lang]["depends"]
    return T[lang]["yes"] if raw else T[lang]["no"]


def aliases(registry) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    real = set(registry.names())
    for name in registry.names(aliases=True):
        if name not in real:
            target = next(n for n in sorted(real) if registry.get(n) is registry.get(name))
            out.setdefault(target, []).append(name)
    return out


def linked_name(obj, name: str, alias_map: dict, lang: str) -> str:
    """```name``` linked to the source file, then its registry aliases."""
    alias = alias_map.get(name)
    word = "alias" if lang == "en" else "别名"
    return source_link(obj, f"`{name}`") + (f"<br>{word}: " + ", ".join(f"`{a}`" for a in alias) if alias else "")


# --------------------------------------------------------------------------- datasets
SPLIT_ORDER = ("train", "valid", "validation", "test", "all", "unlabeled", "trajectory")


def _counts(n_samples: dict) -> str:
    order = sorted(n_samples, key=lambda s: (SPLIT_ORDER.index(s) if s in SPLIT_ORDER else 99, s))
    return " · ".join(f"{s} {n_samples[s]:,}" for s in order)


def recorded_sizes() -> dict:
    """{dataset: {options json: (options, n_samples)}} from benchmarks/results/*.json."""
    out: dict = {}
    for path in sorted((ROOT / "benchmarks" / "results").glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for tab in doc.get("tables", []):
            ds = tab.get("dataset", {})
            opts = ds.get("options") or {}
            out.setdefault(ds.get("name"), {})[json.dumps(opts, sort_keys=True)] = (opts, ds.get("n_samples", {}))
    return out


def dataset_kind(cls) -> str:
    meta = cls.meta
    if meta.get("source") == "simulated":
        return "sim"
    if "imu" in (meta.get("modalities") or ()):
        return "multi"
    return {"wifi_rssi": "wifi", "ble_rssi": "ble", "csi": "csi", "csi_amp": "csi"}.get(meta.get("modality"), "other")


def _class_entries(cls) -> list[str]:
    """Relative paths of every file a dataset class declares (class level: all floors, rooms, ...)."""
    paths = []
    for entry in cls.files.values():
        pairs = (entry,) if len(entry) == 2 and isinstance(entry[0], str) and not isinstance(entry[1], tuple) else entry
        paths += [rel for rel, _ in pairs]
    return paths


def dataset_rows(name: str, cls, recorded: dict, lang: str) -> tuple[str, dict]:
    """The row-count cell and, for a generated dataset, the meta of its default tables."""
    t = T[lang]
    if name in recorded:
        entries = recorded[name]
        if "{}" in entries:
            return _counts(entries["{}"][1]), {}
        parts = []
        for opts, counts in entries.values():
            label = ", ".join(f"{k}={v}" for k, v in opts.items())
            parts.append(f"{label}: {_counts(counts)}")
        return "<br>".join(parts), {}
    if cls.meta.get("source") == "simulated":
        try:  # generated locally and deterministically; a scenario-based simulator needs its files
            source = cls(None, download=False)
            tables = {s: source.load(s) for s in source.splits}
            return t["generated"].format(counts=_counts({s: len(v) for s, v in tables.items()})), \
                dict(next(iter(tables.values())).meta)
        except (ImportError, FileNotFoundError, RuntimeError, OSError, ValueError):
            return t["per_scenario"], {}
    paths = _class_entries(cls)
    traces = [p for p in paths if "/path_data_files/" in p]
    if traces:
        floors = {p.split("/path_data_files/")[0] for p in traces}
        return t["manifest"].format(traces=len(traces), floors=len(floors)), {}
    return t["manifest_files"].format(n=len(paths)), {}


def dataset_source(meta: dict) -> str:
    text = meta.get("citation") or ""
    doi = str(meta.get("doi") or "")
    if doi.startswith("10."):
        text += f". [doi:{doi}](https://doi.org/{doi})"
    elif doi.lower().startswith("arxiv:"):
        text += f". [{doi}](https://arxiv.org/abs/{doi.split(':', 1)[1]})"
    elif meta.get("url"):
        text += f". <{meta['url']}>"
    return text


def gen_datasets(lang: str) -> str:
    from indoorloc.datasets import DATASETS

    t = T[lang]
    recorded = recorded_sizes()
    alias_map = aliases(DATASETS)
    groups: dict[str, list] = {}
    for name in DATASETS.names():
        cls = DATASETS.get(name)
        groups.setdefault(dataset_kind(cls), []).append((name, cls))
    out = []
    for kind in ("wifi", "ble", "csi", "multi", "sim", "other"):
        if kind not in groups:
            continue
        rows = []
        for name, cls in groups[kind]:
            meta = dict(cls.meta)
            mods = meta.get("modalities")
            modality = (", ".join(f"**`{m}`**" if m == meta.get("modality") else f"`{m}`" for m in mods)
                        if mods else f"`{meta.get('modality', '?')}`")
            pos = meta.get("pos_units")
            crs = meta.get("crs")
            if pos is None:
                pos_text = t["none"] + (f" ({meta['task']})" if meta.get("task") else "")
            else:
                pos_text = str(pos) + (f"; crs `{crs}`" if crs and crs != "local" else "")
            count, generated = dataset_rows(name, cls, recorded, lang)
            rows.append([linked_name(cls, name, alias_map, lang), modality, count,
                         meta.get("units") or generated.get("units") or t["none"],
                         pos_text, meta.get("license"), dataset_source(meta)])
        out.append(f"**{t['groups'][kind]}**\n\n" + table(
            [t["name"], t["modality"], t["rows"], t["xunits"], t["pos"], t["license"], t["source"]], rows))
    note = (f"{t['default_mod']}. {t['recorded'].capitalize() if lang == 'en' else t['recorded']} "
            f"(`benchmarks/results/*.json`).")
    return "\n\n".join(out) + "\n\n" + note


def _param(p: inspect.Parameter) -> str:
    if p.default is inspect.Parameter.empty:
        return f"`{p.name}`"
    default = p.default
    if hasattr(default, "shape") and getattr(default, "ndim", 0):
        return f"`{p.name}=<{len(default)} values>`"
    return f"`{p.name}={default!r}`"


def gen_dataset_options(lang: str) -> str:
    from indoorloc.datasets import DATASETS

    t = T[lang]
    rows = []
    for name in DATASETS.names():
        cls = DATASETS.get(name)
        params = [p for p in inspect.signature(cls.__init__).parameters.values()
                  if p.name not in ("self", "root", "download", "verify")]
        splits = ", ".join(f"`{s}`" for s in cls.files)
        if cls.split_aliases:
            splits += " (" + ", ".join(f"`{a}`→`{b}`" for a, b in cls.split_aliases.items()) + ")"
        rows.append([f"`{name}`", splits, ", ".join(_param(p) for p in params) or t["none"]])
    return table([t["name"], t["splits"], t["options"]], rows)


# --------------------------------------------------------------------------- methods
# Classes whose docstring has no References section yet (none today): name -> reference text.
FALLBACK_REFERENCES: dict[str, str] = {}


def method_families() -> dict[str, str]:
    from indoorloc import methods

    doc = methods.__doc__ or ""
    block = doc.split("Families:", 1)[1] if "Families:" in doc else ""
    out = {}
    for line in block.splitlines():
        m = re.match(r"^\s{2}(\S+)\s{2,}(.+)$", line)
        if m:
            for name in re.sub(r"\([^)]*\)", "", m.group(2)).split(","):
                if name.strip():
                    out[name.strip()] = m.group(1)
    return out


def gen_methods(lang: str) -> str:
    from indoorloc.methods import METHODS

    t = T[lang]
    families = method_families()
    alias_map = aliases(METHODS)
    groups: dict[str, list] = {}
    for name in METHODS.names():
        cls = METHODS.get(name)
        extras = extras_of(sys.modules[cls.__module__])
        family = families.get(name, "other")
        nan = class_flag(cls, "_allow_nan", lang, members=True)
        if nan == t["no"] and family in ("fingerprinting", "deep"):  # RSSI fingerprints: FillMissing is the fix
            nan = t["fill_first"]
        row = [linked_name(cls, name, alias_map, lang) + f"<br>`{cls.__name__}`", first_line(cls),
               ", ".join(f"`[{e}]`" for e in extras) or t["numpy"], nan,
               ref_cell(cls, lang, FALLBACK_REFERENCES, name)]
        groups.setdefault(family, []).append(row)
    out = []
    for fam in ("fingerprinting", "model-based", "sequence", "deep", "other"):
        if fam in groups:
            out.append(f"**{t['families'][fam]}**\n\n" + table(
                [t["name"], t["what"], t["extra"], t["nan"], t["ref"]], groups[fam]))
    return "\n\n".join(out)


def gen_transfer(lang: str) -> str:
    from indoorloc.methods import transfer

    t = T[lang]
    rows = []
    for name in transfer.__all__:
        obj = getattr(transfer, name)
        kind = t["cls"] if inspect.isclass(obj) else t["fn"]
        extras = ", ".join(f"`[{e}]`" for e in extras_of(transfer)) if name == "SkadaAdapter" else t["numpy"]
        rows.append([source_link(obj, f"`{name}`"), kind, first_line(obj), extras, ref_cell(obj, lang)])
    return table([t["name"], t["kind"], t["what"], t["extra"], t["ref"]], rows)


# --------------------------------------------------------------------------- signals
def gen_transforms(lang: str) -> str:
    from indoorloc import signals
    from indoorloc.signals.transforms import Transform

    t = T[lang]
    rows = []
    order = ("transforms", "augment", "calibration", "csi", "magnetic")
    classes = [getattr(signals, n) for n in signals.__all__
               if inspect.isclass(getattr(signals, n)) and issubclass(getattr(signals, n), Transform)
               and n not in ("Transform", "Augmentation")]
    classes.sort(key=lambda c: (order.index(c.__module__.rsplit(".", 1)[1]), c.__name__))
    for cls in classes:
        rows.append([source_link(cls, f"`{cls.__name__}`"), f"`{cls.__module__.replace('indoorloc.', '')}`",
                     first_line(cls), class_flag(cls, "_requires_fit", lang), ref_cell(cls, lang)])
    return table([t["class"], t["module"], t["what"], t["fit"], t["ref"]], rows)


def gen_views(lang: str) -> str:
    from indoorloc import signals

    t = T[lang]
    rows = [[source_link(getattr(signals, n), f"`{n}`"), first_line(getattr(signals, n))]
            for n in ("WiFiSignal", "BLESignal")]
    return table([t["class"], t["what"]], rows)


def _functions(module, lang: str, names=None) -> str:
    t = T[lang]
    names = names or [n for n, f in inspect.getmembers(module, inspect.isfunction)
                      if not n.startswith("_") and f.__module__ == module.__name__]
    rows = [[f"`{n}`", first_line(getattr(module, n))] for n in names]
    return table([t["function"], t["what"]], rows)


def gen_signal_functions(lang: str) -> str:
    from indoorloc import signals

    out = []
    for mod in ("functional", "csi", "ranging", "imu", "magnetic", "vlc"):
        module = getattr(signals, mod)
        out.append(f"**`indoorloc.signals.{mod}`**: {first_line(module)}\n\n" + _functions(module, lang))
    return "\n\n".join(out)


# --------------------------------------------------------------------------- evaluation
def gen_protocols(lang: str) -> str:
    from indoorloc.evaluation import get_protocol, list_protocols

    t = T[lang]
    rows = []
    for name in list_protocols():
        p = get_protocol(name)
        rows.append([f"`{name}`", p.summary, ", ".join(f"`{n}`" for n in p.needs) or t["none"]])
    return table([t["protocol"], t["summary"], t["needs"]], rows)


def gen_results_fields(lang: str) -> str:
    from indoorloc.evaluation import EvaluationResults

    return ", ".join(f"`{f.name}`" for f in dataclasses.fields(EvaluationResults))


def gen_eval_functions(lang: str) -> str:
    from indoorloc.evaluation import bounds, functional, plot, protocols, report, scoring

    out = []
    sections = [("functional", functional), ("protocols", protocols), ("scoring", scoring), ("bounds", bounds),
                ("report", report), ("plot", plot)]
    for label, module in sections:
        out.append(f"**`indoorloc.evaluation.{label}`**: {first_line(module)}\n\n" + _functions(module, lang))
    return "\n\n".join(out)


def gen_literature(lang: str) -> str:
    from indoorloc.evaluation import literature

    t = T[lang]
    checks = literature.CHECKS
    rows = []
    for name in literature.list_tables():
        entries = literature.load(name)["entries"]
        rows.append([f"`{name}`", len(entries), *(sum(e.get("check") == c for e in entries) for c in checks)])
    return table([t["dataset"], t["entries"], *(f"`{c}`" for c in checks)], rows)


# --------------------------------------------------------------------------- apps
def gen_apps(lang: str) -> str:
    from indoorloc import apps

    t = T[lang]
    order = ("tracking", "particle", "maps", "pdr", "fusion", "streaming", "navigation")
    items = [(n, getattr(apps, n)) for n in apps.__all__]
    items.sort(key=lambda it: (order.index(it[1].__module__.rsplit(".", 1)[1]), not inspect.isclass(it[1]), it[0]))
    rows = []
    for name, obj in items:
        rows.append([source_link(obj, f"`{name}`"), f"`{obj.__module__.replace('indoorloc.', '')}`",
                     t["cls"] if inspect.isclass(obj) else t["fn"], first_line(obj), ref_cell(obj, lang)])
    return table([t["name"], t["module"], t["kind"], t["what"], t["ref"]], rows)


# --------------------------------------------------------------------------- cli
def gen_cli(lang: str) -> str:
    from indoorloc.cli import build_parser

    t = T[lang]
    parser = build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    for p in (parser, *sub.choices.values()):
        p.color = False  # Python 3.14 colours help text; the docs want plain text
    helps = {c.dest: c.help for c in sub._choices_actions}
    out = []
    for name, p in sub.choices.items():
        usage = re.sub(r"\s+", " ", p.format_usage().replace("usage: ", "")).strip()
        rows = []
        for a in p._actions:
            if isinstance(a, argparse._HelpAction):
                continue
            flag = ", ".join(a.option_strings) if a.option_strings else a.dest
            if a.choices:
                flag += " {" + ",".join(a.choices) + "}"
            rows.append([f"`{flag}`", a.help or ""])
        out.append(f"#### `indoorloc {name}`\n\n{helps.get(name, '')}\n\n```text\n{usage}\n```\n\n"
                   + table([t["option"], t["help"]], rows))
    return "\n\n".join(out)


def gen_extras(lang: str) -> str:
    """The pip extras of pyproject.toml with their packages and the comment that explains each."""
    t = T[lang]
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = text.split("[project.optional-dependencies]", 1)[1].split("\n[", 1)[0]
    rows, name, buf = [], None, ""
    for line in block.splitlines():
        m = re.match(r"^([\w-]+)\s*=\s*\[(.*)$", line)
        if m:
            name, buf = m.group(1), m.group(2)
        elif name:
            buf += " " + line.strip()
        if name and "]" in buf:
            packages, _, comment = buf.partition("]")
            pkgs = ", ".join(f"`{p.strip().strip(chr(34))}`" for p in packages.split(",") if p.strip())
            rows.append([f"`indoorloc[{name}]`", pkgs, comment.strip().lstrip("#").strip()])
            name, buf = None, ""
    deps = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.MULTILINE | re.DOTALL).group(1)
    base = ", ".join(f"`{d.strip().strip(chr(34))}`" for d in deps.split(",") if d.strip())
    rows.insert(0, ["`indoorloc`", base, "L1-L5 on numpy alone" if lang == "en" else "五层全部只需 numpy"])
    return table([t["install"], t["packages"], t["purpose"]], rows)


GENERATORS = {
    "datasets": gen_datasets, "dataset-options": gen_dataset_options, "methods": gen_methods,
    "transfer": gen_transfer, "transforms": gen_transforms, "views": gen_views,
    "signal-functions": gen_signal_functions, "protocols": gen_protocols,
    "results-fields": gen_results_fields, "eval-functions": gen_eval_functions, "literature": gen_literature,
    "apps": gen_apps, "cli": gen_cli, "extras": gen_extras,
}
MARK = re.compile(r"(<!-- catalog:([\w-]+) -->\n)(.*?)(<!-- /catalog:\2 -->)", re.DOTALL)


# --------------------------------------------------------------------------- driver
def render(text: str, lang: str, cache: dict) -> str:
    def replace(m):
        key = (m.group(2), lang)
        if key not in cache:
            cache[key] = GENERATORS[m.group(2)](lang)
        return f"{m.group(1)}{cache[key]}\n{m.group(4)}"
    return MARK.sub(replace, text)


def update(check: bool) -> int:
    cache: dict = {}
    stale = []
    for lang, folder in PAGES.items():
        for page in sorted(folder.glob("*.md")) + EXTRA_PAGES[lang]:
            if not page.is_file():
                continue
            text = page.read_text(encoding="utf-8")
            new = render(text, lang, cache)
            if new != text:
                stale.append(page.relative_to(ROOT).as_posix())
                if not check:
                    page.write_text(new, encoding="utf-8")
    from indoorloc.methods import METHODS

    for name in FALLBACK_REFERENCES:
        if not references(METHODS.get(name)):
            print(f"note: {name} has no References section in its docstring; the catalog uses FALLBACK_REFERENCES",
                  file=sys.stderr)
    verb = "out of date" if check else "updated"
    print(f"{len(stale)} page(s) {verb}" + (": " + ", ".join(stale) if stale else ""))
    return 1 if check and stale else 0


def _blocks(text: str):
    for m in re.finditer(r"^```python\n(.*?)^```", text, re.DOTALL | re.MULTILINE):
        yield text[:m.start()].count("\n") + 2, m.group(1)


def _header(code: str) -> list[str]:
    """The comment lines a block starts with (where ``# data:`` and ``# requires:`` markers live)."""
    lines = []
    for line in code.lstrip().splitlines():
        if not line.startswith("#"):
            break
        lines.append(line)
    return lines


def _missing_data(code: str, no_data: bool = False) -> list[str]:
    names = [n.strip() for line in _header(code) if (m := re.match(r"#\s*data:\s*(.+)", line))
             for n in m.group(1).split(",")]
    if not names or no_data:
        return names
    from indoorloc.datasets import DATASETS, default_root

    missing = []
    for name in names:
        cls = DATASETS.get(name)
        source = cls(default_root() / cls.name)
        paths = [source.root / rel for split in source.files for rel, _ in source._entries(split)]
        if not paths or not all(p.is_file() for p in paths):
            missing.append(name)
    return missing


def _expected_output(code: str) -> list[str]:
    r"""Comment lines that directly follow a ``print(...)`` call: the output the page promises.

    >>> block = "x = 1  # a note\nprint(x,\n      x + 1)  # trailing\n# 1 2\n# 3\n"
    >>> _expected_output(block)
    ['1 2', '3']
    >>> _expected_output("print(1)\ny = 2\n# a comment, not output\n")
    []
    """
    expected, after_print, depth = [], False, 0
    for line in code.splitlines():
        stripped = line.strip()
        if depth == 0 and after_print and stripped.startswith("# "):
            expected.append(stripped[2:])
            continue
        code_part = line.split("  #")[0]
        if depth > 0:
            depth += code_part.count("(") - code_part.count(")")
            after_print = depth == 0
            continue
        if "print(" in code_part and not stripped.startswith("#"):
            depth = code_part[code_part.index("print("):].count("(") - code_part[code_part.index("print("):].count(")")
            after_print = depth == 0
        else:
            after_print = False
    return expected


def _missing_modules(code: str) -> list[str]:
    import importlib.util

    names = [n.strip() for line in _header(code) if (m := re.match(r"#\s*requires:\s*(.+)", line))
             for n in m.group(1).split(",")]
    return [n for n in names if importlib.util.find_spec(n) is None]


def run_page(page: Path, verbose: bool, no_data: bool = False) -> tuple[int, int, int]:
    """Run the Python blocks of one page in order, in one namespace, inside a temporary folder."""
    failures, ran, skipped = 0, 0, 0
    namespace: dict = {"__name__": "__main__"}
    home = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="indoorloc-docs-") as tmp:
        os.chdir(tmp)
        try:
            for line, code in _blocks(page.read_text(encoding="utf-8")):
                where = f"{page.relative_to(ROOT).as_posix()}:{line}"
                if re.match(r"#\s*0\.1\b", code.lstrip()):
                    skipped += 1
                    print(f"skip {where} (0.1 code, shown for comparison)")
                    continue
                missing = _missing_data(code, no_data) + _missing_modules(code)
                if missing:
                    skipped += 1
                    print(f"skip {where} (not available: {', '.join(missing)})")
                    continue
                out = io.StringIO()
                start = time.perf_counter()
                try:
                    with contextlib.redirect_stdout(out):
                        exec(compile(code, where, "exec"), namespace)  # noqa: S102 (the guide's own code)
                except Exception:  # noqa: BLE001 (report every failing block, keep going)
                    failures += 1
                    print(f"FAIL {where}\n{traceback.format_exc()}")
                    continue
                ran += 1
                expected, actual = _expected_output(code), out.getvalue().rstrip("\n").splitlines()
                if expected and [a.rstrip() for a in actual] != expected:
                    failures += 1
                    print(f"DIFF {where}: printed\n  " + "\n  ".join(actual)
                          + "\n  but the page shows\n  " + "\n  ".join(expected))
                    continue
                print(f"ok   {where} ({time.perf_counter() - start:.1f} s)")
                if verbose and out.getvalue():
                    print("     " + out.getvalue().rstrip().replace("\n", "\n     "))
        finally:
            os.chdir(home)
    print(f"#counts {ran} {skipped} {failures}")
    return ran, skipped, failures


def run_snippets(pattern: str | None, verbose: bool, no_data: bool = False) -> int:
    """Each page in its own interpreter, so registrations and imports never leak between pages."""
    import subprocess

    totals = [0, 0, 0]
    for lang, folder in PAGES.items():
        for page in sorted(folder.glob("*.md")) + EXTRA_PAGES[lang]:
            if not page.is_file() or (pattern and pattern not in page.as_posix()):
                continue
            cmd = [sys.executable, str(Path(__file__).resolve()), "--run-page", str(page)] + (["-v"] if verbose else [])
            cmd += ["--no-data"] if no_data else []
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            for line in proc.stdout.splitlines():
                if line.startswith("#counts "):
                    for i, n in enumerate(line.split()[1:]):
                        totals[i] += int(n)
                else:
                    print(line)
            if proc.returncode != 0:
                totals[2] += 1
                print(f"FAIL {page.relative_to(ROOT).as_posix()} (the page runner exited with {proc.returncode})\n"
                      + proc.stderr[-2000:])
    print(f"{totals[0]} block(s) ran, {totals[1]} skipped, {totals[2]} failed")
    return 1 if totals[2] else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if a generated block is out of date")
    parser.add_argument("--snippets", action="store_true", help="run the Python blocks of the guide pages")
    parser.add_argument("--page", help="with --snippets: only pages whose path contains this text")
    parser.add_argument("-v", "--verbose", action="store_true", help="with --snippets: print each block's output")
    parser.add_argument("--no-data", action="store_true",
                        help="with --snippets: skip every `# data:` block, as on CI where no dataset is cached")
    parser.add_argument("--run-page", help=argparse.SUPPRESS)  # internal: one page, in this interpreter
    args = parser.parse_args(argv)
    if args.run_page:
        run_page(Path(args.run_page).resolve(), args.verbose, args.no_data)
        return 0
    if args.snippets:
        return run_snippets(args.page, args.verbose, args.no_data)
    return update(args.check)


if __name__ == "__main__":
    raise SystemExit(main())
