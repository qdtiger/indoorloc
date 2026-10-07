"""Run the Python blocks of ``README.md`` in order, in one namespace, and check what they promise.

The README states numbers in comments: the output below a ``print(...)`` and the value after a
statement (``horus.evaluate(test).mean_error  # 7.994``). ``CHECKS`` lists the key statements
with the value their comment promises and how that value is read from a run. The tests:

* the README still contains every checked statement and promises the checked value (no data,
  no execution), so a README edit cannot drop or change a number unnoticed;
* the blocks that need no measured data (the L5 block on the simulated office) run on their own
  and give the promised values (every CI job);
* every block runs in order in one namespace on the cached datasets and gives every promised
  value, and every ``print`` prints exactly the comment lines below it (skipped when a dataset
  the README loads is not cached under ``$INDOORLOC_DATA``; about 10 s and 0.8 GB);
* MIGRATION.md and CHANGELOG.md quote the recorded 0.1 run
  (``examples/readme_case/v0.1_results.json``) correctly.

Nothing is downloaded: the dataset downloader is replaced by a failure for the whole run.
"""
from __future__ import annotations

import ast
import builtins
import contextlib
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from conftest import PROJECT

README = PROJECT / "README.md"


# --------------------------------------------------------------------------- reading the README
def readme_blocks(text: str | None = None) -> list[tuple[int, str]]:
    """``(line of the first code line, code)`` of every ```python block, in order."""
    text = README.read_text(encoding="utf-8") if text is None else text
    return [(text[:m.start()].count("\n") + 2, m.group(1))
            for m in re.finditer(r"^```python\n(.*?)^```", text, re.DOTALL | re.MULTILINE)]


@dataclass
class Statement:
    node: ast.stmt
    source: str  # the statement as written
    trailing: str | None  # the comment on its last line, after it
    below: list[str]  # the comment lines directly below it

    @property
    def is_print(self) -> bool:
        return (isinstance(self.node, ast.Expr) and isinstance(self.node.value, ast.Call)
                and isinstance(self.node.value.func, ast.Name) and self.node.value.func.id == "print")

    @property
    def promise(self) -> str | None:
        """What the README says the statement gives: the lines below a print or a bare expression,
        else the trailing comment."""
        if self.below and isinstance(self.node, ast.Expr):
            return "\n".join(self.below)
        return self.trailing


def statements(code: str) -> list[Statement]:
    lines = code.splitlines()
    out = []
    for node in ast.parse(code).body:
        rest = lines[node.end_lineno - 1].encode()[node.end_col_offset:].decode().strip()  # offsets are in bytes
        below = []
        for line in lines[node.end_lineno:]:
            if not line.strip().startswith("#"):
                break
            below.append(line.strip()[1:].strip())
        out.append(Statement(node, ast.get_source_segment(code, node),
                             rest[1:].strip() if rest.startswith("#") else None, below))
    return out


def load_calls(code: str):
    """``(dataset name, split, options)`` of every literal ``load_dataset(...)`` call of a block."""
    for node in ast.walk(ast.parse(code)):
        func = getattr(node, "func", None)
        if isinstance(node, ast.Call) and getattr(func, "attr", getattr(func, "id", None)) == "load_dataset":
            args = [ast.literal_eval(a) for a in node.args]
            options = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
            yield args[0], args[1] if len(args) > 1 else options.pop("split", None), options


def data_status(code: str) -> dict[str, str]:
    """dataset -> ``"simulated"`` (no files), ``"cached"`` or ``"missing"`` for the block's loads."""
    from indoorloc.datasets import DATASETS

    status = {}
    for name, split, options in load_calls(code):
        source = DATASETS.get(name)(None, download=False, verify=False, **options)
        splits = (split,) if isinstance(split, str) else tuple(split or source.default_splits)
        try:
            files = [p for s in splits for p in source.check(s)]
        except FileNotFoundError:
            status[name] = "missing"
            continue
        if status.get(name) != "missing":
            status[name] = "cached" if files else status.get(name, "simulated")
    return status


def free_names(code: str) -> set[str]:
    """Names a block reads without defining them itself (it needs an earlier block for these)."""
    tree = ast.parse(code)
    stored = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load)}
    stored |= {(a.asname or a.name).split(".")[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
               for a in n.names}
    loaded = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return loaded - stored - set(dir(builtins))


def _missing_measured() -> list[str]:
    try:
        return sorted({name for _, code in readme_blocks() for name, s in data_status(code).items() if s == "missing"})
    except Exception as err:  # noqa: BLE001 (an unparsable README fails test_readme_promises_the_checked_values)
        return [f"README not readable ({err})"]


# --------------------------------------------------------------------------- what is checked
@dataclass
class Result:
    value: object  # the value of a bare expression, else None
    printed: str
    ns: dict
    workdir: Path


def _mean_error(pred, table) -> float:
    return float(np.linalg.norm(pred.pos - table.pos, axis=1).mean())


CHECKS = (  # (statement exactly as in the README, what its comment promises, the same text from a run)
    ("print(model.fit(train).evaluate(test))",
     "mean 8.7937  median 5.3546  P90 19.1537  floor 90.46 %  building 99.73 %  (n=1111)",
     lambda r: r.printed.rstrip("\n")),
    ('train.X.shape, train.meta["crs"], sorted(train.groups)',
     "(19937, 520) EPSG:3857 ['device', 'relative_position', 'space', 'time', 'user']",
     lambda r: "{} {} {}".format(*r.value)),
    ('imu = iloc.load_dataset("ilc2020", site="site1", floor="F1", modality="imu")',
     "50 Hz phone IMU + floor plan",
     lambda r: f"{r.ns['imu'].meta['rate_hz']:g} Hz phone IMU" + " + floor plan" * ("floor_plan" in r.ns["imu"].meta)),
    ('csi = iloc.load_dataset("haloc", split="test")', "complex64 (14277, 1, 1, 52)",
     lambda r: f"{r.ns['csi'].X.dtype} {r.ns['csi'].X.shape}"),
    ("amplitude = iloc.CSIAmplitude()(csi)", '|H|, modality "csi_amp"',
     lambda r: f'|H|, modality "{r.ns["amplitude"].meta["modality"]}"'),
    ("horus.evaluate(test).mean_error", "7.994", lambda r: f"{r.value:.3f}"),
    ("print(geo.evaluate(te))",
     "mean 0.6035  median 0.4921  P90 1.2515  floor n/a  building n/a  (n=200)",
     lambda r: r.printed.rstrip("\n")),
    ('horus = iloc.load_model("horus_uji")', "config.json + arrays.npz",
     lambda r: " + ".join(sorted((p.name for p in (r.workdir / "horus_uji").iterdir()), reverse=True))),
    ("ipin_score(test, pred)", "12.201", lambda r: f"{r.value:.3f}"),
    ("model.evaluate(test).cdf([1, 5, 10])", "[0.096 0.476 0.736]",
     lambda r: "[" + " ".join(f"{v:.3f}" for v in r.value) + "]"),
    ('folds = get_protocol("cross-device").folds(train, random_state=0)', "one fold per phone (16)",
     lambda r: f"one fold per phone ({len(r.ns['folds'])})"),
    ('smooth = iloc.KalmanTracker().smooth(fixes, t=walk.groups["time"])', "mean error 2.97 m -> 2.11 m",
     lambda r: f"mean error {_mean_error(r.ns['fixes'], r.ns['walk']):.2f} m -> "
               f"{_mean_error(r.ns['smooth'], r.ns['walk']):.2f} m"),
    ("nav.route((2.0, 2.0), (36.0, 17.0)).instructions[1].text", "'Turn right, then walk 33.5 m'",
     lambda r: repr(r.value)),
)
CHECKED = {source: (promised, actual) for source, promised, actual in CHECKS}


def _promises(promise: str | None, value: str) -> bool:
    """``value`` is the whole promise or its start, up to a separator (``12.201: P75 of ...``)."""
    return promise is not None and re.match(re.escape(value) + r"(?![\w.])", promise) is not None


# --------------------------------------------------------------------------- running the blocks
def run_blocks(blocks, workdir: Path, ns: dict | None = None) -> tuple[dict, list[str]]:
    """Execute ``blocks`` statement by statement in one namespace, inside ``workdir``.

    Returns ``{statement: text read from the run}`` for the ``CHECKS`` statements that ran, and
    the problems found on the way: a ``print`` whose output differs from the comment lines below
    it. An exception propagates with its README line number.
    """
    ns = {"__name__": "__main__"} if ns is None else ns
    actual, problems = {}, []
    import indoorloc.datasets._base as base

    def no_download(*args, **kwargs):
        raise AssertionError("the README test tried to download a dataset")

    home = Path.cwd()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(base, "_fetch", no_download)
        os.chdir(workdir)
        try:
            for first_line, code in blocks:
                for st in statements(code):
                    node = ast.increment_lineno(st.node, first_line - 1)  # tracebacks point at README.md lines
                    out = io.StringIO()
                    with contextlib.redirect_stdout(out):
                        if isinstance(node, ast.Expr):
                            value = eval(compile(ast.Expression(node.value), "README.md", "eval"), ns)  # noqa: S307
                        else:
                            exec(compile(ast.Module([node], type_ignores=[]), "README.md", "exec"), ns)  # noqa: S102
                            value = None
                    printed = out.getvalue()
                    if st.source in CHECKED:
                        actual[st.source] = CHECKED[st.source][1](Result(value, printed, ns, workdir))
                    if st.is_print and st.below and printed.rstrip("\n").splitlines() != st.below:
                        problems.append(f"README.md:{node.lineno}: {st.source} printed\n  {printed.rstrip()}\n"
                                        "but the README shows\n  " + "\n  ".join(st.below))
        finally:
            os.chdir(home)
    return actual, problems


def _compare(actual: dict) -> list[str]:
    return [f"{source}: the README promises {CHECKED[source][0]!r}, the run gives {text!r}"
            for source, text in actual.items() if text != CHECKED[source][0]]


# --------------------------------------------------------------------------- tests
def test_readme_promises_the_checked_values():
    found = {st.source: st for _, code in readme_blocks() for st in statements(code)}  # also: every block parses
    for source, promised, _ in CHECKS:
        assert source in found, f"README.md no longer has the statement {source!r}: update CHECKS with the README"
        assert _promises(found[source].promise, promised), \
            f"README.md: {source!r} now promises {found[source].promise!r}, the test checks {promised!r}"


def test_readme_blocks_without_measured_data_run_on_their_own(tmp_path):
    """The blocks that load only simulated data and need nothing but ``iloc`` from earlier blocks."""
    alone = [(line, code) for line, code in readme_blocks()
             if free_names(code) <= {"iloc"} and set(data_status(code).values()) <= {"simulated"}]
    assert alone, "no README block runs without measured data"
    ns = {"__name__": "__main__"}
    exec("import indoorloc as iloc", ns)  # noqa: S102 (the README's first line)
    actual, problems = run_blocks(alone, tmp_path, ns)
    assert {'smooth = iloc.KalmanTracker().smooth(fixes, t=walk.groups["time"])',
            "nav.route((2.0, 2.0), (36.0, 17.0)).instructions[1].text"} <= set(actual)
    problems += _compare(actual)
    assert not problems, "\n".join(problems)


MISSING = _missing_measured()


@pytest.mark.skipif(bool(MISSING), reason=f"datasets the README loads are not cached: {', '.join(MISSING)}")
def test_readme_runs_in_order_on_the_cached_data(tmp_path):
    actual, problems = run_blocks(readme_blocks(), tmp_path)
    problems += [f"{source}: not run" for source in CHECKED if source not in actual]
    problems += _compare(actual)
    assert not problems, "\n".join(problems)


def test_migration_and_changelog_quote_the_recorded_01_run():
    """MIGRATION.md and CHANGELOG.md cite the 0.1 run record and quote its numbers correctly."""
    record = json.loads((PROJECT / "examples" / "readme_case" / "v0.1_results.json").read_text(encoding="utf-8"))
    knn, wknn = (record["methods"][m]["metrics"] for m in ("knn", "wknn"))
    assert record["environment"]["cpu_threads"] == 2 and knn["mean_error"] == 8.889091885956587
    migration, changelog = (" ".join((PROJECT / name).read_text(encoding="utf-8").split())
                            for name in ("MIGRATION.md", "CHANGELOG.md"))
    for text in (migration, changelog):
        assert "examples/readme_case/v0.1_results.json" in text
        assert "has since been replaced" not in text and "now holds the 0.2 run" not in text
    assert f"had {knn['mean_error']:.4f} / {wknn['mean_error']:.4f} m (the 0.1 record is" in changelog
    assert f"gave k-NN {knn['mean_error']:.4f} m and WKNN {wknn['mean_error']:.4f} m" in migration
    assert f"(mean errors {knn['mean_error']!r} and {wknn['mean_error']!r} EPSG:3857 m, `cpu_threads: 2`)" in migration
    assert f"| {knn['mean_error']:.4f} | {wknn['mean_error']:.4f} |" in migration  # the reproduced 2-thread row
    assert f"WKNN floor accuracy changed from {wknn['floor_accuracy']:.2f} % to 90.46 %" in migration
    assert f"k-NN floor accuracy stayed at {knn['floor_accuracy']:.2f} %" in migration


def test_the_statement_parser_reads_promises():
    code = ('x = f(1)  # 12.201: a note\n'
            'print(x,\n      2)  # trailing note\n# 1 2\n# 3\n'
            'x.shape\n# (2, 3)\n'
            'a = 1; b = 2   # about b\n')
    by = {st.source: st for st in statements(code)}
    assert by["x = f(1)"].promise == "12.201: a note" and _promises(by["x = f(1)"].promise, "12.201")
    assert not _promises(by["x = f(1)"].promise, "12.20") and not _promises(None, "1")
    printed = by["print(x,\n      2)"]
    assert printed.is_print and printed.below == ["1 2", "3"] and printed.promise == "1 2\n3"
    assert by["x.shape"].promise == "(2, 3)" and not by["x.shape"].is_print
    assert by["a = 1"].trailing is None and by["b = 2"].trailing == "about b"
    assert free_names("import numpy as np\ny = np.zeros(3) + x\nprint(y)") == {"x"}
    assert list(load_calls('t = il.load_dataset("haloc", split="test")\nload_dataset("x", "all", seed=1)')) == \
        [("haloc", "test", {}), ("x", "all", {"seed": 1})]
