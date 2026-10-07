"""Smoke runs of the example scripts (``examples/quickstart.py``, ``examples/0*_*.py``) with tiny settings.

Every script exposes ``main(**settings)``; each test imports the script by path (the file names
start with digits), runs it into a temporary folder and checks the figure and a few returned
numbers, so the examples cannot silently drift from the library. Scripts on real data run only
when the files are cached (``$INDOORLOC_DATA``, default ``~/.cache/indoorloc/datasets``); no
test downloads anything. The known values asserted on real data are the recorded cells of
``benchmarks/results`` (UJIIndoorLoc WKNN 8.7937 / exponential WKNN 7.9880 EPSG:3857 m); on
simulated data, closed-form results: the Cramer-Rao bound is attained by maximum likelihood at
small noise, the example's own VLC Fisher information equals its closed form for four LEDs
around the receiver, and a route's cost is its grid walk plus the documented connector costs.
"""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys

import numpy as np
import pytest
from conftest import DATA_ROOT, HEAVY, PROJECT, UJI_ROOT

pytest.importorskip("matplotlib")

EXAMPLES = PROJECT / "examples"
SCRIPTS = sorted(EXAMPLES.glob("0*_*.py")) + [EXAMPLES / "quickstart.py"]
HAS_UJI = (UJI_ROOT / "trainingData.csv").is_file() and (UJI_ROOT / "validationData.csv").is_file()
HAS_HALOC = (DATA_ROOT / "haloc" / "HALOC.zip").is_file()
HAS_ILC = (DATA_ROOT / "ilc2020" / "data" / "site1" / "F1" / "geojson_map.json").is_file()
uji = pytest.mark.skipif(not HAS_UJI, reason=f"UJIIndoorLoc not cached under {UJI_ROOT}")


def load(name: str):
    path = EXAMPLES / name
    spec = importlib.util.spec_from_file_location(f"example_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_example_is_covered_and_shares_one_figure_style():
    names = {p.name for p in SCRIPTS}
    assert {"01_fingerprinting_benchmark.py", "02_model_based_vs_crlb.py", "03_csi_pipeline.py",
            "04_tracking_and_fusion.py", "05_domain_adaptation.py", "06_navigation.py", "quickstart.py"} <= names
    styles = [load(p.name) for p in SCRIPTS if p.name != "quickstart.py"]
    for module in styles:
        assert callable(module.main)
        assert module.STYLE == styles[0].STYLE and module.PALETTE == styles[0].PALETTE, module.__name__
        assert module.STYLE["figure.facecolor"] == "white" and module.STYLE["savefig.dpi"] >= 150


def test_examples_import_without_the_heavy_packages():
    """Importing a script (e.g. to read its settings) loads numpy and indoorloc only; matplotlib, torch
    and the rest are imported inside the functions that need them, as in the library (CONTRACTS §1)."""
    code = ("import importlib.util, sys\n"
            f"for name in {HEAVY!r}:\n"
            "    sys.modules[name] = None\n"
            f"for path in {[str(p) for p in SCRIPTS]!r}:\n"
            "    spec = importlib.util.spec_from_file_location('example', path)\n"
            "    spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
            "print(sorted(m for m, v in sys.modules.items() if v is not None and m.split('.')[0] in "
            f"{HEAVY!r}))\n")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(PROJECT), os.environ.get("PYTHONPATH")]))}
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=PROJECT,
                         check=False, timeout=120)
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.strip() == "[]"


def test_examples_use_only_public_names():
    private = re.compile(r"from indoorloc[\w.]* import [^\n]*\b_\w|indoorloc(\.\w+)*\._\w|iloc(\.\w+)*\._\w")
    for path in SCRIPTS:
        hits = [m.group(0) for m in private.finditer(path.read_text())]
        assert not hits, f"{path.name} uses private names: {hits}"


# ------------------------------------------------------------------------------------ quickstart
def test_quickstart_runs_offline_on_the_simulated_office(tmp_path):
    out = load("quickstart.py").main("synthetic_office", download=False, workdir=tmp_path, verbose=False)
    assert out["reload_identical"] is True
    assert np.isfinite(out["mean_error"]) and 0 < out["mean_error"] < 5  # simulated 40 m x 20 m floor
    assert (tmp_path / "wknn_model" / "config.json").is_file()


@uji
def test_quickstart_reproduces_the_recorded_uji_wknn_result(tmp_path):
    out = load("quickstart.py").main(download=False, workdir=tmp_path, verbose=False)
    assert out["mean_error"] == pytest.approx(8.7937, abs=1e-4)  # benchmarks/results: WKNN (k=5), fill -104
    assert out["building_accuracy"] == pytest.approx(99.73, abs=0.01)
    assert out["mean_error_ground"] == pytest.approx(8.7937 * 0.76613, abs=1e-3)
    assert out["reload_identical"] is True


# ------------------------------------------------------------------------------------ 01
def test_fingerprinting_example_on_the_simulated_office(tmp_path):
    res = load("01_fingerprinting_benchmark.py").main(dataset="synthetic_office", methods=["k-NN (k=5)", "WKNN (k=5)"],
                                                      mlp=False, download=False, out=tmp_path, verbose=False)
    assert (tmp_path / "fingerprinting_cdf.png").stat().st_size > 10_000
    for r in res.values():
        lo, hi = r["mean_ci95"]
        assert lo <= r["mean"] <= hi and r["n_failed"] == 0 and r["recorded_mean_native"] is None


@uji
def test_fingerprinting_example_matches_the_benchmark_matrix(tmp_path):
    res = load("01_fingerprinting_benchmark.py").main(methods=["WKNN (k=5)", "WKNN, exponential repr."], mlp=False,
                                                      download=False, out=tmp_path, verbose=False)
    assert res["WKNN (k=5)"]["mean_native"] == pytest.approx(8.7937, abs=1e-4)
    assert res["WKNN, exponential repr."]["mean_native"] == pytest.approx(7.9880, abs=1e-4)
    for r in res.values():
        if r["recorded_mean_native"] is not None:  # the repository's recorded cell, when present
            assert r["mean_native"] == pytest.approx(r["recorded_mean_native"], abs=1e-9)


# ------------------------------------------------------------------------------------ 02
def test_crlb_example_estimators_reach_the_bound_at_small_noise(tmp_path):
    res = load("02_model_based_vs_crlb.py").main(n_points=40, n_draws=3, n_boot=50, out=tmp_path, verbose=False)
    assert (tmp_path / "model_based_vs_crlb.png").is_file()
    assert set(res) == {"toa", "tdoa", "aoa", "vlc"}
    for case, (closed, ml) in (("toa", ("linear least squares", "Gauss-Newton (ML)")),
                               ("tdoa", ("Chan-Ho closed form", "Chan-Ho + Gauss-Newton (ML)")),
                               ("aoa", ("Stansfield (linear)", "Gauss-Newton (ML)"))):
        small_ml, small_lin = res[case][ml][0], res[case][closed][0]
        assert 0.8 < small_ml["ratio"] < 1.25, (case, small_ml)  # ML attains the CRLB at small noise
        assert small_lin["rmse"] >= small_ml["rmse"]            # the closed form does not beat ML there
        for r in res[case][ml]:
            assert r["crlb"] > 0 and r["n_failed"] == 0
    # the bound scales linearly with the noise std (Gaussian models): CRLB(sigma) / sigma is constant
    toa = res["toa"]["Gauss-Newton (ML)"]
    scale = [r["crlb"] / r["sigma"] for r in toa]
    assert np.allclose(scale, scale[0], rtol=1e-9)
    # VLC trials that are not placed are either short of LEDs or see only collinear LEDs; never more than failed
    for r in res["vlc"]["model fit, Gauss-Newton (ML)"]:
        assert 0 <= r["n_too_few"] <= r["n_failed"] <= r["n_trials"]


def test_crlb_example_vlc_fisher_information_matches_the_closed_form():
    """Four LEDs facing down at the corners of a square centred above the receiver (facing up): each gives
    P = Pt C h^(m+1) / d^(m+3), C = (m + 1) A / (2 pi), so |dP/dr| = (m + 3) Pt C h^(m+1) r / d^(m+5) along
    the horizontal line to the LED; the four unit vectors sum to u u^T = 2 I, hence J = 2 (dP/dr)^2 / sigma^2 I
    and the bound sqrt(trace J^-1) = sigma / |dP/dr| (Kahn & Barry 1997 channel gain; Kay 1993, Ch. 3)."""
    from indoorloc.evaluation import bounds
    from indoorloc.methods.vlc import LambertianLocalizer

    example = load("02_model_based_vs_crlb.py")
    a, top, z0, m, tx, area, sigma = 1.5, 3.0, 0.8, 1.7, 2.0, 1e-4, 1e-7
    centre = np.array([10.0, 5.0])
    leds = np.array([[a, a, top], [-a, a, top], [-a, -a, top], [a, -a, top]]) + np.r_[centre, 0.0]
    model = LambertianLocalizer(leds, order=m, tx_power=tx, area=area, receiver_height=z0)
    h, r = top - z0, a * np.sqrt(2.0)
    d, C = np.hypot(r, h), (m + 1.0) * area / (2.0 * np.pi)
    dp_dr = (m + 3.0) * tx * C * h ** (m + 1.0) * r / d ** (m + 5.0)
    J = example.vlc_fisher(model, centre[None], np.ones((1, 4), dtype=bool), sigma)
    assert np.allclose(J[0], 2.0 * dp_dr ** 2 / sigma ** 2 * np.eye(2), rtol=1e-6, atol=1e-6 * J[0, 0, 0])
    assert bounds.crlb_rmse(J)[0] == pytest.approx(sigma / dp_dr, rel=1e-6)
    # an LED the noise model drops carries no information: three LEDs give a larger bound than four
    three = example.vlc_fisher(model, centre[None], np.array([[True, True, True, False]]), sigma)
    assert bounds.crlb_rmse(three)[0] > bounds.crlb_rmse(J)[0]


# ------------------------------------------------------------------------------------ 03
@pytest.mark.skipif(not HAS_HALOC, reason=f"HALOC not cached under {DATA_ROOT / 'haloc'}")
def test_csi_example_on_a_haloc_subset(tmp_path):
    res = load("03_csi_pipeline.py").main(download=False, sequences=[3, 4, 5], span_s=20.0, out=tmp_path,
                                          kalman_grid={"process_noise": (0.1,), "min_meas_std": (1.0,)}, every=8,
                                          verbose=False)
    assert (tmp_path / "csi_pipeline.png").is_file()
    spread = res["phase_spread_median"]
    assert spread["sanitized"] < 0.1 and spread["raw"] > 1.0  # sanitizing removes the per-packet phase offsets
    for r in res["wknn"].values():
        assert np.isfinite(r["mean_error"]) and r["n_failed"] == 0
    assert res["kalman"]["process_noise"] == 0.1 and np.isfinite(res["kalman"]["smooth"]["mean_error"])


# ------------------------------------------------------------------------------------ 04
@pytest.mark.skipif(not HAS_ILC, reason=f"ILC 2020 site1/F1 not cached under {DATA_ROOT / 'ilc2020'}")
def test_tracking_example_on_two_ilc_traces(tmp_path):
    res = load("04_tracking_and_fusion.py").main(stride=60, max_test=2, n_particles=200, download=False,
                                                 out=tmp_path, verbose=False)
    assert (tmp_path / "tracking_and_fusion.png").is_file()
    assert res["n_traces"] == 120 and res["test_traces"] == 2
    for stats in res["summary"].values():
        assert stats["n"] > 0 and np.isfinite(stats["mean"]) and stats["mean"] < 30.0
        change, lo, hi = stats["change_vs_wifi"]  # paired, resampling whole traces
        assert lo <= change <= hi
    assert res["summary"]["WiFi WKNN"]["change_vs_wifi"] == (0.0, 0.0, 0.0)


# ------------------------------------------------------------------------------------ 05
@uji
def test_domain_adaptation_example(tmp_path):
    res = load("05_domain_adaptation.py").main(download=False, n_boot=100, out=tmp_path, verbose=False)
    assert (tmp_path / "domain_adaptation.png").is_file()
    runs = list(res)
    assert len(runs) == 3
    for run in res.values():
        assert run["no adaptation"]["delta"] == (0.0, 0.0, 0.0)
        for r in run.values():
            mean, lo, hi = r["delta"]
            assert lo <= mean <= hi
    # the offsets learned 13 -> 14 and 14 -> 13 have opposite signs and similar size (one phone reads lower)
    b_ab = next(iter(res[runs[0]]["DeviceCalibration (offset)"]["maps"].values()))[1]
    b_ba = next(iter(res[runs[1]]["DeviceCalibration (offset)"]["maps"].values()))[1]
    assert b_ab * b_ba < 0 and 2.0 < abs(b_ab) < 8.0 and 2.0 < abs(b_ba) < 8.0


# ------------------------------------------------------------------------------------ 06
def test_navigation_example_plans_both_routes(tmp_path):
    res = load("06_navigation.py").main(out=tmp_path, verbose=False)
    assert (tmp_path / "navigation.png").is_file()
    cheapest, step_free = res["cheapest (any connector)"], res["step-free (elevator only)"]
    assert cheapest["cost"] <= step_free["cost"]  # A* is optimal: removing the stairs cannot lower the cost
    # rides from the documented connector costs: stairs 10 m per storey, elevator 20 m wait + 3 m per storey
    assert cheapest["rides"] == pytest.approx(2 * 10.0) and step_free["rides"] == pytest.approx(20.0 + 2 * 3.0)
    for route in (cheapest, step_free):
        assert route["cost"] >= route["length"] > 0
        # cost = grid walk + rides; smoothing only shortens the grid walk, and only by a little here
        walk_on_grid = route["cost"] - route["rides"]
        assert route["length"] <= walk_on_grid + 1e-9 <= 1.05 * route["length"]
        assert route["floors"][0] == 0 and route["floors"][-1] == 2
        assert route["instructions"][0].startswith("Start") and route["instructions"][-1].startswith("Arrive")
    assert not any("stairs" in text for text in step_free["instructions"])
    assert any("elevator" in text for text in step_free["instructions"])
