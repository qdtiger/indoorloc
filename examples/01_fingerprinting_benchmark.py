"""WiFi fingerprinting on UJIIndoorLoc: seven classic methods and an MLP, one protocol, error CDFs.

    python examples/01_fingerprinting_benchmark.py              # about 45 s (25 s without the MLP)
    python examples/01_fingerprinting_benchmark.py --no-mlp     # skip the torch model

The official UJIIndoorLoc split (19,937 training scans, 1,111 validation scans, 520 access
points, 3 buildings, 5 floors; Torres-Sospedra et al., IPIN 2014, CC BY 4.0) is downloaded
on first use and sha256-checked. Every method is a registry name plus an L2 preprocessing
step, built with ``create_model(name, preprocess=...)``, fitted on the training file and
scored with ``evaluate`` on the validation file (``--no-download`` refuses the download):

    k-NN (k=5), WKNN (k=5)            fill "not heard" with -104 dBm
    WKNN, exponential representation  Torres-Sospedra et al. 2015, alpha = 24
    Horus                              Youssef & Agrawala 2005 (per-AP Gaussian likelihood)
    random forest, extra trees         100 trees, random_state = 0 (scikit-learn)
    ensemble                           median of WKNN, random forest and Horus
    MLP (optional)                     512-256-128, random_state = 0 (torch, CPU)

Units: UJIIndoorLoc positions are EPSG:3857 (Web Mercator) metres, which is what published
UJIIndoorLoc numbers use; at the campus's latitude one such metre is 0.766 ground metres
(``meta["ground_scale"]``). The figure and the ground-metre columns use ``scale=0.766``; the
EPSG:3857 column is checked against the recorded run of the same cell in
``benchmarks/results/ujiindoorloc.json`` (the repository's benchmark matrix, made with the
``indoorloc benchmark`` command line), when that file is present. Times are wall-clock on the
machine that runs the script and vary between runs; the errors do not (every method here is
deterministic).

The MLP is deterministic for identical input (and torch thread count), but its training is
chaotic in the last bit: the benchmark command line reaches the same training rows through a
pooled train+test table, whose position mean differs from this script's by a relative 4e-17,
and scores 11.294 EPSG:3857 m where this script scores 11.738; ``random_state`` 1 and 2 give
12.135 and 11.709 (measured on 2026-09-28, 8 threads). Read any single-seed MLP number on this
split as uncertain by about +-0.4 m, beyond the bootstrap interval of the test set.

Output: ``assets/figures/fingerprinting_cdf.png`` (``--out`` to change the folder):
(a) error CDFs in ground metres (k-NN and extra trees, whose curves nearly coincide with WKNN
and random forest, are left out of this panel only); (b) mean error with its 95 % bootstrap
interval (resampling test scans), and the floor hit rate, of every method.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

import indoorloc as iloc

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "assets" / "figures"
RECORD = ROOT / "benchmarks" / "results" / "ujiindoorloc.json"
# One style for every example figure: white background, readable sizes, hairline grid, and one
# categorical palette in a fixed slot order. Neighbouring slots pass colour-vision-deficiency checks,
# but only the first three stay apart as ALL pairs; a figure that shows more series at once picks
# a subset and adds a second cue (dashes or labels), as 01 and 04 do.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, MUTED = "#0b0b0b", "#52514e"
STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 9.0, "axes.titlesize": 10.0, "axes.labelsize": 9.0,
    "legend.fontsize": 8.0, "xtick.labelsize": 8.0, "ytick.labelsize": 8.0, "axes.titleweight": "bold",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#dddcd8",
    "grid.linewidth": 0.6, "grid.linestyle": "-", "lines.linewidth": 1.8, "lines.markersize": 5.0,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
}

DISPLAY = {"ujiindoorloc": "UJIIndoorLoc", "synthetic_office": "SyntheticOffice (simulated)"}
FILL_DBM = -104.0  # UJIIndoorLoc's weakest reading is -104 dBm; "not heard" is filled with it
# label -> (registry name, parameters, preprocessing, the matching cell of the benchmark matrix)
METHODS = {
    "k-NN (k=5)": ("knn", {"k": 5}, "fill", ("knn", "fill")),
    "WKNN (k=5)": ("wknn", {"k": 5}, "fill", ("wknn", "fill")),
    "WKNN, exponential repr.": ("wknn", {"k": 5}, "exponential", ("wknn", "exponential")),
    "Horus": ("horus", {}, "fill", ("horus", "fill")),
    "random forest": ("rf", {"n_jobs": 8, "random_state": 0}, "fill", ("rf(n_jobs=8)", "fill")),
    "extra trees": ("extratrees", {"n_jobs": 8, "random_state": 0}, "fill", ("extratrees(n_jobs=8)", "fill")),
    "ensemble (median)": ("ensemble", {"localizers": ["wknn", "rf", "horus"], "combine": "median"}, "fill",
                          ('ensemble(localizers=["wknn","rf","horus"],combine="median")', "fill")),
    "MLP": ("mlp", {"random_state": 0, "device": "cpu"}, "fill", ("mlp", "fill")),
}
# A method keeps its colour in both panels. Six CDFs cross on one plot, so every pair of colours must stay
# apart (not only neighbours): blue, yellow, pink, dark green and purple pass the all-pairs colour-vision
# checks; green-pink is the one weak pair under colour-vision deficiency, so the green MLP line is also
# dashed. The two methods left out of the CDF (near-duplicates of a drawn one) are grey in panel (b).
CDF_HIDDEN = ("k-NN (k=5)", "extra trees")
COLOUR = {"WKNN, exponential repr.": PALETTE[0], "ensemble (median)": PALETTE[6], "WKNN (k=5)": PALETTE[5],
          "Horus": PALETTE[3], "random forest": PALETTE[4], "MLP": PALETTE[2], "k-NN (k=5)": MUTED,
          "extra trees": MUTED}
DASHED = ("MLP",)


def preprocessing(kind: str):
    if kind == "fill":
        return iloc.FillMissing(FILL_DBM)
    return iloc.ExponentialRepresentation(alpha=24.0)  # learns the minimum reading from the training scans


def recorded_means() -> dict:
    """(method spec, preprocess) -> recorded pooled mean error of the official-split table."""
    if not RECORD.is_file():
        return {}
    payload = json.loads(RECORD.read_text())
    table = next((t for t in payload["tables"] if t["id"] == "official"), None)
    return {} if table is None else {(c["method"], c["preprocess"]): c["pooled"]["mean_error"]
                                     for c in table["cells"] if c.get("status") == "ok"}


def figure(results: dict, path: Path, unit: str, title: str, cdf_hidden=CDF_HIDDEN) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from indoorloc.evaluation.plot import plot_cdf

    names = list(results)
    shown = [n for n in sorted(names, key=lambda n: results[n]["mean"]) if n not in cdf_hidden] or names
    colour = {n: COLOUR[n] if n in shown else MUTED for n in names}
    if all(COLOUR[n] == MUTED for n in shown):  # a quick run of hidden methods only: draw them in colour
        colour.update({n: PALETTE[i] for i, n in enumerate(shown)})
    with plt.rc_context(STYLE):
        fig, (ax_cdf, ax_bar) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.25, 1.0],
                                                                                        "wspace": 0.55})
        for name in shown:  # one call per method keeps each method's colour fixed
            plot_cdf(results[name]["errors"], ax=ax_cdf, label=name, unit=unit, color=colour[name], lw=1.6,
                     ls="--" if name in DASHED else "-")
        hidden = [n for n in names if n not in shown]
        ax_cdf.set_xlim(0, 25)
        ax_cdf.legend(loc="lower right", title="lowest mean error first" + (
            f"\nnot drawn: {', '.join(hidden)} (see b)" if hidden else ""), title_fontsize=7.5)
        ax_cdf.set_title("(a) Error CDF", loc="left")
        ax_cdf.grid(True)

        order = sorted(names, key=lambda n: results[n]["mean"])
        y = np.arange(len(order))[::-1]
        for yi, name in zip(y, order):
            r = results[name]
            lo, hi = r["mean_ci95"]
            ax_bar.plot([lo, hi], [yi, yi], color=colour[name], lw=2.6, solid_capstyle="round")
            ax_bar.plot(r["mean"], yi, "o", color=colour[name], ms=7, mec="white", mew=1.2)
            floor = "" if r["floor_accuracy"] is None else f"{r['floor_accuracy']:.1f} %"
            ax_bar.text(1.03, yi, f"{r['mean']:.2f}", transform=ax_bar.get_yaxis_transform(), va="center",
                        ha="left", fontsize=8, color=INK)
            ax_bar.text(1.22, yi, floor, transform=ax_bar.get_yaxis_transform(), va="center", ha="left",
                        fontsize=8, color=INK)
        top = len(order) - 0.35
        ax_bar.text(1.03, top, "mean", transform=ax_bar.get_yaxis_transform(), fontsize=8, color=MUTED, va="bottom")
        ax_bar.text(1.22, top, "floor", transform=ax_bar.get_yaxis_transform(), fontsize=8, color=MUTED, va="bottom")
        ax_bar.set_yticks(y, order)
        ax_bar.set_ylim(-0.6, len(order) - 0.2)
        ax_bar.set_xlabel(f"mean error [{unit}], 95 % bootstrap interval")
        ax_bar.grid(True, axis="x")
        ax_bar.grid(False, axis="y")
        ax_bar.set_title("(b) Mean error and floor hit rate", loc="left")
        fig.suptitle(title, fontsize=9, color=MUTED, y=1.0)
        fig.savefig(path)
        plt.close(fig)


def main(*, dataset: str = "ujiindoorloc", methods=None, mlp: bool | None = None, download: bool = True,
         out: Path | str = FIGURES, dataset_options: dict | None = None, torch_threads: int | None = 8,
         verbose: bool = True) -> dict:
    """Fit and score each method; return ``{label: metrics}`` (errors in ground metres).

    ``methods``: labels of ``METHODS`` (default: all; the MLP only if torch is installed and
    ``mlp`` is not False). ``dataset``/``dataset_options`` allow a quick run on another
    RSSI table with a train/test split, e.g. ``"synthetic_office"``. ``torch_threads``: the
    MLP's result depends on torch's thread count, so it is fixed (8, as in the benchmark
    matrix) before the MLP is trained; None leaves torch as it is.
    """
    start = time.perf_counter()
    if mlp is None:
        mlp = importlib.util.find_spec("torch") is not None
    chosen = list(METHODS) if methods is None else list(methods)
    if not mlp:
        chosen = [m for m in chosen if METHODS[m][0] != "mlp"]
    if torch_threads is not None and any(METHODS[m][0] == "mlp" for m in chosen):
        import torch

        torch.set_num_threads(int(torch_threads))
    train, test = iloc.load_dataset(dataset, download=download, **(dataset_options or {}))
    scale = float(train.meta.get("ground_scale", 1.0))
    unit = "m, ground" if "ground_scale" in train.meta else train.meta.get("pos_units", "m")
    record = recorded_means() if dataset == "ujiindoorloc" else {}
    results = {}
    for label in chosen:
        name, params, pre, cell = METHODS[label]
        t0 = time.perf_counter()
        model = iloc.create_model(name, preprocess=preprocessing(pre), **params).fit(train)
        t1 = time.perf_counter()
        pred = model.localize(test)
        t2 = time.perf_counter()
        native = iloc.evaluate(test, pred)               # dataset units (EPSG:3857 m for UJIIndoorLoc)
        ground = iloc.evaluate(test, pred, scale=scale)  # ground metres
        results[label] = {
            "errors": ground.errors, "mean": ground.mean_error, "median": ground.median_error,
            "p90": ground.p90_error, "mean_ci95": iloc.evaluation.bootstrap_ci(ground.errors, "mean"),
            "mean_native": native.mean_error, "recorded_mean_native": record.get(cell),
            "floor_accuracy": ground.floor_accuracy, "building_accuracy": ground.building_accuracy,
            "n": ground.n, "n_failed": ground.n_failed, "fit_s": t1 - t0, "predict_s": t2 - t1,
        }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "fingerprinting_cdf.png"
    split = "official split" if dataset == "ujiindoorloc" else "train/test split"
    title = (f"{DISPLAY.get(dataset, dataset)}, {split}: fit on {len(train):,} training scans, scored on "
             f"{len(test):,} held-out scans")
    if scale != 1:
        title += f"; EPSG:3857 errors x {scale:.3f} = ground metres"
    figure(results, path, unit, title)
    if verbose:
        print(f"{train.meta.get('name', dataset)}: {len(train):,} train / {len(test):,} test scans, "
              f"{train.X.shape[1]} features; errors x {scale:.4f} = {unit}")
        print(f"{'method':24s} {'mean':>6s} {'95% CI':>13s} {'median':>6s} {'P90':>6s} {'floor%':>7s} {'bldg%':>6s} "
              f"{'native':>7s} {'recorded':>8s} {'fit s':>6s} {'pred s':>6s}")
        for label, r in results.items():
            fl = "n/a" if r["floor_accuracy"] is None else f"{r['floor_accuracy']:.2f}"
            bd = "n/a" if r["building_accuracy"] is None else f"{r['building_accuracy']:.2f}"
            rec = "-" if r["recorded_mean_native"] is None else f"{r['recorded_mean_native']:.3f}"
            print(f"{label:24s} {r['mean']:6.3f} [{r['mean_ci95'][0]:5.2f},{r['mean_ci95'][1]:5.2f}] "
                  f"{r['median']:6.3f} {r['p90']:6.2f} {fl:>7s} {bd:>6s} {r['mean_native']:7.3f} {rec:>8s} "
                  f"{r['fit_s']:6.1f} {r['predict_s']:6.1f}")
        if record:
            same = [abs(r["mean_native"] - r["recorded_mean_native"]) < 1e-6 for r in results.values()
                    if r["recorded_mean_native"] is not None]
            differ = [label for label, r in results.items() if r["recorded_mean_native"] is not None
                      and abs(r["mean_native"] - r["recorded_mean_native"]) >= 1e-6]
            print(f"native-unit means equal to the recorded benchmark run: {sum(same)} of {len(same)}"
                  + (f" (differs: {', '.join(differ)}; see the module docstring)" if differ else ""))
        print(f"figure: {path}  ({time.perf_counter() - start:.1f} s)")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-mlp", action="store_true", help="skip the MLP (needs torch)")
    parser.add_argument("--no-download", action="store_true", help="fail instead of downloading the dataset")
    parser.add_argument("--out", type=Path, default=FIGURES, help="folder for the figure")
    args = parser.parse_args()
    main(mlp=False if args.no_mlp else None, download=not args.no_download, out=args.out)
