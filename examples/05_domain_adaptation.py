"""Cross-device WiFi fingerprinting on UJIIndoorLoc, with and without domain adaptation.

    python examples/05_domain_adaptation.py        # about 10 s

A radio map recorded with one set of phones is used to localize scans from other phones.
Three unsupervised remedies from the library, all fitted on UNLABELLED scans of the new
phones (the "adaptation" scans), are compared with doing nothing, under two protocols:

(a) Two phones, same places. In UJIIndoorLoc's training file, phones 13 and 14 surveyed 244
    common reference points (same building, floor, space and relative position). The radio
    map is phone A's scans at those points; phone B's scans at a random half of the points
    (seed 0) are the adaptation scans, its scans at the other half are scored. Both directions,
    13 -> 14 and 14 -> 13. Only the device differs, so this isolates device heterogeneity.
(b) Deployment. The radio map is the whole training file (16 phones); the target is every
    validation scan from the 9 phones that never appear in training, recorded about four months
    later. The first half (by time) of each phone's scans are its adaptation scans; the second
    half is scored. Device and time both differ, and the radio map is already multi-device.

Remedies (``LocalizerPipeline`` = preprocessing + WKNN, k = 5, "not heard" = -104 dBm):

    DeviceCalibration   signals: one global map per new phone, ``offset`` (x + b), ``linear``
                        (a x + b, Q-Q line) or ``quantile`` (histogram matching), learned from its
                        adaptation scans against the radio map (unpaired; Haeberlen et al. 2004,
                        Laoudias et al. 2013); applied to that phone's scans before WKNN
    CORAL               methods.transfer: the radio map re-coloured with the target covariance
                        (Sun et al., AAAI 2016); ``align_mean=True`` also moves its mean
    TCA                 methods.transfer: a 30-dimensional embedding in which the domain means
                        match (Pan et al., IEEE TNN 2011), linear or RBF kernel

The figure shows each remedy's change of the mean error against no adaptation, with a 95 %
paired bootstrap interval that resamples scored positions (scans at one position are not
independent), and prints the absolute numbers and floor hit rates. Errors are in ground
metres (EPSG:3857 x 0.766). Measured data, deterministic.

What the numbers say (measured 2026-09-28; the script prints them): with two phones at the same
places every remedy lowers the mean error: by 0.1-0.7 m for 13 -> 14 (5.55 m without
adaptation; the intervals of the linear calibration and linear TCA exclude zero, the offset
calibration's ends at 0.00, the others include zero) and by 0.7-1.3 m for 14 -> 13 (5.65 m;
every interval excludes zero); the offsets learned in the two directions agree (-4.1 dB and
+5.0 dB). In the deployment setting only the per-phone linear calibration helps (8.51 ->
8.05 m, interval -0.84 to -0.10 m); CORAL makes WKNN much WORSE (+7.1 m, +9.8 m with
``align_mean``) and TCA slightly worse (+0.6 to +0.7 m, intervals that include zero). A likely
reason, not tested here: 355 adaptation scans from nine phones, four months later and unevenly
spread over three buildings give a poor estimate of a 520 x 520 target covariance, and the
target statistics mix phone, time and coverage differences. Adaptation is not a free win;
measure it on your own protocol.

Output: ``assets/figures/domain_adaptation.png`` (``--out`` to change the folder).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import indoorloc as iloc
from indoorloc.methods.transfer import CORAL, TCA
from indoorloc.signals import Compose, DeviceCalibration, FillMissing

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "assets" / "figures"
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

FILL_DBM = -104.0
SEED = 0
REMEDIES = ("no adaptation", "DeviceCalibration (offset)", "DeviceCalibration (linear)",
            "DeviceCalibration (quantile)", "CORAL", "CORAL (align_mean)", "TCA (linear, 30)", "TCA (RBF, 30)")
FEATURE_LEVEL = {"CORAL": CORAL(), "CORAL (align_mean)": CORAL(align_mean=True),
                 "TCA (linear, 30)": TCA(30), "TCA (RBF, 30)": TCA(30, kernel="rbf")}


# ------------------------------------------------------------------------------------ protocols
def two_phone_split(train, phone_a: int, phone_b: int, seed: int = SEED):
    """Radio map = phone A at the reference points both phones surveyed; phone B's scans at a
    random half of those points = adaptation, at the other half = scored."""
    key = np.stack([train.building, train.floor, train.groups["space"], train.groups["relative_position"]], axis=1)
    point = np.unique(key, axis=0, return_inverse=True)[1].reshape(-1)
    device = train.groups["device"]
    shared = np.intersect1d(point[device == phone_a], point[device == phone_b])
    adapt_points = np.random.default_rng(seed).permutation(shared)[: len(shared) // 2]
    phone_b_rows = (device == phone_b) & np.isin(point, shared)
    return (train[(device == phone_a) & np.isin(point, shared)], train[phone_b_rows & np.isin(point, adapt_points)],
            train[phone_b_rows & ~np.isin(point, adapt_points)], len(shared))


def deployment_split(train, test):
    """Radio map = all training scans; target = validation scans of phones unseen in training,
    first half of each phone's scans (by time) = adaptation, second half = scored."""
    target = test[~np.isin(test.groups["device"], np.unique(train.groups["device"]))]
    adapt = np.zeros(len(target), dtype=bool)
    for phone in np.unique(target.groups["device"]):
        rows = np.flatnonzero(target.groups["device"] == phone)
        rows = rows[np.argsort(target.groups["time"][rows], kind="stable")]
        adapt[rows[: len(rows) // 2]] = True
    return train, target[adapt], target[~adapt]


# ------------------------------------------------------------------------------------ remedies
def run_remedies(source, adapt, scored, scale: float) -> dict:
    """Every remedy on one protocol: ``{remedy: {"errors", "result", ...}}``."""
    out = {}
    base = iloc.create_model("wknn", k=5, preprocess=FillMissing(FILL_DBM)).fit(source)
    out["no adaptation"] = {"pred": base.localize(scored)}
    for method in ("offset", "linear", "quantile"):
        X = np.array(scored.X, dtype=np.float32)
        offsets = {}
        for phone in np.unique(scored.groups["device"]):  # one calibration per new phone
            cal = DeviceCalibration(method=method).fit(adapt.X[adapt.groups["device"] == phone], reference=source.X)
            rows = scored.groups["device"] == phone
            X[rows] = cal.transform(scored.X[rows])
            if method != "quantile":
                offsets[int(phone)] = (float(cal.coef_), float(cal.intercept_))
        out[f"DeviceCalibration ({method})"] = {"pred": base.localize(scored.replace(X=X)), "maps": offsets}
    for name, adapter in FEATURE_LEVEL.items():
        model = iloc.create_model("wknn", k=5, preprocess=Compose([FillMissing(FILL_DBM), adapter]))
        model.fit(source, preprocess__target=adapt.X)  # unlabelled target scans; Compose fills them first
        out[name] = {"pred": model.localize(scored)}
    for r in out.values():
        r["result"] = iloc.evaluate(scored, r["pred"], scale=scale)
    return out


def paired_ci(errors, reference, clusters, *, n_boot: int = 2000, seed: int = SEED) -> tuple[float, float, float]:
    """Mean of ``errors - reference`` and its 95 % bootstrap interval, resampling whole clusters."""
    diff = np.asarray(errors) - np.asarray(reference)
    ids, inverse = np.unique(clusters, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    sums, counts = np.bincount(inverse, weights=diff), np.bincount(inverse)
    draws = np.random.default_rng(seed).integers(0, len(ids), size=(n_boot, len(ids)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


# ------------------------------------------------------------------------------------ figure
def figure(panels: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(panels), figsize=(12.0, 4.6), sharey=True, gridspec_kw={"wspace": 0.08})
        axes = np.atleast_1d(axes)
        y = np.arange(len(REMEDIES))[::-1]
        slot = 0  # every run keeps its own colour across the panels
        for ax, (title, runs) in zip(axes, panels.items()):
            ax.axvline(0.0, color=INK, lw=1.1)
            n_runs = len(runs)
            for j, (label, run) in enumerate(runs.items()):
                colour = PALETTE[slot]
                slot += 1
                offset = ((n_runs - 1) / 2 - j) * 0.22
                base = run["no adaptation"]["result"]
                for yi, name in zip(y, REMEDIES):
                    mean, lo, hi = run[name]["delta"]
                    ax.plot([lo, hi], [yi + offset] * 2, color=colour, lw=2.2, solid_capstyle="round")
                    ax.plot(mean, yi + offset, "o", color=colour, ms=6, mec="white", mew=1.0,
                            label=f"{label}: no adaptation {base.mean_error:.2f} m, floor {base.floor_accuracy:.1f} %"
                            if name == REMEDIES[0] else None)
            ax.set_yticks(y, REMEDIES)
            ax.set_xlabel("change of mean error vs. no adaptation [m, ground]  (< 0: better)")
            ax.set_title(title, loc="left")
            ax.grid(True, axis="x")
            ax.grid(False, axis="y")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=7.5)
        fig.suptitle("UJIIndoorLoc (measured), WKNN (k = 5); dots: mean change, bars: 95 % paired bootstrap interval "
                     "over scored positions", fontsize=9, color=MUTED, y=1.0)
        fig.savefig(path)
        plt.close(fig)


# ------------------------------------------------------------------------------------ main
def main(*, phones=(13, 14), download: bool = True, out: Path | str = FIGURES, n_boot: int = 2000,
         verbose: bool = True) -> dict:
    """Run both protocols; return ``{panel: {run: {remedy: {mean, delta, floor, ...}}}}``."""
    start = time.perf_counter()
    train, test = iloc.load_dataset("ujiindoorloc", download=download)
    scale = float(train.meta["ground_scale"])
    a, b = phones
    panels, sizes = {}, {}
    two = {}
    for src, tgt in ((a, b), (b, a)):
        source, adapt, scored, n_points = two_phone_split(train, src, tgt)
        two[f"phone {src} -> {tgt}"] = (source, adapt, scored)
        sizes[f"phone {src} -> {tgt}"] = (len(source), len(adapt), len(scored), n_points)
    source, adapt, scored = deployment_split(train, test)
    runs = {"(a)": two, "(b)": {"16 training phones -> 9 new phones": (source, adapt, scored)}}
    sizes["16 training phones -> 9 new phones"] = (len(source), len(adapt), len(scored),
                                                   len(np.unique(scored.groups["device"])))
    results = {}
    for key, protocol in runs.items():
        panel = {}
        for label, (source, adapt, scored) in protocol.items():
            run = run_remedies(source, adapt, scored, scale)
            ref = run["no adaptation"]["result"].errors
            for r in run.values():
                r["delta"] = paired_ci(r["result"].errors, ref, scored.pos, n_boot=n_boot)
            panel[label] = run
        results[key] = panel
    title_a = f"(a) Two phones, same {sizes[next(iter(two))][3]} reference points"
    panels = {title_a: results["(a)"], "(b) Deployment: new phones, 4 months later": results["(b)"]}

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "domain_adaptation.png"
    figure(panels, path)
    summary = {label: {name: {"mean_error": r["result"].mean_error, "median_error": r["result"].median_error,
                              "floor_accuracy": r["result"].floor_accuracy, "delta": r["delta"],
                              **({"maps": r["maps"]} if "maps" in r else {})}
                       for name, r in run.items()}
               for panel in results.values() for label, run in panel.items()}
    if verbose:
        for label, run in summary.items():
            n_src, n_adapt, n_scored, extra = sizes[label]
            what = "shared points" if "->" in label and "phones" not in label else "new phones"
            print(f"\n{label}: radio map {n_src:,} scans, adaptation {n_adapt:,} unlabelled scans, "
                  f"scored {n_scored:,} scans ({extra} {what})")
            print(f"  {'remedy':30s} {'mean':>6s} {'median':>7s} {'floor%':>7s} {'change':>7s} {'95% CI':>16s}")
            for name, r in run.items():
                d, lo, hi = r["delta"]
                print(f"  {name:30s} {r['mean_error']:6.2f} {r['median_error']:7.2f} {r['floor_accuracy']:7.1f} "
                      f"{d:+7.2f} [{lo:+6.2f}, {hi:+6.2f}]")
            maps = run["DeviceCalibration (offset)"]["maps"]
            print("  learned offsets b [dB] per new phone: " + ", ".join(f"{p}: {v[1]:+.1f}" for p, v in maps.items()))
        print(f"\nfigure: {path}  ({time.perf_counter() - start:.1f} s)")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-download", action="store_true", help="fail instead of downloading UJIIndoorLoc")
    parser.add_argument("--out", type=Path, default=FIGURES, help="folder for the figure")
    args = parser.parse_args()
    main(download=not args.no_download, out=args.out)
