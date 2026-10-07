"""WiFi CSI on real data (HALOC): raw vs sanitized phase, then CSI fingerprints -> WKNN.

    python examples/03_csi_pipeline.py        # about 50 s; downloads HALOC.zip (28.6 MB) once

HALOC (Strohmayer & Kampel, ICLR 2024 Tiny Papers; Zenodo 10.5281/zenodo.10715595, CC BY 4.0,
non-commercial research use requested): an ESP32-S3 behind a directional antenna records the
CSI of about 100 packets/s from a transmitter while one person walks up and down a 20 m
hallway; every packet carries the person's position. ``load_dataset("haloc")`` returns the
authors' split: train = sequences 0-3 (96,491 packets), test = sequence 5 (14,277 packets).
``X`` is complex CSI ``(N, 1, 1, 52)`` over the 52 L-LTF data subcarriers (-26..26 without 0).

1. L2, phase. The raw phase of a commodity receiver carries a slope across subcarriers
   (sampling-time offset, packet-detection delay) and an offset (carrier frequency and phase
   offsets) that change from packet to packet. ``CSIPhaseSanitize`` unwraps the phase along the
   subcarriers and removes the least-squares line in the subcarrier index (Sen et al., MobiSys
   2012). The figure shows 20 consecutive packets (0.2 s) before and after, and the spread of
   the phase over every window of 10 consecutive packets of the test sequence (circular
   standard deviation per subcarrier, averaged over subcarriers).
2. L3, fingerprints. ``create_model("wknn", k=5, preprocess=...)`` with the amplitude
   ``CSIAmplitude()`` (linear |H|; HALOC's raw I/Q are not gain-calibrated) and with the
   sanitized phase ``CSIPhaseSanitize(output="phase")``, fitted on the training sequences and
   scored on the test sequence with ``evaluate`` (3-D error, metres). Context: a model that
   always answers the mean training position.
3. L5, tracking. The per-packet |CSI| fixes are noisy; ``KalmanTracker.smooth`` (constant
   velocity, measurement noise from each fix's ``Prediction.spread``, RTS smoother) turns them
   into a track. Its two parameters (``process_noise``, ``min_meas_std``) are chosen on the
   authors' validation sequence 4 from a small grid, never on the test sequence. The RTS pass
   uses the whole recording (offline); ``KalmanTracker.filter`` is the causal variant, also
   printed. The grid holds process noises that allow a walker's speed to change by 0.3-2 m/s
   within 10 s. Both the validation and the test walk are one slow, straight pass (about
   0.14 m/s), so the lowest process noise wins, and lower values than the grid's would score
   better still on these two walks (measured on sequence 4: 1.70 m at q = 0.001 against 1.78 m
   at q = 0.01, ``min_meas_std`` 1) while failing on a walk that turns around. Read the tracking
   gain as specific to this recording.

Result (measured 2026-09-28): WKNN on |CSI| 3.669 m mean error (median 2.847 m), on the
sanitized phase 3.807 m, the mean training position 4.897 m; the Kalman filter on the |CSI|
fixes 2.394 m and the RTS track 1.971 m (median 1.537 m). The phase spread over 10 packets is
1.899 rad raw and 0.026 rad sanitized (medians over 1,427 windows).

In the raw ESP32-S3 phase of (a), consecutive packets differ mostly by a constant offset
(carrier phase) and only a little in slope: this receiver already removes most of the timing
offset. The sanitized phase of (b) is stable to 0.03 rad within 0.1 s.

The |CSI| WKNN cell reproduces the repository's benchmark matrix (``benchmarks/results/
haloc.json``, mean 3.669 m) when that file is present. Measured data only; nothing here is
simulated. The person walks along x (0-20 m); y and z vary by 1.8 m and 0.1 m, so the error is
almost entirely along the hallway.

Output: ``assets/figures/csi_pipeline.png`` (``--out`` to change the folder).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import indoorloc as iloc

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "assets" / "figures"
RECORD = ROOT / "benchmarks" / "results" / "haloc.json"
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

KALMAN_GRID = {"process_noise": (0.01, 0.1, 0.5), "min_meas_std": (1.0, 3.0)}  # chosen on sequence 4
FEATURES = {  # label -> L2 preprocessing of the complex CSI
    "|CSI| (linear amplitude)": iloc.CSIAmplitude(),
    "sanitized phase": iloc.CSIPhaseSanitize(output="phase"),
}


def circular_std(phase: np.ndarray, axis: int = 0) -> np.ndarray:
    """sqrt(-2 ln R), R = |mean exp(j phase)| (Mardia & Jupp 2000); 0 = identical angles."""
    R = np.abs(np.mean(np.exp(1j * phase), axis=axis))
    return np.sqrt(-2.0 * np.log(np.clip(R, 1e-12, 1.0)))


def window_spread(phase: np.ndarray, width: int) -> np.ndarray:
    """Mean over subcarriers of the circular std over ``width`` consecutive packets, per window."""
    n = len(phase) // width
    blocks = phase[: n * width].reshape(n, width, -1)
    return circular_std(blocks, axis=1).mean(axis=1)


def recorded_wknn_mean() -> float | None:
    if not RECORD.is_file():
        return None
    for table in json.loads(RECORD.read_text())["tables"]:
        for cell in table["cells"]:
            if cell["method"] == "wknn" and cell.get("status") == "ok" and table["id"] == "official":
                return float(cell["pooled"]["mean_error"])
    return None


def figure(test, raw_phase, clean_phase, start, spreads, results, baseline, track, track_result, path: Path,
           span_s: float | None, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from indoorloc.evaluation.plot import plot_cdf

    k = np.asarray(test.meta["subcarriers"])
    rows = slice(start, start + 20)
    t = np.asarray(test.groups["time"], dtype=np.float64)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(12.0, 7.4))
        grid = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.05], hspace=0.42, wspace=1.1)
        ax_raw, ax_clean, ax_spread = (fig.add_subplot(grid[0, 2 * i:2 * i + 2]) for i in range(3))
        ax_track, ax_cdf = fig.add_subplot(grid[1, :4]), fig.add_subplot(grid[1, 4:])

        for line in np.unwrap(raw_phase[rows], axis=1):
            ax_raw.plot(k, line, color=PALETTE[0], lw=1.0, alpha=0.75)
        ax_raw.set_title(f"(a) Raw phase, 20 packets ({t[start + 19] - t[start]:.2f} s)", loc="left")
        ax_raw.set_ylabel("phase, unwrapped [rad]")
        for line in clean_phase[rows]:
            ax_clean.plot(k, line, color=PALETTE[1], lw=1.0, alpha=0.75)
        ax_clean.set_title("(b) After CSIPhaseSanitize", loc="left")
        ax_clean.set_ylabel("phase [rad]")
        for ax in (ax_raw, ax_clean):
            ax.set_xlabel("subcarrier index")

        bins = np.logspace(-2.5, 0.6, 50)
        for i, (name, spread) in enumerate(spreads.items()):
            ax_spread.hist(spread, bins=bins, color=PALETTE[i], alpha=0.8,
                           label=f"{name}: median {np.median(spread):.3f}")
        ax_spread.set_xscale("log")
        ax_spread.set_ylim(0, ax_spread.get_ylim()[1] * 1.3)
        ax_spread.set_xlabel("circular std over 10 packets [rad]")
        ax_spread.set_ylabel("windows")
        ax_spread.set_title("(c) Phase spread, test sequence", loc="left")
        ax_spread.legend(loc="upper center")

        span = t[-1] if span_s is None else span_s
        shown = t <= span
        first = next(iter(results))
        ax_track.plot(t[shown], results[first]["pred"].pos[shown, 0], ".", color=PALETTE[2], ms=1.4, alpha=0.35,
                      label=f"WKNN on {first}, one fix per packet", zorder=1)
        ax_track.plot(t[shown], track.pos[shown, 0], color=PALETTE[6], lw=1.8,
                      label="the same fixes, Kalman RTS smoothed (L5)", zorder=3)
        ax_track.plot(t[shown], test.pos[shown, 0], color=INK, lw=1.6, label="true position", zorder=4)
        ax_track.set_xlim(0, span)
        ax_track.set_xlabel("time in test sequence 5 [s]")
        ax_track.set_ylabel("position along the hallway, x [m]")
        ax_track.set_title("(d) The test walk: fixes and track (x only)", loc="left")
        ax_track.legend(loc="upper left", markerscale=6)

        for i, (name, r) in enumerate(results.items()):
            plot_cdf(r["result"], ax=ax_cdf, label=f"WKNN, {name}", color=PALETTE[2 + i], lw=1.6)
        plot_cdf(track_result, ax=ax_cdf, label="WKNN |CSI| + Kalman RTS", color=PALETTE[6], lw=1.6)
        plot_cdf(baseline, ax=ax_cdf, label="mean training position", color=MUTED, lw=1.2, ls="--")
        ax_cdf.set_xlim(0, 12)
        ax_cdf.set_title("(e) Error CDF, test sequence", loc="left")
        ax_cdf.legend(loc="lower right", fontsize=7.5)
        fig.suptitle(title, fontsize=9, color=MUTED, y=0.995)
        fig.savefig(path)
        plt.close(fig)


def main(*, download: bool = True, out: Path | str = FIGURES, sequences=None, window_start_s: float = 30.0,
         span_s: float | None = None, kalman_grid: dict | None = None, every: int = 1, verbose: bool = True) -> dict:
    """Run the three parts; return the printed numbers. ``sequences`` (e.g. ``[0, 4, 5]``) loads
    a subset of HALOC's sequences for a quick run; the train, valid and test splits must remain.
    ``span_s``: seconds of the test walk drawn in panel (d) (None = all of it). ``kalman_grid``:
    the candidate Kalman parameters (default ``KALMAN_GRID``). ``every``: keep every n-th packet of
    each split (a quicker, coarser run; 1 = all packets, as reported in the module docstring)."""
    start_time = time.perf_counter()
    options = {} if sequences is None else {"sequences": sequences}
    train, valid, test = (t[::every] for t in iloc.load_dataset("haloc", split=("train", "valid", "test"),
                                                                 download=download, **options))

    # 1. phase sanitization (L2)
    raw_phase = np.angle(test.X[:, 0, 0, :]).astype(np.float64)
    clean_phase = iloc.CSIPhaseSanitize(output="phase").transform(test).X[:, 0, 0, :].astype(np.float64)
    spreads = {"raw": window_spread(raw_phase, 10), "sanitized": window_spread(clean_phase, 10)}
    t = np.asarray(test.groups["time"])
    start = int(min(np.searchsorted(t, window_start_s), len(test) - 20))

    # 2. fingerprints (L2 -> L3 -> L4)
    results = {}
    for name, pre in FEATURES.items():
        t0 = time.perf_counter()
        model = iloc.create_model("wknn", k=5, preprocess=pre).fit(train)
        pred = model.localize(test)
        results[name] = {"model": model, "pred": pred, "result": iloc.evaluate(test, pred), "n_train": len(train),
                         "seconds": time.perf_counter() - t0}
    centroid = np.broadcast_to(train.pos.mean(axis=0), test.pos.shape)
    baseline = iloc.evaluate(test.pos, centroid)

    # 3. tracking (L5): Kalman parameters chosen on the validation sequence, then applied to the test one
    amplitude = next(iter(FEATURES))
    model = results[amplitude]["model"]  # the |CSI| WKNN of part 2, fitted on the training sequences only
    fixes_valid, t_valid = model.localize(valid), np.asarray(valid.groups["time"], dtype=np.float64)
    candidates = KALMAN_GRID if kalman_grid is None else kalman_grid
    grid = [(q, m) for q in candidates["process_noise"] for m in candidates["min_meas_std"]]
    scores = {qm: iloc.evaluate(valid, iloc.KalmanTracker(process_noise=qm[0], min_meas_std=qm[1])
                                .smooth(fixes_valid, t_valid)).mean_error for qm in grid}
    q, m = min(grid, key=lambda qm: (scores[qm], qm))
    t_test = np.asarray(test.groups["time"], dtype=np.float64)
    fixes = results[amplitude]["pred"]
    tracks = {"filter": iloc.KalmanTracker(process_noise=q, min_meas_std=m).filter(fixes, t_test),
              "smooth": iloc.KalmanTracker(process_noise=q, min_meas_std=m).smooth(fixes, t_test)}
    tracking = {"process_noise": q, "min_meas_std": m, "valid_mean_error": {f"{k[0]}/{k[1]}": v for k, v in
                                                                             scores.items()},
                **{name: iloc.evaluate(test, track) for name, track in tracks.items()}}

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "csi_pipeline.png"
    title = (f"HALOC (measured, ESP32-S3): trained on sequences {', '.join(map(str, train.meta['sequences']))} "
             f"({len(train):,} packets), tested on sequence 5 ({len(test):,} packets)"
             + (f", 1 packet in {every} kept" if every > 1 else ""))
    figure(test, raw_phase, clean_phase, start, spreads, results, baseline, tracks["smooth"], tracking["smooth"],
           path, span_s, title)
    summary = {"phase_spread_median": {k: float(np.median(v)) for k, v in spreads.items()},
               "n_windows": len(spreads["raw"]),
               "wknn": {name: r["result"].to_dict() for name, r in results.items()},
               "baseline": baseline.to_dict(), "recorded_wknn_amplitude_mean": recorded_wknn_mean(),
               "kalman": {"process_noise": q, "min_meas_std": m, "valid_mean_error": tracking["valid_mean_error"],
                          "filter": tracking["filter"].to_dict(), "smooth": tracking["smooth"].to_dict()}}
    if verbose:
        print(f"HALOC: train {len(train):,} packets (sequences {train.meta.get('sequences')}), test {len(test):,} "
              f"packets ({t[-1] - t[0]:.1f} s), X {test.X.shape[1:]} {test.X.dtype}")
        print(f"phase spread over {summary['n_windows']:,} windows of 10 packets (circular std, rad, median): "
              f"raw {summary['phase_spread_median']['raw']:.3f}, sanitized "
              f"{summary['phase_spread_median']['sanitized']:.4f}")
        for name, r in results.items():
            print(f"WKNN (k=5) on {name:26s}: {r['result']}  [{r['seconds']:.1f} s]")
        print(f"mean training position (context)        : {baseline}")
        print("Kalman parameters on validation sequence 4 (mean error of the RTS track, m): "
              + ", ".join(f"q={k} min_std={v}: {e:.3f}" for (k, v), e in scores.items()))
        print(f"chosen process_noise={q}, min_meas_std={m}; on the test sequence:")
        print(f"  |CSI| WKNN + Kalman filter (causal)    : {tracking['filter']}")
        print(f"  |CSI| WKNN + Kalman RTS (offline)      : {tracking['smooth']}")
        rec = summary["recorded_wknn_amplitude_mean"]
        if rec is not None:
            now = results["|CSI| (linear amplitude)"]["result"].mean_error
            print(f"benchmark matrix, WKNN |CSI| mean: {rec:.4f} m (this run {now:.4f} m)")
        print(f"figure: {path}  ({time.perf_counter() - start_time:.1f} s)")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-download", action="store_true", help="fail instead of downloading HALOC")
    parser.add_argument("--out", type=Path, default=FIGURES, help="folder for the figure")
    args = parser.parse_args()
    main(download=not args.no_download, out=args.out)
