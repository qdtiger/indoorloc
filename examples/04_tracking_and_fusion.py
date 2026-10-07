"""Tracking and sensor fusion on real smartphone traces (ILC 2020, site1 / F1): WiFi, Kalman, PDR, particle filter.

    python examples/04_tracking_and_fusion.py      # about 12 s; downloads the site1/F1 files (~250 MB) once

Data: the Indoor Location Competition 2.0 sample (Microsoft Research and XYZ10; Hu et al.,
MobiCom 2023, DOI 10.1145/3570361.3592507; MIT licence), floor F1 of site 1, a shopping mall:
120 traces recorded with Android phones (accelerometer, gyroscope, magnetometer and rotation
vector at 50 Hz, WiFi scans every few seconds) along paths whose surveyor-labelled waypoints
are the ground truth, plus a GeoJSON floor plan (``meta["floor_plan"]``, walls in metres).

Protocol: leave one trajectory out. Every 12th trace (10 traces spread over the survey
sessions, 67 waypoints) is a test trace; for each of them every model is fitted on the other
119 traces only. Errors are measured at the waypoints after the test trace's first usable
WiFi scan (the fused and WiFi-only tracks exist only from then on); every method is scored on
the same waypoints. The same split and settings as ``tests/apps/test_apps_realdata.py``,
whose models are trained on the other 110 traces instead of 119.

Methods, from the layers they use:

    WiFi WKNN (L2 + L3)     FillMissing(-100) + WKNN (k=5) on scans that hear at least 5 APs;
                            the training positions are the waypoints interpolated in time
    + Kalman filter (L5)    KalmanTracker (constant velocity, noise from Prediction.spread), causal
    + Kalman RTS (L5)       the same with the Rauch-Tung-Striebel smoother (offline)
    PDR (L5)                StepDetector + Weinberg step length (constant fitted on the training
                            traces' walked distances) + the phone's rotation-vector heading, its
                            offset to the map fitted on the training traces; STARTED AT THE TRUE
                            FIRST WAYPOINT, which the other methods do not get
    PDR + WiFi (L5)         PDRFusion: a 1,000-particle filter, PDR steps as the motion model,
                            WKNN fixes as measurements, unknown start and heading offset
    PDR + WiFi + map (L5)   the same, particles that cross a wall of the floor plan die

Result (measured 2026-09-28, re-run 2026-09-29 with identical values; mean / median error over
the 67 waypoints, then the change of the mean against WiFi WKNN with a 95 % paired bootstrap
interval that resamples whole test traces): WiFi WKNN 6.95 / 5.79 m; + Kalman filter 5.83 /
5.28 m, -1.12 m [-2.39, -0.37]; + Kalman RTS 5.20 / 4.78 m, -1.75 m [-3.47, -0.67]; PDR from
the true start 4.99 / 4.94 m, -1.97 m [-4.11, -0.34]; PDR + WiFi 5.98 / 5.28 m, -0.97 m
[-2.29, +0.06]; PDR + WiFi + map 5.85 / 5.06 m, -1.10 m [-2.32, -0.21]. So the Kalman tracks
and the map-constrained fusion improve on the raw fixes by more than the trace-to-trace
variation; fusion without the map does not clearly do so, and the floor plan's own effect on
the fused error (0.13 m) is far below what ten traces resolve. Per trace the ranking changes
(the script prints the table). Ten traces make a coarse bootstrap: read the intervals as a
rough guide, not a test.

The Kalman RTS track is offline (it uses later fixes too); the Kalman filter and both particle
filters are causal. The tracker and particle-filter settings are fixed constants (the values of
``tests/apps/test_apps_realdata.py``), not chosen on a validation split, so these gains are not
tuned-then-held-out results. Measured data; the particle filters are seeded
(``random_state=0``), so every number is reproducible. The example shows how the layers
compose, not a ranking.

Output: ``assets/figures/tracking_and_fusion.png`` (``--out`` to change the folder):
(a) one test trace on the floor plan, (b) error CDFs at the waypoints of all test traces.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import indoorloc as iloc
from indoorloc.apps import imu_arrays, step_lengths

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

METHODS = ("WiFi WKNN", "+ Kalman filter", "+ Kalman RTS", "PDR (true start)", "PDR + WiFi", "PDR + WiFi + map")
# Six lines cross on one CDF, so every pair of colours must stay apart (not only neighbours): blue, yellow,
# purple, pink and dark green pass the all-pairs colour-vision checks; green-pink is the one weak pair
# under colour-vision deficiency, so the green PDR line is also dashed (a second cue besides colour).
COLOUR = dict(zip(METHODS, (PALETTE[0], PALETTE[3], PALETTE[6], PALETTE[2], PALETTE[4], PALETTE[5])))
DASHED = ("PDR (true start)",)
MIN_APS = 5  # a scan hearing fewer access points gives no fix
N_BOOT = 2000


# ------------------------------------------------------------------------------------ data
def rotation_heading(rv: np.ndarray) -> np.ndarray:
    """Heading of the phone's +y axis, radians counter-clockwise from east, from Android's
    rotation vector (x, y, z, w): the second column of ``SensorManager.getRotationMatrixFromVector``."""
    x, y, z, w = np.asarray(rv, dtype=np.float64).T
    return np.arctan2(1 - 2 * (x * x + z * z), 2 * (x * y - z * w))


def load_traces(site: str, floor: str, download: bool, root=None) -> tuple[list[dict], dict]:
    """Per trace: IMU arrays, rotation-vector heading, waypoints and WiFi scans; plus the floor plan."""
    def table(modality):
        return iloc.load_dataset("ilc2020", root=root, download=download, site=site, floor=floor,
                                 modality=modality, outside_waypoints="nan")

    imu, wp, wifi = table("imu"), table("waypoints"), table("wifi")
    rv = [imu.meta["channels"].index(c) for c in ("rv_x", "rv_y", "rv_z", "rv_w")]
    detector = iloc.StepDetector()
    traces = []
    for code, name in enumerate(wp.meta["trajectory_names"]):
        rows = imu.groups["trajectory"] == code
        d = imu_arrays(imu[rows])
        d["name"], d["heading"] = str(name), rotation_heading(imu.X[rows][:, rv])
        mine = wp.groups["trajectory"] == code
        d["wp_t"], d["wp"] = np.asarray(wp.groups["time"][mine], np.float64), wp.pos[mine]
        mine = wifi.groups["trajectory"] == code
        d["scan_t"], d["X"], d["scan_pos"] = np.asarray(wifi.groups["time"][mine], np.float64), wifi.X[mine], \
            wifi.pos[mine]
        d["steps"] = detector.detect(d["acc"], d["t"])
        d["length"] = float(np.linalg.norm(np.diff(d["wp"], axis=0), axis=1).sum())
        if len(d["wp"]) >= 2 and len(d["scan_t"]):
            traces.append(d)
    return traces, wp.meta["floor_plan"]


def interpolate(t_query, t, pos) -> np.ndarray:
    ok = np.all(np.isfinite(pos), axis=1)
    return np.stack([np.interp(t_query, t[ok], pos[ok, i]) for i in range(2)], axis=1)


# ------------------------------------------------------------------------------------ one test trace
def run_trace(d: dict, train: list[dict], fmap, n_particles: int) -> dict:
    """Fit on ``train`` (the other traces), track ``d``; errors at its waypoints."""
    heard = np.isfinite(np.concatenate([p["X"] for p in train])).any(axis=0)
    inside = [np.all(np.isfinite(p["scan_pos"]), axis=1) for p in train]
    X = np.concatenate([p["X"][m][:, heard] for p, m in zip(train, inside)])
    Y = np.concatenate([p["scan_pos"][m] for p, m in zip(train, inside)])
    model = iloc.create_model("wknn", k=5, preprocess=iloc.FillMissing(-100.0)).fit(X, Y)
    # Weinberg constant: walked distance / sum of the unit-constant step lengths, training traces only
    k = sum(p["length"] for p in train) / sum(step_lengths(p["steps"], "weinberg", 1.0).sum() for p in train)
    # PDR heading offset (phone heading -> map), circular mean over the training traces' steps
    diffs = []
    for p in train:
        leg = np.diff(p["wp"], axis=0)
        seg = np.clip(np.searchsorted(p["wp_t"], p["steps"].t) - 1, 0, len(leg) - 1)
        diffs.append(np.arctan2(leg[seg, 1], leg[seg, 0]) - p["heading"][p["steps"].index])
    offset = float(np.angle(np.mean(np.exp(1j * np.concatenate(diffs)))))

    Xt = d["X"][:, heard]
    usable = np.isfinite(Xt).sum(axis=1) >= MIN_APS
    fixes, scan_t = model.localize(Xt[usable]), d["scan_t"][usable]
    sel = d["wp_t"] >= scan_t[0]
    wt, truth = d["wp_t"][sel], d["wp"][sel]
    steps = d["steps"]
    lengths = step_lengths(steps, "weinberg", k)
    heading = d["heading"][steps.index]
    tracks = {  # (times, positions) of each method
        "WiFi WKNN": (scan_t, fixes.pos),
        "+ Kalman filter": (scan_t, iloc.KalmanTracker(process_noise=0.5, min_meas_std=2.0).filter(fixes, scan_t).pos),
        "+ Kalman RTS": (scan_t, iloc.KalmanTracker(process_noise=0.5, min_meas_std=2.0).smooth(fixes, scan_t).pos),
    }
    pdr = iloc.PDR(k=k).run(d["acc"], t=d["t"], yaw=d["heading"] + offset, start=d["wp"][0])
    tracks["PDR (true start)"] = (d["t"], pdr.position_at(d["t"]))
    for name, floor_map in (("PDR + WiFi", None), ("PDR + WiFi + map", fmap)):
        pf = iloc.ParticleFilter(n_particles, step_length_std=0.1, heading_std=0.08, heading_drift_std=0.01,
                                 min_meas_std=3.0, floor_map=floor_map, recovery=(0.05, 0.5), random_state=0)
        t, est = iloc.PDRFusion(pf, heading_bias_std=None).run((steps.t, lengths, heading), fixes, scan_t)
        tracks[name] = (t, est.pos)
    errors = {name: np.linalg.norm(interpolate(wt, t, pos) - truth, axis=1) for name, (t, pos) in tracks.items()}
    return {"name": d["name"], "errors": errors, "tracks": tracks, "wp": d["wp"], "wp_used": truth,
            "n_fixes": int(usable.sum()), "n_scans": len(usable), "k": k, "offset": offset}


def trace_bootstrap(results: list[dict], name: str, reference: str = METHODS[0], *, n_boot: int = N_BOOT,
                    seed: int = 0) -> tuple[float, float, float]:
    """Change of the pooled mean error of ``name`` against ``reference`` with a 95 % paired bootstrap
    interval that resamples whole test traces (the waypoints of one walk are not independent)."""
    sums = np.array([np.sum(r["errors"][name] - r["errors"][reference]) for r in results])
    counts = np.array([len(r["errors"][name]) for r in results], dtype=np.float64)
    draws = np.random.default_rng(seed).integers(0, len(results), size=(n_boot, len(results)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(sums.sum() / counts.sum()), float(lo), float(hi)


# ------------------------------------------------------------------------------------ figure
def figure(results: list[dict], floor_plan: dict, show: dict, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from indoorloc.datasets.plot import plot_floor_plan
    from indoorloc.evaluation.plot import plot_cdf

    with plt.rc_context(STYLE):
        fig, (ax_map, ax_cdf) = plt.subplots(1, 2, figsize=(12.0, 5.8), gridspec_kw={"width_ratios": [1.3, 1.0],
                                                                                        "wspace": 0.16})
        plot_floor_plan(floor_plan, ax_map, colors="#9a9994", linewidths=0.7)
        wp = show["wp"]
        tracks = show["tracks"]
        drawn = np.concatenate([wp] + [tracks[n][1] for n in METHODS])
        drawn = drawn[np.all(np.isfinite(drawn), axis=1)]
        lo, hi = drawn.min(axis=0) - 4.0, drawn.max(axis=0) + 4.0
        fix = tracks["WiFi WKNN"][1]
        ax_map.plot(fix[:, 0], fix[:, 1], "o", color=COLOUR["WiFi WKNN"], ms=3.5, alpha=0.8, label="WiFi WKNN fixes",
                    zorder=3)
        for name in ("+ Kalman RTS", "PDR (true start)", "PDR + WiFi + map"):
            pos = tracks[name][1]
            ax_map.plot(pos[:, 0], pos[:, 1], color=COLOUR[name], lw=1.6, label=name, zorder=4,
                        ls="--" if name in DASHED else "-")
        ax_map.plot(wp[:, 0], wp[:, 1], "-", color=INK, lw=2.0, label="waypoints (truth)", zorder=5)
        ax_map.plot(wp[:, 0], wp[:, 1], "o", color=INK, ms=3.5, zorder=5)
        ax_map.plot(*wp[0], "o", color=INK, mfc="white", ms=8, zorder=6, label="start")
        ax_map.set_xlim(lo[0], hi[0])
        ax_map.set_ylim(lo[1], hi[1])
        ax_map.set_aspect("equal")
        ax_map.set_xlabel("x [m]")
        ax_map.set_ylabel("y [m]")
        ax_map.grid(False)
        mean = {n: float(np.mean(show["errors"][n])) for n in METHODS}
        ax_map.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=7.5, title_fontsize=7.5,
                      title="mean error on this trace: " + ", ".join(
                          f"{n} {mean[n]:.1f} m" for n in ("WiFi WKNN", "+ Kalman RTS", "PDR (true start)",
                                                            "PDR + WiFi + map")))
        ax_map.set_title(f"(a) Median test trace (by fused error), {len(show['wp_used'])} scored waypoints", loc="left")

        pooled = {n: np.concatenate([r["errors"][n] for r in results]) for n in METHODS}
        for name in METHODS:
            plot_cdf(pooled[name], ax=ax_cdf, label=f"{name} ({np.mean(pooled[name]):.2f} m)", color=COLOUR[name],
                     lw=1.6, ls="--" if name in DASHED else "-")
        ax_cdf.set_xlim(0, 20)
        ax_cdf.legend(loc="lower right", title="method (mean error)", title_fontsize=7.5, fontsize=7.5)
        ax_cdf.set_title(f"(b) Error CDF, {len(pooled[METHODS[0]])} waypoints of {len(results)} test traces",
                         loc="left")
        fig.suptitle(title, fontsize=9, color=MUTED, y=1.0)
        fig.savefig(path)
        plt.close(fig)


# ------------------------------------------------------------------------------------ main
def main(*, site: str = "site1", floor: str = "F1", stride: int = 12, max_test: int | None = None,
         n_particles: int = 1000, download: bool = True, root=None, out: Path | str = FIGURES,
         verbose: bool = True) -> dict:
    """Leave-one-trajectory-out over every ``stride``-th trace (at most ``max_test`` of them);
    return ``{method: pooled error statistics}`` plus per-trace means."""
    start = time.perf_counter()
    traces, plan = load_traces(site, floor, download, root)
    fmap = iloc.FloorMap.from_dict(plan)
    test_ids = list(range(0, len(traces), stride))[:max_test]
    results = [run_trace(traces[i], [p for j, p in enumerate(traces) if j != i], fmap, n_particles)
               for i in test_ids]
    pooled = {n: np.concatenate([r["errors"][n] for r in results]) for n in METHODS}
    summary = {n: {"mean": float(np.mean(e)), "median": float(np.median(e)), "p90": float(np.percentile(e, 90)),
                   "n": len(e), "change_vs_wifi": trace_bootstrap(results, n)} for n, e in pooled.items()}
    per_trace = {r["name"]: {n: float(np.mean(r["errors"][n])) for n in METHODS} for r in results}
    pairwise = trace_bootstrap(results, "+ Kalman RTS", reference="PDR + WiFi + map")
    # the trace drawn in (a): the one whose fused error is the median over the test traces (a typical case)
    fused = np.array([np.mean(r["errors"]["PDR + WiFi + map"]) for r in results])
    show = results[int(np.argsort(fused)[len(fused) // 2])]

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "tracking_and_fusion.png"
    title = (f"ILC 2020 {site}/{floor} (measured, Android phones), leave one trajectory out: {len(traces)} traces, "
             f"{len(results)} test traces; errors at the surveyor's waypoints")
    figure(results, plan, show, path, title)
    if verbose:
        n_fix = sum(r["n_fixes"] for r in results)
        n_scan = sum(r["n_scans"] for r in results)
        print(f"ILC 2020 {site}/{floor}: {len(traces)} traces; test traces {len(results)} (every {stride}th), "
              f"{summary[METHODS[0]]['n']} scored waypoints, {n_fix} of {n_scan} test scans hear >= {MIN_APS} APs")
        k = [r["k"] for r in results]
        offset = np.rad2deg([r["offset"] for r in results])
        print(f"Weinberg constant (fitted per test trace on the other traces): {min(k):.4f}-{max(k):.4f}; "
              f"heading offset {offset.min():.1f} to {offset.max():.1f} deg")
        print(f"{'method':20s} {'mean':>6s} {'median':>7s} {'P90':>6s} {'vs WiFi':>8s} {'95% CI over traces':>19s}"
              "   (metres, pooled over the scored waypoints)")
        for n, s in summary.items():
            d, lo, hi = s["change_vs_wifi"]
            print(f"{n:20s} {s['mean']:6.2f} {s['median']:7.2f} {s['p90']:6.2f} {d:+8.2f} [{lo:+6.2f}, {hi:+6.2f}]")
        d, lo, hi = pairwise
        print(f"+ Kalman RTS vs PDR + WiFi + map: {d:+.2f} m [{lo:+.2f}, {hi:+.2f}] (95 % CI over traces; an interval "
              "that contains 0 cannot rank the two)")
        print(f"per test trace, mean error [m]:\n  {'trace':24s} " + " ".join(f"{n:>17s}" for n in METHODS))
        for name, v in per_trace.items():
            print(f"  {name:24s} " + " ".join(f"{v[n]:17.2f}" for n in METHODS))
        print(f"figure: {path}  (trace in (a): {show['name']})  ({time.perf_counter() - start:.1f} s)")
    return {"summary": summary, "per_trace": per_trace, "n_traces": len(traces), "test_traces": len(results),
            "rts_vs_fused": pairwise}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-download", action="store_true", help="fail instead of downloading the traces")
    parser.add_argument("--out", type=Path, default=FIGURES, help="folder for the figure")
    args = parser.parse_args()
    main(download=not args.no_download, out=args.out)
