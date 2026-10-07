"""Model-based estimators against the Cramer-Rao lower bound (simulated data, Monte Carlo).

    python examples/02_model_based_vs_crlb.py            # about 30 s on a laptop CPU
    python examples/02_model_based_vs_crlb.py --points 100 --draws 5   # a quick look

Four measurement models, each solved by a closed-form estimator and by maximum likelihood
(Gauss-Newton), each compared with its Cramer-Rao lower bound (CRLB) from
``indoorloc.evaluation.bounds``:

    (a) ranging (ToA / UWB), 4 anchors   TrilaterationLocalizer  "linear" vs "gauss_newton"   toa_crlb
    (b) TDoA, 5 anchors                  TDOALocalizer           Chan-Ho vs Chan-Ho + GN      tdoa_crlb
    (c) angle of arrival, 4 arrays       AoALocalizer            Stansfield vs GN             aoa_crlb
    (d) visible light (RSS), 176 LEDs    LambertianLocalizer     "ranges" vs "nls"            Fisher information
                                                                                              of the Lambertian model

The closed forms are the library's: linear trilateration differences the squared ranges;
Stansfield's bearing intersection is used with equal weights (Stansfield 1947 weights each line
by 1 / (sigma d)^2, which needs the unknown distances d); Chan-Ho is the two-step weighted least
squares of Chan & Ho (1994); "ranges" inverts each power to a distance and trilaterates.

Set-up (every step uses the public API):

* Geometry from L1: ``load_dataset("synthetic_office", modality=..., noise_std=0)`` gives the
  exact, noise-free measurements at ``--points`` random test points of a 40 m x 20 m office
  floor (seed 0, line of sight only: the NLOS bias of the simulator is switched off, because
  the bound assumes unbiased Gaussian noise). With ``dim=2`` the ranging and angle anchors
  sit at device height, so the 2-D geometry is exact.
* Noise: ``--draws`` independent Gaussian draws per point and noise level (numpy
  ``default_rng(seed)``): range noise per anchor (TDoA differences share the reference
  anchor's noise, as ``tdoa_crlb`` models), bearing noise per array, optical power noise
  per LED. VLC: an LED counts as seen when its noise-free power is at least 5 sigma (the
  simulator's detection rule, applied to the noise-free power so that estimator and bound
  use the same LEDs); an LED behind a wall is never seen.
* Metric: position RMSE over all trials with an estimate, against the bound's RMS over the
  same trials, ``sqrt(sum |e|^2 / sum tr(J^-1))``. A ratio of 1 means the estimator attains
  the bound; the 95 % interval comes from a bootstrap over points (1,000 resamples).
  Trials without an estimate (fewer measurements than unknowns, or the mirror ambiguity of
  collinear LEDs) are counted as ``n_failed`` and left out, as ``evaluate`` does.
* The VLC bound is not in ``evaluation.bounds``: its Fisher information
  ``J = sum_i grad P_i grad P_i^T / sigma^2`` over the seen LEDs is built here, with central
  differences (1e-6 m) of ``LambertianLocalizer.predict_power`` (the exact model the data was
  simulated with), and inverted by ``evaluation.bounds.crlb_rmse``.

Result (400 points x 20 draws, measured 2026-09-29; the script prints every value): the
maximum-likelihood estimators attain the bound. RMSE / CRLB is 0.99-1.02 for ToA over
sigma = 1 mm-3 m, 0.99-1.00 for TDoA up to 0.3 m, 0.99-1.03 for AoA over 0.03-30 degrees and
0.99-1.02 for VLC up to 1e-7 W (each 95 % interval includes 1 or ends within 0.01 of it). Where
they do not: TDoA Gauss-Newton leaves the bound at 1 m (1.41) and 3 m (2.35; RMSE 1.57 m and
7.8 m, median error 0.88 m and 2.6 m), and at 3 m 20 of 8,000 trials are not placed (estimates
farther than ``FAR_FIELD`` anchor spreads from the anchors, see ``TDOALocalizer``). This is the
likelihood's own threshold effect, not a local-search failure: in a separate check (noise seed
1, 400 points x 20 draws), all 25 TDoA estimates more than 10 m off at sigma = 1 m, and all 60
more than 30 m off at 3 m, have a LOWER likelihood cost than the true position (0.09-0.80x and
0.003-0.65x of it), so the maximum-likelihood estimate itself lies there, and the bound, a
small-error approximation, no longer describes it. VLC model fitting reaches 1.19-1.32x the
bound at 3e-7-1e-6 W. The closed forms stay above the bound: linear trilateration by 1.28-1.31x,
Stansfield by 1.14-1.18x, the Chan-Ho closed form by 1.05x at 1 mm rising to 3.2x at 3 m (up to
0.1 m its median error is within 6 % of the ML one: a tail of large errors drives its RMSE), and
VLC power-to-range trilateration by 4.5-4.9x up to 3e-7 W.
VLC trials not placed (both solvers alike): 20 of 8,000 up to 3e-7 W (one point whose seen LEDs are
collinear), 140 at 1e-6 W (20 see fewer than 3 LEDs above 5 sigma, 120 only collinear LEDs) and
2,260 (28 %) at 3e-6 W (2,200 fewer than 3 LEDs, 60 collinear); the ratios at 1e-6 and 3e-6 W
cover the placed trials only, i.e. the better-lit points.
Everything here is SIMULATED: the bound is a statement about this noise model, not about real
radios. Off-scale points would be drawn as triangles on the upper edge (none at these settings).

Output: ``assets/figures/model_based_vs_crlb.png`` (``--out`` to change the folder).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import indoorloc as iloc
from indoorloc.evaluation import bounds
from indoorloc.methods.vlc import LambertianLocalizer

FIGURES = Path(__file__).resolve().parents[1] / "assets" / "figures"
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

SEED = 0
R_HI = 6.0  # top of the RMSE / CRLB axis
DETECT_SNR = 5.0  # VLC: an LED is seen when its noise-free power is >= 5 sigma (SyntheticOffice's rule)


# ------------------------------------------------------------------------------------ the cases
def geometry(modality: str, n_points: int):
    """Noise-free measurements at ``n_points`` test points of the simulated office (one storey, 2-D)."""
    physics = {"ranges": {"nlos_bias_mean": 0.0}, "tdoa": {"nlos_bias_mean": 0.0}, "aoa": {"nlos_std": 0.0}}
    return iloc.load_dataset("synthetic_office", split="test", seed=SEED, modality=modality, n_test=n_points,
                             noise_std=0.0, physics=physics.get(modality))


def vlc_fisher(model: LambertianLocalizer, pos: np.ndarray, seen: np.ndarray, sigma: float, h: float = 1e-6):
    """Fisher information (N, 2, 2) of Gaussian power noise ``sigma`` over the seen LEDs."""
    grads = []
    for axis in range(2):
        step = np.zeros(2)
        step[axis] = h
        grads.append((model.predict_power(pos + step) - model.predict_power(pos - step)) / (2 * h))
    G = np.stack(grads, axis=-1) * seen[..., None]  # (N, A, 2), unseen LEDs carry no information
    return np.einsum("nai,naj->nij", G, G) / sigma ** 2


def cases(n_points: int) -> dict:
    """name -> table, noise levels, noise model, bound and the two estimators."""
    toa, tdoa, aoa, vlc = (geometry(m, n_points) for m in ("ranges", "tdoa", "aoa", "vlc"))
    deg = np.pi / 180.0
    vlc_nls = LambertianLocalizer.from_meta(vlc.meta, solver="nls")
    anchors = {name: t.meta["anchors"] for name, t in (("toa", toa), ("tdoa", tdoa), ("aoa", aoa))}
    orient = aoa.meta["anchor_orientations"]

    def wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    return {
        "toa": {
            "table": toa, "title": f"(a) Ranging (ToA), {len(anchors['toa'])} anchors", "unit": "m", "show": 1.0,
            "sigmas": (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0),
            "noise": lambda X0, s, rng: X0 + s * rng.standard_normal(X0.shape),
            "bound": lambda s: bounds.toa_crlb(anchors["toa"], toa.pos, s),
            "estimators": {"linear least squares": iloc.TrilaterationLocalizer(anchors["toa"], solver="linear"),
                           "Gauss-Newton (ML)": iloc.TrilaterationLocalizer(anchors["toa"], solver="gauss_newton")}},
        "tdoa": {
            "table": tdoa, "title": f"(b) TDoA, {len(anchors['tdoa'])} anchors", "unit": "m", "show": 1.0,
            "sigmas": (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0),
            "noise": lambda X0, s, rng: X0 + _tdoa_noise(s, rng, X0),
            "bound": lambda s: bounds.tdoa_crlb(anchors["tdoa"], tdoa.pos, s),
            "estimators": {"Chan-Ho closed form": iloc.TDOALocalizer(anchors["tdoa"], refine=False),
                           "Chan-Ho + Gauss-Newton (ML)": iloc.TDOALocalizer(anchors["tdoa"], refine=True)}},
        "aoa": {
            "table": aoa, "title": f"(c) Angle of arrival, {len(anchors['aoa'])} arrays", "unit": "deg",
            "show": 1 / deg, "sigmas": tuple(v * deg for v in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)),
            "noise": lambda X0, s, rng: wrap(X0 + s * rng.standard_normal(X0.shape)),
            "bound": lambda s: bounds.aoa_crlb(anchors["aoa"], aoa.pos, s),
            "estimators": {"Stansfield (linear)": iloc.AoALocalizer(anchors["aoa"], orient, solver="linear"),
                           "Gauss-Newton (ML)": iloc.AoALocalizer(anchors["aoa"], orient, solver="gauss_newton")}},
        "vlc": {
            "table": vlc, "title": f"(d) Visible light (RSS), {vlc.X.shape[1]} LEDs", "unit": "W", "show": 1.0,
            "sigmas": (1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6),
            "noise": lambda X0, s, rng: np.where(np.isfinite(X0) & (X0 >= DETECT_SNR * s),
                                                 X0 + s * rng.standard_normal(X0.shape), np.nan),
            "bound": lambda s: _vlc_bound(vlc_nls, vlc, s),
            "too_few": lambda s: (np.isfinite(vlc.X) & (vlc.X >= DETECT_SNR * s)).sum(axis=1) < 3,
            "estimators": {"power -> range -> trilateration": LambertianLocalizer.from_meta(vlc.meta, solver="ranges"),
                           "model fit, Gauss-Newton (ML)": vlc_nls}},
    }


def _tdoa_noise(s: float, rng: np.random.Generator, X0: np.ndarray) -> np.ndarray:
    n = s * rng.standard_normal((len(X0), X0.shape[1] + 1))  # one range error per anchor
    return n[:, 1:] - n[:, :1]


def _vlc_bound(model: LambertianLocalizer, table, sigma: float) -> np.ndarray:
    seen = np.isfinite(table.X) & (table.X >= DETECT_SNR * sigma)  # the LEDs the noise model keeps
    return bounds.crlb_rmse(vlc_fisher(model, table.pos, seen, sigma))


# ------------------------------------------------------------------------------------ Monte Carlo
def monte_carlo(case: dict, n_draws: int, n_boot: int, rng: np.random.Generator) -> dict:
    """RMSE, bound and their ratio (bootstrap CI over points) per estimator and noise level."""
    t = case["table"]
    out = {name: [] for name in case["estimators"]}
    for s in case["sigmas"]:
        crlb = case["bound"](s)
        draws = [case["noise"](t.X, s, rng) for _ in range(n_draws)]
        boot = rng.integers(0, len(t), size=(n_boot, len(t)))
        for name, model in case["estimators"].items():
            sq = np.stack([np.sum((model.localize(X).pos - t.pos) ** 2, axis=1) for X in draws])  # (R, N)
            ok = np.isfinite(sq) & np.isfinite(crlb)[None, :]
            err_pt = np.where(ok, sq, 0.0).sum(axis=0)                      # per point: sum of squared errors
            bnd_pt = np.where(ok, crlb[None, :] ** 2, 0.0).sum(axis=0)       # and of the bound, same trials
            n_ok = int(ok.sum())
            rmse = float(np.sqrt(err_pt.sum() / n_ok)) if n_ok else np.nan
            rms_bound = float(np.sqrt(bnd_pt.sum() / n_ok)) if n_ok else np.nan
            with np.errstate(invalid="ignore", divide="ignore"):
                ratios = np.sqrt(err_pt[boot].sum(axis=1) / bnd_pt[boot].sum(axis=1))
            finite_err = np.sqrt(sq[np.isfinite(sq)])
            out[name].append({
                "sigma": float(s), "rmse": rmse, "crlb": rms_bound, "ratio": rmse / rms_bound,
                "ratio_ci95": [float(v) for v in np.nanpercentile(ratios, [2.5, 97.5])],
                "median_error": float(np.median(finite_err)) if finite_err.size else np.nan,
                "max_error": float(finite_err.max()) if finite_err.size else np.nan,
                "n_trials": int(sq.size), "n_failed": int((~np.isfinite(sq)).sum()),
                "n_too_few": int(case["too_few"](s).sum()) * n_draws if "too_few" in case else 0,
                "n_no_bound": int((np.isfinite(sq) & ~np.isfinite(crlb)[None, :]).sum()),
            })
    return out


# ------------------------------------------------------------------------------------ figure
def figure(all_cases: dict, results: dict, path: Path, n_points: int, n_draws: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 4, figsize=(12.6, 5.8), sharex="col",
                                 gridspec_kw={"height_ratios": [2.3, 1.0], "hspace": 0.08, "wspace": 0.32})
        for col, (key, case) in enumerate(all_cases.items()):
            top, bottom = axes[0, col], axes[1, col]
            show = case["show"]
            rows = results[key]
            first = next(iter(rows.values()))
            x = np.array([r["sigma"] for r in first]) * show
            bound = np.array([r["crlb"] for r in first])
            top.loglog(x, bound, color=INK, lw=2.2, label="CRLB", zorder=4)
            y_lo, y_hi = np.nanmin(bound) / 2.5, np.nanmax(bound) * 12
            for i, (name, recs) in enumerate(rows.items()):
                colour, marker = PALETTE[i], "os"[i]
                rmse = np.array([r["rmse"] for r in recs])
                ratio = np.array([r["ratio"] for r in recs])
                ci = np.array([r["ratio_ci95"] for r in recs])
                inside = rmse <= y_hi
                top.loglog(x[inside], rmse[inside], marker=marker, color=colour, ms=5.5, lw=1.5, label=name,
                           mfc="white" if i else colour, zorder=5)
                if (~inside).any():  # divergent trials: mark the level and print the values
                    top.plot(x[~inside], np.full((~inside).sum(), y_hi / 1.3), "^", color=colour, ms=7, zorder=6)
                    values = ", ".join(f"{v:.0e}".replace("e+0", "e").replace("e+", "e") for v in rmse[~inside])
                    top.text(0.97, 0.04 + 0.12 * i, f"\u25b2 off scale:\nRMSE {values} m", transform=top.transAxes,
                             ha="right", va="bottom", fontsize=7.5, color=colour)
                keep = ratio <= R_HI
                yerr = np.abs(ci[keep] - ratio[keep, None]).T
                bottom.errorbar(x[keep] * (1.0 + 0.06 * (i - 0.5)), ratio[keep], yerr=yerr, fmt=marker, color=colour,
                                mfc="white" if i else colour, ms=4.5, lw=1.0, capsize=2)
                bottom.plot(x[keep], ratio[keep], color=colour, lw=1.0, alpha=0.6)
                if (~keep).any():
                    bottom.plot(x[~keep], np.full((~keep).sum(), R_HI / 1.06), "^", color=colour, ms=6,
                                clip_on=False)
            top.set_ylim(y_lo, y_hi)
            rec = max((r for recs in rows.values() for r in recs), key=lambda r: (r["n_failed"] / r["n_trials"],
                                                                                   r["sigma"]))
            worst = rec["n_failed"] / rec["n_trials"]
            if worst > 0.01:  # say how many trials an estimator could not place (left out of the RMSE), and why
                few, rest = rec["n_too_few"] / rec["n_trials"], (rec["n_failed"] - rec["n_too_few"]) / rec["n_trials"]
                why = (": fewer measure-\nments than unknowns" if "too_few" not in case else
                       f": {few:.1%} see\n< 3 LEDs above 5 sigma,\n{rest:.1%} only collinear LEDs")
                top.text(0.97, 0.04, f"{worst:.0%} of trials not placed\nat {rec['sigma'] * show:.0e} {case['unit']}{why}",
                         transform=top.transAxes, ha="right", va="bottom", fontsize=7.5, color=MUTED)
            top.set_title(case["title"], loc="left")
            bottom.axhline(1.0, color=INK, lw=1.2)
            bottom.set_yscale("log")
            bottom.set_ylim(0.85, R_HI)
            bottom.set_yticks([1, 1.5, 2, 3, 5])
            bottom.set_yticklabels(["1", "1.5", "2", "3", "5"])
            bottom.minorticks_off()
            bottom.set_xscale("log")
            bottom.set_xlabel(f"noise std [{case['unit']}]")
            top.legend(loc="upper left", handlelength=1.8)
            if col == 0:
                top.set_ylabel("position RMSE [m]")
                bottom.set_ylabel("RMSE / CRLB")
        fig.suptitle(f"Simulated (SyntheticOffice, line of sight, Gaussian noise), {n_points} points x {n_draws} "
                     "noise draws per level; bars: 95 % bootstrap interval; triangles: off scale",
                     fontsize=9, color=MUTED, y=0.985)
        fig.savefig(path)
        plt.close(fig)


# ------------------------------------------------------------------------------------ main
def main(*, n_points: int = 400, n_draws: int = 20, n_boot: int = 1000, out: Path | str = FIGURES,
         verbose: bool = True) -> dict:
    """Run the Monte-Carlo study; return ``{case: {estimator: [one record per noise level]}}``."""
    start = time.perf_counter()
    rng = np.random.default_rng(SEED)
    all_cases = cases(n_points)
    results = {key: monte_carlo(case, n_draws, n_boot, rng) for key, case in all_cases.items()}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "model_based_vs_crlb.png"
    figure(all_cases, results, path, n_points, n_draws)
    if verbose:
        print(f"SyntheticOffice seed {SEED}, {n_points} points x {n_draws} Gaussian draws per noise level "
              "(simulated, line of sight)")
        for key, case in all_cases.items():
            print(f"\n{case['title']}")
            print(f"  {'estimator':34s} {'sigma':>9s} {'RMSE [m]':>10s} {'CRLB [m]':>10s} {'ratio':>6s} "
                  f"{'95% CI':>13s} {'median':>8s} {'max':>10s} {'failed':>7s}")
            for name, recs in results[key].items():
                for r in recs:
                    lo, hi = r["ratio_ci95"]
                    print(f"  {name:34s} {r['sigma'] * case['show']:9.3g} {r['rmse']:10.4g} {r['crlb']:10.4g} "
                          f"{r['ratio']:6.4g} [{lo:6.4g}, {hi:6.4g}] {r['median_error']:8.3g} {r['max_error']:10.3g} "
                          f"{r['n_failed']:4d}/{r['n_trials']}"
                          + (f" ({r['n_too_few']} with < 3 LEDs)" if r["n_too_few"] else ""))
        print(f"\nfigure: {path}  ({time.perf_counter() - start:.1f} s)")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--points", type=int, default=400, help="test points (default 400)")
    parser.add_argument("--draws", type=int, default=20, help="noise draws per point and level (default 20)")
    parser.add_argument("--out", type=Path, default=FIGURES, help="folder for the figure")
    args = parser.parse_args()
    main(n_points=args.points, n_draws=args.draws, out=args.out)
