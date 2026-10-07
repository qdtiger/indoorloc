"""Figures of a dataset and of localization results with the IndoorLoc 0.2 API.

    python examples/dataset_distribution.py                              # simulated office, no download
    python examples/dataset_distribution.py --dataset ujiindoorloc       # a public dataset (cached or --download)
    python examples/dataset_distribution.py --out /tmp/figs --html       # also an interactive page (needs plotly)

Figures written into ``--out`` (default ``work_dirs/dataset_distribution/<dataset>``):

    distribution_2d.png   where the train and test samples lie, one panel per floor, over the
                          floor plan when the dataset has one (datasets.plot.plot_distribution)
    distribution_3d.png   the same samples with the floors stacked
    density_2d.png        training samples per cell
    error_cdf.png         k-NN and WKNN (k=5) on the test split: error CDFs (evaluation.plot)
    error_map.png         the test positions of the busiest floor coloured by the WKNN error
                          (scale capped at the 95th percentile), arrows to the estimates
    trajectories.png      simulated office only: one walk, the raw WKNN fixes and a Kalman
                          (RTS) smoothed track over the walls (evaluation.plot.plot_trajectories)
    distribution_3d.html  with --html: the 3-D view as a plotly page (datasets.plot.distribution_html)

Localization figures are made for RSSI datasets (WiFi / BLE) only. Every number printed is
computed here from the data; the default dataset is simulated (``synthetic_office``, seed 0),
so its errors say nothing about real buildings. This script replaces the 0.1 examples
``visualization_demo.py`` and ``dataset_distribution_demo.py``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import indoorloc as iloc

RSSI = ("wifi_rssi", "ble_rssi", "rssi")


def load(name: str, download: bool, n_floors: int):
    options = {"n_floors": n_floors, "seed": 0} if name == "synthetic_office" else {}
    tables = iloc.load_dataset(name, download=download, **options)
    return tables if isinstance(tables, tuple) else (tables,)


def dataset_figures(tables, out: Path, html: bool) -> None:
    from indoorloc.datasets.plot import distribution_html, plot_distribution

    fig = plot_distribution(tables)
    fig.savefig(out / "distribution_2d.png", dpi=150)
    has_3d = all(t.floor is not None or t.pos.shape[1] >= 3 for t in tables)
    if has_3d:
        plot_distribution(tables, view="3d").savefig(out / "distribution_3d.png", dpi=150)
    plot_distribution(tables[0], kind="density").savefig(out / "density_2d.png", dpi=150)
    if html and has_3d:
        try:
            distribution_html(tables, out / "distribution_3d.html", view="3d")
        except ImportError as err:
            print(f"  skipped the HTML page: {err}")


def localization_figures(train, test, out: Path) -> None:
    from indoorloc.evaluation.plot import plot_cdf, plot_error_map

    fill = float(np.floor(np.nanmin(train.X))) - 1.0  # below the weakest reading: "not heard"
    scale = float(train.meta.get("ground_scale", 1.0))  # EPSG:3857 -> ground metres (UJIIndoorLoc)
    results, preds = {}, {}
    for method in ("knn", "wknn"):
        model = iloc.create_model(method, k=5, preprocess=iloc.FillMissing(fill)).fit(train)
        preds[method] = model.localize(test)
        results[method] = iloc.evaluate(test, preds[method], scale=scale)
        print(f"  {method:5s} {results[method].summary()}")
    unit = "m (ground)" if "ground_scale" in train.meta else (train.meta.get("pos_units") or "m")
    ax = plot_cdf({m.upper(): r for m, r in results.items()}, unit=unit)
    ax.set_title(f"{train.meta.get('name')}: test split, k = 5")
    ax.figure.savefig(out / "error_cdf.png", dpi=150)
    plan = train.meta.get("floor_plan")
    floor = None
    rows = np.arange(len(test))
    if test.floor is not None:  # one floor keeps the map readable
        floor = int(np.bincount(test.floor - test.floor.min()).argmax() + test.floor.min())
        rows = np.flatnonzero(test.floor == floor)
    p95 = results["wknn"].p95_error  # colour scale capped at P95: one gross error would flatten the rest
    ax = plot_error_map(test[rows], preds["wknn"][rows], errors=results["wknn"].errors[rows], arrows=True,
                        floor_plan=plan, floor=floor, unit=unit, pos_unit=train.meta.get("pos_units") or "m",
                        vmax=p95)
    ax.set_title("WKNN errors" + ("" if floor is None else f", floor {floor}") + f" (n={len(rows)}; colours "
                 f"capped at P95 = {p95:.1f})")
    ax.figure.savefig(out / "error_map.png", dpi=150)


def trajectory_figure(name: str, train, out: Path, n_floors: int) -> None:
    from indoorloc.apps import KalmanTracker
    from indoorloc.evaluation.plot import plot_trajectories

    walks = iloc.load_dataset(name, split="trajectory", n_floors=n_floors, seed=0)
    first = walks[walks.groups["trajectory"] == walks.groups["trajectory"][0]]
    fill = float(np.floor(np.nanmin(train.X))) - 1.0
    fixes = iloc.create_model("wknn", k=5, preprocess=iloc.FillMissing(fill)).fit(train).localize(first)
    smoothed = KalmanTracker(process_noise=0.3).smooth(fixes, t=first.groups["time"])
    for label, track in (("WKNN fixes", fixes), ("Kalman RTS", smoothed)):
        print(f"  walk: {label:10s} {iloc.evaluate(first, track).summary()}")
    floor = int(first.floor[0]) if first.floor is not None else None
    ax = plot_trajectories(first, {"WKNN fixes": fixes, "Kalman RTS smoothed": smoothed},
                           floor_plan=first.meta.get("floor_plan"), floor=floor)
    ax.set_title(f"simulated walk, {len(first)} scans" + ("" if floor is None else f", floor {floor}"))
    ax.figure.savefig(out / "trajectories.png", dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="synthetic_office", help=f"one of {iloc.list_datasets()}")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--download", action="store_true", help="download a public dataset if not cached")
    parser.add_argument("--floors", type=int, default=3, help="storeys of the simulated office")
    parser.add_argument("--html", action="store_true", help="also write an interactive plotly page")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")  # files only, no window
    out = args.out or Path("work_dirs") / "dataset_distribution" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    tables = load(args.dataset, args.download, args.floors)
    meta = tables[0].meta
    print(f"{meta.get('name')}: " + ", ".join(f"{t.meta.get('split')} {len(t):,}" for t in tables)
          + f" samples, modality {meta.get('modality')}, crs {meta.get('crs')}"
          + (" (simulated)" if meta.get("source") == "simulated" else ""))
    if tables[0].pos.shape[1] < 2:
        print("  this dataset labels rooms, not coordinates: there is nothing to draw on a plan")
        return
    dataset_figures(tables, out, args.html)
    if meta.get("modality") in RSSI and len(tables) == 2:
        localization_figures(*tables, out)
        if args.dataset == "synthetic_office":
            trajectory_figure(args.dataset, tables[0], out, args.floors)
    else:
        print("  localization figures need an RSSI dataset with a train/test split; skipped")
    print(f"figures in {out}: {', '.join(sorted(p.name for p in out.iterdir()))}")


if __name__ == "__main__":
    main()
