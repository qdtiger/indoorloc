"""Run or rebuild the UJIIndoorLoc example shown in the README (IndoorLoc 0.2 API).

    python examples/readme_demo.py             # replay the recorded case in examples/readme_case
    python examples/readme_demo.py --rebuild   # fit both baselines again from the original CSV files

The recorded case holds the full official validation set (samples.npz), the two fitted
models in IndoorLoc's pickle-free format (knn/, wknn/: config.json + arrays.npz), their
predictions and the metrics. Nearest-neighbour search breaks distance ties by training
index, so every number here is reproducible bit for bit on any machine and thread count.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

import indoorloc as iloc
from indoorloc import SampleTable

CASE_DIR = Path(__file__).resolve().parent / "readme_case"
SOURCE = "https://archive.ics.uci.edu/dataset/310/ujiindoorloc"
SOURCE_HASHES = {
    "trainingData.csv": "45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168",
    "validationData.csv": "5f90c536648cd657b2c516d20c4e0968d4003279ea6bd5d5d5322d3f1e8905c0",
}
METHODS = ("knn", "wknn")
K = 5
FILL_DBM = -104.0  # below the weakest reading in the dataset (-104 dBm)
CASE_FILES = ("samples.npz", "protocol.json", "results.json", "predictions.npz",
              "knn/config.json", "knn/arrays.npz", "wknn/config.json", "wknn/arrays.npz", "apps_ilc2020.json")
APPS_SHOWN = ("WiFi WKNN", "+ Kalman RTS", "PDR + WiFi + map")  # the L5 tracks drawn in the replay


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def verify_case(directory):
    directory = Path(directory)
    hashes = json.loads((directory / "checksums.json").read_text())
    for filename, expected in hashes.items():
        if sha256(directory / filename) != expected:
            raise ValueError(f"Example artifact checksum mismatch: {filename}")


def build_model(method):
    """The recorded recipe: missing readings -> -104 dBm, then k-NN (uniform) or WKNN (1/d) with k=5."""
    return iloc.create_model(method, k=K, preprocess=iloc.FillMissing(FILL_DBM))


def load_samples(directory=CASE_DIR) -> SampleTable:
    """The 1,111 official validation scans stored with the case (dBm, NaN = not heard)."""
    with np.load(Path(directory) / "samples.npz", allow_pickle=False) as data:
        table = SampleTable(data["rssi"], data["xy"], data["floor"], data["building"],
                            ids=data["ids"], meta={"feature_names": tuple(data["ap_names"].tolist()),
                                                   "modality": "wifi_rssi", "units": "dBm", "crs": "EPSG:3857"})
    if len(table) != 1111 or table.meta["feature_names"] != tuple(f"WAP{i:03d}" for i in range(1, 521)):
        raise ValueError("The example requires all 1,111 official validation scans over WAP001..WAP520")
    return table


def load_example(method="wknn", directory=CASE_DIR):
    """A fitted baseline and the validation table it is evaluated on."""
    if method not in METHODS:
        raise ValueError(f"Choose one of {METHODS}")
    directory = Path(directory)
    verify_case(directory)
    return iloc.load_model(directory / method), load_samples(directory)


def prediction_array(pred):
    """(N, 4): x, y, floor, building (the layout the 3D replay reads)."""
    return np.c_[pred.pos, pred.floor, pred.building].astype(np.float64)


def rebuild_case(directory, data_root=None):
    """Fit the two fixed k=5 baselines on the original training file and record everything."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    train, test = iloc.load_dataset("ujiindoorloc", root=data_root)
    for table in (train, test):
        name = table.meta["source_files"][0]
        if table.meta["sha256"] != SOURCE_HASHES[name]:
            raise ValueError(f"Source file differs from the published recipe: {name}")
    if (len(train), len(test)) != (19937, 1111):
        raise ValueError("Unexpected official split sizes")

    np.savez_compressed(directory / "samples.npz", rssi=test.X, xy=test.pos, floor=test.floor,
                        building=test.building, ids=test.ids,
                        ap_names=np.asarray(test.meta["feature_names"]))
    write_json(directory / "protocol.json", {
        "dataset": "UJIIndoorLoc", "source": SOURCE, "loader": "indoorloc.load_dataset('ujiindoorloc')",
        "dataset_doi": "10.24432/C5MS59", "dataset_license": "CC BY 4.0",
        "dataset_attribution": "Torres-Sospedra, J., Montoliu, R., Martínez-Usó, A., Arnau, T., and Avariento, J. "
                               "(2014). UJIIndoorLoc. UCI Machine Learning Repository.",
        "split": {"train": "trainingData.csv", "evaluation": "validationData.csv",
                  "train_samples": 19937, "evaluation_samples": 1111},
        "source_sha256": SOURCE_HASHES,
        "input": {"signal": "WiFi RSSI", "unit": "dBm", "shape": ["N", 520], "ap_order": "WAP001..WAP520",
                  "missing": "the file's 100 ('not detected') is loaded as NaN",
                  "preprocessing": f"FillMissing({FILL_DBM:g}): NaN -> {FILL_DBM:g} dBm; no fitted statistics"},
        "target": {"columns": ["LONGITUDE", "LATITUDE"], "crs": "EPSG:3857 (Web Mercator metres)",
                   "ground_scale": train.meta["ground_scale"],
                   "note": "errors are reported in EPSG:3857 metres, as in the literature; multiply by "
                           "ground_scale for ground metres", "floor_labels": [0, 1, 2, 3, 4],
                   "building_labels": [0, 1, 2]},
        "models": {"knn": {"k": K, "weights": "uniform"}, "wknn": {"k": K, "weights": "distance (1/d)"}},
        "neighbour_search": "exact Euclidean; ties broken by training index, independent of BLAS/OpenMP threads",
        "selection": "k=5 fixed in advance; no hyperparameter search on the evaluation file",
        "metrics": {"position": "2D Euclidean distance over all 1111 samples",
                    "classification": "floor and building accuracy in percent, reported separately",
                    "percentiles": "numpy.percentile, linear interpolation", "floor_penalty": "none"},
    })
    recorded = {"environment": {"python": platform.python_version(), "platform": platform.platform(),
                                "indoorloc": iloc.__version__, "numpy": np.__version__},
                "timing": "wall time of fit and localize on this CPU", "methods": {}}
    predictions = {}
    for method in METHODS:
        model = build_model(method)
        started = time.perf_counter()
        model.fit(train)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        pred = model.localize(test)
        localize_seconds = time.perf_counter() - started
        results = iloc.evaluate(test, pred)
        model.save(directory / method, info={"dataset": "ujiindoorloc", "split": "train",
                                             "sha256": train.meta["sha256"]})
        restored = iloc.load_model(directory / method)
        np.testing.assert_array_equal(prediction_array(restored.localize(test)), prediction_array(pred))
        predictions[method] = prediction_array(pred)
        recorded["methods"][method] = {"metrics": results.to_dict(), "fit_seconds": fit_seconds,
                                       "localize_seconds": localize_seconds, "evaluation_samples": len(test)}
        print(f"Fitted {method.upper()}: {results.summary()}; save/load identical")
    np.savez_compressed(directory / "predictions.npz", **predictions)
    write_json(directory / "results.json", recorded)
    record_apps(directory)
    write_json(directory / "checksums.json", {name: sha256(directory / name) for name in CASE_FILES})


def record_apps(directory, data_root=None):
    """The L5 part of the replay: examples/04 (leave one trajectory out on ILC 2020 site1/F1),
    the pooled errors of every method and one held-out trace (the median by fused error), its
    tracks resampled to 1 Hz, and the floor-plan walls around it."""
    import importlib

    example = importlib.import_module("examples.04_tracking_and_fusion")
    traces, plan = example.load_traces("site1", "F1", download=True, root=data_root)
    fmap = iloc.FloorMap.from_dict(plan)
    test_ids = list(range(0, len(traces), 12))
    results = [example.run_trace(traces[i], [p for j, p in enumerate(traces) if j != i], fmap, 1000)
               for i in test_ids]
    pooled = {n: np.concatenate([r["errors"][n] for r in results]) for n in example.METHODS}
    fused = np.array([np.mean(r["errors"]["PDR + WiFi + map"]) for r in results])
    show = results[int(np.argsort(fused)[len(fused) // 2])]
    t0 = show["tracks"]["WiFi WKNN"][0][0]
    finite = lambda a: np.asarray(a)[np.all(np.isfinite(a), axis=1)]  # noqa: E731
    everything = np.concatenate([finite(show["wp_used"])] + [finite(pos) for _, pos in show["tracks"].values()])
    lo, hi = everything.min(axis=0) - 6, everything.max(axis=0) + 6
    walls = np.asarray(plan["walls"], dtype=np.float64).reshape(-1, 4)
    near = walls[(np.maximum(walls[:, 0], walls[:, 2]) >= lo[0]) & (np.minimum(walls[:, 0], walls[:, 2]) <= hi[0])
                 & (np.maximum(walls[:, 1], walls[:, 3]) >= lo[1]) & (np.minimum(walls[:, 1], walls[:, 3]) <= hi[1])]

    def resample(name, step=1.0):
        t, pos = show["tracks"][name]
        grid = np.arange(t0, t[np.all(np.isfinite(pos), axis=1)][-1], step)
        return finite(example.interpolate(grid, t, pos)).round(2).tolist()

    record = {
        "dataset": "ILC 2020 site1/F1 (measured, Android phones; MIT licence; Hu et al., MobiCom 2023, "
                   "DOI 10.1145/3570361.3592507)",
        "protocol": "leave one trajectory out, every 12th of the traces tested; errors at the surveyor's waypoints",
        "recorded_by": "examples/readme_demo.py --rebuild (runs examples/04_tracking_and_fusion.py's run_trace)",
        "traces": len(traces), "test_traces": len(results), "waypoints": int(len(pooled["WiFi WKNN"])),
        "methods": {n: {"mean": float(e.mean()), "median": float(np.median(e)), "p90": float(np.percentile(e, 90))}
                    for n, e in pooled.items()},
        "shown": {"name": show["name"], "waypoints": finite(show["wp_used"]).round(2).tolist(),
                  "fixes": finite(show["tracks"]["WiFi WKNN"][1]).round(2).tolist(),
                  "tracks": {n: resample(n) for n in APPS_SHOWN[1:]},
                  "errors": {n: float(np.mean(show["errors"][n])) for n in APPS_SHOWN},
                  "walls": near.round(2).tolist(), "bounds": [lo.round(2).tolist(), hi.round(2).tolist()]},
    }
    write_json(Path(directory) / "apps_ilc2020.json", record)
    print("L5 (ILC 2020): " + ", ".join(f"{n} {v['mean']:.2f} m" for n, v in record["methods"].items()))


def run_example(directory, output, render=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    measured, predictions = {}, {}
    for method in METHODS:
        model, test = load_example(method, directory)
        pred = model.localize(test)
        results = iloc.evaluate(test, pred)
        measured[method] = results.to_dict()
        predictions[method] = prediction_array(pred)
        with (output / f"{method}_predictions.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "true_x_m", "true_y_m", "pred_x_m", "pred_y_m",
                             "true_floor", "pred_floor", "true_building", "pred_building", "error_m"])
            for row in zip(test.ids, *test.pos.T, *pred.pos.T, test.floor, pred.floor,
                           test.building, pred.building, results.errors):
                writer.writerow(row)
        print(f"{method.upper():5} {results.summary()}")
    write_json(output / "metrics.json", measured)
    if render:
        from examples.readme_figure import render_figure
        scene = render_figure(directory, predictions, measured, output / "localization")
        print(f"3D replay: {scene.resolve()}")
    print(f"Results: {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="Fit both baselines again from the original data")
    parser.add_argument("--data-root", type=Path, help="Directory holding trainingData.csv and validationData.csv")
    parser.add_argument("--case", type=Path, default=CASE_DIR, help="Directory with the recorded example")
    parser.add_argument("--output", type=Path, default=Path("work_dirs/readme_demo"))
    parser.add_argument("--no-render", action="store_true", help="Skip building the HTML replay")
    args = parser.parse_args()
    directory = args.case
    if args.rebuild:
        directory = args.output / "case"
        rebuild_case(directory, args.data_root)
    run_example(directory, args.output, render=not args.no_render)


if __name__ == "__main__":
    main()
