"""IndoorLoc in one screen: load a public dataset, fit a pipeline, evaluate it, save and reload it.

    python examples/quickstart.py                                # UJIIndoorLoc, downloaded once
    python examples/quickstart.py --dataset synthetic_office     # simulated, no download

Five layers exchange plain numpy arrays:

    L1  load_dataset("ujiindoorloc")        -> (train, test) SampleTables, sha256-checked files
    L2  FillMissing(-104)                    "not heard" (NaN) -> -104 dBm
    L3  create_model("wknn", k=5, ...)       sklearn-style fit / predict / localize
    L4  model.evaluate(test)                 mean / median / P90 error, floor and building hit rates
        model.save(path), load_model(path)   JSON + npz, no pickle

The last part shows the two core types with your own arrays: any ``(N, F)`` feature array
and ``(N, D)`` positions make a ``SampleTable``; every localizer returns a ``Prediction``;
L4 scores plain arrays. UJIIndoorLoc positions are EPSG:3857 (Web Mercator) metres, the
unit of published UJIIndoorLoc results; ``scale=meta["ground_scale"]`` (0.766) converts
errors to ground metres. All numbers are printed; every step is deterministic (the one random
split is seeded), so a re-run prints the same values.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

import indoorloc as iloc


def main(dataset: str = "ujiindoorloc", *, download: bool = True, workdir: Path | str | None = None,
         verbose: bool = True) -> dict:
    """Run the walkthrough; return the printed numbers."""
    say = print if verbose else (lambda *a, **k: None)

    # L1 -- a public dataset as two SampleTables (arrays + labels + provenance)
    train, test = iloc.load_dataset(dataset, download=download)
    digest = train.meta.get("sha256")
    say(f"{train.meta.get('citation', dataset)}: train {train.X.shape}, test {test.X.shape}, "
        f"crs {train.meta.get('crs')}, " + (f"sha256 {digest[:12]}..." if digest else "generated from a seed"))

    # L2 + L3 -- preprocessing and localizer in one estimator
    model = iloc.create_model("wknn", k=5, preprocess=iloc.FillMissing(-104.0))
    model.fit(train)

    # L4 -- held-out scores (dataset units; ground metres where the frame is not metric)
    native = model.evaluate(test)
    scale = float(train.meta.get("ground_scale", 1.0))
    ground = model.evaluate(test, scale=scale)
    say(f"WKNN (k=5), {train.meta.get('pos_units', 'm')}: {native}")
    if scale != 1.0:
        say(f"WKNN (k=5), ground metres (x {scale:.4f}): {ground}")

    # save / load: config.json + arrays.npz, no pickle; the reloaded model predicts identically
    with tempfile.TemporaryDirectory() as tmp:
        path = model.save(Path(workdir or tmp) / "wknn_model", info={"dataset": dataset})
        loaded = iloc.load_model(path)
        same = bool(np.array_equal(loaded.predict(test), model.predict(test)))
        size_kb = sum(f.stat().st_size for f in Path(path).iterdir()) / 1024
    say(f"saved to {path.name}/ ({size_kb:.0f} kB: config.json + arrays.npz); reloaded predictions identical: {same}")

    # The core types with your own arrays: one building's scans as a new table
    rows = train.building == train.building.min() if train.building is not None else slice(None)
    X, pos = train.X[rows], train.pos[rows]                    # plain numpy arrays (read-only views)
    mine = iloc.SampleTable(X, pos, floor=None if train.floor is None else train.floor[rows])
    fit_rows, test_rows = iloc.evaluation.random_split(len(mine), 0.3, random_state=0)  # L4: index arrays
    filled = np.nan_to_num(mine.X, nan=-104.0)
    knn = iloc.KNNLocalizer(k=3).fit(filled[fit_rows], mine.pos[fit_rows])  # sklearn style on arrays
    pred = knn.localize(filled[test_rows])                           # Prediction(pos, floor, building, spread)
    scores = iloc.evaluate(mine.pos[test_rows], pred.pos, scale=scale)  # L4 on plain arrays
    say(f"own arrays: SampleTable {mine.X.shape} -> Prediction pos {pred.pos.shape}, spread[:3] "
        f"{np.round(pred.spread[:3] * scale, 2).tolist()}; 3-NN, random 70/30 split of these scans: "
        f"mean {scores.mean_error:.2f} m, median {scores.median_error:.2f} m (scans of the same reference points "
        "land on both sides of a random split, so it is optimistic next to a held-out test set)")
    return {"mean_error": native.mean_error, "median_error": native.median_error,
            "floor_accuracy": native.floor_accuracy, "building_accuracy": native.building_accuracy,
            "mean_error_ground": ground.mean_error, "reload_identical": same,
            "own_arrays_mean_error": scores.mean_error}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default="ujiindoorloc", help="a dataset with train and test splits")
    parser.add_argument("--no-download", action="store_true", help="fail instead of downloading")
    args = parser.parse_args()
    main(args.dataset, download=not args.no_download)
