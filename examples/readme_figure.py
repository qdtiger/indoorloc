"""Build a self-contained 3D replay from the README experiment artifacts."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "readme_scene_vendor"
METHODS = ("knn", "wknn")
K = 5  # Fixed in the recorded protocol for both baselines.
TIE_GAP = 1e-4  # Featured scans need an unambiguous 5th neighbor in signal space.
SELECTION = ("Every featured scan has a 5th/6th neighbor gap above 1e-4 and the correct building for both methods. "
             "The first also has the correct floor, neighbors from at least 3 surveyed positions, and the KNN/WKNN "
             "mean error closest to the median over all held-out scans; the next ones apply that rule within each "
             "remaining building; the last is the scan both methods place on the wrong floor whose mean error is "
             "closest to the median, with no minimum number of positions.")
GEOMETRY = ("Schematic per-building envelope of the training and held-out positions on all floors (grown 9 m, "
            "then shrunk 5 m: about 4 m beyond the outermost positions), drawn identically on every floor; "
            "floor spacing exaggerated for legibility, not to scale")


def radio_map(directory):
    """The training radio map stored inside the fitted KNN model, and each scan's K+1 neighbours."""
    from examples.readme_demo import FILL_DBM, load_example
    from indoorloc.methods.neighbors import kneighbors

    model, test = load_example("knn", directory)
    knn = model.localizer_  # LocalizerPipeline(FillMissing, KNNLocalizer): the fitted radio map
    features = model.preprocess_.transform(test.X).astype(np.float64)
    # Exact search with ties broken by training index: identical on every machine and thread count.
    distances, neighbors = kneighbors(knn.X_fit_, features, K + 1, fit_sq=knn.fit_sq_)
    return {"fingerprints": knn.X_fit_, "xy": knn.pos_, "floor": knn.floor_, "building": knn.building_,
            "features": features, "distances": distances, "neighbors": neighbors, "fill": FILL_DBM}


def footprint(points, grow=9.0, shrink=5.0, cell=1.0, reach=10.0):
    """Signed distance (meters, negative inside) to the envelope of recorded positions."""
    origin = points.min(axis=0) - grow - reach
    size = np.ceil((points.max(axis=0) + grow + reach - origin) / cell).astype(int) + 1
    xs, ys = (origin[i] + np.arange(size[i]) * cell for i in range(2))
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    nearest = cKDTree(points).query(grid)[0].reshape(size[1], size[0])
    covered = ndimage.distance_transform_edt(nearest <= grow) * cell > shrink
    parts, count = ndimage.label(covered)
    area = ndimage.sum(covered, parts, range(1, count + 1)) * cell**2
    covered = np.isin(parts, 1 + np.flatnonzero(area >= 150))
    gaps, count = ndimage.label(~covered)
    open_gaps = np.unique(np.r_[gaps[0], gaps[-1], gaps[:, 0], gaps[:, -1]])
    area = ndimage.sum(~covered, gaps, range(1, count + 1)) * cell**2
    covered |= np.isin(gaps, [i + 1 for i in range(count) if area[i] < 100 and i + 1 not in open_gaps])
    signed = (ndimage.distance_transform_edt(~covered) - ndimage.distance_transform_edt(covered)) * cell
    signed = ndimage.gaussian_filter(signed, 0.8)
    encoded = np.clip(np.round((signed + reach) / (2 * reach) * 255), 0, 255).astype(np.uint8)
    return {"origin": origin.round(4).tolist(), "cell": cell, "size": size.tolist(), "reach": reach,
            "sdf": base64.b64encode(encoded.tobytes()).decode()}


def featured_scans(data, predictions, errors, radio):
    """Pick the showcased scans by the fixed rule in SELECTION."""
    average = np.mean([errors[method] for method in METHODS], axis=0)
    target = np.median(average)
    gap = radio["distances"][:, K] - radio["distances"][:, K - 1]
    reference = np.unique(np.c_[radio["xy"], radio["floor"], radio["building"]], axis=0, return_inverse=True)[1]
    distinct = np.array([len(set(reference[row[:K]])) for row in radio["neighbors"]])
    floor_ok = np.all([predictions[m][:, 2] == data["floor"] for m in METHODS], axis=0)
    floor_wrong = np.all([predictions[m][:, 2] != data["floor"] for m in METHODS], axis=0)
    building_ok = np.all([predictions[m][:, 3] == data["building"] for m in METHODS], axis=0)
    usable = (gap > TIE_GAP) & building_ok

    def closest(mask):
        candidates = np.flatnonzero(mask)
        if not len(candidates):
            return None
        return int(candidates[np.argsort(np.abs(average[candidates] - target), kind="stable")[0]])

    typical = usable & floor_ok & (distinct >= 3)
    first = closest(typical)
    if first is None:
        raise ValueError("No held-out scan satisfies the featured-scan rule")
    chosen = [(first, "median")]
    chosen += [(closest(typical & (data["building"] == b)), f"building {b}")
               for b in sorted(set(data["building"].tolist()) - {int(data["building"][first])})]
    chosen.append((closest(usable & floor_wrong), "floor error"))
    return [(index, rule) for index, rule in chosen if index is not None]


def fingerprint_rows(raw, neighbor_features, names, fill, limit=20):
    """Query RSSI plus its neighbors' stored fingerprints, in dBm, over the APs any of them heard."""
    heard = np.flatnonzero(~np.isnan(raw))
    columns = heard[np.argsort(-raw[heard], kind="stable")].tolist()
    stored = np.where(neighbor_features > fill, neighbor_features, np.nan)
    extra = np.flatnonzero(np.any(~np.isnan(stored), axis=0) & np.isnan(raw))
    counts = np.sum(~np.isnan(stored[:, extra]), axis=0)
    strength = np.nanmean(stored[:, extra], axis=0)
    columns += extra[np.lexsort((-strength, -counts))].tolist()
    columns = columns[:limit]
    rows = [[None if np.isnan(raw[c]) else int(raw[c]) for c in columns]]
    rows += [[None if np.isnan(row[c]) else int(row[c]) for c in columns] for row in stored]
    return {"aps": [str(names[c]) for c in columns], "rows": rows}


def dataset_catalog():
    """Registered datasets grouped by the signal they carry (read from class attributes, nothing loaded)."""
    from indoorloc.datasets import DATASETS, list_datasets

    kind = {"wifi_rssi": "wifi", "ble_rssi": "ble", "csi": "csi", "csi_amp": "csi"}
    catalog = {key: [] for key in ("wifi", "ble", "csi", "multi", "simulated")}
    for name in list_datasets():
        cls = DATASETS.get(name)
        if getattr(cls, "name", name) != name:  # an alias
            continue
        if ".simulated." in cls.__module__:
            group = "simulated"
        elif name == "ilc2020":  # IMU + WiFi + BLE + floor plans in one recording
            group = "multi"
        else:
            group = kind.get(cls.meta.get("modality"), "other")
        catalog.setdefault(group, []).append(name)
    return {key: ids for key, ids in catalog.items() if ids}


def scene_data(directory, predictions, metrics):
    directory = Path(directory)
    with np.load(directory / "samples.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    data["source_row"] = np.arange(len(data["xy"]))
    radio = radio_map(directory)
    positions = np.r_[radio["xy"].astype(np.float64), data["xy"]]
    center = (positions.min(axis=0) + positions.max(axis=0)) / 2
    errors = {method: np.linalg.norm(predictions[method][:, :2] - data["xy"], axis=1) for method in METHODS}

    keys, reference, counts = np.unique(np.c_[radio["xy"], radio["floor"], radio["building"]], axis=0,
                                        return_inverse=True, return_counts=True)
    buildings = []
    for b in sorted(set(radio["building"].tolist())):
        points = np.r_[keys[keys[:, 3] == b, :2].astype(np.float64), data["xy"][data["building"] == b]]
        outline = footprint(np.unique(points.round(1), axis=0) - center)
        buildings.append({"id": b, "floors": sorted(set(radio["floor"][radio["building"] == b].tolist())),
                          "footprint": outline})

    featured = []
    fingerprints = radio["fingerprints"].astype(np.float64)
    for index, rule in featured_scans(data, predictions, errors, radio):
        nearest = radio["neighbors"][index, :K]
        distance = radio["distances"][index, :K]
        signal = np.linalg.norm(fingerprints - radio["features"][index], axis=1)
        closest = np.full(len(keys), np.inf)
        np.minimum.at(closest, reference, signal)
        cutoff = (distance[-1] + radio["distances"][index, K]) / 2
        # Exact counts for the narrowing animation: keeping the n closest positions (by their closest
        # fingerprint) keeps positions[n-1] positions (ties included) and fingerprints[n-1] fingerprints.
        by_position = np.sort(closest)
        weights = 1 / distance
        featured.append({
            "index": index, "rule": rule, "observed": int(np.sum(~np.isnan(data["rssi"][index]))),
            "neighbors": [{"fingerprint": int(j), "reference": int(reference[j]), "distance": float(d)}
                          for j, d in zip(nearest, distance)],
            "weights": {"knn": [1 / K] * K, "wknn": (weights / weights.sum()).tolist()},
            "cutoff": float(cutoff), "referenceDistance": closest.round(5).tolist(),
            "referenceRank": np.searchsorted(by_position, closest, side="left").tolist(),
            "narrowing": {"positions": np.searchsorted(by_position, by_position, side="right").tolist(),
                          "fingerprints": np.searchsorted(np.sort(signal), by_position, side="right").tolist()},
            "fingerprint": fingerprint_rows(data["rssi"][index], radio["fingerprints"][nearest], data["ap_names"],
                                            radio["fill"]),
            # The full scan: dBm (null = not heard) and the model input drawn on a 0..1 scale.
            "raw": [None if np.isnan(v) else int(v) for v in data["rssi"][index]],
            "features": ((radio["features"][index] - radio["fill"]) / -radio["fill"]).round(5).tolist(),
        })

    return {
        "dataset": "UJIIndoorLoc", "coordinateOrigin": center.tolist(), "k": K,
        # Loader ids by signal type; only the dataset with a committed recorded run counts as verified.
        "catalog": dataset_catalog(), "verified": ["ujiindoorloc"],
        "reference": {"xy": (keys[:, :2].astype(np.float64) - center).round(3).tolist(),
                      "floor": keys[:, 2].astype(int).tolist(), "building": keys[:, 3].astype(int).tolist(),
                      "count": counts.tolist()},
        "trainingFingerprints": len(radio["xy"]), "accessPoints": int(data["rssi"].shape[1]),
        "buildings": buildings,
        "scans": {
            "row": data["source_row"].astype(int).tolist(),
            "truth": (data["xy"] - center).round(7).tolist(),
            "floor": data["floor"].astype(int).tolist(), "building": data["building"].astype(int).tolist(),
            "results": {method: {
                "prediction": (predictions[method][:, :2] - center).round(7).tolist(),
                "floor": predictions[method][:, 2].astype(int).tolist(),
                "building": predictions[method][:, 3].astype(int).tolist(),
                "error": errors[method].tolist(),
            } for method in METHODS},
        },
        "featured": featured,
        "methods": {method: {"metrics": metrics[method], "errors": np.sort(errors[method]).tolist()}
                    for method in METHODS},
        "apps": apps_data(directory),
        "evaluationSamples": len(data["xy"]), "duration": 22, "selection": SELECTION,
        "geometry": GEOMETRY,
    }


def _clip(walls, lo, hi):
    """Wall segments (W, 4) clipped to the box [lo, hi] (Liang-Barsky); segments outside are dropped."""
    out = []
    for x0, y0, x1, y1 in walls:
        t0, t1, dx, dy = 0.0, 1.0, x1 - x0, y1 - y0
        for p, q in ((-dx, x0 - lo[0]), (dx, hi[0] - x0), (-dy, y0 - lo[1]), (dy, hi[1] - y0)):
            if p == 0:
                if q < 0:
                    t0, t1 = 1.0, 0.0
            elif p < 0:
                t0 = max(t0, q / p)
            else:
                t1 = min(t1, q / p)
        if t0 < t1:
            out.append([x0 + t0 * dx, y0 + t0 * dy, x0 + t1 * dx, y0 + t1 * dy])
    return np.asarray(out, dtype=np.float64).reshape(-1, 4)


def apps_data(directory):
    """The recorded L5 case (examples/readme_demo.py record_apps), centred on the shown trace."""
    record = json.loads((Path(directory) / "apps_ilc2020.json").read_text(encoding="utf-8"))
    shown = record["shown"]
    center = (np.asarray(shown["bounds"][0]) + np.asarray(shown["bounds"][1])) / 2
    rel = lambda xy: (np.asarray(xy, dtype=np.float64).reshape(-1, 2) - center).round(2).tolist()  # noqa: E731
    walls = _clip(np.asarray(shown["walls"], dtype=np.float64).reshape(-1, 4), *shown["bounds"]) - np.r_[center, center]
    return {"dataset": "ILC 2020 site1/F1", "traces": record["traces"], "testTraces": record["test_traces"],
            "waypoints": record["waypoints"], "methods": record["methods"],
            "size": (np.asarray(shown["bounds"][1]) - np.asarray(shown["bounds"][0])).round(2).tolist(),
            "truth": rel(shown["waypoints"]), "fixes": rel(shown["fixes"]),
            "rts": rel(shown["tracks"]["+ Kalman RTS"]), "fused": rel(shown["tracks"]["PDR + WiFi + map"]),
            "walls": walls.round(2).tolist(), "shownErrors": shown["errors"]}


def render_figure(directory, predictions, metrics, destination):
    """Write an offline HTML animation; measurement and prediction arrays stay exact."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = scene_data(directory, predictions, metrics)
    encode = lambda path: base64.b64encode((VENDOR / path).read_bytes()).decode()
    module = encode("three.module.min.js")
    fonts = {"__FONT_SANS__": encode("fonts/inter-latin-wght.woff2"),
             "__FONT_MONO__": encode("fonts/jetbrains-mono-latin-wght.woff2"),
             "__FONT_CJK__": encode("fonts/noto-sans-sc-subset.woff2")}
    template = (HERE / "readme_scene.html").read_text(encoding="utf-8")
    script = (HERE / "readme_scene.js").read_text(encoding="utf-8")
    payload = json.dumps(data, separators=(",", ":"))
    signature = hashlib.sha256((payload + template + script + module).encode()).hexdigest()
    fallback = None
    for previous in (destination.with_suffix(".html"), HERE.parent / "assets/readme/localization.html"):
        if not previous.is_file():
            continue
        source = previous.read_text(encoding="utf-8")
        if f'name="indoorloc-scene" content="{signature}"' in source:
            match = re.search(r'<script id="scene-fallback" type="application/json">(.*?)</script>', source)
            if match:
                fallback = json.loads(match.group(1))
                if fallback:
                    break
    document = template.replace("__SCENE_DATA__", payload).replace("__SCENE_SIGNATURE__", signature)
    document = document.replace("__SCENE_FALLBACK__", json.dumps(fallback))
    for placeholder, font in fonts.items():
        document = document.replace(placeholder, font)
    document = document.replace("__THREE_MODULE__", module).replace("__SCENE_SCRIPT__", script)
    destination.with_suffix(".html").write_text(document, encoding="utf-8")
    return destination.with_suffix(".html")


def embed_animation_fallback(html, animation):
    """Keep the rendered replay available when the viewer has no WebGL context."""
    html = Path(html)
    encoded = "data:image/webp;base64," + base64.b64encode(Path(animation).read_bytes()).decode()
    source = re.sub(r'(<script id="scene-fallback" type="application/json">).*?(</script>)',
                    lambda match: match.group(1) + json.dumps(encoded) + match.group(2), html.read_text(encoding="utf-8"))
    html.write_text(source, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=HERE / "readme_case")
    parser.add_argument("--output", type=Path, default=HERE.parent / "assets/readme/localization")
    args = parser.parse_args()
    from examples.readme_demo import verify_case
    verify_case(args.case)
    with np.load(args.case / "predictions.npz", allow_pickle=False) as archive:
        predictions = {key: archive[key] for key in archive.files}
    recorded = json.loads((args.case / "results.json").read_text(encoding="utf-8"))
    metrics = {key: value["metrics"] for key, value in recorded["methods"].items()}
    print(render_figure(args.case, predictions, metrics, args.output))


if __name__ == "__main__":
    main()
