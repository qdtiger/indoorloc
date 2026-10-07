"""Recompute the cross-check cells of the matrix outside the library and write ``results/crosscheck.json``.

The *Cross-checks* table of ``docs/benchmarks.md`` compares a few matrix cells with values known
before the matrix ran. Four of those values are regression values of this library itself, so a
match only shows that nothing changed. This script adds an outside view: the same k-NN cells
recomputed from the loaders' arrays by a short plain-numpy k-NN (and, for comparison, by
scikit-learn's brute-force ``KNeighborsRegressor``) that shares no code with
``indoorloc.methods`` or ``indoorloc.signals``::

    python -m benchmarks.crosscheck [--results-dir benchmarks/results]

The numpy k-NN orders neighbours by (squared distance, training index), the rule the library
documents, and computes the squared distances exactly whenever the readings allow it. With
``g`` the smallest integer for which every value times 2**g is an integer (whole dBm: g = 0;
BBIL's averaged float32 readings: g = 18), and F features:

* ``exact-gemm``: if 4 F (2**g max|x|)**2 < 2**53, every term of ``|q|^2 + |t|^2 - 2 q.t`` on the
  scaled values is an integer below 2**53, so the matrix product is exact in any order
  (UJIIndoorLoc, TUJI1);
* ``exact-diff``: else if F (2**g max|q - t|)**2 < 2**53, the differences, their squares and
  their sum are exact integers (BBIL);
* ``float``: otherwise (HALOC's CSI amplitudes) an ordinary float64 matrix product; the result
  records how many test rows have their k-th and (k+1)-th distances within 1e-9 (relative),
  the only rows rounding could rank differently.

Whole-dBm data have many exact ties, and an implementation that breaks them differently gets a
slightly different mean; the result records how many test rows tie at the k-th neighbour and
what scikit-learn's search (which does not order ties by index) gives.

References
----------
T. Cover, P. Hart, "Nearest neighbor pattern classification", IEEE Transactions on Information
Theory 13(1):21-27, 1967. DOI: 10.1109/TIT.1967.1053964
J. Torres-Sospedra, R. Montoliu, S. Trilles, O. Belmonte, J. Huerta, "Comprehensive analysis of
distance and similarity measures for Wi-Fi fingerprinting indoor positioning systems", Expert
Systems with Applications 42(23):9263-9278, 2015. DOI: 10.1016/j.eswa.2015.08.013 (the positive
representation used by the TUJI1 check).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FORMAT = "indoorloc-benchmark-crosscheck"
FORMAT_VERSION = 1


# ------------------------------------------------------------ features (independent of indoorloc.signals)
def fill(value: float):
    """Missing readings (NaN) -> ``value``; returns float64 rows."""
    return lambda train, X: np.where(np.isnan(X), value, X).astype(np.float64).reshape(len(X), -1)


def positive(train, X):
    """Torres-Sospedra et al.'s positive representation: ``x - min`` for heard readings, 0 for missing
    ones and readings at or below ``min``, with ``min`` = lowest training reading - 1 dB."""
    low = float(np.nanmin(train)) - 1.0
    X = np.asarray(X, dtype=np.float64).reshape(len(X), -1)
    return np.where(np.isnan(X) | (X <= low), 0.0, X - low)


def amplitude(train, X):
    """``|H|`` of complex CSI, flattened."""
    return np.abs(X).astype(np.float64).reshape(len(X), -1)


# --------------------------------------------------------------------------- k-NN
def knn_positions(A, P, Q, k: int, weights: str, *, chunk_bytes: float = 2e8) -> tuple[np.ndarray, dict]:
    """Plain-numpy k-NN estimates of ``Q`` from training rows ``A`` at positions ``P``.

    Neighbours are ordered by (squared distance, training index). ``weights``: ``"uniform"`` (mean
    position) or ``"distance"`` (weights 1/d; a query with an exact match takes the mean of its exact
    matches, as scikit-learn does). Returns the (len(Q), D) estimates and a dict: ``mode`` (how the
    distances were computed, see the module docstring), ``grid_bits`` (g), ``ties_at_k`` (test rows
    with more than k training rows at or below the k-th distance) and, in float mode, ``near_ties``.
    """
    A, Q, P = np.asarray(A, np.float64), np.asarray(Q, np.float64), np.asarray(P, np.float64)
    if weights not in ("uniform", "distance"):
        raise ValueError(f"weights must be 'uniform' or 'distance', got {weights!r}")
    mode, g = "float", None
    for bits in range(41):  # the coarsest binary grid holding every value
        if np.all(np.ldexp(A, bits) % 1 == 0) and np.all(np.ldexp(Q, bits) % 1 == 0):
            g = bits
            break
    if g is not None:
        top = max(np.abs(A).max(), np.abs(Q).max()) * 2.0 ** g
        span = (max(A.max(), Q.max()) - min(A.min(), Q.min())) * 2.0 ** g
        if 4 * A.shape[1] * top ** 2 < 2.0 ** 53:
            mode = "exact-gemm"
        elif A.shape[1] * span ** 2 < 2.0 ** 53:
            mode = "exact-diff"
        if mode != "float":  # scaled values are integers; distances only change by the factor 2**g
            A, Q = np.ldexp(A, g), np.ldexp(Q, g)
    info = {"mode": mode, "grid_bits": g if mode != "float" else None, "ties_at_k": 0,
            "near_ties": 0 if mode == "float" else None}
    a2 = np.einsum("ij,ij->i", A, A)
    per_query = A.shape[0] * (A.shape[1] if mode == "exact-diff" else 3) * 8.0
    step = max(1, int(chunk_bytes // per_query))
    out = np.empty((len(Q), P.shape[1]))
    for s in range(0, len(Q), step):
        q = Q[s:s + step]
        if mode == "exact-diff":
            d2 = ((q[:, None, :] - A[None]) ** 2).sum(-1)
        else:
            d2 = np.einsum("ij,ij->i", q, q)[:, None] + a2[None] - 2.0 * (q @ A.T)
        idx = np.argsort(d2, axis=1, kind="stable")[:, :k]  # equal distances keep training-index order
        dk = np.take_along_axis(d2, idx, 1)
        info["ties_at_k"] += int(np.sum(np.sum(d2 <= dk[:, -1:], axis=1) > k))
        if mode == "float" and k < A.shape[0]:
            nxt = np.partition(d2, k, axis=1)[:, k]
            info["near_ties"] += int(np.sum(nxt - dk[:, -1] <= 1e-9 * np.maximum(np.abs(nxt), 1.0)))
        if weights == "uniform":
            out[s:s + step] = P[idx].mean(axis=1)
            continue
        d = np.sqrt(np.maximum(dk, 0.0))
        w = np.zeros_like(d)
        np.divide(1.0, d, out=w, where=d > 0)
        exact = d == 0
        rows = exact.any(axis=1)
        w[rows] = exact[rows]
        out[s:s + step] = (w[..., None] * P[idx]).sum(axis=1) / w.sum(axis=1)[:, None]
    return out, info


def sklearn_positions(A, P, Q, k: int, weights: str) -> np.ndarray | None:
    """The same estimate by scikit-learn's brute-force search (its ties are not ordered by index)."""
    try:
        from sklearn.neighbors import KNeighborsRegressor
    except ImportError:
        return None
    return KNeighborsRegressor(n_neighbors=k, weights=weights, algorithm="brute", n_jobs=1).fit(A, P).predict(Q)


# --------------------------------------------------------------------------- the checks
@dataclass(frozen=True)
class Check:
    """One matrix cell and how to recompute it: the dataset's official train/test files, features, k, weights."""

    dataset: str
    table: str
    preprocess: str
    method: str
    features: object
    k: int
    weights: str
    options: dict = field(default_factory=dict)


CHECKS = (
    Check("ujiindoorloc", "official", "fill", "knn", fill(-104.0), 5, "uniform"),
    Check("ujiindoorloc", "official", "fill", "wknn", fill(-104.0), 5, "distance"),
    Check("tuji1", "official", "positive", "knn(k=1)", positive, 1, "uniform"),
    Check("haloc", "official", "CSIAmplitude", "wknn", amplitude, 5, "distance"),
    Check("ble_indoor", "official-office", "fill(value=-110)", "wknn", fill(-110.0), 5, "distance",
          {"room": "office"}),
)


def recompute(check: Check, root=None) -> dict:
    """Mean error of one check on the dataset's train/test files, by numpy (and scikit-learn).

    ``root`` is the dataset's own folder (default: the library's cache folder for that dataset)."""
    from indoorloc.datasets import DATASETS  # the loaders only: data in, arrays out

    source = DATASETS.get(check.dataset)(root, download=False, **check.options)
    train, test = source.load("train"), source.load("test")
    A, Q = check.features(train.X, train.X), check.features(train.X, test.X)
    est, info = knn_positions(A, train.pos, Q, check.k, check.weights)
    row = {"dataset": check.dataset, "table": check.table, "preprocess": check.preprocess, "method": check.method,
           "options": check.options, "k": check.k, "weights": check.weights, "n_train": len(A), "n_test": len(Q),
           "sha256": {"train": train.meta.get("sha256"), "test": test.meta.get("sha256")},
           "numpy_mean_error": float(np.linalg.norm(est - test.pos, axis=1).mean()), **info}
    sk = sklearn_positions(A, train.pos, Q, check.k, check.weights)
    row["sklearn_mean_error"] = None if sk is None else float(np.linalg.norm(sk - test.pos, axis=1).mean())
    return row


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m benchmarks.crosscheck", description=__doc__.split("\n")[0])
    p.add_argument("--results-dir", type=Path, default=HERE / "results")
    p.add_argument("--data", type=Path, help="folder holding one sub-folder per dataset (default: $INDOORLOC_DATA "
                                             "or ~/.cache/indoorloc/datasets)")
    args = p.parse_args(argv)
    rows = []
    for check in CHECKS:
        row = recompute(check, args.data / check.dataset if args.data else None)
        rows.append(row)
        print(f"{check.dataset}/{check.table} | {check.preprocess} | {check.method}: numpy "
              f"{row['numpy_mean_error']:.4f} ({row['mode']}, {row['ties_at_k']} rows tie at the k-th neighbour)"
              + (f", scikit-learn {row['sklearn_mean_error']:.4f}" if row["sklearn_mean_error"] is not None else ""),
              file=sys.stderr)
    versions = {"numpy": np.__version__, "python": sys.version.split()[0]}
    if "sklearn" in sys.modules:
        versions["sklearn"] = sys.modules["sklearn"].__version__
    doc = {"format": FORMAT, "format_version": FORMAT_VERSION,
           "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "command": "python -m benchmarks.crosscheck", "versions": versions, "cells": rows}
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "crosscheck.json").write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n",
                                                      encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent))
    raise SystemExit(main())
