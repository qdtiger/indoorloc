"""Trivial reference answers that every benchmark table includes.

A method that does not beat the training centroid has learned nothing about position: the
loader of ``csi_fingerprint`` shows that k-NN on held-out reference points can be *worse* than
this constant answer. The class follows the L3 contract (``BaseLocalizer``), so the command line
runs it like any registered method::

    indoorloc benchmark --dataset tuji1 --method benchmarks.baselines:TrainingCentroid
"""
from __future__ import annotations

import numpy as np

from indoorloc.core import Prediction
from indoorloc.methods import BaseLocalizer

__all__ = ["TrainingCentroid"]


def _mode(labels):
    """Most frequent label (ties: the smallest), or None without labels."""
    if labels is None:
        return None
    values, counts = np.unique(labels, return_counts=True)
    return values[np.argmax(counts)]  # np.unique sorts, argmax takes the first maximum


class TrainingCentroid(BaseLocalizer):
    """Answer the mean training position, the most frequent floor and building for every query.

    The prediction ignores the signal entirely, so its error is the spread of the test
    positions around the centre of the training positions: the floor any localization method
    must beat. Floor and building are the training majority (ties: smallest label).
    ``Prediction.spread`` is the RMS distance of the training positions from their centroid.
    The scan is never read, so any input is accepted (missing readings, complex CSI).

    References
    ----------
    P. Bahl, V. N. Padmanabhan, "RADAR: an in-building RF-based user location and tracking
    system", IEEE INFOCOM 2000. DOI: 10.1109/INFCOM.2000.832252 (compares RADAR against a
    random and a strongest-base-station answer: a trivial baseline in every table).
    """

    _allow_nan = True
    _allow_complex = True

    def __init__(self):
        pass

    def _fit(self, X, pos, floor, building):
        self.centroid_ = pos.mean(axis=0)
        self.spread_ = float(np.sqrt(((pos - self.centroid_) ** 2).sum(axis=1).mean()))
        self.floor_ = _mode(floor)
        self.building_ = _mode(building)

    def _localize(self, X) -> Prediction:
        n = len(X)
        full = lambda v: None if v is None else np.full(n, v, dtype=np.int64)  # noqa: E731
        return Prediction(np.tile(self.centroid_, (n, 1)), floor=full(self.floor_), building=full(self.building_),
                          spread=np.full(n, self.spread_))
