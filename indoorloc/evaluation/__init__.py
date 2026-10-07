"""L4: evaluation of plain arrays (numpy only).

functional   position errors, CDFs, label accuracies, ``evaluate`` -> ``EvaluationResults``
protocols    train/test index splits from a table's groups (official, random, k-fold,
             cross-device, cross-time, leave-one-group-out) and the named ``PROTOCOLS``
scoring      IPIN / EvAAL competition scores, floor-aware errors, CEP, bootstrap intervals
bounds       Cramér-Rao lower bounds (ToA, RSS, AoA, TDoA) and dilution of precision
literature   published numbers with provenance; ``compare`` keeps them apart from results
report       Markdown / text tables of results and of benchmark files
plot         CDF, error-map, bound-map and trajectory figures (``import indoorloc.evaluation.plot``; matplotlib)
"""
from __future__ import annotations

from . import bounds, literature, protocols, report, scoring
from .bounds import aoa_crlb, dop, gdop, rss_crlb, tdoa_crlb, toa_crlb
from .functional import EvaluationResults, error_cdf, evaluate, label_accuracy, position_errors
from .protocols import (PROTOCOLS, Fold, Protocol, cross_device_split, cross_time_split, get_protocol, group_split,
                        kfold, leave_one_group_out, list_protocols, pool_splits, random_split, register_protocol,
                        split_summary)
from .scoring import (bootstrap_ci, cep, evaal_etri_score, ipin_score, penalized_errors, percentile_error,
                      success_rate)

__all__ = [
    # functional
    "EvaluationResults", "error_cdf", "evaluate", "label_accuracy", "position_errors",
    # protocols
    "PROTOCOLS", "Fold", "Protocol", "cross_device_split", "cross_time_split", "get_protocol", "group_split", "kfold",
    "leave_one_group_out", "list_protocols", "pool_splits", "random_split", "register_protocol", "split_summary",
    # scoring
    "bootstrap_ci", "cep", "evaal_etri_score", "ipin_score", "penalized_errors", "percentile_error", "success_rate",
    # bounds
    "aoa_crlb", "dop", "gdop", "rss_crlb", "tdoa_crlb", "toa_crlb",
    # submodules
    "bounds", "literature", "protocols", "report", "scoring",
]
