"""Design-based plotting statistics for the frozen JEV demonstration.

All estimates target the automated silver reference and condition on valid
first-pass JEV probability vectors. No API calls, fitting, or threshold tuning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .jev import reliability_bins


def _check_vectors(
    confidence: np.ndarray, correct: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if (
        confidence.ndim != 1
        or correct.shape != confidence.shape
        or weights.shape != confidence.shape
        or not len(confidence)
    ):
        raise ValueError("Nonempty one-dimensional confidence, correctness and weights required")
    if (
        not np.isfinite(confidence).all()
        or ((confidence < 0) | (confidence > 1)).any()
        or not np.isfinite(correct).all()
        or ((correct < 0) | (correct > 1)).any()
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or weights.sum() <= 0
    ):
        raise ValueError("Invalid classification values or bootstrap weights")
    return confidence, correct, weights


def selective_risk(
    confidence: np.ndarray,
    correct: np.ndarray,
    weights: np.ndarray,
    coverages: np.ndarray,
) -> np.ndarray:
    """Expected design-weighted error at fixed coverage, fractionally averaging ties.

    For a confidence tie straddling the desired coverage, select a random
    proportion of its weight in expectation. This is deterministic in aggregate
    and independent of the input row order. The estimand is not an implemented
    adjudication policy: it assumes neither error-free reviewers nor a realised
    random assignment of tied cases.
    """
    confidence, correct, weights = _check_vectors(confidence, correct, weights)
    c = np.asarray(coverages, dtype=float)
    if c.ndim != 1 or not len(c) or not np.isfinite(c).all() or ((c <= 0) | (c > 1)).any():
        raise ValueError("Coverage grid must lie in (0, 1]")
    order = np.argsort(-confidence, kind="stable")
    scores = confidence[order]
    sorted_weight = weights[order]
    sorted_error = sorted_weight * (1.0 - correct[order])
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(scores) != 0)]
    group_weights = np.add.reduceat(sorted_weight, starts)
    group_errors = np.add.reduceat(sorted_error, starts)
    # Bootstrapped units can have zero replicate weight; skip empty groups.
    nonempty = group_weights > 0
    group_weights = group_weights[nonempty]
    group_errors = group_errors[nonempty]
    cumulative_weight = np.cumsum(group_weights)
    cumulative_error = np.cumsum(group_errors)
    desired = c * cumulative_weight[-1]
    positions = np.minimum(np.searchsorted(cumulative_weight, desired, side="left"),
                           len(cumulative_weight) - 1)
    before_weight = np.r_[0, cumulative_weight[:-1]][positions]
    before_error = np.r_[0, cumulative_error[:-1]][positions]
    fraction = np.clip(
        (desired - before_weight) / group_weights[positions], 0, 1
    )
    included_error = before_error + fraction * group_errors[positions]
    return np.clip(included_error / desired, 0, 1)


def plot_data_with_uncertainty(
    confidence: np.ndarray,
    correct: np.ndarray,
    weights: np.ndarray,
    replicate_weights: np.ndarray,
    *,
    coverage_grid: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build reliability and risk curves with stratified bootstrap percentile CIs.

    Bootstrap rows must be in the same observation order as the point estimate.
    Empty replicate reliability bins do not contribute fake zero accuracies:
    their nonempty counts are reported explicitly.
    """
    confidence, correct, weights = _check_vectors(confidence, correct, weights)
    boot = np.asarray(replicate_weights, dtype=float)
    if boot.ndim != 2 or boot.shape[1] != len(confidence) or not len(boot):
        raise ValueError("Bootstrap weights must have shape [replications, valid cases]")
    if not np.isfinite(boot).all() or (boot < 0).any() or (boot.sum(axis=1) <= 0).any():
        raise ValueError("Invalid bootstrap weights")

    coverages = (
        np.linspace(0.10, 1.0, 181)
        if coverage_grid is None
        else np.asarray(coverage_grid, dtype=float)
    )
    if coverages.ndim != 1 or not len(coverages) or not np.isfinite(coverages).all():
        raise ValueError("Invalid coverage grid")
    if ((coverages <= 0) | (coverages > 1)).any() or np.any(np.diff(coverages) <= 0):
        raise ValueError("Coverages must be strictly increasing within (0, 1]")

    bins = reliability_bins(confidence, correct, weights)
    bin_ids = np.minimum(np.floor(confidence * 10).astype(int), 9)
    bin_boot = np.full((len(boot), 10), np.nan, dtype=float)
    risk_boot = np.empty((len(boot), len(coverages)), dtype=float)
    overall_boot = np.empty(len(boot), dtype=float)

    for index, replicate in enumerate(boot):
        mass = np.bincount(bin_ids, weights=replicate, minlength=10)
        hits = np.bincount(bin_ids, weights=replicate * correct, minlength=10)
        np.divide(hits, mass, out=bin_boot[index], where=mass > 0)
        risk_boot[index] = selective_risk(
            confidence, correct, replicate, coverages
        )
        overall_boot[index] = np.dot(replicate, 1 - correct) / replicate.sum()

    observed_risk = selective_risk(confidence, correct, weights, coverages)
    risk = pd.DataFrame({
        "coverage": coverages,
        "selective_error": observed_risk,
        "ci_low": np.quantile(risk_boot, .025, axis=0),
        "ci_high": np.quantile(risk_boot, .975, axis=0),
        "random_review_expected_error": float(
            np.dot(weights, 1 - correct) / weights.sum()
        ),
    })
    rows = []
    for base in bins:
        index = min(int(base["lower"] * 10 + 1e-8), 9)
        observed = bin_boot[:, index]
        observed = observed[np.isfinite(observed)]
        if len(observed) < 2:
            raise ValueError(f"Insufficient nonempty bootstrap draws for confidence bin {index}")
        rows.append({
            **base,
            "ci_low": float(np.quantile(observed, .025)),
            "ci_high": float(np.quantile(observed, .975)),
            "nonempty_bootstrap_replicates": int(len(observed)),
        })
    calibration = pd.DataFrame(rows)
    reference = {
        "random_review_expected_error": float(risk["random_review_expected_error"].iloc[0]),
        "random_review_ci_low": float(np.quantile(overall_boot, .025)),
        "random_review_ci_high": float(np.quantile(overall_boot, .975)),
        "bootstrap_reps": len(boot),
        "valid_cases": len(confidence),
        "tie_policy": "uniform_random_within_equal_top_probabilities_in_expectation",
        "interpretation": "conditional_on_valid_outputs_and_automated_silver_labels",
    }
    return calibration, risk, reference
