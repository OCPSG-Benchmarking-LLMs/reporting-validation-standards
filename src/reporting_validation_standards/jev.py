"""Strict parsing and design-weighted metrics for the JEV demonstration."""
from __future__ import annotations

import hashlib
import json

import numpy as np


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":")).encode()).hexdigest()


def extract_choice(response: dict, labels: list[str]) -> dict:
    """Fail closed on malformed distributions rather than silently normalising them."""
    answer = response["answers"]["policy"]
    if answer["type"] != "choice":
        raise ValueError("Expected a choice answer")
    probabilities = answer["probabilities"]
    if set(probabilities) != set(labels):
        raise ValueError("Probability keys do not match the complete label inventory")
    values = np.asarray([probabilities[k] for k in labels], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Invalid probability values")
    if not np.isclose(values.sum(), 1, atol=1e-4):
        raise ValueError("Probabilities do not sum to one")
    winner = answer["choice"]
    if winner not in labels or not np.isclose(values[labels.index(winner)], values.max()):
        raise ValueError("Reported choice is inconsistent with probabilities")
    confidence = float(answer["confidence"])
    if not np.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Invalid confidence")
    return {"prediction": winner, "confidence": confidence,
            "probabilities": {k: float(probabilities[k]) for k in labels}}


def probability_response_diagnostics(response: dict, labels: list[str]) -> dict:
    """Preserve provider metadata and raw probabilities from rejected responses."""
    answer = response.get("answers", {}).get("policy", {})
    result = {"model_returned": response.get("model"), "usage": response.get("usage")}
    if not isinstance(answer, dict):
        return result
    for source, target in (("choice", "raw_choice"), ("confidence", "raw_confidence")):
        if source in answer:
            result[target] = answer[source]
    raw = answer.get("probabilities")
    if not isinstance(raw, dict):
        result["probability_diagnostics_status"] = "absent_or_unexpected_type"
        return result
    result["raw_probabilities"] = raw
    result["probability_missing_labels"] = sorted(set(labels) - set(raw))
    result["probability_unexpected_labels"] = sorted(set(raw) - set(labels))
    try:
        values = np.asarray([raw[label] for label in labels], dtype=float)
    except (KeyError, TypeError, ValueError):
        result["probability_diagnostics_status"] = "incomplete_or_non_numeric"
        return result
    if not np.isfinite(values).all():
        result["probability_diagnostics_status"] = "non_finite"
        return result
    result["probability_sum"] = float(values.sum())
    result["probability_sum_minus_one"] = float(values.sum() - 1.0)
    result["probability_diagnostics_status"] = "complete"
    return result


def weighted_metrics(probabilities: np.ndarray, truth: np.ndarray,
                     weights: np.ndarray) -> dict:
    """Multiclass proper scores, top-label calibration and risk at fixed coverage."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(truth, dtype=int)
    w = np.asarray(weights, dtype=float)
    if p.ndim != 2 or len(p) != len(y) or len(y) != len(w):
        raise ValueError("Shape mismatch")
    if not len(y) or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("Invalid weights")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-4):
        raise ValueError("Invalid probabilities")
    if (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError("Truth index outside inventory")
    z = np.eye(p.shape[1])[y]
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(float)
    conf = p.max(axis=1)
    brier = ((p - z) ** 2).sum(axis=1)
    logloss = -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1))
    scale = w.sum()
    out = {
        "n": len(y),
        "accuracy": float(np.dot(w, correct) / scale),
        "brier_multiclass": float(np.dot(w, brier) / scale),
        "log_loss": float(np.dot(w, logloss) / scale),
        "mean_top_probability": float(np.dot(w, conf) / scale),
        "top_probability_calibration_gap": float(np.dot(w, conf - correct) / scale),
    }
    # Stable tie-breaking by original observation order; report achieved weighted coverage.
    ordering = np.argsort(-conf, kind="stable")
    for target in (0.75, 0.90, 1.0):
        cumulative = np.cumsum(w[ordering]) / scale
        take = np.searchsorted(cumulative, target, side="left") + 1
        indices = ordering[:take]
        out[f"coverage_{int(target*100)}"] = float(w[indices].sum() / scale)
        out[f"risk_{int(target*100)}"] = float(
            np.dot(w[indices], 1 - correct[indices]) / w[indices].sum()
        )
    return out


def reliability_bins(confidence: np.ndarray, correct: np.ndarray,
                     weights: np.ndarray) -> list[dict]:
    """Exhaustive weighted 0.1-wide bins; rightmost bin includes exactly 1.0.

    Bin indices, rather than repeated floating-point bounds, prevent
    boundary values such as 0.6 from falling between adjacent bins.
    """
    conf = np.asarray(confidence, dtype=float)
    acc = np.asarray(correct, dtype=float)
    w = np.asarray(weights, dtype=float)
    if conf.ndim != 1 or acc.shape != conf.shape or w.shape != conf.shape:
        raise ValueError("Reliability vector shapes must agree")
    if not np.isfinite(conf).all() or ((conf < 0) | (conf > 1)).any():
        raise ValueError("Invalid confidence values")
    if not np.isfinite(acc).all() or ((acc < 0) | (acc > 1)).any():
        raise ValueError("Invalid correctness values")
    if not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("Reliability weights must be finite and positive")

    indices = np.minimum(np.floor(conf * 10).astype(int), 9)
    results = []
    total = float(w.sum())
    for index in range(10):
        selected = indices == index
        if not selected.any():
            continue
        results.append({
            "lower": index / 10,
            "upper": (index + 1) / 10,
            "n": int(selected.sum()),
            "estimated_population_mass": float(w[selected].sum() / total),
            "mean_confidence": float(np.average(conf[selected], weights=w[selected])),
            "observed_accuracy": float(np.average(acc[selected], weights=w[selected])),
        })
    if sum(row["n"] for row in results) != len(conf):
        raise RuntimeError("Some valid classifications were omitted from reliability bins")
    if not np.isclose(sum(row["estimated_population_mass"] for row in results), 1):
        raise RuntimeError("Reliability bin weights do not sum to one")
    return results
