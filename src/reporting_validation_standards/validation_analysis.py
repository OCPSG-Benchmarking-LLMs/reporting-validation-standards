from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .stability_analysis import INVALID_LABEL


@dataclass(frozen=True)
class MetricBundle:
    valid_rate: float
    operational_accuracy: float
    conditional_accuracy: float
    operational_macro_precision: float
    operational_macro_recall: float
    operational_macro_f1: float
    conditional_macro_precision: float
    conditional_macro_recall: float
    conditional_macro_f1: float


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.zeros_like(numerator, dtype=float)
    mask = denominator > 0
    output[mask] = numerator[mask] / denominator[mask]
    return output


def confusion_metrics(
    confusion: np.ndarray,
    *,
    n_reference_labels: int,
    invalid_prediction_index: int,
) -> tuple[MetricBundle, pd.DataFrame]:
    """Compute operational and valid-output-conditional multiclass metrics.

    Rows are target-population reference labels. Columns are all allowed prediction
    labels plus one explicit invalid-output column. Macro metrics average only over
    the reference labels represented in the target population. Predictions into
    official labels absent from that reference population still count as errors for
    the true reference class.
    """
    matrix = np.asarray(confusion, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("confusion must be a two-dimensional matrix.")
    if matrix.shape[0] != n_reference_labels:
        raise ValueError("confusion row count does not match reference-label count.")
    if not 0 <= invalid_prediction_index < matrix.shape[1]:
        raise ValueError("invalid_prediction_index is outside the confusion matrix.")

    total_mass = float(matrix.sum())
    invalid_mass = float(matrix[:, invalid_prediction_index].sum())
    valid_mass = total_mass - invalid_mass
    valid_rate = valid_mass / total_mass if total_mass > 0 else np.nan

    target_columns = np.arange(n_reference_labels)
    tp = np.diag(matrix[:, target_columns])
    row_total_operational = matrix.sum(axis=1)
    col_total = matrix[:, target_columns].sum(axis=0)

    operational_precision = _safe_divide(tp, col_total)
    operational_recall = _safe_divide(tp, row_total_operational)
    operational_f1 = _safe_divide(
        2 * operational_precision * operational_recall,
        operational_precision + operational_recall,
    )

    operational_accuracy = (
        float(tp.sum() / total_mass) if total_mass > 0 else np.nan
    )

    conditional_matrix = matrix.copy()
    conditional_matrix[:, invalid_prediction_index] = 0.0
    row_total_conditional = conditional_matrix.sum(axis=1)
    conditional_tp = np.diag(conditional_matrix[:, target_columns])
    conditional_col_total = conditional_matrix[:, target_columns].sum(axis=0)

    conditional_precision = _safe_divide(
        conditional_tp,
        conditional_col_total,
    )
    conditional_recall = _safe_divide(
        conditional_tp,
        row_total_conditional,
    )
    conditional_f1 = _safe_divide(
        2 * conditional_precision * conditional_recall,
        conditional_precision + conditional_recall,
    )
    conditional_accuracy = (
        float(conditional_tp.sum() / valid_mass)
        if valid_mass > 0
        else np.nan
    )

    bundle = MetricBundle(
        valid_rate=float(valid_rate),
        operational_accuracy=float(operational_accuracy),
        conditional_accuracy=float(conditional_accuracy),
        operational_macro_precision=float(operational_precision.mean()),
        operational_macro_recall=float(operational_recall.mean()),
        operational_macro_f1=float(operational_f1.mean()),
        conditional_macro_precision=float(conditional_precision.mean()),
        conditional_macro_recall=float(conditional_recall.mean()),
        conditional_macro_f1=float(conditional_f1.mean()),
    )

    class_metrics = pd.DataFrame(
        {
            "reference_index": np.arange(n_reference_labels),
            "support_operational": row_total_operational,
            "support_valid": row_total_conditional,
            "true_positive": tp,
            "operational_precision": operational_precision,
            "operational_recall": operational_recall,
            "operational_f1": operational_f1,
            "conditional_precision": conditional_precision,
            "conditional_recall": conditional_recall,
            "conditional_f1": conditional_f1,
        }
    )
    return bundle, class_metrics


def build_confusion_matrix(
    records: pd.DataFrame,
    *,
    reference_labels: list[str],
    prediction_labels: list[str],
    weights: np.ndarray,
) -> np.ndarray:
    if len(records) != len(weights):
        raise ValueError("records and weights must have equal length.")

    reference_lookup = {
        label: index for index, label in enumerate(reference_labels)
    }
    prediction_lookup = {
        label: index for index, label in enumerate(prediction_labels)
    }

    try:
        reference_index = records["reference_label"].map(reference_lookup).to_numpy()
        prediction_index = records["analysis_label"].map(prediction_lookup).to_numpy()
    except KeyError as exc:
        raise ValueError("Could not map labels into the confusion matrix.") from exc

    if pd.isna(reference_index).any():
        bad = records.loc[
            pd.isna(reference_index),
            "reference_label",
        ].drop_duplicates().tolist()
        raise ValueError(f"Unknown reference labels: {bad}")
    if pd.isna(prediction_index).any():
        bad = records.loc[
            pd.isna(prediction_index),
            "analysis_label",
        ].drop_duplicates().tolist()
        raise ValueError(f"Unknown prediction labels: {bad}")

    n_rows = len(reference_labels)
    n_cols = len(prediction_labels)
    flat = (
        reference_index.astype(int) * n_cols
        + prediction_index.astype(int)
    )
    counts = np.bincount(
        flat,
        weights=np.asarray(weights, dtype=float),
        minlength=n_rows * n_cols,
    )
    return counts.reshape(n_rows, n_cols)


def rao_wu_rescaled_bootstrap_weights(
    sample_design: pd.DataFrame,
    *,
    bootstrap_reps: int,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    """Generate paired Rao-Wu rescaled bootstrap weights for stratified SRSWOR.

    For stratum h, draw n_h - 1 sample units with replacement and use the
    rescaling factor

        1 + sqrt(1 - f_h) * (n_h m_hi / (n_h - 1) - 1),

    where m_hi is the bootstrap multiplicity and f_h = n_h / N_h.

    The replicate weights sum to N_h within each stratum. Census strata (f_h=1)
    therefore contribute no sampling variance.
    """
    if bootstrap_reps <= 0:
        raise ValueError("bootstrap_reps must be positive.")

    design = sample_design.copy()
    design["id"] = design["id"].astype(str)
    ids = design["id"].tolist()
    n = len(design)
    weights = np.zeros((bootstrap_reps, n), dtype=float)
    rng = np.random.Generator(np.random.PCG64(seed))

    for _, stratum in design.groupby("predicted_label", sort=True):
        indices = stratum.index.to_numpy(dtype=int)
        n_h = len(indices)
        N_values = stratum["stratum_population_n"].astype(int).unique()
        if len(N_values) != 1:
            raise ValueError("Inconsistent stratum_population_n within stratum.")
        N_h = int(N_values[0])
        if N_h < n_h:
            raise ValueError("Stratum population cannot be smaller than sample.")
        base_weight = N_h / n_h
        f_h = n_h / N_h

        if n_h == 1 or np.isclose(f_h, 1.0):
            weights[:, indices] = base_weight
            continue

        probabilities = np.full(n_h, 1.0 / n_h)
        multiplicities = rng.multinomial(
            n_h - 1,
            probabilities,
            size=bootstrap_reps,
        )
        factor = 1.0 + np.sqrt(1.0 - f_h) * (
            n_h * multiplicities / (n_h - 1) - 1.0
        )
        weights[:, indices] = base_weight * factor

    expected_population = float(design["sampling_weight"].sum())
    replicate_totals = weights.sum(axis=1)
    if not np.allclose(replicate_totals, expected_population, rtol=0, atol=1e-7):
        raise RuntimeError(
            "Rao-Wu bootstrap weights do not preserve the target population total."
        )

    return weights, ids


def metric_bundle_to_dict(bundle: MetricBundle) -> dict[str, float]:
    return {
        field: float(getattr(bundle, field))
        for field in bundle.__dataclass_fields__
    }


def percentile_interval(
    values: np.ndarray,
    *,
    level: float = 0.95,
) -> tuple[float, float]:
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1.")
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan

    alpha = 1.0 - level
    lower, upper = np.quantile(
        values,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )
    return float(lower), float(upper)


def primary_bootstrap(
    records: pd.DataFrame,
    *,
    model_key: str,
    reference_labels: list[str],
    prediction_labels: list[str],
    observation_ids: list[str],
    bootstrap_weights: np.ndarray,
    repetitions: int,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]], pd.DataFrame]:
    model_records = records.loc[records["model_key"].eq(model_key)].copy()
    id_lookup = {
        observation_id: index
        for index, observation_id in enumerate(observation_ids)
    }
    observation_index = (
        model_records["observation_id"].map(id_lookup).to_numpy(dtype=int)
    )

    reference_lookup = {
        label: index for index, label in enumerate(reference_labels)
    }
    prediction_lookup = {
        label: index for index, label in enumerate(prediction_labels)
    }
    ref_idx = model_records["reference_label"].map(reference_lookup).to_numpy(dtype=int)
    pred_idx = model_records["analysis_label"].map(prediction_lookup).to_numpy(dtype=int)

    n_rows = len(reference_labels)
    n_cols = len(prediction_labels)
    flat = ref_idx * n_cols + pred_idx
    invalid_index = prediction_lookup[INVALID_LABEL]

    metric_rows: list[dict[str, Any]] = []
    class_f1_operational = np.empty(
        (bootstrap_weights.shape[0], n_rows),
        dtype=float,
    )
    class_f1_conditional = np.empty(
        (bootstrap_weights.shape[0], n_rows),
        dtype=float,
    )

    for bootstrap_index in range(bootstrap_weights.shape[0]):
        decision_weights = (
            bootstrap_weights[bootstrap_index, observation_index]
            / repetitions
        )
        confusion = np.bincount(
            flat,
            weights=decision_weights,
            minlength=n_rows * n_cols,
        ).reshape(n_rows, n_cols)

        bundle, classes = confusion_metrics(
            confusion,
            n_reference_labels=n_rows,
            invalid_prediction_index=invalid_index,
        )
        row = {
            "bootstrap_rep": bootstrap_index + 1,
            "model_key": model_key,
            **metric_bundle_to_dict(bundle),
        }
        metric_rows.append(row)
        class_f1_operational[bootstrap_index, :] = (
            classes["operational_f1"].to_numpy(dtype=float)
        )
        class_f1_conditional[bootstrap_index, :] = (
            classes["conditional_f1"].to_numpy(dtype=float)
        )

    metrics = pd.DataFrame(metric_rows)
    intervals = {
        metric: percentile_interval(metrics[metric].to_numpy())
        for metric in MetricBundle.__dataclass_fields__
    }

    class_rows: list[dict[str, Any]] = []
    for index, label in enumerate(reference_labels):
        op_low, op_high = percentile_interval(
            class_f1_operational[:, index]
        )
        cond_low, cond_high = percentile_interval(
            class_f1_conditional[:, index]
        )
        class_rows.append(
            {
                "model_key": model_key,
                "reference_label": label,
                "operational_f1_ci_low": op_low,
                "operational_f1_ci_high": op_high,
                "conditional_f1_ci_low": cond_low,
                "conditional_f1_ci_high": cond_high,
            }
        )

    return metrics, intervals, pd.DataFrame(class_rows)
