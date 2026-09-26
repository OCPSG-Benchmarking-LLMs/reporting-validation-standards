#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from reporting_validation_standards.error_propagation import (
    matrix_diagnostics,
    prevalence_error_metrics,
    row_normalize,
    solve_simplex_least_squares,
)
from reporting_validation_standards.experiment_specs import (
    load_model_config,
    load_prompt_config,
)
from reporting_validation_standards.sample_io import load_frozen_population
from reporting_validation_standards.stability_analysis import (
    prepare_records,
    read_inference_jsonl,
    sha256_file,
)
from reporting_validation_standards.validation_analysis import (
    build_confusion_matrix,
    rao_wu_rescaled_bootstrap_weights,
)


DEPLOYMENT_CONFIG = Path("config/deployment.yaml")
SAMPLE_MANIFEST = Path("data/manifests/uk_test_1000_ids.csv")
VALIDATION_MODEL_KEY = "openai_gpt54_mini"
VALIDATION_ROOT = Path("runs/production/stability")
VALIDATION_RESULTS_ROOT = Path("results/production")
DEPLOYMENT_ROOT = Path("runs/deployment")
DEPLOYMENT_RESULTS_ROOT = Path("results/deployment")
OUTPUT_ROOT = Path("outputs/propagation")
DEFAULT_BOOTSTRAP_REPS = 2000
DEFAULT_BOOTSTRAP_SEED = 20260926


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Propagate validation uncertainty into corrected GB policy-agenda "
            "prevalence estimates after the frozen full-population deployment."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPS,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_deployment_spec() -> dict[str, Any]:
    payload = yaml.safe_load(DEPLOYMENT_CONFIG.read_text(encoding="utf-8")) or {}
    deployment = payload.get("deployment")
    if not isinstance(deployment, dict):
        raise ValueError("Invalid deployment configuration.")
    return deployment


def _verify_raw(
    *,
    records_path: Path,
    qa_path: Path,
    expected_model_key: str,
    expected_n: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    qa = _load_json(qa_path)
    if not qa.get("structural_qa_pass") or not qa.get("completion_qa_pass"):
        raise RuntimeError(f"{qa_path} does not describe a finalized complete run.")

    expected_hash = str(qa.get("raw_inference_records_sha256") or "")
    observed_hash = sha256_file(records_path)
    if observed_hash != expected_hash:
        raise RuntimeError(
            f"Raw inference hash mismatch for {records_path}: "
            f"expected {expected_hash}, observed {observed_hash}."
        )

    records = read_inference_jsonl(records_path)
    if len(records) != expected_n:
        raise RuntimeError(
            f"{records_path} contains {len(records):,} records; "
            f"expected {expected_n:,}."
        )
    if records["model_key"].nunique() != 1:
        raise RuntimeError(f"{records_path} contains multiple model keys.")
    if str(records["model_key"].iloc[0]) != expected_model_key:
        raise RuntimeError(
            f"{records_path} model key does not match {expected_model_key}."
        )
    return records, qa


def _prevalence_vector(
    labels: pd.Series,
    *,
    official_labels: list[str],
) -> np.ndarray:
    counts = labels.astype(str).value_counts()
    unknown = sorted(set(counts.index) - set(official_labels))
    if unknown:
        raise ValueError(f"Observed labels outside official inventory: {unknown}")
    total = int(counts.sum())
    if total <= 0:
        raise ValueError("Cannot calculate prevalence from an empty label vector.")
    return np.asarray(
        [float(counts.get(label, 0)) / total for label in official_labels],
        dtype=float,
    )


def _bootstrap_interval(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan, np.nan
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    args = parse_args()
    if args.bootstrap_reps <= 0:
        raise ValueError("--bootstrap-reps must be positive.")

    deployment = _load_deployment_spec()
    deployment_id = str(deployment["deployment_id"])
    deployment_model_key = str(deployment["model_key"])
    prompt_id = str(deployment["prompt_id"])

    model_config = load_model_config()
    prompt_config = load_prompt_config()
    model_lookup = {
        str(model["model_key"]): model
        for model in model_config["models"]
    }
    deployment_model = model_lookup[deployment_model_key]
    official_labels = [str(label) for label in prompt_config["official_labels"]]

    population, population_metadata = load_frozen_population(
        token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )
    population_id_column = str(population_metadata["id_column"])
    reference_column = str(population_metadata["label_column"])
    population[population_id_column] = population[population_id_column].astype(str)

    deployment_records_path = (
        DEPLOYMENT_ROOT / deployment_id / "inference_records.jsonl"
    )
    deployment_qa_path = (
        DEPLOYMENT_RESULTS_ROOT / deployment_id / "qa_summary.json"
    )
    deployment_records, deployment_qa = _verify_raw(
        records_path=deployment_records_path,
        qa_path=deployment_qa_path,
        expected_model_key=deployment_model_key,
        expected_n=len(population),
    )
    if not deployment_records["is_valid"].all():
        raise RuntimeError(
            "The finalized deployment contains invalid outputs. This version of "
            "the propagation estimator is frozen for the observed all-valid deployment."
        )
    if set(deployment_records["prompt_id"].astype(str)) != {prompt_id}:
        raise RuntimeError("Deployment records do not match the frozen P0 prompt.")

    population_ids = set(population[population_id_column].astype(str))
    deployment_ids = set(deployment_records["observation_id"].astype(str))
    if deployment_ids != population_ids:
        raise RuntimeError(
            "Deployment observation IDs do not equal the frozen population IDs."
        )

    sample_design = pd.read_csv(SAMPLE_MANIFEST)
    sample_design["id"] = sample_design["id"].astype(str)
    validation_records_path = (
        VALIDATION_ROOT / VALIDATION_MODEL_KEY / "inference_records.jsonl"
    )
    validation_qa_path = (
        VALIDATION_RESULTS_ROOT / VALIDATION_MODEL_KEY / "qa_summary.json"
    )
    validation_records, validation_qa = _verify_raw(
        records_path=validation_records_path,
        qa_path=validation_qa_path,
        expected_model_key=VALIDATION_MODEL_KEY,
        expected_n=12_000,
    )
    validation_records = prepare_records(validation_records, sample_design)
    validation_p0 = validation_records.loc[
        validation_records["prompt_id"].eq(prompt_id)
    ].copy()
    if len(validation_p0) != len(sample_design) * 3:
        raise RuntimeError(
            "Expected three P0 validation repetitions for every sampled speech."
        )
    if not validation_p0["is_valid"].all():
        raise RuntimeError(
            "GPT-5.4 Mini P0 validation contains invalid outputs; "
            "the frozen propagation estimator expects the observed all-valid panel."
        )

    population_reference_labels = set(
        population[reference_column].astype(str).unique()
    )
    reference_labels = [
        label for label in official_labels
        if label in population_reference_labels
    ]
    absent_reference_labels = [
        label for label in official_labels
        if label not in population_reference_labels
    ]
    target_indices = np.asarray(
        [official_labels.index(label) for label in reference_labels],
        dtype=int,
    )

    # Exact silver prevalence in the complete target population.
    true_prevalence = _prevalence_vector(
        population[reference_column],
        official_labels=official_labels,
    )
    # Naive measured prevalence from one frozen full-population classifier pass.
    naive_prevalence = _prevalence_vector(
        deployment_records["analysis_label"],
        official_labels=official_labels,
    )

    # Point estimate of the 17 x 23 misclassification mechanism from the
    # probability sample. Each speech contributes its sampling weight exactly once
    # across its three repeated P0 classifications.
    validation_weights = (
        validation_p0["sampling_weight"].to_numpy(dtype=float) / 3.0
    )
    point_confusion = build_confusion_matrix(
        validation_p0,
        reference_labels=reference_labels,
        prediction_labels=official_labels,
        weights=validation_weights,
    )
    point_misclassification = row_normalize(point_confusion)
    forward_operator = point_misclassification.T

    constrained = solve_simplex_least_squares(
        forward_operator,
        naive_prevalence,
    )
    if not constrained.converged:
        raise RuntimeError(
            "Point-estimate simplex least-squares solver did not converge."
        )
    corrected_reference = constrained.solution
    corrected_prevalence = np.zeros(len(official_labels), dtype=float)
    corrected_prevalence[target_indices] = corrected_reference

    unconstrained_solution, residuals, _, _ = np.linalg.lstsq(
        forward_operator,
        naive_prevalence,
        rcond=None,
    )
    unconstrained_diagnostics = {
        "sum": float(unconstrained_solution.sum()),
        "minimum": float(unconstrained_solution.min()),
        "maximum": float(unconstrained_solution.max()),
        "negative_components": int((unconstrained_solution < 0).sum()),
        "negative_mass": float(
            -unconstrained_solution[unconstrained_solution < 0].sum()
        ),
        "residual_sum_squares": (
            float(residuals[0]) if len(residuals) else None
        ),
        "solution_by_reference_label": {
            label: float(value)
            for label, value in zip(reference_labels, unconstrained_solution)
        },
    }

    naive_errors = prevalence_error_metrics(
        naive_prevalence,
        true_prevalence,
        target_indices=target_indices,
    )
    corrected_errors = prevalence_error_metrics(
        corrected_prevalence,
        true_prevalence,
        target_indices=target_indices,
    )
    absent_indices = np.asarray(
        [official_labels.index(label) for label in absent_reference_labels],
        dtype=int,
    )
    naive_absent_mass = float(naive_prevalence[absent_indices].sum())
    corrected_absent_mass = float(corrected_prevalence[absent_indices].sum())

    print("Verified frozen inputs")
    print(
        f"  deployment: {deployment_id}  "
        f"N={len(deployment_records):,}  hash=PASS"
    )
    print(
        f"  validation: {VALIDATION_MODEL_KEY} P0  "
        f"n={len(sample_design):,} × 3 repetitions  hash=PASS"
    )
    print(
        f"  misclassification operator: "
        f"{forward_operator.shape[0]} predictions × "
        f"{forward_operator.shape[1]} reference labels"
    )
    print(
        f"  point solver: converged={constrained.converged}  "
        f"iterations={constrained.iterations:,}  "
        f"objective={constrained.objective:.10f}"
    )

    bootstrap_weights, bootstrap_ids = rao_wu_rescaled_bootstrap_weights(
        sample_design.reset_index(drop=True),
        bootstrap_reps=args.bootstrap_reps,
        seed=args.bootstrap_seed,
    )
    id_lookup = {
        observation_id: index
        for index, observation_id in enumerate(bootstrap_ids)
    }
    observation_index = validation_p0["observation_id"].map(id_lookup)
    if observation_index.isna().any():
        raise RuntimeError("Validation records contain IDs outside the frozen sample.")
    observation_index_array = observation_index.to_numpy(dtype=int)

    reference_lookup = {
        label: index for index, label in enumerate(reference_labels)
    }
    prediction_lookup = {
        label: index for index, label in enumerate(official_labels)
    }
    ref_index = validation_p0["reference_label"].map(reference_lookup)
    pred_index = validation_p0["analysis_label"].map(prediction_lookup)
    if ref_index.isna().any() or pred_index.isna().any():
        raise RuntimeError("Could not map validation labels into propagation matrix.")

    n_reference = len(reference_labels)
    n_prediction = len(official_labels)
    flat = (
        ref_index.to_numpy(dtype=int) * n_prediction
        + pred_index.to_numpy(dtype=int)
    )

    corrected_bootstrap = np.empty(
        (args.bootstrap_reps, n_reference),
        dtype=float,
    )
    bootstrap_rows: list[dict[str, Any]] = []
    warm_start = corrected_reference.copy()

    print(
        f"\nPropagating {args.bootstrap_reps:,} paired stratified bootstrap "
        f"replicates (seed={args.bootstrap_seed})"
    )
    for bootstrap_index in range(args.bootstrap_reps):
        decision_weights = (
            bootstrap_weights[bootstrap_index, observation_index_array] / 3.0
        )
        confusion = np.bincount(
            flat,
            weights=decision_weights,
            minlength=n_reference * n_prediction,
        ).reshape(n_reference, n_prediction)
        mechanism = row_normalize(confusion)

        fit = solve_simplex_least_squares(
            mechanism.T,
            naive_prevalence,
            initial=warm_start,
        )
        if not fit.converged:
            raise RuntimeError(
                f"Bootstrap solver failed to converge at replicate "
                f"{bootstrap_index + 1}."
            )
        warm_start = fit.solution
        corrected_bootstrap[bootstrap_index, :] = fit.solution

        embedded = np.zeros(len(official_labels), dtype=float)
        embedded[target_indices] = fit.solution
        errors = prevalence_error_metrics(
            embedded,
            true_prevalence,
            target_indices=target_indices,
        )
        bootstrap_rows.append(
            {
                "bootstrap_rep": bootstrap_index + 1,
                "objective": fit.objective,
                "iterations": fit.iterations,
                "projected_gradient_norm": fit.projected_gradient_norm,
                **errors,
                "improvement_total_variation_distance": (
                    naive_errors["total_variation_distance"]
                    - errors["total_variation_distance"]
                ),
                "improvement_mae_target_labels": (
                    naive_errors["mae_target_labels"]
                    - errors["mae_target_labels"]
                ),
            }
        )

        if (
            bootstrap_index == 0
            or (bootstrap_index + 1) % 250 == 0
            or bootstrap_index + 1 == args.bootstrap_reps
        ):
            print(
                f"  bootstrap {bootstrap_index + 1:,}/{args.bootstrap_reps:,}"
            )

    bootstrap_summary = pd.DataFrame(bootstrap_rows)

    prevalence_rows: list[dict[str, Any]] = []
    for label_index, label in enumerate(official_labels):
        if label in reference_lookup:
            ref_position = reference_lookup[label]
            ci_low, ci_high = _bootstrap_interval(
                corrected_bootstrap[:, ref_position]
            )
        else:
            ci_low, ci_high = 0.0, 0.0

        prevalence_rows.append(
            {
                "label": label,
                "present_in_silver_reference_population": (
                    label in population_reference_labels
                ),
                "silver_true_prevalence": float(true_prevalence[label_index]),
                "naive_classifier_prevalence": float(
                    naive_prevalence[label_index]
                ),
                "corrected_prevalence": float(
                    corrected_prevalence[label_index]
                ),
                "corrected_ci_low": ci_low,
                "corrected_ci_high": ci_high,
                "naive_minus_true": float(
                    naive_prevalence[label_index] - true_prevalence[label_index]
                ),
                "corrected_minus_true": float(
                    corrected_prevalence[label_index] - true_prevalence[label_index]
                ),
            }
        )
    prevalence_comparison = pd.DataFrame(prevalence_rows)

    mechanism_rows: list[dict[str, Any]] = []
    for row_index, reference_label in enumerate(reference_labels):
        for column_index, prediction_label in enumerate(official_labels):
            mechanism_rows.append(
                {
                    "reference_label": reference_label,
                    "prediction_label": prediction_label,
                    "estimated_probability": float(
                        point_misclassification[row_index, column_index]
                    ),
                    "estimated_population_confusion_count": float(
                        point_confusion[row_index, column_index]
                    ),
                }
            )
    mechanism_long = pd.DataFrame(mechanism_rows)

    summary_rows: list[dict[str, Any]] = []
    metric_names = sorted(set(naive_errors) | set(corrected_errors))
    for metric in metric_names:
        bootstrap_values = bootstrap_summary[metric].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_interval(bootstrap_values)
        summary_rows.append(
            {
                "metric": metric,
                "naive": float(naive_errors[metric]),
                "corrected": float(corrected_errors[metric]),
                "corrected_ci_low": ci_low,
                "corrected_ci_high": ci_high,
                "improvement_naive_minus_corrected": float(
                    naive_errors[metric] - corrected_errors[metric]
                ),
            }
        )
    propagation_summary = pd.DataFrame(summary_rows)

    improvement_tvd_low, improvement_tvd_high = _bootstrap_interval(
        bootstrap_summary["improvement_total_variation_distance"].to_numpy(
            dtype=float
        )
    )
    improvement_mae_low, improvement_mae_high = _bootstrap_interval(
        bootstrap_summary["improvement_mae_target_labels"].to_numpy(dtype=float)
    )

    solver_diagnostics = {
        "estimator": (
            "simplex-constrained least squares: "
            "min ||M^T pi - q||_2^2 subject to pi >= 0 and sum(pi)=1"
        ),
        "forward_operator": matrix_diagnostics(forward_operator),
        "point_solution": {
            "converged": constrained.converged,
            "iterations": constrained.iterations,
            "objective": constrained.objective,
            "projected_gradient_norm": constrained.projected_gradient_norm,
            "solution_by_reference_label": {
                label: float(value)
                for label, value in zip(reference_labels, corrected_reference)
            },
        },
        "unconstrained_least_squares": unconstrained_diagnostics,
        "naive_absent_reference_label_mass": naive_absent_mass,
        "corrected_absent_reference_label_mass": corrected_absent_mass,
        "bootstrap": {
            "reps": args.bootstrap_reps,
            "seed": args.bootstrap_seed,
            "all_converged": True,
            "improvement_total_variation_distance_ci": [
                improvement_tvd_low,
                improvement_tvd_high,
            ],
            "improvement_mae_target_labels_ci": [
                improvement_mae_low,
                improvement_mae_high,
            ],
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prevalence_comparison.to_csv(
        args.output_dir / "prevalence_comparison.csv",
        index=False,
    )
    propagation_summary.to_csv(
        args.output_dir / "propagation_summary.csv",
        index=False,
    )
    mechanism_long.to_csv(
        args.output_dir / "misclassification_matrix.csv",
        index=False,
    )

    bootstrap_prevalence = pd.DataFrame(
        corrected_bootstrap,
        columns=reference_labels,
    )
    bootstrap_output = pd.concat(
        [
            pd.DataFrame(
                {
                    "bootstrap_rep": np.arange(
                        1,
                        args.bootstrap_reps + 1,
                        dtype=int,
                    )
                }
            ),
            bootstrap_prevalence,
            bootstrap_summary.drop(columns="bootstrap_rep"),
        ],
        axis=1,
    )
    bootstrap_output.to_csv(
        args.output_dir / "bootstrap_propagation.csv",
        index=False,
    )
    _write_json(
        args.output_dir / "solver_diagnostics.json",
        solver_diagnostics,
    )

    analysis_manifest = {
        "schema_version": 1,
        "analysis": "prevalence_error_propagation",
        "deployment_id": deployment_id,
        "deployment_model_key": deployment_model_key,
        "requested_model_id": deployment_model["model_id"],
        "prompt_id": prompt_id,
        "dataset_revision": population_metadata["dataset_revision"],
        "population_n": len(population),
        "population_id_sha256": deployment_qa["population_id_sha256"],
        "deployment_raw_inference_sha256": (
            deployment_qa["raw_inference_records_sha256"]
        ),
        "validation_model_key": VALIDATION_MODEL_KEY,
        "validation_raw_inference_sha256": (
            validation_qa["raw_inference_records_sha256"]
        ),
        "reference_labels": reference_labels,
        "official_prediction_labels": official_labels,
        "absent_reference_labels": absent_reference_labels,
        "validation_repetitions": 3,
        "validation_repetition_treatment": (
            "Each sampled observation's three P0 classifications are averaged so "
            "its inverse-inclusion weight contributes once to the estimated "
            "misclassification mechanism."
        ),
        "estimator": {
            "type": "simplex_constrained_least_squares",
            "equation": "q approximately equals M^T pi",
            "constraints": ["pi_j >= 0", "sum(pi)=1"],
            "reason_for_constraint": (
                "The 23-by-17 forward operator is rectangular; constrained "
                "least squares respects prevalence geometry and avoids negative "
                "or non-unit-mass corrected prevalences."
            ),
        },
        "uncertainty": {
            "source": "validation probability-sample uncertainty",
            "method": "Rao-Wu rescaled bootstrap within the 17 sampling strata",
            "bootstrap_reps": args.bootstrap_reps,
            "seed": args.bootstrap_seed,
            "rng": "NumPy PCG64",
            "confidence_level": 0.95,
            "interval": "percentile",
            "scope": (
                "The full deployment is treated as a census of one frozen model "
                "realization. Intervals propagate validation-sample uncertainty "
                "in the misclassification mechanism, not model stochasticity, "
                "benchmark-construction uncertainty, or provider drift."
            ),
        },
        "known_silver_truth_usage": (
            "Silver prevalence is used only after correction to demonstrate "
            "whether the adjustment moves estimates toward the known benchmark "
            "population distribution; it is not used to fit the constrained "
            "prevalence solution."
        ),
        "outputs": [
            "prevalence_comparison.csv",
            "propagation_summary.csv",
            "misclassification_matrix.csv",
            "bootstrap_propagation.csv",
            "solver_diagnostics.json",
        ],
    }
    _write_json(
        args.output_dir / "analysis_manifest.json",
        analysis_manifest,
    )

    print("\nPrevalence error relative to known silver population")
    print(
        f"  naive TV distance:     "
        f"{100 * naive_errors['total_variation_distance']:.2f} pp"
    )
    print(
        f"  corrected TV distance: "
        f"{100 * corrected_errors['total_variation_distance']:.2f} pp"
    )
    print(
        f"  naive target-label MAE:     "
        f"{100 * naive_errors['mae_target_labels']:.2f} pp"
    )
    print(
        f"  corrected target-label MAE: "
        f"{100 * corrected_errors['mae_target_labels']:.2f} pp"
    )
    print(
        f"  naive mass assigned to labels absent from silver GB: "
        f"{100 * naive_absent_mass:.2f}%"
    )
    print(
        f"  constrained correction mass on absent labels: "
        f"{100 * corrected_absent_mass:.2f}%"
    )
    print(
        f"  bootstrap CI for TV improvement: "
        f"[{100 * improvement_tvd_low:+.2f}, "
        f"{100 * improvement_tvd_high:+.2f}] pp"
    )
    print(
        f"  bootstrap CI for target-label MAE improvement: "
        f"[{100 * improvement_mae_low:+.2f}, "
        f"{100 * improvement_mae_high:+.2f}] pp"
    )
    print(f"\nAnalysis outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
