#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reporting_validation_standards.experiment_specs import (
    load_model_config,
    load_prompt_config,
)
from reporting_validation_standards.stability_analysis import (
    INVALID_LABEL,
    prepare_records,
    read_inference_jsonl,
    sha256_file,
)
from reporting_validation_standards.validation_analysis import (
    MetricBundle,
    build_confusion_matrix,
    confusion_metrics,
    metric_bundle_to_dict,
    primary_bootstrap,
    rao_wu_rescaled_bootstrap_weights,
)


PRODUCTION_ROOT = Path("runs/production/stability")
FINALIZED_ROOT = Path("results/production")
SAMPLE_MANIFEST = Path("data/manifests/uk_test_1000_ids.csv")
OUTPUT_ROOT = Path("outputs/validation")
CANONICAL_PROMPT_ID = "p0_capbench_canonical"
DEFAULT_BOOTSTRAP_REPS = 2000
DEFAULT_BOOTSTRAP_SEED = 20260926


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate silver-reference classifier performance with inverse-probability "
            "weights and Rao-Wu rescaled bootstrap uncertainty."
        )
    )
    parser.add_argument("--production-root", type=Path, default=PRODUCTION_ROOT)
    parser.add_argument("--finalized-root", type=Path, default=FINALIZED_ROOT)
    parser.add_argument("--sample-manifest", type=Path, default=SAMPLE_MANIFEST)
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_and_verify_model(
    *,
    model_key: str,
    production_root: Path,
    finalized_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records_path = production_root / model_key / "inference_records.jsonl"
    qa_path = finalized_root / model_key / "qa_summary.json"

    if not qa_path.exists():
        raise FileNotFoundError(
            f"Finalized QA summary not found for {model_key}: {qa_path}"
        )
    qa = _load_json(qa_path)
    if not qa.get("structural_qa_pass") or not qa.get("completion_qa_pass"):
        raise RuntimeError(
            f"{model_key} is not a finalized structurally complete run."
        )

    observed_hash = sha256_file(records_path)
    expected_hash = str(qa.get("raw_inference_records_sha256") or "")
    if observed_hash != expected_hash:
        raise RuntimeError(
            f"Raw inference hash mismatch for {model_key}: "
            f"expected {expected_hash}, observed {observed_hash}."
        )

    records = read_inference_jsonl(records_path)
    expected_n = int(qa["expected_decisions"])
    if len(records) != expected_n:
        raise RuntimeError(
            f"{model_key} has {len(records):,} records; expected {expected_n:,}."
        )
    return records, qa


def _metric_row(
    records: pd.DataFrame,
    *,
    reference_labels: list[str],
    prediction_labels: list[str],
    weighted_weights: np.ndarray,
    unweighted_weights: np.ndarray,
) -> dict[str, float]:
    invalid_index = prediction_labels.index(INVALID_LABEL)

    weighted_confusion = build_confusion_matrix(
        records,
        reference_labels=reference_labels,
        prediction_labels=prediction_labels,
        weights=weighted_weights,
    )
    unweighted_confusion = build_confusion_matrix(
        records,
        reference_labels=reference_labels,
        prediction_labels=prediction_labels,
        weights=unweighted_weights,
    )

    weighted, _ = confusion_metrics(
        weighted_confusion,
        n_reference_labels=len(reference_labels),
        invalid_prediction_index=invalid_index,
    )
    unweighted, _ = confusion_metrics(
        unweighted_confusion,
        n_reference_labels=len(reference_labels),
        invalid_prediction_index=invalid_index,
    )

    output: dict[str, float] = {}
    for metric in MetricBundle.__dataclass_fields__:
        output[f"weighted_{metric}"] = float(getattr(weighted, metric))
        output[f"unweighted_{metric}"] = float(getattr(unweighted, metric))
    return output


def _paired_model_differences(
    *,
    primary_performance: pd.DataFrame,
    bootstrap_metrics: pd.DataFrame,
    model_keys: list[str],
) -> pd.DataFrame:
    metrics = [
        "operational_accuracy",
        "operational_macro_f1",
        "conditional_macro_f1",
    ]
    point_lookup = primary_performance.set_index("model_key")
    rows: list[dict[str, Any]] = []

    for left_index, left in enumerate(model_keys):
        for right in model_keys[left_index + 1 :]:
            left_boot = (
                bootstrap_metrics.loc[
                    bootstrap_metrics["model_key"].eq(left),
                    ["bootstrap_rep", *metrics],
                ]
                .set_index("bootstrap_rep")
                .sort_index()
            )
            right_boot = (
                bootstrap_metrics.loc[
                    bootstrap_metrics["model_key"].eq(right),
                    ["bootstrap_rep", *metrics],
                ]
                .set_index("bootstrap_rep")
                .sort_index()
            )
            if not left_boot.index.equals(right_boot.index):
                raise RuntimeError(
                    f"Paired bootstrap replicate indices differ for {left} and {right}."
                )

            for metric in metrics:
                differences = (
                    left_boot[metric].to_numpy(dtype=float)
                    - right_boot[metric].to_numpy(dtype=float)
                )
                lower, upper = np.quantile(differences, [0.025, 0.975])
                point_difference = float(
                    point_lookup.loc[left, f"weighted_{metric}"]
                    - point_lookup.loc[right, f"weighted_{metric}"]
                )
                rows.append(
                    {
                        "model_key_left": left,
                        "model_key_right": right,
                        "metric": metric,
                        "difference_left_minus_right": point_difference,
                        "ci_low": float(lower),
                        "ci_high": float(upper),
                        "ci_excludes_zero": bool(lower > 0 or upper < 0),
                    }
                )

    return pd.DataFrame(rows)


def _condition_performance(
    records: pd.DataFrame,
    *,
    model_keys: list[str],
    prompt_ids: list[str],
    reference_labels: list[str],
    prediction_labels: list[str],
    repetitions: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for model_key in model_keys:
        for prompt_id in prompt_ids:
            prompt_records = records.loc[
                records["model_key"].eq(model_key)
                & records["prompt_id"].eq(prompt_id)
            ].copy()

            for repetition in range(1, repetitions + 1):
                subset = prompt_records.loc[
                    prompt_records["repetition"].eq(repetition)
                ]
                metrics = _metric_row(
                    subset,
                    reference_labels=reference_labels,
                    prediction_labels=prediction_labels,
                    weighted_weights=subset["sampling_weight"].to_numpy(dtype=float),
                    unweighted_weights=np.ones(len(subset), dtype=float),
                )
                rows.append(
                    {
                        "model_key": model_key,
                        "prompt_id": prompt_id,
                        "repetition": repetition,
                        "aggregation": "single_repetition",
                        "n_scheduled": len(subset),
                        "n_valid": int(subset["is_valid"].sum()),
                        **metrics,
                    }
                )

            pooled_metrics = _metric_row(
                prompt_records,
                reference_labels=reference_labels,
                prediction_labels=prediction_labels,
                weighted_weights=(
                    prompt_records["sampling_weight"].to_numpy(dtype=float)
                    / repetitions
                ),
                unweighted_weights=(
                    np.ones(len(prompt_records), dtype=float) / repetitions
                ),
            )
            rows.append(
                {
                    "model_key": model_key,
                    "prompt_id": prompt_id,
                    "repetition": 0,
                    "aggregation": "expected_over_repetitions",
                    "n_scheduled": len(prompt_records),
                    "n_valid": int(prompt_records["is_valid"].sum()),
                    **pooled_metrics,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.bootstrap_reps <= 0:
        raise ValueError("--bootstrap-reps must be positive.")

    model_config = load_model_config()
    prompt_config = load_prompt_config()
    model_keys = [str(model["model_key"]) for model in model_config["models"]]
    model_stage = {
        str(model["model_key"]): str(model.get("stage") or "")
        for model in model_config["models"]
    }
    prompt_ids = [str(prompt["prompt_id"]) for prompt in prompt_config["prompts"]]
    if CANONICAL_PROMPT_ID not in prompt_ids:
        raise RuntimeError("Canonical P0 prompt is missing from prompt configuration.")

    official_labels = [str(label) for label in prompt_config["official_labels"]]
    sample_design = pd.read_csv(args.sample_manifest)
    sample_design["id"] = sample_design["id"].astype(str)

    # The validation target is the GB silver-reference population. Only labels with
    # positive reference support in that target population enter the macro average.
    population_labels = set(sample_design["predicted_label"].astype(str))
    reference_labels = [
        label for label in official_labels if label in population_labels
    ]
    if len(reference_labels) != len(population_labels):
        missing = sorted(population_labels - set(reference_labels))
        raise RuntimeError(
            f"Frozen sample contains labels absent from the official inventory: {missing}"
        )

    # Put target-population reference labels first because metric extraction uses
    # matching row/column positions for the diagonal. Official labels absent from
    # GB remain allowable prediction columns and therefore still create errors.
    prediction_labels = (
        reference_labels
        + [label for label in official_labels if label not in population_labels]
        + [INVALID_LABEL]
    )

    repetitions = int(model_config["experiment_policy"]["repetitions"])
    expected_population_n = float(
        sample_design[["predicted_label", "stratum_population_n"]]
        .drop_duplicates()["stratum_population_n"]
        .sum()
    )

    qa_by_model: dict[str, Any] = {}
    frames: list[pd.DataFrame] = []

    print("Verifying finalized production inputs")
    for model_key in model_keys:
        frame, qa = _read_and_verify_model(
            model_key=model_key,
            production_root=args.production_root,
            finalized_root=args.finalized_root,
        )
        qa_by_model[model_key] = qa
        frames.append(frame)
        print(
            f"  {model_key:28} records={len(frame):,}  "
            f"hash=PASS  completion=PASS"
        )

    records = pd.concat(frames, ignore_index=True)
    records = prepare_records(records, sample_design)

    primary = records.loc[
        records["prompt_id"].eq(CANONICAL_PROMPT_ID)
    ].copy()
    expected_primary_n = len(model_keys) * len(sample_design) * repetitions
    if len(primary) != expected_primary_n:
        raise RuntimeError(
            f"Expected {expected_primary_n:,} P0 records, found {len(primary):,}."
        )

    print(
        f"\nGenerating {args.bootstrap_reps:,} paired Rao-Wu rescaled "
        f"bootstrap replicates (seed={args.bootstrap_seed})"
    )
    bootstrap_weights, bootstrap_ids = rao_wu_rescaled_bootstrap_weights(
        sample_design.reset_index(drop=True),
        bootstrap_reps=args.bootstrap_reps,
        seed=args.bootstrap_seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    primary_rows: list[dict[str, Any]] = []
    class_frames: list[pd.DataFrame] = []
    confusion_rows: list[dict[str, Any]] = []
    bootstrap_frames: list[pd.DataFrame] = []

    invalid_index = prediction_labels.index(INVALID_LABEL)

    for model_key in model_keys:
        model_primary = primary.loc[
            primary["model_key"].eq(model_key)
        ].copy()
        if len(model_primary) != len(sample_design) * repetitions:
            raise RuntimeError(
                f"{model_key} P0 record count is incomplete: "
                f"{len(model_primary):,}."
            )

        weighted_weights = (
            model_primary["sampling_weight"].to_numpy(dtype=float)
            / repetitions
        )
        unweighted_weights = np.ones(len(model_primary), dtype=float) / repetitions

        weighted_confusion = build_confusion_matrix(
            model_primary,
            reference_labels=reference_labels,
            prediction_labels=prediction_labels,
            weights=weighted_weights,
        )
        unweighted_confusion = build_confusion_matrix(
            model_primary,
            reference_labels=reference_labels,
            prediction_labels=prediction_labels,
            weights=unweighted_weights,
        )
        if not np.isclose(
            weighted_confusion.sum(),
            expected_population_n,
            rtol=0,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"{model_key} weighted P0 confusion matrix does not sum to "
                f"the target population size."
            )

        weighted_bundle, weighted_classes = confusion_metrics(
            weighted_confusion,
            n_reference_labels=len(reference_labels),
            invalid_prediction_index=invalid_index,
        )
        unweighted_bundle, _ = confusion_metrics(
            unweighted_confusion,
            n_reference_labels=len(reference_labels),
            invalid_prediction_index=invalid_index,
        )

        bootstrap_metrics, intervals, class_intervals = primary_bootstrap(
            model_primary,
            model_key=model_key,
            reference_labels=reference_labels,
            prediction_labels=prediction_labels,
            observation_ids=bootstrap_ids,
            bootstrap_weights=bootstrap_weights,
            repetitions=repetitions,
        )
        bootstrap_frames.append(bootstrap_metrics)

        row: dict[str, Any] = {
            "model_key": model_key,
            "stage": model_stage[model_key],
            "prompt_id": CANONICAL_PROMPT_ID,
            "aggregation": "expected_over_three_repetitions",
            "n_observations": len(sample_design),
            "n_scheduled_decisions": len(model_primary),
            "n_valid_decisions": int(model_primary["is_valid"].sum()),
        }
        for metric in MetricBundle.__dataclass_fields__:
            weighted_value = float(getattr(weighted_bundle, metric))
            unweighted_value = float(getattr(unweighted_bundle, metric))
            ci_low, ci_high = intervals[metric]
            row[f"weighted_{metric}"] = weighted_value
            row[f"weighted_{metric}_ci_low"] = ci_low
            row[f"weighted_{metric}_ci_high"] = ci_high
            row[f"unweighted_{metric}"] = unweighted_value
            row[f"weighted_minus_unweighted_{metric}"] = (
                weighted_value - unweighted_value
            )
        primary_rows.append(row)

        weighted_classes = weighted_classes.copy()
        weighted_classes["reference_label"] = reference_labels
        weighted_classes.insert(0, "model_key", model_key)
        weighted_classes = weighted_classes.merge(
            class_intervals,
            on=["model_key", "reference_label"],
            how="left",
            validate="one_to_one",
        )
        class_frames.append(weighted_classes)

        for row_index, reference_label in enumerate(reference_labels):
            for column_index, prediction_label in enumerate(prediction_labels):
                confusion_rows.append(
                    {
                        "model_key": model_key,
                        "reference_label": reference_label,
                        "prediction_label": prediction_label,
                        "estimated_population_count": float(
                            weighted_confusion[row_index, column_index]
                        ),
                    }
                )

        print(
            f"  {model_key:28} "
            f"valid={100 * weighted_bundle.valid_rate:6.2f}%  "
            f"accuracy={100 * weighted_bundle.operational_accuracy:6.2f}%  "
            f"macro-F1={100 * weighted_bundle.operational_macro_f1:6.2f}%  "
            f"conditional macro-F1="
            f"{100 * weighted_bundle.conditional_macro_f1:6.2f}%"
        )

    primary_performance = pd.DataFrame(primary_rows)
    per_class = pd.concat(class_frames, ignore_index=True)
    confusion_long = pd.DataFrame(confusion_rows)
    bootstrap_metrics = pd.concat(bootstrap_frames, ignore_index=True)

    paired_model_differences = _paired_model_differences(
        primary_performance=primary_performance,
        bootstrap_metrics=bootstrap_metrics,
        model_keys=model_keys,
    )

    condition_performance = _condition_performance(
        records,
        model_keys=model_keys,
        prompt_ids=prompt_ids,
        reference_labels=reference_labels,
        prediction_labels=prediction_labels,
        repetitions=repetitions,
    )

    primary_performance.to_csv(
        args.output_dir / "primary_model_performance.csv",
        index=False,
    )
    condition_performance.to_csv(
        args.output_dir / "condition_performance.csv",
        index=False,
    )
    per_class.to_csv(
        args.output_dir / "per_class_performance.csv",
        index=False,
    )
    confusion_long.to_csv(
        args.output_dir / "confusion_matrix_population.csv",
        index=False,
    )
    bootstrap_metrics.to_csv(
        args.output_dir / "bootstrap_metrics.csv",
        index=False,
    )
    paired_model_differences.to_csv(
        args.output_dir / "paired_model_differences.csv",
        index=False,
    )

    source_hashes = {
        model_key: qa_by_model[model_key]["raw_inference_records_sha256"]
        for model_key in model_keys
    }
    manifest = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "analysis": "design_based_silver_reference_validation",
        "reference_standard": (
            "OCPSG automated silver-standard consensus labels; not human gold labels"
        ),
        "primary_estimand": {
            "prompt": CANONICAL_PROMPT_ID,
            "repetition_treatment": (
                "Average each observation's three repeated P0 decisions so each "
                "sampled observation contributes one population weight in total."
            ),
            "target_population": "GB frozen silver-standard test population",
            "population_n": int(expected_population_n),
            "macro_average_labels": reference_labels,
            "prediction_inventory": official_labels,
            "failure_handling": (
                "Operational metrics retain invalid API/parse outcomes as measurement "
                "failures that contribute false negatives for the true reference class. "
                "Conditional metrics separately describe performance among valid outputs."
            ),
        },
        "sampling_design": {
            "estimator": (
                "Inverse-inclusion weighted confusion-matrix totals under the frozen "
                "label-stratified SRSWOR design."
            ),
            "weight_column": "sampling_weight",
            "strata_column": "predicted_label",
            "positivity": True,
        },
        "uncertainty": {
            "method": "Rao-Wu rescaled bootstrap within sampling strata",
            "bootstrap_reps": args.bootstrap_reps,
            "seed": args.bootstrap_seed,
            "rng": "NumPy PCG64",
            "confidence_level": 0.95,
            "interval": "percentile",
            "cluster_unit": "sampled observation",
            "paired_across_models": True,
            "finite_population_note": (
                "Rescaling uses each stratum sampling fraction; census strata "
                "contribute no sampling variance."
            ),
            "scope": (
                "Intervals quantify uncertainty from the frozen probability sample. "
                "They do not claim to integrate all model-infrastructure or benchmark-"
                "construction uncertainty."
            ),
        },
        "model_keys": model_keys,
        "model_stages": model_stage,
        "sample_id_sha256": qa_by_model[model_keys[0]]["sample_id_sha256"],
        "dataset_revision": qa_by_model[model_keys[0]]["dataset_revision"],
        "schedule_sha256": qa_by_model[model_keys[0]]["schedule_sha256"],
        "raw_inference_sha256": source_hashes,
        "outputs": [
            "primary_model_performance.csv",
            "condition_performance.csv",
            "per_class_performance.csv",
            "confusion_matrix_population.csv",
            "bootstrap_metrics.csv",
            "paired_model_differences.csv",
        ],
    }
    _write_json(args.output_dir / "analysis_manifest.json", manifest)

    print("\nPrimary P0 design-based performance")
    display = primary_performance[
        [
            "model_key",
            "weighted_valid_rate",
            "weighted_operational_accuracy",
            "weighted_operational_macro_f1",
            "weighted_operational_macro_f1_ci_low",
            "weighted_operational_macro_f1_ci_high",
            "unweighted_operational_macro_f1",
            "weighted_minus_unweighted_operational_macro_f1",
            "weighted_conditional_macro_f1",
        ]
    ].copy()
    percentage_columns = [
        column
        for column in display.columns[1:]
        if column != "weighted_minus_unweighted_operational_macro_f1"
    ]
    for column in percentage_columns:
        display[column] = display[column].map(
            lambda value: f"{100 * value:.2f}%"
        )
    display["weighted_minus_unweighted_operational_macro_f1"] = display[
        "weighted_minus_unweighted_operational_macro_f1"
    ].map(lambda value: f"{100 * value:+.2f} pp")
    print(display.to_string(index=False))

    print("\nPaired model differences in weighted operational macro-F1")
    pair_display = paired_model_differences.loc[
        paired_model_differences["metric"].eq("operational_macro_f1"),
        [
            "model_key_left",
            "model_key_right",
            "difference_left_minus_right",
            "ci_low",
            "ci_high",
            "ci_excludes_zero",
        ],
    ].copy()
    for column in ["difference_left_minus_right", "ci_low", "ci_high"]:
        pair_display[column] = pair_display[column].map(
            lambda value: f"{100 * value:+.2f} pp"
        )
    print(pair_display.to_string(index=False))

    print(f"\nAnalysis outputs: {args.output_dir}")
    print(
        "Interpretation: these are performance estimates relative to the automated "
        "silver reference, not estimates against human gold labels."
    )


if __name__ == "__main__":
    main()
