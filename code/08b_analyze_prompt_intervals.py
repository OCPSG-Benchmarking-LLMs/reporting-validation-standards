#!/usr/bin/env python3
"""Per-prompt design-based intervals for the validation figure.

For every model x prompt, the three repetitions of each sampled speech are averaged so
that the speech contributes its inverse-inclusion weight once, exactly as in the primary
P0 analysis of script 08. The same Rao-Wu rescaled bootstrap replicates (seed and
replicate count read from outputs/validation/analysis_manifest.json) give 95% percentile
intervals. The script fails closed unless the point estimates reproduce the
repetition-averaged rows of condition_performance.csv and the P0 intervals reproduce
primary_model_performance.csv. No committed output is rewritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from runpy import run_path

import pandas as pd

from reporting_validation_standards.experiment_specs import (
    load_model_config,
    load_prompt_config,
)
from reporting_validation_standards.stability_analysis import (
    INVALID_LABEL,
    prepare_records,
)
from reporting_validation_standards.validation_analysis import (
    build_confusion_matrix,
    confusion_metrics,
    metric_bundle_to_dict,
    primary_bootstrap,
    rao_wu_rescaled_bootstrap_weights,
)

SCRIPT_08 = Path(__file__).resolve().with_name("08_analyze_validation.py")
VALIDATION_DIR = Path("outputs/validation")
CANONICAL_PROMPT_ID = "p0_capbench_canonical"
METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1")
TOLERANCE = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path,
                        default=Path("runs/production/stability"))
    parser.add_argument("--finalized-root", type=Path, default=Path("results/production"))
    parser.add_argument("--sample-manifest", type=Path,
                        default=Path("data/manifests/uk_test_1000_ids.csv"))
    parser.add_argument("--validation-dir", type=Path, default=VALIDATION_DIR)
    return parser.parse_args()


def interval_row(model_key: str, prompt_id: str, point: dict, intervals: dict) -> dict:
    row = {"model_key": model_key, "prompt_id": prompt_id}
    for metric in METRICS:
        key = f"operational_{metric}"
        column = f"weighted_{key}"
        row[column] = float(point[key])
        row[f"{column}_ci_low"] = float(intervals[key][0])
        row[f"{column}_ci_high"] = float(intervals[key][1])
    return row


def check_against_committed(
    intervals: pd.DataFrame,
    condition: pd.DataFrame,
    primary: pd.DataFrame,
    tolerance: float = TOLERANCE,
) -> None:
    """Fail unless the new table reproduces every committed estimate it overlaps."""
    pooled = condition.loc[condition["repetition"].eq(0)].set_index(["model_key", "prompt_id"])
    table = intervals.set_index(["model_key", "prompt_id"])
    if set(table.index) != set(pooled.index):
        raise ValueError("Interval table does not cover the same model x prompt cells.")
    p0 = table.xs(CANONICAL_PROMPT_ID, level="prompt_id")
    committed = primary.set_index("model_key")
    for metric in METRICS:
        column = f"weighted_operational_{metric}"
        gap = (table[column] - pooled.loc[table.index, column]).abs().max()
        if not gap <= tolerance:
            raise ValueError(f"{column} differs from condition_performance.csv ({gap:g}).")
        for bound in ("ci_low", "ci_high"):
            name = f"{column}_{bound}"
            gap = (p0[name] - committed.loc[p0.index, name]).abs().max()
            if not gap <= tolerance:
                raise ValueError(
                    f"P0 {name} differs from primary_model_performance.csv ({gap:g})."
                )


def main() -> None:
    args = parse_args()
    read_and_verify = run_path(str(SCRIPT_08))["_read_and_verify_model"]
    manifest = json.loads((args.validation_dir / "analysis_manifest.json").read_text(
        encoding="utf-8"))
    uncertainty = manifest["uncertainty"]
    reps, seed = int(uncertainty["bootstrap_reps"]), int(uncertainty["seed"])

    model_config = load_model_config()
    prompt_config = load_prompt_config()
    model_keys = [str(model["model_key"]) for model in model_config["models"]]
    prompt_ids = [str(prompt["prompt_id"]) for prompt in prompt_config["prompts"]]
    repetitions = int(model_config["experiment_policy"]["repetitions"])

    sample_design = pd.read_csv(args.sample_manifest)
    sample_design["id"] = sample_design["id"].astype(str)
    official = [str(label) for label in prompt_config["official_labels"]]
    population = set(sample_design["predicted_label"].astype(str))
    reference_labels = [label for label in official if label in population]
    prediction_labels = (
        reference_labels
        + [label for label in official if label not in population]
        + [INVALID_LABEL]
    )
    invalid_index = prediction_labels.index(INVALID_LABEL)

    frames, raw_hashes = [], {}
    for model_key in model_keys:
        frame, qa = read_and_verify(
            model_key=model_key,
            production_root=args.production_root,
            finalized_root=args.finalized_root,
        )
        frames.append(frame)
        raw_hashes[model_key] = qa["raw_inference_records_sha256"]
    records = prepare_records(pd.concat(frames, ignore_index=True), sample_design)

    weights, ids = rao_wu_rescaled_bootstrap_weights(
        sample_design.reset_index(drop=True), bootstrap_reps=reps, seed=seed
    )
    rows = []
    for model_key in model_keys:
        for prompt_id in prompt_ids:
            subset = records.loc[
                records["model_key"].eq(model_key) & records["prompt_id"].eq(prompt_id)
            ].copy()
            if len(subset) != len(sample_design) * repetitions:
                raise RuntimeError(f"{model_key} x {prompt_id} is incomplete: {len(subset):,}.")
            confusion = build_confusion_matrix(
                subset, reference_labels=reference_labels, prediction_labels=prediction_labels,
                weights=subset["sampling_weight"].to_numpy(dtype=float) / repetitions,
            )
            bundle, _ = confusion_metrics(
                confusion, n_reference_labels=len(reference_labels),
                invalid_prediction_index=invalid_index,
            )
            _, intervals, _ = primary_bootstrap(
                subset, model_key=model_key, reference_labels=reference_labels,
                prediction_labels=prediction_labels, observation_ids=ids,
                bootstrap_weights=weights, repetitions=repetitions,
            )
            rows.append(interval_row(model_key, prompt_id, metric_bundle_to_dict(bundle),
                                     intervals))
            print(f"  {model_key:28} {prompt_id:28} done")

    table = pd.DataFrame(rows)
    check_against_committed(
        table,
        pd.read_csv(args.validation_dir / "condition_performance.csv"),
        pd.read_csv(args.validation_dir / "primary_model_performance.csv"),
    )
    out = args.validation_dir / "prompt_performance_intervals.csv"
    table.to_csv(out, index=False, float_format="%.12g")
    (args.validation_dir / "prompt_intervals_manifest.json").write_text(json.dumps({
        "analysis": "per_prompt_repetition_averaged_validation_intervals",
        "estimand": (
            "Design-weighted operational accuracy, macro-precision, macro-recall and "
            "macro-F1 against the OCPSG silver reference for each model x prompt, with "
            "the three repetitions of a speech averaged so it counts once."
        ),
        "bootstrap": {"method": "Rao-Wu rescaled within sampling strata",
                      "replicates": reps, "seed": seed, "interval": "95% percentile",
                      "pointwise": True},
        "checks": [
            "point estimates equal the repetition = 0 rows of condition_performance.csv",
            "P0 intervals equal primary_model_performance.csv",
        ],
        "note": ("P1-P3 intervals are new descriptive outputs from the same replicates "
                 "as the primary P0 analysis; no committed estimate changes."),
        "raw_inference_records_sha256": raw_hashes,
        "output": out.as_posix(),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out.as_posix()} ({len(table)} rows); P0 intervals reproduce the primary "
          "analysis.")


if __name__ == "__main__":
    main()
