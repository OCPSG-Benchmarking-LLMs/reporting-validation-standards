#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from reporting_validation_standards.experiment_specs import (
    load_model_config,
    load_prompt_config,
)
from reporting_validation_standards.stability_analysis import (
    build_model_pair_agreement,
    build_observation_model_summary,
    build_panel_agreement,
    build_prompt_comparisons,
    build_prompt_quads,
    build_repeat_triads,
    prepare_records,
    read_inference_jsonl,
    sha256_file,
    summarize_by_reference_label,
    summarize_model,
    summarize_repeat_by_prompt,
)


PRODUCTION_ROOT = Path("runs/production/stability")
FINALIZED_ROOT = Path("results/production")
SAMPLE_MANIFEST = Path("data/manifests/uk_test_1000_ids.csv")
OUTPUT_ROOT = Path("outputs/stability")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse repeated-inference, prompt, and model-family stability for "
            "the completed production panel."
        )
    )
    parser.add_argument("--production-root", type=Path, default=PRODUCTION_ROOT)
    parser.add_argument("--finalized-root", type=Path, default=FINALIZED_ROOT)
    parser.add_argument("--sample-manifest", type=Path, default=SAMPLE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
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


def _failure_diagnostics(records: pd.DataFrame) -> pd.DataFrame:
    failures = records.loc[~records["is_valid"]].copy()
    if failures.empty:
        return pd.DataFrame(
            columns=[
                "model_key",
                "error_type",
                "last_status_code",
                "count",
            ]
        )

    def last_status_code(metadata: Any) -> int | None:
        if not isinstance(metadata, dict):
            return None
        events = metadata.get("retry_events") or []
        if not events:
            return None
        status = events[-1].get("status_code")
        return int(status) if status is not None else None

    failures["error_type"] = failures.get(
        "error_type",
        pd.Series(index=failures.index, dtype="object"),
    ).fillna("UnknownError")
    failures["last_status_code"] = failures.get(
        "provider_metadata",
        pd.Series(index=failures.index, dtype="object"),
    ).map(last_status_code)

    return (
        failures.groupby(
            ["model_key", "error_type", "last_status_code"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .sort_values(["model_key", "count"], ascending=[True, False])
        .reset_index(drop=True)
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
    if not qa.get("structural_qa_pass"):
        raise RuntimeError(f"{model_key} did not pass structural QA.")
    if not qa.get("completion_qa_pass"):
        raise RuntimeError(f"{model_key} did not pass completion QA.")

    observed_hash = sha256_file(records_path)
    expected_hash = str(qa.get("raw_inference_records_sha256") or "")
    if not expected_hash:
        raise RuntimeError(
            f"{model_key} finalized QA summary lacks raw inference SHA-256."
        )
    if observed_hash != expected_hash:
        raise RuntimeError(
            f"Raw inference hash mismatch for {model_key}: "
            f"expected {expected_hash}, observed {observed_hash}. "
            "Refusing analysis of data that do not match finalized provenance."
        )

    records = read_inference_jsonl(records_path)
    expected_n = int(qa["expected_decisions"])
    if len(records) != expected_n:
        raise RuntimeError(
            f"{model_key} has {len(records):,} raw records but finalized QA "
            f"expects {expected_n:,}."
        )
    if records["model_key"].nunique() != 1 or records["model_key"].iloc[0] != model_key:
        raise RuntimeError(f"{model_key} raw records contain inconsistent model keys.")

    return records, qa


def main() -> None:
    args = parse_args()

    model_config = load_model_config()
    prompt_config = load_prompt_config()

    model_keys = [str(model["model_key"]) for model in model_config["models"]]
    model_stage = {
        str(model["model_key"]): str(model.get("stage") or "")
        for model in model_config["models"]
    }

    prompt_ids = [str(prompt["prompt_id"]) for prompt in prompt_config["prompts"]]
    if "p0_capbench_canonical" not in prompt_ids:
        raise RuntimeError("Canonical P0 prompt is missing from prompt configuration.")
    canonical_prompt_id = "p0_capbench_canonical"
    perturbation_prompt_ids = [
        prompt_id for prompt_id in prompt_ids if prompt_id != canonical_prompt_id
    ]

    sample_design = pd.read_csv(args.sample_manifest)
    if len(sample_design) != 1000:
        raise RuntimeError(
            f"Expected the frozen 1,000-case sample manifest, found {len(sample_design):,} rows."
        )

    qa_by_model: dict[str, Any] = {}
    frames: list[pd.DataFrame] = []

    print("Verifying finalized production inputs")
    for model_key in model_keys:
        records, qa = _read_and_verify_model(
            model_key=model_key,
            production_root=args.production_root,
            finalized_root=args.finalized_root,
        )
        qa_by_model[model_key] = qa
        frames.append(records)
        parse_counts = qa.get("parse_status") or {}
        print(
            f"  {model_key:28} "
            f"records={len(records):,}  "
            f"ok={int(parse_counts.get('ok', 0)):,}  "
            f"errors={int(parse_counts.get('error', 0)):,}  "
            f"hash=PASS"
        )

    records = pd.concat(frames, ignore_index=True)
    records = prepare_records(records, sample_design)

    repetitions = int(model_config["experiment_policy"]["repetitions"])
    expected_prompts = len(prompt_ids)
    expected_decisions_per_observation = repetitions * expected_prompts

    expected_total = (
        len(model_keys)
        * len(sample_design)
        * expected_prompts
        * repetitions
    )
    if len(records) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total:,} total records across the configured panel, "
            f"found {len(records):,}."
        )

    repeat_triads = build_repeat_triads(
        records,
        expected_repetitions=repetitions,
    )
    prompt_quads = build_prompt_quads(
        records,
        expected_prompts=expected_prompts,
    )
    observation_summary = build_observation_model_summary(
        records,
        expected_decisions_per_observation=expected_decisions_per_observation,
    )

    model_summary = pd.DataFrame(
        [
            summarize_model(
                model_key=model_key,
                decisions=records,
                repeat_triads=repeat_triads,
                prompt_quads=prompt_quads,
                observation_summary=observation_summary,
            )
            for model_key in model_keys
        ]
    )
    model_summary.insert(
        1,
        "stage",
        model_summary["model_key"].map(model_stage),
    )

    repeat_by_prompt = summarize_repeat_by_prompt(repeat_triads)
    prompt_comparisons = build_prompt_comparisons(
        records,
        canonical_prompt_id=canonical_prompt_id,
        perturbation_prompt_ids=perturbation_prompt_ids,
    )
    model_pair_agreement = build_model_pair_agreement(
        records,
        model_keys=model_keys,
        canonical_prompt_id=canonical_prompt_id,
    )

    original_six = [
        model_key
        for model_key in model_keys
        if model_stage[model_key] in {"core", "secondary"}
    ]
    panels = {
        "original_six": original_six,
        "all_seven": model_keys,
    }
    panel_agreement = build_panel_agreement(
        records,
        panels=panels,
        canonical_prompt_id=canonical_prompt_id,
    )

    by_reference_label = summarize_by_reference_label(
        records=records,
        observation_summary=observation_summary,
    )
    failure_diagnostics = _failure_diagnostics(records)

    observation_output = observation_summary.copy()
    observation_output["instability_rank"] = (
        observation_output.groupby("model_key")["entropy_bits_valid"]
        .rank(method="min", ascending=False, na_option="bottom")
        .astype("Int64")
    )
    observation_output = observation_output.sort_values(
        ["model_key", "instability_rank", "observation_id"],
        kind="stable",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "model_summary.csv": model_summary,
        "repetition_by_prompt.csv": repeat_by_prompt,
        "prompt_comparisons.csv": prompt_comparisons,
        "model_pair_agreement.csv": model_pair_agreement,
        "panel_agreement.csv": panel_agreement,
        "stability_by_reference_label.csv": by_reference_label,
        "failure_diagnostics.csv": failure_diagnostics,
        "observation_instability.csv": observation_output,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False)

    source_hashes = {
        model_key: qa_by_model[model_key]["raw_inference_records_sha256"]
        for model_key in model_keys
    }
    manifest = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "analysis": "prompt_and_inference_stability",
        "model_keys": model_keys,
        "model_stages": model_stage,
        "original_panel_model_keys": original_six,
        "supplementary_model_keys": [
            model_key
            for model_key in model_keys
            if model_stage[model_key] == "supplementary"
        ],
        "canonical_prompt_id": canonical_prompt_id,
        "perturbation_prompt_ids": perturbation_prompt_ids,
        "repetitions": repetitions,
        "n_observations": len(sample_design),
        "n_total_records": len(records),
        "sample_id_sha256": qa_by_model[model_keys[0]]["sample_id_sha256"],
        "dataset_revision": qa_by_model[model_keys[0]]["dataset_revision"],
        "schedule_sha256": qa_by_model[model_keys[0]]["schedule_sha256"],
        "raw_inference_sha256": source_hashes,
        "error_handling": {
            "operational_failures": (
                "Parse/API failures remain observed outcomes for reliability metrics. "
                "Classification-stability metrics are reported both unconditionally "
                "(a failure prevents stability) and conditionally on all required "
                "outputs being valid."
            ),
            "invalid_label_sentinel": "__INVALID_OUTPUT__",
        },
        "weighting": {
            "primary_weighted_columns": (
                "Use inverse-inclusion sampling weights from the frozen "
                "label-stratified probability sample."
            ),
            "unweighted_columns": (
                "Retained as descriptive sample quantities for transparency."
            ),
            "uncertainty_intervals": (
                "Not estimated in this script. Design-based uncertainty is handled "
                "in the subsequent validation analysis rather than mixed into the "
                "stability diagnostics."
            ),
        },
        "outputs": list(outputs),
    }
    _write_json(args.output_dir / "analysis_manifest.json", manifest)

    print("\nModel-level stability summary")
    display_cols = [
        "model_key",
        "valid_decision_rate_weighted",
        "repeat_unanimity_given_complete_weighted",
        "four_prompt_unanimity_given_complete_weighted",
        "all12_unanimity_unconditional_weighted",
    ]
    display = model_summary[display_cols].copy()
    for column in display_cols[1:]:
        display[column] = display[column].map(
            lambda value: "NA" if pd.isna(value) else f"{100 * value:.2f}%"
        )
    print(display.to_string(index=False))

    if not failure_diagnostics.empty:
        print("\nObserved production failures")
        print(failure_diagnostics.to_string(index=False))

    print("\nPanel-level agreement")
    panel_display = panel_agreement.copy()
    for column in [
        "complete_rate_weighted",
        "unanimity_unconditional_weighted",
        "unanimity_given_complete_weighted",
    ]:
        panel_display[column] = panel_display[column].map(
            lambda value: "NA" if pd.isna(value) else f"{100 * value:.2f}%"
        )
    print(
        panel_display[
            [
                "panel",
                "scope",
                "complete_rate_weighted",
                "unanimity_unconditional_weighted",
                "unanimity_given_complete_weighted",
            ]
        ].to_string(index=False)
    )

    print(f"\nAnalysis outputs: {args.output_dir}")
    print(
        "Primary interpretation rule: use weighted columns for GB test-population "
        "descriptives; use unweighted columns only as sample diagnostics."
    )


if __name__ == "__main__":
    main()
