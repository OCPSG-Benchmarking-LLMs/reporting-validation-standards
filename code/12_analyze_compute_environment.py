#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reporting_validation_standards.compute_analysis import (
    aggregate_local_sessions,
    summarize_local_timing,
    summarize_records,
)
from reporting_validation_standards.experiment_specs import load_model_config
from reporting_validation_standards.stability_analysis import (
    read_inference_jsonl,
    sha256_file,
)


PRODUCTION_ROOT = Path("runs/production/stability")
PRODUCTION_RESULTS_ROOT = Path("results/production")
DEPLOYMENT_ID = "gb_test_p0_gpt54_mini_v1"
DEPLOYMENT_ROOT = Path("runs/deployment")
DEPLOYMENT_RESULTS_ROOT = Path("results/deployment")
OUTPUT_ROOT = Path("outputs/compute_environment")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _verify_records(
    *,
    records_path: Path,
    qa_path: Path,
    expected_model_key: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    qa = _load_json(qa_path)
    if not qa.get("structural_qa_pass") or not qa.get("completion_qa_pass"):
        raise RuntimeError(f"{qa_path} is not a finalized complete run.")

    observed_hash = sha256_file(records_path)
    expected_hash = str(qa.get("raw_inference_records_sha256") or "")
    if observed_hash != expected_hash:
        raise RuntimeError(
            f"Raw inference hash mismatch for {records_path}: "
            f"expected {expected_hash}, observed {observed_hash}."
        )

    records = read_inference_jsonl(records_path)
    expected_n = int(qa["expected_decisions"])
    if len(records) != expected_n:
        raise RuntimeError(
            f"{records_path} contains {len(records):,} records; "
            f"expected {expected_n:,}."
        )
    if records["model_key"].nunique() != 1:
        raise RuntimeError(f"{records_path} contains multiple model keys.")
    if str(records["model_key"].iloc[0]) != expected_model_key:
        raise RuntimeError(
            f"{records_path} model key differs from {expected_model_key}."
        )
    return records, qa


def _load_finalized_sessions(model_key: str) -> list[dict[str, Any]]:
    session_paths = sorted(
        (PRODUCTION_RESULTS_ROOT / model_key / "sessions").glob(
            "*/session_manifest.json"
        )
    )
    if not session_paths:
        raise RuntimeError(f"No finalized session manifests found for {model_key}.")
    return [_load_json(path) for path in session_paths]


def _validate_against_qa(
    summary: dict[str, Any],
    qa: dict[str, Any],
    *,
    model_key: str,
) -> None:
    if summary["scheduled_decisions"] != int(qa["expected_decisions"]):
        raise RuntimeError(f"{model_key} decision count disagrees with finalized QA.")

    qa_retries = int(qa.get("total_adapter_retries") or 0)
    if summary["total_adapter_retries"] != qa_retries:
        raise RuntimeError(f"{model_key} retry count disagrees with finalized QA.")

    qa_cost = qa.get("recorded_cost_total_usd")
    record_cost = summary["recorded_cost_total_usd"]
    if qa_cost is None and record_cost is not None:
        raise RuntimeError(f"{model_key} raw records contain unexpected cost values.")
    if qa_cost is not None:
        if record_cost is None or not np.isclose(
            float(qa_cost),
            float(record_cost),
            rtol=0,
            atol=1e-10,
        ):
            raise RuntimeError(f"{model_key} cost total disagrees with finalized QA.")


def main() -> None:
    model_config = load_model_config()
    models = model_config["models"]
    model_keys = [str(model["model_key"]) for model in models]

    production_rows: list[dict[str, Any]] = []
    local_timing_rows: list[dict[str, Any]] = []
    local_environment_rows: list[dict[str, Any]] = []
    local_session_frames: list[pd.DataFrame] = []
    source_hashes: dict[str, str] = {}

    print("Verifying finalized production inputs")
    for model in models:
        model_key = str(model["model_key"])
        records_path = (
            PRODUCTION_ROOT / model_key / "inference_records.jsonl"
        )
        qa_path = PRODUCTION_RESULTS_ROOT / model_key / "qa_summary.json"
        records, qa = _verify_records(
            records_path=records_path,
            qa_path=qa_path,
            expected_model_key=model_key,
        )
        source_hashes[model_key] = qa["raw_inference_records_sha256"]

        summary = summarize_records(records)
        _validate_against_qa(summary, qa, model_key=model_key)

        is_local = str(model["provider"]) == "ollama"
        production_rows.append(
            {
                "model_key": model_key,
                "provider": str(model["provider"]),
                "stage": str(model.get("stage") or ""),
                "requested_model_id": str(model["model_id"]),
                "infrastructure_scope": (
                    "local_observed_machine"
                    if is_local
                    else "hosted_provider_hidden"
                ),
                "energy_measurement_scope": (
                    "machine_level_session"
                    if is_local
                    else "not_observed"
                ),
                "carbon_measurement_scope": (
                    "machine_level_session_estimate"
                    if is_local
                    else "not_observed"
                ),
                **summary,
            }
        )

        print(
            f"  {model_key:28} records={len(records):,}  "
            f"valid={100 * summary['valid_rate']:.2f}%  hash=PASS"
        )

        if is_local:
            timing = summarize_local_timing(records)
            timing["model_key"] = model_key
            timing["requested_model_id"] = str(model["model_id"])
            local_timing_rows.append(timing)

            sessions = _load_finalized_sessions(model_key)
            environment, session_frame, runtime = aggregate_local_sessions(
                sessions,
                expected_decisions=int(qa["expected_decisions"]),
            )
            local_environment_rows.append(
                {
                    "model_key": model_key,
                    "requested_model_id": str(model["model_id"]),
                    **environment,
                    **runtime,
                }
            )
            session_frame.insert(0, "model_key", model_key)
            local_session_frames.append(session_frame)

    production_summary = pd.DataFrame(production_rows)
    local_timing = pd.DataFrame(local_timing_rows)
    local_environment = pd.DataFrame(local_environment_rows)
    local_sessions = pd.concat(local_session_frames, ignore_index=True)

    print("\nVerifying finalized full-population deployment")
    deployment_records_path = (
        DEPLOYMENT_ROOT / DEPLOYMENT_ID / "inference_records.jsonl"
    )
    deployment_qa_path = (
        DEPLOYMENT_RESULTS_ROOT / DEPLOYMENT_ID / "qa_summary.json"
    )
    deployment_records, deployment_qa = _verify_records(
        records_path=deployment_records_path,
        qa_path=deployment_qa_path,
        expected_model_key="openai_gpt54_mini",
    )
    deployment_summary_dict = summarize_records(deployment_records)
    _validate_against_qa(
        deployment_summary_dict,
        deployment_qa,
        model_key="openai_gpt54_mini deployment",
    )
    deployment_summary = pd.DataFrame(
        [
            {
                "deployment_id": DEPLOYMENT_ID,
                "model_key": "openai_gpt54_mini",
                "provider": "openai",
                "requested_model_id": deployment_qa["requested_model_id"],
                "infrastructure_scope": "hosted_provider_hidden",
                "energy_measurement_scope": "not_observed",
                "carbon_measurement_scope": "not_observed",
                **deployment_summary_dict,
            }
        ]
    )
    print(
        f"  {DEPLOYMENT_ID}: "
        f"records={len(deployment_records):,}  "
        f"valid={100 * deployment_summary_dict['valid_rate']:.2f}%  hash=PASS"
    )

    reporting_rows: list[dict[str, Any]] = []
    for row in production_rows:
        local = row["infrastructure_scope"] == "local_observed_machine"
        reporting_rows.append(
            {
                "model_key": row["model_key"],
                "provider": row["provider"],
                "request_latency_reportable": True,
                "token_usage_reportable": True,
                "retry_diagnostics_reportable": True,
                "monetary_cost_reportable": (
                    row["recorded_cost_total_usd"] is not None
                ),
                "machine_energy_reportable": local,
                "machine_carbon_estimate_reportable": local,
                "provider_side_energy_reportable": False,
                "provider_side_carbon_reportable": False,
                "environmental_claim_scope": (
                    "Dedicated local machine sessions measured with CodeCarbon; "
                    "report run/session totals and normalized run-average intensity."
                    if local
                    else
                    "No provider-side infrastructure or energy telemetry available; "
                    "do not infer carbon from latency, tokens, or price."
                ),
            }
        )
    reporting_matrix = pd.DataFrame(reporting_rows)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    production_summary.to_csv(
        OUTPUT_ROOT / "production_compute_summary.csv",
        index=False,
    )
    deployment_summary.to_csv(
        OUTPUT_ROOT / "deployment_compute_summary.csv",
        index=False,
    )
    local_timing.to_csv(
        OUTPUT_ROOT / "local_inference_timing.csv",
        index=False,
    )
    local_environment.to_csv(
        OUTPUT_ROOT / "local_environment_summary.csv",
        index=False,
    )
    local_sessions.to_csv(
        OUTPUT_ROOT / "local_session_environment.csv",
        index=False,
    )
    reporting_matrix.to_csv(
        OUTPUT_ROOT / "environmental_reporting_scope.csv",
        index=False,
    )

    manifest = {
        "schema_version": 1,
        "analysis": "computational_performance_and_environmental_transparency",
        "production_model_keys": model_keys,
        "production_raw_inference_sha256": source_hashes,
        "deployment_id": DEPLOYMENT_ID,
        "deployment_raw_inference_sha256": (
            deployment_qa["raw_inference_records_sha256"]
        ),
        "request_level_metrics": {
            "latency": "end_to_end_including_retries_and_backoff",
            "tokens": "provider/runtime reported token accounting",
            "retries": "explicit adapter retries; SDK retries disabled where applicable",
            "monetary_cost": (
                "recorded only when dated provider pricing and usage accounting "
                "were available"
            ),
        },
        "local_environmental_measurement": {
            "scope": "dedicated local production sessions",
            "tool": "CodeCarbon",
            "aggregation": (
                "Session-level CPU, GPU, RAM, total energy, and emissions estimates "
                "are summed across all production sessions contributing to the "
                "12,000 finalized decisions for each local model."
            ),
            "normalized_metrics": (
                "Per-1,000-decision energy/emissions values are run-average "
                "normalizations of measured session totals, not direct per-request "
                "carbon measurements."
            ),
            "load_timing": (
                "First valid request per session is reported as session-start load. "
                "It is not labelled a perfect cold start because OS/filesystem/driver "
                "caches may persist between sessions."
            ),
        },
        "hosted_environmental_measurement": {
            "scope": "not observed",
            "rule": (
                "No server-side energy or carbon estimate is produced for hosted "
                "providers because infrastructure, utilization, power, and electricity "
                "mix are not exposed by the recorded API telemetry."
            ),
        },
        "interpretive_principle": (
            "Environmental precision should not exceed infrastructure knowledge."
        ),
        "outputs": [
            "production_compute_summary.csv",
            "deployment_compute_summary.csv",
            "local_inference_timing.csv",
            "local_environment_summary.csv",
            "local_session_environment.csv",
            "environmental_reporting_scope.csv",
        ],
    }
    _write_json(OUTPUT_ROOT / "analysis_manifest.json", manifest)

    print("\nProduction computational summary")
    display = production_summary[
        [
            "model_key",
            "valid_rate",
            "latency_median_seconds",
            "latency_p95_seconds",
            "total_tokens_mean_valid",
            "records_with_retries",
            "recorded_cost_total_usd",
        ]
    ].copy()
    display["valid_rate"] = display["valid_rate"].map(
        lambda value: f"{100 * value:.2f}%"
    )
    print(display.to_string(index=False))

    print("\nLocal machine-level environmental summary")
    env_display = local_environment[
        [
            "model_key",
            "sessions",
            "tracked_duration_seconds",
            "energy_consumed_kwh",
            "emissions_kg_co2e",
            "run_average_energy_wh_per_1000_decisions",
            "run_average_emissions_g_co2e_per_1000_decisions",
        ]
    ].copy()
    env_display["tracked_duration_hours"] = (
        env_display["tracked_duration_seconds"] / 3600
    )
    env_display = env_display.drop(columns="tracked_duration_seconds")
    print(env_display.to_string(index=False))

    print("\nLocal inference timing")
    timing_display = local_timing[
        [
            "model_key",
            "session_start_load_median_seconds",
            "warm_load_median_seconds",
            "prompt_eval_median_seconds",
            "generation_median_seconds",
            "generation_throughput_median_tokens_per_second",
        ]
    ]
    print(timing_display.to_string(index=False))

    print("\nFull-population deployment")
    dep = deployment_summary.iloc[0]
    print(
        f"  decisions={int(dep['scheduled_decisions']):,}  "
        f"median_latency={dep['latency_median_seconds']:.3f}s  "
        f"p95_latency={dep['latency_p95_seconds']:.3f}s  "
        f"retries={int(dep['total_adapter_retries'])}  "
        f"cost_usd={dep['recorded_cost_total_usd']:.6f}"
    )
    print(f"\nAnalysis outputs: {OUTPUT_ROOT}")
    print(
        "Hosted-provider carbon is intentionally left unestimated; local carbon "
        "figures remain machine/session-level estimates."
    )


if __name__ == "__main__":
    main()
