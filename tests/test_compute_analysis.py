import math

import pandas as pd

from reporting_validation_standards.compute_analysis import (
    aggregate_local_sessions,
    summarize_local_timing,
    summarize_records,
)


def test_summarize_records_separates_validity_latency_retries_and_cost():
    frame = pd.DataFrame(
        {
            "is_valid": [True, True, False],
            "latency_seconds": [1.0, 2.0, 4.0],
            "retry_count": [0, 1, 2],
            "input_tokens": [100, 200, None],
            "output_tokens": [10, 20, None],
            "total_tokens": [110, 220, None],
            "cost_usd": [0.1, 0.2, None],
        }
    )
    summary = summarize_records(frame)

    assert summary["scheduled_decisions"] == 3
    assert summary["valid_decisions"] == 2
    assert math.isclose(summary["valid_rate"], 2 / 3)
    assert summary["records_with_retries"] == 2
    assert summary["total_adapter_retries"] == 3
    assert summary["input_tokens_total"] == 300
    assert math.isclose(summary["recorded_cost_total_usd"], 0.3)


def test_summarize_local_timing_distinguishes_session_start_and_warm_load():
    frame = pd.DataFrame(
        {
            "is_valid": [True, True, True, True],
            "run_id": ["s1", "s1", "s2", "s2"],
            "request_started_utc": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:10Z",
                "2026-01-02T00:00:00Z",
                "2026-01-02T00:00:10Z",
            ],
            "provider_metadata": [
                {
                    "load_duration_seconds": 10.0,
                    "prompt_eval_duration_seconds": 1.0,
                    "eval_duration_seconds": 0.5,
                    "eval_tokens_per_second": 20.0,
                    "total_duration_seconds": 12.0,
                },
                {
                    "load_duration_seconds": 0.01,
                    "prompt_eval_duration_seconds": 2.0,
                    "eval_duration_seconds": 0.6,
                    "eval_tokens_per_second": 18.0,
                    "total_duration_seconds": 2.8,
                },
                {
                    "load_duration_seconds": 12.0,
                    "prompt_eval_duration_seconds": 1.5,
                    "eval_duration_seconds": 0.4,
                    "eval_tokens_per_second": 22.0,
                    "total_duration_seconds": 14.0,
                },
                {
                    "load_duration_seconds": 0.02,
                    "prompt_eval_duration_seconds": 2.5,
                    "eval_duration_seconds": 0.7,
                    "eval_tokens_per_second": 17.0,
                    "total_duration_seconds": 3.4,
                },
            ],
        }
    )

    summary = summarize_local_timing(frame)
    assert summary["sessions_observed"] == 2
    assert math.isclose(summary["session_start_load_median_seconds"], 11.0)
    assert math.isclose(summary["warm_load_median_seconds"], 0.015)


def test_aggregate_local_sessions_sums_machine_level_measurements():
    base = {
        "carbon_tracking_enabled": True,
        "ollama_version": "0.34.3",
        "local_model_digest": "abc",
        "nvidia_metadata": {
            "gpus": [
                {
                    "name": "GPU",
                    "driver_version": "1.0",
                    "memory_total_mb": "12000",
                    "power_limit_w": "200",
                }
            ]
        },
        "local_model_registry_metadata": {
            "details": {
                "quantization_level": "Q4_K_M",
                "parameter_size": "10B",
            }
        },
        "runtime_residency_metadata": {
            "context_length": 8192,
            "size_bytes": 100,
            "size_vram_bytes": 50,
            "vram_fraction": 0.5,
        },
    }
    sessions = []
    for index, processed in enumerate([40, 60], start=1):
        session = dict(base)
        session.update(
            {
                "session_id": f"s{index}",
                "status": "complete_or_staged",
                "n_processed": processed,
                "n_errors": 0,
                "elapsed_seconds": 10.0 * index,
                "carbon_summary": {
                    "codecarbon_version": "3.3.1",
                    "country_iso_code": "NLD",
                    "tracking_mode": "machine",
                    "measure_power_secs": 1,
                    "duration_seconds": 9.0 * index,
                    "energy_consumed_kwh": 0.1 * index,
                    "emissions_kg_co2e": 0.02 * index,
                    "cpu_energy_kwh": 0.04 * index,
                    "gpu_energy_kwh": 0.05 * index,
                    "ram_energy_kwh": 0.01 * index,
                },
            }
        )
        sessions.append(session)

    summary, session_frame, environment = aggregate_local_sessions(
        sessions,
        expected_decisions=100,
    )

    assert len(session_frame) == 2
    assert math.isclose(summary["energy_consumed_kwh"], 0.3)
    assert math.isclose(summary["emissions_kg_co2e"], 0.06)
    assert environment["country_iso_code"] == "NLD"
    assert environment["runtime_context_length"] == "8192"
