from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def numeric_summary(values: list[Any]) -> dict[str, float | None]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


def summarize_records(records: pd.DataFrame) -> dict[str, Any]:
    valid = records["is_valid"].astype(bool)
    latency = numeric_summary(records["latency_seconds"].tolist())
    retries = pd.to_numeric(records["retry_count"], errors="coerce").fillna(0)

    def total(column: str) -> int | None:
        if column not in records:
            return None
        values = pd.to_numeric(records[column], errors="coerce")
        return int(values.fillna(0).sum()) if values.notna().any() else None

    def valid_mean(column: str) -> float | None:
        if column not in records:
            return None
        values = pd.to_numeric(records.loc[valid, column], errors="coerce")
        return float(values.mean()) if values.notna().any() else None

    costs = pd.to_numeric(records.get("cost_usd"), errors="coerce")
    cost_total = float(costs.fillna(0).sum()) if costs.notna().any() else None

    return {
        "scheduled_decisions": int(len(records)),
        "valid_decisions": int(valid.sum()),
        "valid_rate": float(valid.mean()),
        "latency_mean_seconds": latency["mean"],
        "latency_median_seconds": latency["median"],
        "latency_p95_seconds": latency["p95"],
        "latency_max_seconds": latency["max"],
        "records_with_retries": int((retries > 0).sum()),
        "total_adapter_retries": int(retries.sum()),
        "input_tokens_total": total("input_tokens"),
        "output_tokens_total": total("output_tokens"),
        "total_tokens_total": total("total_tokens"),
        "input_tokens_mean_valid": valid_mean("input_tokens"),
        "output_tokens_mean_valid": valid_mean("output_tokens"),
        "total_tokens_mean_valid": valid_mean("total_tokens"),
        "recorded_cost_total_usd": cost_total,
    }


def summarize_local_timing(records: pd.DataFrame) -> dict[str, Any]:
    valid = records.loc[records["is_valid"].astype(bool)].copy()
    if valid.empty:
        raise ValueError("No valid local inference records.")

    def meta_value(item: Any, key: str) -> Any:
        return item.get(key) if isinstance(item, dict) else None

    for column, key in {
        "_load": "load_duration_seconds",
        "_prompt": "prompt_eval_duration_seconds",
        "_generation": "eval_duration_seconds",
        "_throughput": "eval_tokens_per_second",
        "_ollama_total": "total_duration_seconds",
    }.items():
        valid[column] = valid["provider_metadata"].map(
            lambda item, k=key: meta_value(item, k)
        )

    ordered = valid.sort_values(["run_id", "request_started_utc"], kind="stable")
    first = ordered.groupby("run_id", sort=False).head(1)
    warm = ordered.drop(index=first.index)

    load_all = numeric_summary(valid["_load"].tolist())
    load_first = numeric_summary(first["_load"].tolist())
    load_warm = numeric_summary(warm["_load"].tolist())
    prompt = numeric_summary(valid["_prompt"].tolist())
    generation = numeric_summary(valid["_generation"].tolist())
    throughput = numeric_summary(valid["_throughput"].tolist())
    total_duration = numeric_summary(valid["_ollama_total"].tolist())

    return {
        "valid_records": int(len(valid)),
        "sessions_observed": int(valid["run_id"].nunique()),
        "load_duration_mean_seconds": load_all["mean"],
        "load_duration_median_seconds": load_all["median"],
        "session_start_load_median_seconds": load_first["median"],
        "session_start_load_max_seconds": load_first["max"],
        "warm_load_median_seconds": load_warm["median"],
        "prompt_eval_mean_seconds": prompt["mean"],
        "prompt_eval_median_seconds": prompt["median"],
        "prompt_eval_p95_seconds": prompt["p95"],
        "generation_mean_seconds": generation["mean"],
        "generation_median_seconds": generation["median"],
        "generation_p95_seconds": generation["p95"],
        "generation_throughput_mean_tokens_per_second": throughput["mean"],
        "generation_throughput_median_tokens_per_second": throughput["median"],
        "generation_throughput_p95_tokens_per_second": throughput["p95"],
        "ollama_total_duration_mean_seconds": total_duration["mean"],
        "ollama_total_duration_median_seconds": total_duration["median"],
    }


def aggregate_local_sessions(
    sessions: list[dict[str, Any]],
    expected_decisions: int,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    environment_values: dict[str, set[str]] = {
        key: set()
        for key in [
            "codecarbon_version",
            "country_iso_code",
            "tracking_mode",
            "ollama_version",
            "local_model_digest",
            "gpu_name",
            "driver_version",
            "quantization_level",
            "parameter_size",
            "runtime_context_length",
        ]
    }

    for session in sessions:
        carbon = session.get("carbon_summary")
        if not session.get("carbon_tracking_enabled") or not isinstance(carbon, dict):
            raise RuntimeError(f"Missing carbon summary for {session.get('session_id')}.")

        gpu_list = (session.get("nvidia_metadata") or {}).get("gpus") or []
        gpu = gpu_list[0] if gpu_list else {}
        registry = session.get("local_model_registry_metadata") or {}
        details = registry.get("details") or {}
        residency = session.get("runtime_residency_metadata") or {}

        observed = {
            "codecarbon_version": carbon.get("codecarbon_version"),
            "country_iso_code": carbon.get("country_iso_code"),
            "tracking_mode": carbon.get("tracking_mode"),
            "ollama_version": session.get("ollama_version"),
            "local_model_digest": session.get("local_model_digest"),
            "gpu_name": gpu.get("name"),
            "driver_version": gpu.get("driver_version"),
            "quantization_level": details.get("quantization_level"),
            "parameter_size": details.get("parameter_size"),
            "runtime_context_length": residency.get("context_length"),
        }
        for key, value in observed.items():
            environment_values[key].add(str(value))

        rows.append({
            "session_id": session.get("session_id"),
            "status": session.get("status"),
            "n_processed": int(session.get("n_processed") or 0),
            "n_errors": int(session.get("n_errors") or 0),
            "elapsed_seconds": float(session.get("elapsed_seconds") or 0),
            "tracked_duration_seconds": float(carbon.get("duration_seconds") or 0),
            "energy_consumed_kwh": float(carbon.get("energy_consumed_kwh") or 0),
            "emissions_kg_co2e": float(carbon.get("emissions_kg_co2e") or 0),
            "cpu_energy_kwh": float(carbon.get("cpu_energy_kwh") or 0),
            "gpu_energy_kwh": float(carbon.get("gpu_energy_kwh") or 0),
            "ram_energy_kwh": float(carbon.get("ram_energy_kwh") or 0),
        })

    for key, values in environment_values.items():
        if len(values) != 1:
            raise RuntimeError(f"Runtime metadata changed across sessions for {key}: {values}")

    frame = pd.DataFrame(rows)
    if int(frame["n_processed"].sum()) != expected_decisions:
        raise RuntimeError("Local session manifests do not account for all decisions.")

    energy = float(frame["energy_consumed_kwh"].sum())
    emissions = float(frame["emissions_kg_co2e"].sum())
    duration = float(frame["tracked_duration_seconds"].sum())
    summary = {
        "sessions": int(len(frame)),
        "decisions": expected_decisions,
        "tracked_duration_seconds": duration,
        "session_elapsed_seconds": float(frame["elapsed_seconds"].sum()),
        "energy_consumed_kwh": energy,
        "emissions_kg_co2e": emissions,
        "cpu_energy_kwh": float(frame["cpu_energy_kwh"].sum()),
        "gpu_energy_kwh": float(frame["gpu_energy_kwh"].sum()),
        "ram_energy_kwh": float(frame["ram_energy_kwh"].sum()),
        "run_average_energy_wh_per_1000_decisions": energy * 1_000_000 / expected_decisions,
        "run_average_emissions_g_co2e_per_1000_decisions": emissions * 1_000_000 / expected_decisions,
        "tracked_decisions_per_hour": expected_decisions / (duration / 3600),
        "implied_emissions_intensity_kg_co2e_per_kwh": emissions / energy,
    }

    first = sessions[0]
    first_gpu = ((first.get("nvidia_metadata") or {}).get("gpus") or [{}])[0]
    residency = first.get("runtime_residency_metadata") or {}
    environment = {key: next(iter(values)) for key, values in environment_values.items()}
    environment.update({
        "gpu_memory_total_mb": first_gpu.get("memory_total_mb"),
        "gpu_power_limit_w": first_gpu.get("power_limit_w"),
        "model_size_bytes": residency.get("size_bytes"),
        "model_vram_bytes": residency.get("size_vram_bytes"),
        "vram_fraction": residency.get("vram_fraction"),
        "measure_power_secs": (first.get("carbon_summary") or {}).get("measure_power_secs"),
    })
    return summary, frame, environment
