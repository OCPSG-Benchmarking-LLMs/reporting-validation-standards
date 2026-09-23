#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reporting_validation_standards.experiment_specs import (
    load_model_config,
    load_prompt_config,
    render_prompt,
)
from reporting_validation_standards.fireworks_adapter import FireworksClassifier
from reporting_validation_standards.local_compute import (
    CodeCarbonRunTracker,
    collect_nvidia_metadata,
    collect_platform_metadata,
)
from reporting_validation_standards.ollama_adapter import OllamaClassifier
from reporting_validation_standards.openai_adapter import OpenAIClassifier
from reporting_validation_standards.production import (
    SCHEDULE_ALGORITHM_VERSION,
    make_stability_schedule,
    prompt_hashes,
    schedule_sha256,
    validate_existing_records,
)
from reporting_validation_standards.sample_io import (
    load_frozen_sample,
    select_input_text,
)


PRODUCTION_ROOT = Path("runs/production/stability")
MAX_OUTPUT_TOKENS = 40
TEMPERATURE = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one model in the frozen 1,000-case × 4-prompt × 3-repetition "
            "stability experiment. The runner is append-only and resumable."
        )
    )
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--input-mode", choices=["text_en", "text"], default="text_en")
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=None,
        help=(
            "Process at most this many currently pending cells, then stop cleanly. "
            "Useful for a staged launch; rerun the same command without this option to resume."
        ),
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--track-carbon", action="store_true")
    parser.add_argument(
        "--country-iso-code",
        default=None,
        help="Three-letter ISO code required with --track-carbon.",
    )
    parser.add_argument("--measure-power-secs", type=int, default=1)
    parser.add_argument("--ollama-host", default=None)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Malformed JSONL in {path} at line {line_number}. "
                "Do not continue until the file is repaired."
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Expected a JSON object in {path} at line {line_number}."
            )
        records.append(payload)
    return records


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except Exception:
            return None
        return completed.stdout.strip()

    head = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": head,
        "working_tree_dirty": bool(status) if status is not None else None,
    }


def _build_classifier(
    model: dict[str, Any],
    *,
    timeout_seconds: float | None,
    max_retries: int | None,
    ollama_host: str | None,
):
    provider = model["provider"]
    if provider == "openai":
        return OpenAIClassifier(
            timeout_seconds=timeout_seconds or 120.0,
            max_retries=3 if max_retries is None else max_retries,
        )
    if provider == "fireworks":
        return FireworksClassifier(
            timeout_seconds=timeout_seconds or 120.0,
            max_retries=3 if max_retries is None else max_retries,
        )
    if provider == "ollama":
        return OllamaClassifier(
            host=ollama_host,
            timeout_seconds=timeout_seconds or 600.0,
            max_retries=1 if max_retries is None else max_retries,
        )
    raise ValueError(f"Unsupported provider: {provider}")


def _effective_retry_count(model: dict[str, Any], requested: int | None) -> int:
    if requested is not None:
        return requested
    return 1 if model["provider"] == "ollama" else 3


def _effective_timeout(model: dict[str, Any], requested: float | None) -> float:
    if requested is not None:
        return requested
    return 600.0 if model["provider"] == "ollama" else 120.0


def main() -> None:
    args = parse_args()

    if args.max_decisions is not None and args.max_decisions <= 0:
        raise ValueError("--max-decisions must be positive.")
    if args.measure_power_secs <= 0:
        raise ValueError("--measure-power-secs must be positive.")

    model_config = load_model_config()
    prompt_config = load_prompt_config()
    model_lookup = {model["model_key"]: model for model in model_config["models"]}
    if args.model_key not in model_lookup:
        raise KeyError(f"Unknown model key: {args.model_key}")
    model = model_lookup[args.model_key]

    if args.track_carbon and model["provider"] != "ollama":
        raise ValueError("--track-carbon is only valid for local Ollama runs.")
    if args.track_carbon and not args.country_iso_code:
        raise ValueError("--country-iso-code is required with --track-carbon.")
    if model["provider"] == "ollama" and not args.plan_only and not args.track_carbon:
        raise ValueError(
            "Production Ollama runs require --track-carbon so every local "
            "production session contributes to run-level energy accounting."
        )

    policy = model_config["experiment_policy"]
    repetitions = int(policy["repetitions"])
    schedule_seed = int(policy["schedule_seed"])
    schedule_algorithm = str(policy["schedule_algorithm"])
    if schedule_algorithm != SCHEDULE_ALGORITHM_VERSION:
        raise RuntimeError(
            "Configured schedule algorithm does not match the implemented version: "
            f"{schedule_algorithm!r} != {SCHEDULE_ALGORITHM_VERSION!r}."
        )

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    sample, sample_metadata = load_frozen_sample(token=token)
    id_column = sample_metadata["id_column"]
    observation_ids = sample[id_column].astype(str).tolist()
    sample_by_id = {
        str(row[id_column]): row
        for _, row in sample.iterrows()
    }

    prompt_ids = [prompt["prompt_id"] for prompt in prompt_config["prompts"]]
    schedule = make_stability_schedule(
        observation_ids=observation_ids,
        prompt_ids=prompt_ids,
        repetitions=repetitions,
        seed=schedule_seed,
    )
    schedule_hash = schedule_sha256(schedule)
    expected_keys = {cell.key for cell in schedule}
    hashes = prompt_hashes(prompt_ids=prompt_ids)

    run_dir = PRODUCTION_ROOT / args.model_key
    records_path = run_dir / "inference_records.jsonl"
    manifest_path = run_dir / "experiment_manifest.json"

    frozen_spec: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "prompt_and_inference_stability_production",
        "model_key": args.model_key,
        "provider": model["provider"],
        "requested_model_id": model["model_id"],
        "snapshot_pinned": bool(model.get("snapshot_pinned", False)),
        "stage": model.get("stage"),
        "sample_id_sha256": sample_metadata["sample_id_sha256"],
        "dataset_revision": sample_metadata["dataset_revision"],
        "input_mode": args.input_mode,
        "n_cases": len(sample),
        "prompt_ids": prompt_ids,
        "prompt_spec_sha256": hashes,
        "repetitions": repetitions,
        "schedule_seed": schedule_seed,
        "schedule_algorithm": schedule_algorithm,
        "schedule_sha256": schedule_hash,
        "n_decisions": len(schedule),
        "temperature": TEMPERATURE,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "reasoning_setting": model["reasoning_setting"],
        "structured_output_mode": model["structured_output_mode"],
        "runtime_context_length": model.get("runtime_context_length"),
        "expected_local_digest": model.get("expected_local_digest"),
        "timeout_seconds": _effective_timeout(model, args.timeout_seconds),
        "max_retries": _effective_retry_count(model, args.max_retries),
        "carbon_tracking_required": model["provider"] == "ollama",
        "country_iso_code": (
            args.country_iso_code.upper()
            if model["provider"] == "ollama" and args.country_iso_code
            else None
        ),
        "measure_power_secs": (
            args.measure_power_secs if model["provider"] == "ollama" else None
        ),
    }

    existing_records = _read_jsonl(records_path)
    completed = validate_existing_records(
        existing_records,
        expected_model_key=args.model_key,
        expected_schedule_keys=expected_keys,
    )
    pending = [cell for cell in schedule if cell.key not in completed]

    print(f"Model: {args.model_key}")
    print(f"Provider: {model['provider']}")
    print(f"Frozen sample: {len(sample):,}")
    print(f"Prompts: {len(prompt_ids)}")
    print(f"Repetitions: {repetitions}")
    print(f"Schedule SHA-256: {schedule_hash}")
    print(f"Completed: {len(completed):,}/{len(schedule):,}")
    print(f"Pending: {len(pending):,}")

    if args.plan_only:
        print("Plan-only mode: no inference requests were sent.")
        return

    run_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("frozen_spec") != frozen_spec:
            raise RuntimeError(
                "Existing production manifest does not match the current frozen "
                "specification. Refusing to resume into an incompatible run directory."
            )
    else:
        manifest = {
            "created_at_utc": _utc_now(),
            "frozen_spec": frozen_spec,
        }
        _write_json_atomic(manifest_path, manifest)

    if not pending:
        print("Production model run is already complete.")
        return

    if args.max_decisions is not None:
        pending = pending[: args.max_decisions]

    classifier = _build_classifier(
        model,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        ollama_host=args.ollama_host,
    )

    local_digest = None
    local_model_metadata = None
    ollama_version = None
    runtime_residency_metadata = None

    if model["provider"] == "ollama":
        local_model_metadata = classifier.resolve_local_model(model["model_id"])
        local_digest = str(local_model_metadata.get("digest") or "")
        expected_digest = str(model.get("expected_local_digest") or "")
        if not expected_digest:
            raise RuntimeError(
                f"No expected_local_digest is frozen for {args.model_key}."
            )
        if local_digest != expected_digest:
            raise RuntimeError(
                "Local Ollama digest mismatch. Refusing production inference: "
                f"expected {expected_digest}, observed {local_digest}."
            )
        ollama_version = classifier.version()

    session_id = f"{args.model_key}_{_timestamp()}"
    session_dir = run_dir / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=False)

    carbon_tracker = None
    carbon_summary = None
    if args.track_carbon:
        carbon_tracker = CodeCarbonRunTracker(
            run_dir=session_dir,
            run_id=session_id,
            country_iso_code=args.country_iso_code,
            measure_power_secs=args.measure_power_secs,
        )

    platform_metadata = collect_platform_metadata()
    nvidia_metadata = (
        collect_nvidia_metadata()
        if model["provider"] == "ollama"
        else None
    )
    git_metadata = _git_metadata()

    session_started_utc = _utc_now()
    session_started = time.perf_counter()
    processed_this_session = 0
    errors_this_session = 0
    interrupted = False
    fatal_error: str | None = None

    if carbon_tracker is not None:
        carbon_tracker.start()

    try:
        with records_path.open("a", encoding="utf-8", buffering=1) as handle:
            for position, cell in enumerate(pending, start=1):
                row = sample_by_id[cell.observation_id]
                input_text = select_input_text(row, input_mode=args.input_mode)
                rendered = render_prompt(
                    prompt_id=cell.prompt_id,
                    text=input_text,
                )

                if model["provider"] == "openai":
                    record = classifier.classify(
                        run_id=session_id,
                        observation_id=cell.observation_id,
                        input_text=input_text,
                        model_key=args.model_key,
                        model_id=model["model_id"],
                        prompt=rendered,
                        official_labels=prompt_config["official_labels"],
                        repetition=cell.repetition,
                        reasoning_setting=model["reasoning_setting"],
                        temperature=TEMPERATURE,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                        structured_output_mode=model["structured_output_mode"],
                    )
                elif model["provider"] == "fireworks":
                    record = classifier.classify(
                        run_id=session_id,
                        observation_id=cell.observation_id,
                        input_text=input_text,
                        model_key=args.model_key,
                        model_id=model["model_id"],
                        prompt=rendered,
                        official_labels=prompt_config["official_labels"],
                        repetition=cell.repetition,
                        reasoning_setting=model["reasoning_setting"],
                        temperature=TEMPERATURE,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                        structured_output_mode=model["structured_output_mode"],
                    )
                else:
                    record = classifier.classify(
                        run_id=session_id,
                        observation_id=cell.observation_id,
                        input_text=input_text,
                        model_key=args.model_key,
                        model_id=model["model_id"],
                        local_model_digest=local_digest,
                        prompt=rendered,
                        official_labels=prompt_config["official_labels"],
                        repetition=cell.repetition,
                        reasoning_setting=model["reasoning_setting"],
                        temperature=TEMPERATURE,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                        runtime_context_length=int(model["runtime_context_length"]),
                        structured_output_mode=model["structured_output_mode"],
                        send_think_false=True,
                        keep_alive="5m",
                    )

                record_dict = record.to_dict()
                handle.write(json.dumps(record_dict, ensure_ascii=False) + "\n")
                handle.flush()

                processed_this_session += 1
                if record.parse_status != "ok":
                    errors_this_session += 1

                if (
                    model.get("snapshot_pinned")
                    and record.returned_model_id
                    and record.returned_model_id != model["model_id"]
                ):
                    fatal_error = (
                        "Pinned hosted model identity mismatch: "
                        f"requested {model['model_id']!r}, "
                        f"returned {record.returned_model_id!r}."
                    )
                    raise RuntimeError(fatal_error)

                if (
                    model["provider"] == "ollama"
                    and runtime_residency_metadata is None
                    and record.parse_status == "ok"
                ):
                    running = classifier.resolve_running_model(model["model_id"])
                    if running is not None:
                        size_bytes = running.get("size")
                        size_vram_bytes = running.get("size_vram")
                        vram_fraction = None
                        if (
                            isinstance(size_bytes, (int, float))
                            and isinstance(size_vram_bytes, (int, float))
                            and size_bytes > 0
                        ):
                            vram_fraction = float(size_vram_bytes) / float(size_bytes)
                        runtime_residency_metadata = {
                            "name": running.get("name"),
                            "model": running.get("model"),
                            "size_bytes": size_bytes,
                            "size_vram_bytes": size_vram_bytes,
                            "vram_fraction": vram_fraction,
                            "context_length": running.get("context_length"),
                            "expires_at": running.get("expires_at"),
                            "details": running.get("details"),
                        }

                status = "OK" if record.parse_status == "ok" else "ERROR"
                if position == 1 or position % 25 == 0 or position == len(pending):
                    total_completed = len(completed) + processed_this_session
                    print(
                        f"{status:5}  session={processed_this_session:,}/{len(pending):,}  "
                        f"overall={total_completed:,}/{len(schedule):,}  "
                        f"prompt={cell.prompt_id}  rep={cell.repetition}  "
                        f"id={cell.observation_id}"
                    )

                progress = {
                    "updated_at_utc": _utc_now(),
                    "model_key": args.model_key,
                    "schedule_sha256": schedule_hash,
                    "n_decisions": len(schedule),
                    "n_completed": len(completed) + processed_this_session,
                    "n_pending": len(schedule) - len(completed) - processed_this_session,
                    "session_id": session_id,
                    "session_processed": processed_this_session,
                    "session_errors": errors_this_session,
                }
                _write_json_atomic(run_dir / "progress.json", progress)

    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted by user. Completed records have been checkpointed.")
    except Exception as exc:
        if fatal_error is None:
            fatal_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if model["provider"] == "ollama":
            try:
                classifier.unload_model(model["model_id"])
            except Exception:
                pass
        if carbon_tracker is not None:
            carbon_summary = carbon_tracker.stop()

        elapsed = time.perf_counter() - session_started
        final_total_completed = len(completed) + processed_this_session
        session_manifest = {
            "session_id": session_id,
            "model_key": args.model_key,
            "provider": model["provider"],
            "started_at_utc": session_started_utc,
            "ended_at_utc": _utc_now(),
            "elapsed_seconds": elapsed,
            "status": (
                "interrupted"
                if interrupted
                else "failed"
                if fatal_error
                else "complete_or_staged"
            ),
            "fatal_error": fatal_error,
            "n_processed": processed_this_session,
            "n_errors": errors_this_session,
            "n_completed_before_session": len(completed),
            "n_completed_after_session": final_total_completed,
            "n_total_decisions": len(schedule),
            "max_decisions_requested": args.max_decisions,
            "timeout_seconds": _effective_timeout(model, args.timeout_seconds),
            "max_retries": _effective_retry_count(model, args.max_retries),
            "git": git_metadata,
            "platform_metadata": platform_metadata,
            "nvidia_metadata": nvidia_metadata,
            "requested_model_id": model["model_id"],
            "local_model_digest": local_digest,
            "local_model_registry_metadata": local_model_metadata,
            "ollama_version": ollama_version,
            "runtime_residency_metadata": runtime_residency_metadata,
            "carbon_tracking_enabled": bool(args.track_carbon),
            "carbon_summary": (
                asdict(carbon_summary)
                if carbon_summary is not None
                else None
            ),
        }
        _write_json_atomic(
            session_dir / "session_manifest.json",
            session_manifest,
        )

    if interrupted:
        raise SystemExit(130)

    total_completed = len(completed) + processed_this_session
    print(f"\nSession artefacts: {session_dir}")
    print(f"Session records: {processed_this_session:,}")
    print(f"Session errors: {errors_this_session:,}")
    print(f"Overall completed: {total_completed:,}/{len(schedule):,}")
    print(f"Overall pending: {len(schedule) - total_completed:,}")
    if carbon_summary is not None:
        print(
            "CodeCarbon session: "
            f"energy={carbon_summary.energy_consumed_kwh} kWh  "
            f"emissions={carbon_summary.emissions_kg_co2e} kgCO2e"
        )


if __name__ == "__main__":
    main()
