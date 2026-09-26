#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from reporting_validation_standards.experiment_specs import (
    load_model_config,
    load_prompt_config,
    render_prompt,
)
from reporting_validation_standards.local_compute import collect_platform_metadata
from reporting_validation_standards.openai_adapter import OpenAIClassifier
from reporting_validation_standards.production import (
    SCHEDULE_ALGORITHM_VERSION,
    make_stability_schedule,
    prompt_hashes,
    schedule_sha256,
    validate_existing_records,
)
from reporting_validation_standards.sample_io import (
    load_frozen_population,
    select_input_text,
)


DEPLOYMENT_CONFIG = Path("config/deployment.yaml")
DEPLOYMENT_ROOT = Path("runs/deployment")
TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen one-pass P0 classifier deployment over the complete "
            "GB silver-reference test population. Append-only and resumable."
        )
    )
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=None,
        help="Process at most this many currently pending population cases.",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_deployment_config() -> dict[str, Any]:
    payload = yaml.safe_load(DEPLOYMENT_CONFIG.read_text(encoding="utf-8")) or {}
    deployment = payload.get("deployment")
    if not isinstance(deployment, dict):
        raise ValueError("config/deployment.yaml must contain a deployment mapping.")
    required = {
        "deployment_id",
        "model_key",
        "prompt_id",
        "input_mode",
        "repetitions",
        "schedule_seed",
        "schedule_algorithm",
        "failure_policy",
    }
    missing = sorted(required - set(deployment))
    if missing:
        raise ValueError(f"Deployment config is missing fields: {missing}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
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
                f"Malformed JSONL at {path}:{line_number}."
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Expected JSON object at {path}:{line_number}."
            )
        rows.append(payload)
    return rows


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except Exception:
            return None
        return result.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "working_tree_dirty": bool(status) if status is not None else None,
    }


def main() -> None:
    args = parse_args()
    if args.max_decisions is not None and args.max_decisions <= 0:
        raise ValueError("--max-decisions must be positive.")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive.")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be non-negative.")

    deployment_payload = _load_deployment_config()
    deployment = deployment_payload["deployment"]
    if int(deployment["repetitions"]) != 1:
        raise RuntimeError("Full-population deployment must use exactly one pass.")

    model_config = load_model_config()
    prompt_config = load_prompt_config()
    model_lookup = {
        model["model_key"]: model
        for model in model_config["models"]
    }
    model_key = str(deployment["model_key"])
    if model_key not in model_lookup:
        raise KeyError(f"Unknown deployment model key: {model_key}")
    model = model_lookup[model_key]
    if model["provider"] != "openai":
        raise RuntimeError(
            "The frozen deployment is specified for the OpenAI GPT-5.4 Mini adapter."
        )
    if not model.get("snapshot_pinned"):
        raise RuntimeError("Full-population deployment requires a pinned model snapshot.")

    prompt_id = str(deployment["prompt_id"])
    configured_prompt_ids = {
        str(prompt["prompt_id"]) for prompt in prompt_config["prompts"]
    }
    if prompt_id not in configured_prompt_ids:
        raise KeyError(f"Unknown deployment prompt: {prompt_id}")

    schedule_algorithm = str(deployment["schedule_algorithm"])
    if schedule_algorithm != SCHEDULE_ALGORITHM_VERSION:
        raise RuntimeError(
            "Deployment schedule algorithm does not match the implemented version: "
            f"{schedule_algorithm!r} != {SCHEDULE_ALGORITHM_VERSION!r}."
        )

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    population, population_metadata = load_frozen_population(token=token)
    id_column = population_metadata["id_column"]
    population[id_column] = population[id_column].astype(str)
    population_by_id = {
        str(row[id_column]): row
        for _, row in population.iterrows()
    }

    observation_ids = population[id_column].astype(str).tolist()
    schedule = make_stability_schedule(
        observation_ids=observation_ids,
        prompt_ids=[prompt_id],
        repetitions=1,
        seed=int(deployment["schedule_seed"]),
    )
    schedule_hash = schedule_sha256(schedule)
    expected_keys = {cell.key for cell in schedule}
    prompt_hash = prompt_hashes(prompt_ids=[prompt_id])[prompt_id]

    deployment_id = str(deployment["deployment_id"])
    run_dir = DEPLOYMENT_ROOT / deployment_id
    records_path = run_dir / "inference_records.jsonl"
    manifest_path = run_dir / "experiment_manifest.json"

    frozen_spec = {
        "schema_version": 1,
        "purpose": "full_population_prevalence_deployment",
        "deployment_id": deployment_id,
        "model_key": model_key,
        "provider": model["provider"],
        "requested_model_id": model["model_id"],
        "snapshot_pinned": bool(model.get("snapshot_pinned")),
        "prompt_id": prompt_id,
        "prompt_spec_sha256": prompt_hash,
        "input_mode": str(deployment["input_mode"]),
        "temperature": TEMPERATURE,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "reasoning_setting": model["reasoning_setting"],
        "structured_output_mode": model["structured_output_mode"],
        "dataset_revision": population_metadata["dataset_revision"],
        "population_n": len(population),
        "population_labels_n": int(population_metadata["population_labels_n"]),
        "population_id_sha256": population_metadata["population_id_sha256"],
        "repetitions": 1,
        "schedule_seed": int(deployment["schedule_seed"]),
        "schedule_algorithm": schedule_algorithm,
        "schedule_sha256": schedule_hash,
        "n_decisions": len(schedule),
        "failure_policy": str(deployment["failure_policy"]),
        "timeout_seconds": float(args.timeout_seconds),
        "max_retries": int(args.max_retries),
    }

    existing_records = _read_jsonl(records_path)
    completed = validate_existing_records(
        existing_records,
        expected_model_key=model_key,
        expected_schedule_keys=expected_keys,
    )
    pending = [cell for cell in schedule if cell.key not in completed]

    print(f"Deployment: {deployment_id}")
    print(f"Model: {model_key} ({model['model_id']})")
    print(f"Prompt: {prompt_id}")
    print(f"Frozen population: {len(population):,}")
    print(f"Population ID SHA-256: {population_metadata['population_id_sha256']}")
    print(f"Schedule SHA-256: {schedule_hash}")
    print(f"Completed: {len(completed):,}/{len(schedule):,}")
    print(f"Pending: {len(pending):,}")

    if args.plan_only:
        print("Plan-only mode: no inference requests were sent.")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        existing_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        if existing_manifest.get("frozen_spec") != frozen_spec:
            raise RuntimeError(
                "Existing deployment manifest differs from the current frozen "
                "specification. Refusing to resume."
            )
    else:
        _write_json_atomic(
            manifest_path,
            {
                "created_at_utc": _utc_now(),
                "frozen_spec": frozen_spec,
            },
        )

    if not pending:
        print("Full-population deployment is already complete.")
        return

    if args.max_decisions is not None:
        pending = pending[: args.max_decisions]

    classifier = OpenAIClassifier(
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )

    session_id = f"{deployment_id}_{_timestamp()}"
    session_dir = run_dir / "sessions" / session_id
    session_dir.mkdir(parents=True, exist_ok=False)

    session_started = time.perf_counter()
    session_started_utc = _utc_now()
    processed = 0
    errors = 0
    interrupted = False
    fatal_error: str | None = None
    total_cost = 0.0

    try:
        with records_path.open("a", encoding="utf-8", buffering=1) as handle:
            for position, cell in enumerate(pending, start=1):
                row = population_by_id[cell.observation_id]
                input_text = select_input_text(
                    row,
                    input_mode=str(deployment["input_mode"]),
                )
                rendered = render_prompt(
                    prompt_id=prompt_id,
                    text=input_text,
                )

                record = classifier.classify(
                    run_id=session_id,
                    observation_id=cell.observation_id,
                    input_text=input_text,
                    model_key=model_key,
                    model_id=model["model_id"],
                    prompt=rendered,
                    official_labels=prompt_config["official_labels"],
                    repetition=1,
                    reasoning_setting=model["reasoning_setting"],
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    structured_output_mode=model["structured_output_mode"],
                )

                handle.write(
                    json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
                )
                handle.flush()
                processed += 1
                errors += int(record.parse_status != "ok")
                if record.cost_usd is not None:
                    total_cost += float(record.cost_usd)

                if (
                    record.returned_model_id
                    and record.returned_model_id != model["model_id"]
                ):
                    fatal_error = (
                        "Pinned model identity mismatch: "
                        f"requested {model['model_id']!r}, "
                        f"returned {record.returned_model_id!r}."
                    )
                    raise RuntimeError(fatal_error)

                if position == 1 or position % 100 == 0 or position == len(pending):
                    overall = len(completed) + processed
                    status = "OK" if record.parse_status == "ok" else "ERROR"
                    print(
                        f"{status:5}  session={processed:,}/{len(pending):,}  "
                        f"overall={overall:,}/{len(schedule):,}  "
                        f"errors={errors:,}  id={cell.observation_id}"
                    )

                _write_json_atomic(
                    run_dir / "progress.json",
                    {
                        "updated_at_utc": _utc_now(),
                        "deployment_id": deployment_id,
                        "model_key": model_key,
                        "schedule_sha256": schedule_hash,
                        "n_decisions": len(schedule),
                        "n_completed": len(completed) + processed,
                        "n_pending": len(schedule) - len(completed) - processed,
                        "session_id": session_id,
                        "session_processed": processed,
                        "session_errors": errors,
                        "session_recorded_cost_usd": total_cost,
                    },
                )

    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted by user. Completed deployment records are checkpointed.")
    except Exception as exc:
        if fatal_error is None:
            fatal_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        elapsed = time.perf_counter() - session_started
        _write_json_atomic(
            session_dir / "session_manifest.json",
            {
                "session_id": session_id,
                "deployment_id": deployment_id,
                "model_key": model_key,
                "provider": model["provider"],
                "requested_model_id": model["model_id"],
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
                "n_processed": processed,
                "n_errors": errors,
                "n_completed_before_session": len(completed),
                "n_completed_after_session": len(completed) + processed,
                "n_total_decisions": len(schedule),
                "max_decisions_requested": args.max_decisions,
                "timeout_seconds": args.timeout_seconds,
                "max_retries": args.max_retries,
                "recorded_cost_usd": total_cost,
                "git": _git_metadata(),
                "platform_metadata": collect_platform_metadata(),
            },
        )

    if interrupted:
        raise SystemExit(130)

    overall = len(completed) + processed
    print(f"\nSession artefacts: {session_dir}")
    print(f"Session records: {processed:,}")
    print(f"Session errors: {errors:,}")
    print(f"Session recorded cost (USD): {total_cost:.6f}")
    print(f"Overall completed: {overall:,}/{len(schedule):,}")
    print(f"Overall pending: {len(schedule) - overall:,}")


if __name__ == "__main__":
    main()
