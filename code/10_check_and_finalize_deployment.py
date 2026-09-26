#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml

from reporting_validation_standards.experiment_specs import (
    load_model_config,
    load_prompt_config,
)
from reporting_validation_standards.production import (
    SCHEDULE_ALGORITHM_VERSION,
    make_stability_schedule,
    prompt_hashes,
    record_decision_key,
    schedule_sha256,
)
from reporting_validation_standards.sample_io import load_frozen_population


DEPLOYMENT_CONFIG = Path("config/deployment.yaml")
RUN_ROOT = Path("runs/deployment")
RESULTS_ROOT = Path("results/deployment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check and optionally finalize the frozen GB population deployment."
    )
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    return parser.parse_args()


def _load_config() -> dict[str, Any]:
    payload = yaml.safe_load(DEPLOYMENT_CONFIG.read_text(encoding="utf-8")) or {}
    deployment = payload.get("deployment")
    if not isinstance(deployment, dict):
        raise ValueError("Invalid deployment configuration.")
    return deployment


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Deployment records not found: {path}")
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
            raise RuntimeError(f"Non-object JSON at {path}:{line_number}.")
        rows.append(payload)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_run_files(run_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        hashes[path.relative_to(run_dir).as_posix()] = _sha256(path)
    return hashes


def _write_json_immutable(path: Path, payload: dict[str, Any]) -> str:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(
                f"Finalized artefact differs from existing file: {path}"
            )
        return "unchanged"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return "created"


def _copy_immutable(source: Path, destination: Path) -> str:
    content = source.read_text(encoding="utf-8")
    if destination.exists():
        if destination.read_text(encoding="utf-8") != content:
            raise RuntimeError(
                f"Finalized artefact differs from existing file: {destination}"
            )
        return "unchanged"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return "created"


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def main() -> None:
    args = parse_args()
    deployment = _load_config()
    deployment_id = str(deployment["deployment_id"])
    model_key = str(deployment["model_key"])
    prompt_id = str(deployment["prompt_id"])

    models = load_model_config()
    prompts = load_prompt_config()
    model = next(
        model for model in models["models"]
        if model["model_key"] == model_key
    )

    population, metadata = load_frozen_population(
        token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )
    ids = population[metadata["id_column"]].astype(str).tolist()

    schedule_algorithm = str(deployment["schedule_algorithm"])
    if schedule_algorithm != SCHEDULE_ALGORITHM_VERSION:
        raise RuntimeError("Deployment schedule algorithm mismatch.")
    schedule = make_stability_schedule(
        observation_ids=ids,
        prompt_ids=[prompt_id],
        repetitions=1,
        seed=int(deployment["schedule_seed"]),
    )
    expected_schedule_hash = schedule_sha256(schedule)
    expected_keys = {cell.key for cell in schedule}
    expected_prompt_hash = prompt_hashes(prompt_ids=[prompt_id])[prompt_id]

    run_dir = args.run_root / deployment_id
    records_path = run_dir / "inference_records.jsonl"
    manifest_path = run_dir / "experiment_manifest.json"
    records = _read_jsonl(records_path)
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None
    )

    observed_keys = [record_decision_key(record) for record in records]
    counts = Counter(observed_keys)
    duplicates = [key for key, count in counts.items() if count > 1]
    observed_set = set(observed_keys)
    missing = expected_keys - observed_set
    unexpected = observed_set - expected_keys

    model_mismatches = sum(
        record.get("model_key") != model_key for record in records
    )
    provider_mismatches = sum(
        record.get("provider") != model["provider"] for record in records
    )
    requested_model_mismatches = sum(
        record.get("requested_model_id") != model["model_id"]
        for record in records
    )
    prompt_mismatches = sum(
        record.get("prompt_id") != prompt_id
        or record.get("prompt_spec_sha256") != expected_prompt_hash
        for record in records
    )
    repetition_mismatches = sum(
        int(record.get("repetition") or 0) != 1 for record in records
    )

    manifest_issues: list[str] = []
    if manifest is None:
        manifest_issues.append("experiment_manifest.json is missing")
    else:
        frozen = manifest.get("frozen_spec") or {}
        expected = {
            "deployment_id": deployment_id,
            "model_key": model_key,
            "requested_model_id": model["model_id"],
            "prompt_id": prompt_id,
            "prompt_spec_sha256": expected_prompt_hash,
            "dataset_revision": metadata["dataset_revision"],
            "population_n": len(population),
            "population_id_sha256": metadata["population_id_sha256"],
            "repetitions": 1,
            "schedule_seed": int(deployment["schedule_seed"]),
            "schedule_algorithm": schedule_algorithm,
            "schedule_sha256": expected_schedule_hash,
            "n_decisions": len(schedule),
        }
        for field, value in expected.items():
            if frozen.get(field) != value:
                manifest_issues.append(
                    f"manifest frozen_spec.{field}={frozen.get(field)!r}, "
                    f"expected {value!r}"
                )

    structural_failures = {
        "duplicate_cells": len(duplicates),
        "unexpected_cells": len(unexpected),
        "model_key_mismatches": model_mismatches,
        "provider_mismatches": provider_mismatches,
        "requested_model_mismatches": requested_model_mismatches,
        "prompt_mismatches": prompt_mismatches,
        "repetition_mismatches": repetition_mismatches,
        "manifest_issues": len(manifest_issues),
    }
    structural_pass = all(value == 0 for value in structural_failures.values())
    completion_pass = (
        structural_pass
        and len(missing) == 0
        and len(records) == len(schedule)
    )

    parse_status = Counter(
        str(record.get("parse_status")) for record in records
    )
    returned_ids = Counter(
        str(record["returned_model_id"])
        for record in records
        if record.get("returned_model_id")
    )
    retries = [int(record.get("retry_count") or 0) for record in records]
    latencies = [
        float(record["latency_seconds"])
        for record in records
        if isinstance(record.get("latency_seconds"), (int, float))
    ]
    costs = [
        float(record["cost_usd"])
        for record in records
        if isinstance(record.get("cost_usd"), (int, float))
    ]

    summary = {
        "schema_version": 1,
        "deployment_id": deployment_id,
        "model_key": model_key,
        "provider": model["provider"],
        "requested_model_id": model["model_id"],
        "prompt_id": prompt_id,
        "dataset_revision": metadata["dataset_revision"],
        "population_n": len(population),
        "population_id_sha256": metadata["population_id_sha256"],
        "schedule_sha256": expected_schedule_hash,
        "expected_decisions": len(schedule),
        "observed_records": len(records),
        "unique_decision_cells": len(observed_set),
        "missing_cells": len(missing),
        "duplicate_cells": len(duplicates),
        "unexpected_cells": len(unexpected),
        "structural_qa_pass": structural_pass,
        "completion_qa_pass": completion_pass,
        "structural_failures": structural_failures,
        "manifest_issues": manifest_issues,
        "parse_status": dict(parse_status),
        "records_with_retries": sum(value > 0 for value in retries),
        "total_adapter_retries": sum(retries),
        "returned_model_ids": dict(returned_ids),
        "latency_seconds": (
            {
                "mean": mean(latencies),
                "median": median(latencies),
                "p95": _percentile(latencies, 0.95),
                "max": max(latencies),
            }
            if latencies
            else None
        ),
        "recorded_cost_total_usd": sum(costs) if costs else None,
    }

    print(f"Deployment QA: {deployment_id}")
    print(f"Expected decisions: {len(schedule):,}")
    print(f"Observed records: {len(records):,}")
    print(f"Missing cells: {len(missing):,}")
    print(f"Duplicate cells: {len(duplicates):,}")
    print(f"Unexpected cells: {len(unexpected):,}")
    print(f"Structural QA: {'PASS' if structural_pass else 'FAIL'}")
    print(
        f"Completion QA: "
        f"{'PASS' if completion_pass else 'INCOMPLETE/FAIL'}"
    )
    print(f"parse_status: {dict(parse_status)}")
    print(f"records with retries: {sum(value > 0 for value in retries):,}")
    print(f"total adapter retries: {sum(retries):,}")
    print(f"returned model IDs: {dict(returned_ids)}")
    if latencies:
        print(
            "latency seconds: "
            f"mean={mean(latencies):.3f}  "
            f"median={median(latencies):.3f}  "
            f"p95={_percentile(latencies, 0.95):.3f}  "
            f"max={max(latencies):.3f}"
        )
    if costs:
        print(f"recorded cost total (USD): {sum(costs):.6f}")

    if not args.finalize:
        return

    if not completion_pass:
        raise RuntimeError(
            "Refusing finalization because structural/completion QA did not pass."
        )

    raw_hashes = _hash_run_files(run_dir)
    inference_hash = raw_hashes.get("inference_records.jsonl")
    if not inference_hash:
        raise RuntimeError("Raw inference file is missing from hash inventory.")

    final_summary = dict(summary)
    final_summary["raw_inference_records_sha256"] = inference_hash
    final_summary["raw_file_count"] = len(raw_hashes)

    results_dir = args.results_root / deployment_id
    results_dir.mkdir(parents=True, exist_ok=True)

    statuses = [
        (
            "qa_summary.json",
            _write_json_immutable(
                results_dir / "qa_summary.json",
                final_summary,
            ),
        ),
        (
            "raw_files_sha256.json",
            _write_json_immutable(
                results_dir / "raw_files_sha256.json",
                {
                    "schema_version": 1,
                    "deployment_id": deployment_id,
                    "files": raw_hashes,
                },
            ),
        ),
        (
            "experiment_manifest.json",
            _copy_immutable(
                manifest_path,
                results_dir / "experiment_manifest.json",
            ),
        ),
    ]

    for source in sorted(run_dir.glob("sessions/*/session_manifest.json")):
        session_name = source.parent.name
        destination = (
            results_dir
            / "sessions"
            / session_name
            / "session_manifest.json"
        )
        statuses.append(
            (
                f"sessions/{session_name}/session_manifest.json",
                _copy_immutable(source, destination),
            )
        )

    print("\nFinalization")
    print(f"Raw files hashed: {len(raw_hashes):,}")
    print(f"inference_records.jsonl SHA-256: {inference_hash}")
    print(f"Finalized provenance: {results_dir}")
    for relative, status in statuses:
        print(f"{status:9}  {relative}")


if __name__ == "__main__":
    main()
