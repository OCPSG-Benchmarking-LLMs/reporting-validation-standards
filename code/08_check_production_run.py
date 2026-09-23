#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

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
from reporting_validation_standards.sample_io import load_frozen_sample


PRODUCTION_ROOT = Path("runs/production/stability")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize one completed or partial production stability run."
    )
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--root", type=Path, default=PRODUCTION_ROOT)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Production records not found: {path}")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Malformed JSONL at line {line_number} in {path}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Non-object JSON record at line {line_number} in {path}")
        records.append(payload)
    return records


def pct(values: list[float], q: float) -> float | None:
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


def fmt(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()

    model_config = load_model_config()
    prompt_config = load_prompt_config()
    model_lookup = {model["model_key"]: model for model in model_config["models"]}
    if args.model_key not in model_lookup:
        raise KeyError(f"Unknown model key: {args.model_key}")
    model = model_lookup[args.model_key]

    policy = model_config["experiment_policy"]
    repetitions = int(policy["repetitions"])
    schedule_seed = int(policy["schedule_seed"])
    schedule_algorithm = str(policy["schedule_algorithm"])
    if schedule_algorithm != SCHEDULE_ALGORITHM_VERSION:
        raise RuntimeError(
            f"Configured schedule algorithm {schedule_algorithm!r} does not match "
            f"implementation {SCHEDULE_ALGORITHM_VERSION!r}."
        )

    sample, sample_metadata = load_frozen_sample()
    id_column = sample_metadata["id_column"]
    observation_ids = sample[id_column].astype(str).tolist()
    prompt_ids = [prompt["prompt_id"] for prompt in prompt_config["prompts"]]

    schedule = make_stability_schedule(
        observation_ids=observation_ids,
        prompt_ids=prompt_ids,
        repetitions=repetitions,
        seed=schedule_seed,
    )
    expected_schedule_hash = schedule_sha256(schedule)
    expected_keys = {cell.key for cell in schedule}
    expected_prompt_hashes = prompt_hashes(prompt_ids=prompt_ids)

    run_dir = args.root / args.model_key
    records_path = run_dir / "inference_records.jsonl"
    manifest_path = run_dir / "experiment_manifest.json"

    records = read_jsonl(records_path)
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None
    )

    observed_keys = [record_decision_key(record) for record in records]
    key_counts = Counter(observed_keys)
    duplicate_keys = sorted(key for key, count in key_counts.items() if count > 1)
    observed_key_set = set(observed_keys)
    unexpected_keys = sorted(observed_key_set - expected_keys)
    missing_keys = expected_keys - observed_key_set

    model_key_mismatches = sum(record.get("model_key") != args.model_key for record in records)
    provider_mismatches = sum(record.get("provider") != model["provider"] for record in records)
    requested_model_mismatches = sum(
        record.get("requested_model_id") != model["model_id"] for record in records
    )

    prompt_hash_mismatches = 0
    for record in records:
        prompt_id = record.get("prompt_id")
        expected_hash = expected_prompt_hashes.get(str(prompt_id))
        if expected_hash is None or record.get("prompt_spec_sha256") != expected_hash:
            prompt_hash_mismatches += 1

    cell_counts = Counter(
        (str(record.get("prompt_id")), int(record.get("repetition")))
        for record in records
        if record.get("repetition") is not None
    )

    parse_counts = Counter(str(record.get("parse_status")) for record in records)
    retry_counts = [
        int(record.get("retry_count") or 0)
        for record in records
        if record.get("retry_count") is not None
    ]
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
    total_tokens = [
        int(record["total_tokens"])
        for record in records
        if isinstance(record.get("total_tokens"), int)
    ]

    returned_model_ids = Counter(
        str(record["returned_model_id"])
        for record in records
        if record.get("returned_model_id")
    )
    local_digests = Counter(
        str(record["local_model_digest"])
        for record in records
        if record.get("local_model_digest")
    )
    system_fingerprints = Counter(
        str((record.get("provider_metadata") or {}).get("system_fingerprint"))
        for record in records
        if (record.get("provider_metadata") or {}).get("system_fingerprint")
    )

    retry_event_total = sum(retry_counts)
    records_with_retries = sum(value > 0 for value in retry_counts)

    manifest_issues: list[str] = []
    if manifest is None:
        manifest_issues.append("experiment_manifest.json is missing")
    else:
        frozen = manifest.get("frozen_spec") or {}
        checks = {
            "model_key": args.model_key,
            "provider": model["provider"],
            "requested_model_id": model["model_id"],
            "sample_id_sha256": sample_metadata["sample_id_sha256"],
            "dataset_revision": sample_metadata["dataset_revision"],
            "repetitions": repetitions,
            "schedule_seed": schedule_seed,
            "schedule_algorithm": schedule_algorithm,
            "schedule_sha256": expected_schedule_hash,
            "n_decisions": len(schedule),
        }
        for field, expected in checks.items():
            if frozen.get(field) != expected:
                manifest_issues.append(
                    f"manifest frozen_spec.{field}={frozen.get(field)!r}, expected {expected!r}"
                )

        if frozen.get("prompt_spec_sha256") != expected_prompt_hashes:
            manifest_issues.append(
                "manifest prompt_spec_sha256 does not match current frozen prompts"
            )

        expected_digest = model.get("expected_local_digest")
        if expected_digest and frozen.get("expected_local_digest") != expected_digest:
            manifest_issues.append(
                "manifest expected_local_digest does not match model config"
            )

    structural_failures = {
        "duplicate_cells": len(duplicate_keys),
        "unexpected_cells": len(unexpected_keys),
        "model_key_mismatches": model_key_mismatches,
        "provider_mismatches": provider_mismatches,
        "requested_model_mismatches": requested_model_mismatches,
        "prompt_hash_mismatches": prompt_hash_mismatches,
        "manifest_issues": len(manifest_issues),
    }
    structurally_valid = all(value == 0 for value in structural_failures.values())
    complete = (
        len(missing_keys) == 0
        and len(records) == len(schedule)
        and structurally_valid
    )

    print(f"Production QA: {args.model_key}")
    print(f"Provider: {model['provider']}")
    print(f"Expected decisions: {len(schedule):,}")
    print(f"Observed records: {len(records):,}")
    print(f"Unique decision cells: {len(observed_key_set):,}")
    print(f"Missing cells: {len(missing_keys):,}")
    print(f"Duplicate cells: {len(duplicate_keys):,}")
    print(f"Unexpected cells: {len(unexpected_keys):,}")
    print(f"Schedule SHA-256: {expected_schedule_hash}")
    print(f"Structural QA: {'PASS' if structurally_valid else 'FAIL'}")
    print(f"Completion QA: {'PASS' if complete else 'INCOMPLETE/FAIL'}")

    print("\nPrompt × repetition coverage")
    for prompt_id in prompt_ids:
        for repetition in range(1, repetitions + 1):
            print(
                f"  {prompt_id}  rep={repetition}: "
                f"{cell_counts.get((prompt_id, repetition), 0):,}/1,000"
            )

    print("\nOutcome diagnostics")
    print(f"  parse_status: {dict(parse_counts)}")
    print(f"  records with retries: {records_with_retries:,}")
    print(f"  total adapter retries: {retry_event_total:,}")
    print(f"  returned model IDs: {dict(returned_model_ids)}")
    if system_fingerprints:
        print(f"  system fingerprints: {dict(system_fingerprints)}")
    if local_digests:
        print(f"  local model digests: {dict(local_digests)}")

    if latencies:
        print(
            "  latency seconds: "
            f"mean={fmt(mean(latencies))}  "
            f"median={fmt(median(latencies))}  "
            f"p95={fmt(pct(latencies, 0.95))}  "
            f"max={fmt(max(latencies))}"
        )
    if total_tokens:
        print(
            "  total tokens: "
            f"mean={fmt(mean(total_tokens), 1)}  "
            f"median={fmt(median(total_tokens), 1)}  "
            f"max={max(total_tokens):,}"
        )
    if costs:
        print(f"  recorded cost total (USD): {sum(costs):.6f}")

    if manifest_issues:
        print("\nManifest issues")
        for issue in manifest_issues:
            print(f"  - {issue}")

    if duplicate_keys:
        print("\nFirst duplicate cells")
        for key in duplicate_keys[:5]:
            print(f"  {key}")
    if unexpected_keys:
        print("\nFirst unexpected cells")
        for key in unexpected_keys[:5]:
            print(f"  {key}")

    print("\nInterpretation")
    if complete:
        if parse_counts.get("error", 0):
            print(
                "  The scheduled experiment is structurally complete. Observed "
                "inference/parse errors are retained as outcomes rather than silently rerun."
            )
        else:
            print("  The scheduled experiment is complete and structurally valid.")
    elif structurally_valid:
        print(
            "  The existing records are structurally valid but the scheduled experiment "
            "is not yet complete."
        )
    else:
        print(
            "  Structural QA failed. Do not continue production until the reported "
            "integrity issue is understood."
        )


if __name__ == "__main__":
    main()
