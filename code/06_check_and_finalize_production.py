#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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
RESULTS_ROOT = Path("results/production")
FINALIZATION_SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check one production stability run and, when explicitly requested, "
            "finalize its lightweight provenance artefacts."
        )
    )
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--root", type=Path, default=PRODUCTION_ROOT)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument(
        "--finalize",
        action="store_true",
        help=(
            "Finalize only after structural and completion QA pass. Raw runs remain "
            "under gitignored runs/; compact provenance is written under results/production/."
        ),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Production records not found: {path}")

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
                f"Malformed JSONL at line {line_number} in {path}"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Non-object JSON record at line {line_number} in {path}"
            )
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_run_files(run_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        hashes[path.relative_to(run_dir).as_posix()] = sha256_file(path)
    return hashes


def write_json_immutable(path: Path, payload: dict[str, Any]) -> str:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != rendered:
            raise RuntimeError(
                f"Finalized artefact already exists with different content: {path}"
            )
        return "unchanged"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return "created"


def copy_text_immutable(source: Path, destination: Path) -> str:
    if not source.exists():
        raise FileNotFoundError(f"Required source artefact is missing: {source}")

    content = source.read_text(encoding="utf-8")
    if destination.exists():
        existing = destination.read_text(encoding="utf-8")
        if existing != content:
            raise RuntimeError(
                f"Finalized artefact already exists with different content: {destination}"
            )
        return "unchanged"

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return "created"


def build_qa_summary(
    *,
    args: argparse.Namespace,
    model: dict[str, Any],
    model_config: dict[str, Any],
    prompt_config: dict[str, Any],
    sample,
    sample_metadata: dict[str, Any],
    records: list[dict[str, Any]],
    manifest: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = model_config["experiment_policy"]
    repetitions = int(policy["repetitions"])
    schedule_seed = int(policy["schedule_seed"])
    schedule_algorithm = str(policy["schedule_algorithm"])
    if schedule_algorithm != SCHEDULE_ALGORITHM_VERSION:
        raise RuntimeError(
            f"Configured schedule algorithm {schedule_algorithm!r} does not match "
            f"implementation {SCHEDULE_ALGORITHM_VERSION!r}."
        )

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

    observed_keys = [record_decision_key(record) for record in records]
    key_counts = Counter(observed_keys)
    duplicate_keys = sorted(
        key for key, count in key_counts.items() if count > 1
    )
    observed_key_set = set(observed_keys)
    unexpected_keys = sorted(observed_key_set - expected_keys)
    missing_keys = expected_keys - observed_key_set

    model_key_mismatches = sum(
        record.get("model_key") != args.model_key for record in records
    )
    provider_mismatches = sum(
        record.get("provider") != model["provider"] for record in records
    )
    requested_model_mismatches = sum(
        record.get("requested_model_id") != model["model_id"]
        for record in records
    )

    prompt_hash_mismatches = 0
    for record in records:
        prompt_id = record.get("prompt_id")
        expected_hash = expected_prompt_hashes.get(str(prompt_id))
        if (
            expected_hash is None
            or record.get("prompt_spec_sha256") != expected_hash
        ):
            prompt_hash_mismatches += 1

    cell_counts = Counter(
        (str(record.get("prompt_id")), int(record.get("repetition")))
        for record in records
        if record.get("repetition") is not None
    )
    coverage = {
        prompt_id: {
            str(repetition): cell_counts.get((prompt_id, repetition), 0)
            for repetition in range(1, repetitions + 1)
        }
        for prompt_id in prompt_ids
    }

    parse_counts = Counter(
        str(record.get("parse_status")) for record in records
    )
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
                    f"manifest frozen_spec.{field}={frozen.get(field)!r}, "
                    f"expected {expected!r}"
                )

        if frozen.get("prompt_spec_sha256") != expected_prompt_hashes:
            manifest_issues.append(
                "manifest prompt_spec_sha256 does not match current frozen prompts"
            )

        expected_digest = model.get("expected_local_digest")
        if (
            expected_digest
            and frozen.get("expected_local_digest") != expected_digest
        ):
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
    structurally_valid = all(
        value == 0 for value in structural_failures.values()
    )
    complete = (
        len(missing_keys) == 0
        and len(records) == len(schedule)
        and structurally_valid
    )

    latency_summary = None
    if latencies:
        latency_summary = {
            "mean": mean(latencies),
            "median": median(latencies),
            "p95": pct(latencies, 0.95),
            "max": max(latencies),
        }

    token_summary = None
    if total_tokens:
        token_summary = {
            "mean": mean(total_tokens),
            "median": median(total_tokens),
            "max": max(total_tokens),
        }

    summary = {
        "schema_version": FINALIZATION_SCHEMA_VERSION,
        "model_key": args.model_key,
        "provider": model["provider"],
        "requested_model_id": model["model_id"],
        "schedule_sha256": expected_schedule_hash,
        "sample_id_sha256": sample_metadata["sample_id_sha256"],
        "dataset_revision": sample_metadata["dataset_revision"],
        "expected_decisions": len(schedule),
        "observed_records": len(records),
        "unique_decision_cells": len(observed_key_set),
        "missing_cells": len(missing_keys),
        "duplicate_cells": len(duplicate_keys),
        "unexpected_cells": len(unexpected_keys),
        "structural_qa_pass": structurally_valid,
        "completion_qa_pass": complete,
        "structural_failures": structural_failures,
        "manifest_issues": manifest_issues,
        "prompt_repetition_coverage": coverage,
        "parse_status": dict(parse_counts),
        "records_with_retries": sum(value > 0 for value in retry_counts),
        "total_adapter_retries": sum(retry_counts),
        "returned_model_ids": dict(returned_model_ids),
        "system_fingerprints": dict(system_fingerprints),
        "local_model_digests": dict(local_digests),
        "latency_seconds": latency_summary,
        "total_tokens": token_summary,
        "recorded_cost_total_usd": sum(costs) if costs else None,
    }

    diagnostics = {
        "prompt_ids": prompt_ids,
        "repetitions": repetitions,
        "coverage": coverage,
        "duplicate_keys": duplicate_keys,
        "unexpected_keys": unexpected_keys,
        "missing_keys": missing_keys,
        "manifest_issues": manifest_issues,
    }
    return summary, diagnostics


def print_summary(
    summary: dict[str, Any],
    diagnostics: dict[str, Any],
) -> None:
    print(f"Production QA: {summary['model_key']}")
    print(f"Provider: {summary['provider']}")
    print(f"Expected decisions: {summary['expected_decisions']:,}")
    print(f"Observed records: {summary['observed_records']:,}")
    print(f"Unique decision cells: {summary['unique_decision_cells']:,}")
    print(f"Missing cells: {summary['missing_cells']:,}")
    print(f"Duplicate cells: {summary['duplicate_cells']:,}")
    print(f"Unexpected cells: {summary['unexpected_cells']:,}")
    print(f"Schedule SHA-256: {summary['schedule_sha256']}")
    print(
        "Structural QA: "
        f"{'PASS' if summary['structural_qa_pass'] else 'FAIL'}"
    )
    print(
        "Completion QA: "
        f"{'PASS' if summary['completion_qa_pass'] else 'INCOMPLETE/FAIL'}"
    )

    print("\nPrompt × repetition coverage")
    for prompt_id in diagnostics["prompt_ids"]:
        for repetition in range(1, diagnostics["repetitions"] + 1):
            count = diagnostics["coverage"][prompt_id][str(repetition)]
            print(
                f"  {prompt_id}  rep={repetition}: "
                f"{count:,}/1,000"
            )

    print("\nOutcome diagnostics")
    print(f"  parse_status: {summary['parse_status']}")
    print(
        f"  records with retries: {summary['records_with_retries']:,}"
    )
    print(
        f"  total adapter retries: {summary['total_adapter_retries']:,}"
    )
    print(f"  returned model IDs: {summary['returned_model_ids']}")
    if summary["system_fingerprints"]:
        print(
            f"  system fingerprints: {summary['system_fingerprints']}"
        )
    if summary["local_model_digests"]:
        print(
            f"  local model digests: {summary['local_model_digests']}"
        )

    latency = summary["latency_seconds"]
    if latency:
        print(
            "  latency seconds: "
            f"mean={fmt(latency['mean'])}  "
            f"median={fmt(latency['median'])}  "
            f"p95={fmt(latency['p95'])}  "
            f"max={fmt(latency['max'])}"
        )

    tokens = summary["total_tokens"]
    if tokens:
        print(
            "  total tokens: "
            f"mean={fmt(tokens['mean'], 1)}  "
            f"median={fmt(tokens['median'], 1)}  "
            f"max={tokens['max']:,}"
        )

    if summary["recorded_cost_total_usd"] is not None:
        print(
            "  recorded cost total (USD): "
            f"{summary['recorded_cost_total_usd']:.6f}"
        )

    if diagnostics["manifest_issues"]:
        print("\nManifest issues")
        for issue in diagnostics["manifest_issues"]:
            print(f"  - {issue}")

    if diagnostics["duplicate_keys"]:
        print("\nFirst duplicate cells")
        for key in diagnostics["duplicate_keys"][:5]:
            print(f"  {key}")

    if diagnostics["unexpected_keys"]:
        print("\nFirst unexpected cells")
        for key in diagnostics["unexpected_keys"][:5]:
            print(f"  {key}")

    print("\nInterpretation")
    if summary["completion_qa_pass"]:
        if summary["parse_status"].get("error", 0):
            print(
                "  The scheduled demonstration is structurally complete. Observed "
                "inference/parse errors are retained as outcomes rather than "
                "silently rerun."
            )
        else:
            print(
                "  The scheduled demonstration is complete and structurally valid."
            )
    elif summary["structural_qa_pass"]:
        print(
            "  The existing records are structurally valid but the scheduled "
            "demonstration is not yet complete."
        )
    else:
        print(
            "  Structural QA failed. Do not continue production until the "
            "reported integrity issue is understood."
        )


def finalize_run(
    *,
    run_dir: Path,
    results_dir: Path,
    summary: dict[str, Any],
) -> None:
    if not summary["structural_qa_pass"] or not summary["completion_qa_pass"]:
        raise RuntimeError(
            "Refusing finalization because structural/completion QA did not pass."
        )

    raw_hashes = hash_run_files(run_dir)
    inference_rel = "inference_records.jsonl"
    if inference_rel not in raw_hashes:
        raise RuntimeError(
            "inference_records.jsonl is missing from the raw-run hash inventory."
        )

    final_summary = dict(summary)
    final_summary["raw_inference_records_sha256"] = raw_hashes[inference_rel]
    final_summary["raw_file_count"] = len(raw_hashes)

    results_dir.mkdir(parents=True, exist_ok=True)

    statuses: list[tuple[str, str]] = []
    statuses.append(
        (
            "qa_summary.json",
            write_json_immutable(
                results_dir / "qa_summary.json",
                final_summary,
            ),
        )
    )
    statuses.append(
        (
            "raw_files_sha256.json",
            write_json_immutable(
                results_dir / "raw_files_sha256.json",
                {
                    "schema_version": FINALIZATION_SCHEMA_VERSION,
                    "model_key": summary["model_key"],
                    "files": raw_hashes,
                },
            ),
        )
    )
    statuses.append(
        (
            "experiment_manifest.json",
            copy_text_immutable(
                run_dir / "experiment_manifest.json",
                results_dir / "experiment_manifest.json",
            ),
        )
    )

    session_manifests = sorted(
        run_dir.glob("sessions/*/session_manifest.json")
    )
    for source in session_manifests:
        session_name = source.parent.name
        destination = (
            results_dir
            / "sessions"
            / session_name
            / "session_manifest.json"
        )
        status = copy_text_immutable(source, destination)
        statuses.append(
            (
                f"sessions/{session_name}/session_manifest.json",
                status,
            )
        )

    print("\nFinalization")
    print(f"  Raw files hashed: {len(raw_hashes):,}")
    print(
        "  inference_records.jsonl SHA-256: "
        f"{raw_hashes[inference_rel]}"
    )
    print(f"  Finalized provenance: {results_dir}")
    for relative, status in statuses:
        print(f"  {status:9}  {relative}")


def main() -> None:
    args = parse_args()

    model_config = load_model_config()
    prompt_config = load_prompt_config()
    model_lookup = {
        model["model_key"]: model
        for model in model_config["models"]
    }
    if args.model_key not in model_lookup:
        raise KeyError(f"Unknown model key: {args.model_key}")
    model = model_lookup[args.model_key]

    sample, sample_metadata = load_frozen_sample()

    run_dir = args.root / args.model_key
    records_path = run_dir / "inference_records.jsonl"
    manifest_path = run_dir / "experiment_manifest.json"

    records = read_jsonl(records_path)
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else None
    )

    summary, diagnostics = build_qa_summary(
        args=args,
        model=model,
        model_config=model_config,
        prompt_config=prompt_config,
        sample=sample,
        sample_metadata=sample_metadata,
        records=records,
        manifest=manifest,
    )
    print_summary(summary, diagnostics)

    if args.finalize:
        finalize_run(
            run_dir=run_dir,
            results_dir=args.results_root / args.model_key,
            summary=summary,
        )


if __name__ == "__main__":
    main()
