#!/usr/bin/env python3
"""Separate, opt-in audit of the original JEV probability-sum failures.

This script NEVER modifies the frozen 1,000-observation inference file or its
manifest. Repeated calls are secondary diagnostics, not replacement outcomes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from reporting_validation_standards.jev import digest, extract_choice

DEFAULT_RUN = Path("runs/jev/calibration")
DEFAULT_AUDIT = Path("runs/jev/probability_sum_audit.jsonl")
DEFAULT_SAMPLE = Path("data/generated/uk_test_1000.parquet")
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
TARGET_ERROR = "Probabilities do not sum to one"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def inspect_response(response: dict, labels: list[str]) -> dict:
    """Return diagnostics without modifying/normalising provider probabilities."""
    answer = response.get("answers", {}).get("policy", {})
    raw = answer.get("probabilities", {})
    if not isinstance(raw, dict):
        return {"probability_error": "Missing or non-object probability map"}
    if set(raw) != set(labels):
        return {
            "probability_error": "Label inventory mismatch",
            "missing_labels": sorted(set(labels) - set(raw)),
            "unexpected_labels": sorted(set(raw) - set(labels)),
        }
    try:
        values = [float(raw[label]) for label in labels]
    except (TypeError, ValueError) as exc:
        return {"probability_error": type(exc).__name__}
    if not all(math.isfinite(value) for value in values):
        return {"probability_error": "Non-finite probability"}
    total = math.fsum(values)
    top_index = max(range(len(values)), key=values.__getitem__)
    return {
        "probability_sum": total,
        "probability_sum_minus_one": total - 1.0,
        "probability_min": min(values),
        "probability_max": max(values),
        "n_labels": len(values),
        "top_probability_label": labels[top_index],
        "reported_choice": answer.get("choice"),
        "reported_confidence": answer.get("confidence"),
        "raw_probabilities": raw,
    }


def audit():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    p.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    p.add_argument("--audit-file", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--max-decisions", type=int, default=1,
                   help="Default 1; use 0 to audit every remaining original sum failure")
    p.add_argument("--plan-only", action="store_true")
    args = p.parse_args()
    if args.max_decisions < 0:
        raise ValueError("--max-decisions must be non-negative")
    manifest = json.loads((args.run_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    spec = manifest["specification"]
    if digest(spec) != manifest["sha256"]:
        raise ValueError("Original JEV run specification hash mismatch")
    original_path = args.run_dir / "inference_records.jsonl"
    originals = [
        json.loads(line) for line in original_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(originals) != 1000 or len({r["observation_id"] for r in originals}) != 1000:
        raise ValueError("Expected exactly 1,000 unique frozen observations")
    failures = [
        r for r in originals
        if r.get("status") == "failed" and r.get("error") == TARGET_ERROR
    ]
    for row in originals:
        if row.get("specification_sha256") != manifest["sha256"]:
            raise ValueError("Original inference record specification mismatch")
    audit_entries = {}
    if args.audit_file.exists():
        for line in args.audit_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            obs_id = item["observation_id"]
            if obs_id in audit_entries:
                raise ValueError("Duplicate audit record")
            if item.get("original_specification_sha256") != manifest["sha256"]:
                raise ValueError("Audit record references a different specification")
            audit_entries[obs_id] = item
    failure_ids = {r["observation_id"] for r in failures}
    if set(audit_entries) - failure_ids:
        raise ValueError("Audit includes an observation outside the original sum failures")
    remaining = [r for r in failures if r["observation_id"] not in audit_entries]
    print(f"Original probability-sum failures: {len(failures)}")
    print(f"Already audited: {len(audit_entries)}; remaining: {len(remaining)}")
    if args.plan_only or not remaining:
        return
    key = os.getenv("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError("Set TYPESAFE_API_KEY before running this opt-in audit")
    sample = pd.read_parquet(args.sample).set_index("id")
    if not sample.index.is_unique or not set(r["observation_id"] for r in originals) == set(sample.index):
        raise ValueError("Local sample does not match original observation IDs")
    labels = spec["label_inventory"]
    criteria = spec["criteria"]
    model = spec["model"]
    text_field = spec["input_field"]
    if ENDPOINT != spec["endpoint"] or not set(labels) == set(criteria):
        raise ValueError("Endpoint or label inventory mismatch with frozen specification")
    args.audit_file.parent.mkdir(parents=True, exist_ok=True)
    count = len(remaining) if args.max_decisions == 0 else args.max_decisions
    with args.audit_file.open("a", encoding="utf-8") as output:
        for old in remaining[:count]:
            oid = old["observation_id"]
            payload = {"model": model, "state": str(sample.loc[oid, text_field]),
                       "questions": {"policy": {"type": "choice",
                                                "instructions": spec["instructions"],
                                                "criteria": criteria}}}
            record = {
                "observation_id": oid,
                "original_specification_sha256": manifest["sha256"],
                "original_failure": old.get("error"),
                "audit_started_at": utc_now(),
                "audit_status": "request_failed",
            }
            try:
                request = urllib.request.Request(
                    ENDPOINT, json.dumps(payload).encode("utf-8"),
                    {"Content-Type": "application/json",
                     "Authorization": "Bearer " + key}, method="POST")
                start = time.monotonic()
                with urllib.request.urlopen(request, timeout=120) as response:
                    body = json.load(response)
                record["latency_seconds"] = time.monotonic() - start
                record["model_returned"] = body.get("model")
                record["usage"] = body.get("usage")
                record.update(inspect_response(body, labels))
                try:
                    extract_choice(body, labels)
                    record["audit_status"] = "passes_original_parser"
                except (ValueError, KeyError, TypeError) as exc:
                    record["audit_status"] = "fails_original_parser"
                    record["parser_error"] = str(exc)
            except Exception as exc:
                record["error_type"] = type(exc).__name__
                record["error"] = str(exc)[:240]
            record["audit_completed_at"] = utc_now()
            output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            print(oid, record["audit_status"],
                  f"sum={record.get('probability_sum', 'unavailable')}")


if __name__ == "__main__":
    audit()
