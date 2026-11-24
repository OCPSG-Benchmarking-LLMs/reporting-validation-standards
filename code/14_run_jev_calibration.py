#!/usr/bin/env python3
"""Resumable JEV native-choice demonstration, separate from the frozen seven-model panel."""
import argparse
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from reporting_validation_standards.experiment_specs import load_prompt_config
from reporting_validation_standards.jev import (digest, extract_choice,
                                                probability_response_diagnostics)

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
INSTRUCTIONS = ("Choose the single best-fitting major policy agenda label for "
                "the parliamentary speech. Do not provide explanations.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Use a versioned JEV model, not a moving alias")
    parser.add_argument("--sample", type=Path, default=Path("data/generated/uk_test_1000.parquet"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/uk_test_1000_ids.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/manifests/uk_test_1000_metadata.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/jev/calibration"))
    parser.add_argument("--text-column", default="text_en")
    parser.add_argument("--max-decisions", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.max_decisions < 0:
        raise ValueError("max-decisions must be nonnegative")
    design = pd.read_csv(args.manifest)
    sample = pd.read_parquet(args.sample)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if len(design) != metadata["realised_n"] or len(sample) != len(design):
        raise ValueError("Frozen sample length mismatch")
    if design.id.duplicated().any() or sample.id.duplicated().any():
        raise ValueError("Duplicate frozen IDs")
    if set(design.id) != set(sample.id):
        raise ValueError("Frozen ID membership mismatch")
    sample = sample.set_index("id").loc[design.id].reset_index()
    if not (sample.predicted_label.astype(str).to_numpy() ==
            design.predicted_label.astype(str).to_numpy()).all():
        raise ValueError("Frozen labels differ")
    if args.text_column not in sample or sample[args.text_column].isna().any():
        raise ValueError("Missing text column or values")
    labels = load_prompt_config()["official_labels"]
    specification = {"model": args.model, "endpoint": ENDPOINT,
                     "dataset_revision": metadata["dataset_revision"],
                     "sample_hash": metadata["sample_id_sha256"],
                     "label_inventory": labels, "instructions": INSTRUCTIONS,
                     "criteria": {label: label for label in labels},
                     "input_field": args.text_column, "repetitions": 1,
                     "method": "native-choice (not byte-identical P0 prompting)"}
    stamp = digest(specification)
    record_path = args.output / "inference_records.jsonl"
    specification_path = args.output / "experiment_manifest.json"
    if specification_path.exists():
        old = json.loads(specification_path.read_text(encoding="utf-8"))
        if digest(old["specification"]) != stamp:
            raise ValueError("Cannot change a frozen JEV run specification")
    existing = {}
    if record_path.exists():
        for line in record_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                oid = item["observation_id"]
                if oid in existing or oid not in set(design.id):
                    raise ValueError("Duplicate or unexpected run observation")
                existing[oid] = item
    remaining = sample.loc[~sample.id.isin(existing)]
    print(f"Frozen n={len(sample)}, completed={len(existing)}, remaining={len(remaining)}")
    if args.plan_only:
        return
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError("TYPESAFE_API_KEY is required for live inference")
    args.output.mkdir(parents=True, exist_ok=True)
    if not specification_path.exists():
        specification_path.write_text(json.dumps({"specification": specification,
                                                   "sha256": stamp}, indent=2) + "\n",
                                      encoding="utf-8")
    count = len(remaining) if args.max_decisions == 0 else args.max_decisions
    with record_path.open("a", encoding="utf-8") as output:
        for _, row in remaining.head(count).iterrows():
            payload = {"model": args.model, "state": str(row[args.text_column]),
                       "questions": {"policy": {"type": "choice", "instructions": INSTRUCTIONS,
                                                "criteria": specification["criteria"]}}}
            record = {"observation_id": row.id, "reference_label": row.predicted_label,
                      "specification_sha256": stamp, "started_at":
                      datetime.now(timezone.utc).isoformat(), "status": "failed"}
            response_body = None
            try:
                request = urllib.request.Request(
                    ENDPOINT, json.dumps(payload).encode("utf-8"),
                    {"Content-Type": "application/json",
                     "Authorization": "Bearer " + key}, method="POST")
                before = time.monotonic()
                with urllib.request.urlopen(request, timeout=120) as response:
                    response_body = json.load(response)
                parsed = extract_choice(response_body, labels)
                record.update(parsed)
                record.update({"model_returned": response_body.get("model"),
                               "usage": response_body.get("usage"),
                               "latency_seconds": time.monotonic() - before,
                               "status": "valid"})
            except Exception as exc:
                record.update({"error_type": type(exc).__name__,
                               "error": str(exc)[:240]})
                if isinstance(response_body, dict):
                    record.update(probability_response_diagnostics(response_body, labels))
            record["completed_at"] = datetime.now(timezone.utc).isoformat()
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(record["observation_id"], record["status"])


if __name__ == "__main__":
    main()
