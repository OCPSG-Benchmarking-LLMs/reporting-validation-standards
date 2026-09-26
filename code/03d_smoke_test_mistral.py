#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from reporting_validation_standards.experiment_specs import (
    load_model_config,
    load_prompt_config,
    render_prompt,
)
from reporting_validation_standards.mistral_adapter import MistralClassifier
from reporting_validation_standards.sample_io import (
    load_frozen_sample,
    select_input_text,
)


DEFAULT_MODEL_KEY = "mistral_large3_2512"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test the supplementary Mistral Large 3 snapshot on frozen "
            "UK cases using P0."
        )
    )
    parser.add_argument("--n-cases", type=int, default=5)
    parser.add_argument("--model-key", default=DEFAULT_MODEL_KEY)
    parser.add_argument("--prompt-id", default="p0_capbench_canonical")
    parser.add_argument(
        "--input-mode",
        choices=["text_en", "text"],
        default="text_en",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_cases <= 0:
        raise ValueError("--n-cases must be positive.")

    model_config = load_model_config()
    prompt_config = load_prompt_config()
    official_labels = prompt_config["official_labels"]

    model_lookup = {
        model["model_key"]: model
        for model in model_config["models"]
    }
    if args.model_key not in model_lookup:
        raise KeyError(f"Unknown model key: {args.model_key}")
    model = model_lookup[args.model_key]
    if model["provider"] != "mistral":
        raise ValueError(f"{args.model_key} is not a Mistral model.")

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    sample, sample_metadata = load_frozen_sample(token=token)
    smoke_cases = sample.iloc[: args.n_cases].copy()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"mistral_smoke_{timestamp}"
    run_dir = Path("runs/smoke/mistral") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    classifier = MistralClassifier(
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )

    records = []
    for _, row in smoke_cases.iterrows():
        text = select_input_text(row, input_mode=args.input_mode)
        rendered = render_prompt(prompt_id=args.prompt_id, text=text)

        record = classifier.classify(
            run_id=run_id,
            observation_id=str(row[sample_metadata["id_column"]]),
            input_text=text,
            model_key=args.model_key,
            model_id=model["model_id"],
            prompt=rendered,
            official_labels=official_labels,
            repetition=1,
            reasoning_setting=model["reasoning_setting"],
            temperature=0.0,
            max_output_tokens=40,
            structured_output_mode=model["structured_output_mode"],
        )
        records.append(record.to_dict())

        status = "OK" if record.parse_status == "ok" else "ERROR"
        base = (
            f"{status:5}  {args.model_key:24}  "
            f"id={record.observation_id}  "
            f"label={record.predicted_label}  "
            f"tokens={record.total_tokens}  "
            f"latency={record.latency_seconds:.3f}s"
        )
        if record.cost_usd is not None:
            base += f"  cost_usd={record.cost_usd:.6f}"
        if record.returned_model_id:
            base += f"  returned={record.returned_model_id}"
        print(base)
        if record.parse_status != "ok":
            print(f"       error_type={record.error_type}")
            print(f"       error_message={record.error_message}")

    jsonl_path = run_dir / "inference_records.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    returned_model_ids = sorted(
        {
            str(record["returned_model_id"])
            for record in records
            if record.get("returned_model_id")
        }
    )
    manifest = {
        "run_id": run_id,
        "purpose": "mistral_large3_supplementary_smoke_test",
        "sample_id_sha256": sample_metadata["sample_id_sha256"],
        "dataset_revision": sample_metadata["dataset_revision"],
        "n_cases": args.n_cases,
        "input_mode": args.input_mode,
        "prompt_id": args.prompt_id,
        "model_key": args.model_key,
        "requested_model_id": model["model_id"],
        "returned_model_ids": returned_model_ids,
        "snapshot_pinned": model["snapshot_pinned"],
        "stage": model["stage"],
        "temperature": 0.0,
        "reasoning_setting": model["reasoning_setting"],
        "service_tier": "standard_only",
        "max_output_tokens": 40,
        "n_records": len(records),
        "n_errors": sum(
            record["parse_status"] != "ok"
            for record in records
        ),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"\nRun artefacts: {run_dir}")
    print(
        f"Successful records: "
        f"{len(records) - manifest['n_errors']}/{len(records)}"
    )

    if manifest["n_errors"]:
        raise SystemExit(
            "Smoke test completed with one or more errors. "
            "Inspect inference_records.jsonl before production."
        )

    if (
        returned_model_ids
        and returned_model_ids != [model["model_id"]]
    ):
        raise SystemExit(
            "Smoke test returned a model identifier different from the frozen "
            f"snapshot: requested={model['model_id']!r}, "
            f"returned={returned_model_ids!r}."
        )


if __name__ == "__main__":
    main()
