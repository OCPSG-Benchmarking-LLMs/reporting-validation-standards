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
from reporting_validation_standards.openai_adapter import OpenAIClassifier
from reporting_validation_standards.sample_io import load_frozen_sample, select_input_text


DEFAULT_MODEL_KEYS = ("openai_gpt54", "openai_gpt54_mini")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the two pinned OpenAI models on frozen UK cases using P0."
    )
    parser.add_argument("--n-cases", type=int, default=5)
    parser.add_argument(
        "--model-key",
        action="append",
        dest="model_keys",
        help="Model key to run. Repeat to select multiple models.",
    )
    parser.add_argument("--prompt-id", default="p0_capbench_canonical")
    parser.add_argument("--input-mode", choices=["text_en", "text"], default="text_en")
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

    requested_keys = tuple(args.model_keys or DEFAULT_MODEL_KEYS)
    model_lookup = {model["model_key"]: model for model in model_config["models"]}
    unknown = [key for key in requested_keys if key not in model_lookup]
    if unknown:
        raise KeyError(f"Unknown model key(s): {unknown}")

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    sample, sample_metadata = load_frozen_sample(token=token)
    smoke_cases = sample.iloc[: args.n_cases].copy()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"openai_smoke_{timestamp}"
    run_dir = Path("runs/smoke/openai") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    classifier = OpenAIClassifier(
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )

    records = []
    for model_key in requested_keys:
        model = model_lookup[model_key]
        for _, row in smoke_cases.iterrows():
            text = select_input_text(row, input_mode=args.input_mode)
            rendered = render_prompt(prompt_id=args.prompt_id, text=text)

            record = classifier.classify(
                run_id=run_id,
                observation_id=str(row[sample_metadata["id_column"]]),
                input_text=text,
                model_key=model_key,
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
                f"{status:5}  {model_key:24}  "
                f"id={record.observation_id}  "
                f"label={record.predicted_label}  "
                f"tokens={record.total_tokens}  "
                f"latency={record.latency_seconds:.3f}s"
            )
            if record.cost_usd is not None:
                base += f"  cost_usd={record.cost_usd:.6f}"
            print(base)

    jsonl_path = run_dir / "inference_records.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "run_id": run_id,
        "purpose": "openai_smoke_test",
        "sample_id_sha256": sample_metadata["sample_id_sha256"],
        "dataset_revision": sample_metadata["dataset_revision"],
        "n_cases": args.n_cases,
        "input_mode": args.input_mode,
        "prompt_id": args.prompt_id,
        "model_keys": list(requested_keys),
        "temperature": 0.0,
        "reasoning_effort": "none",
        "max_output_tokens": 40,
        "n_records": len(records),
        "n_errors": sum(record["parse_status"] != "ok" for record in records),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"\nRun artefacts: {run_dir}")
    print(f"Successful records: {len(records) - manifest['n_errors']}/{len(records)}")

    if manifest["n_errors"]:
        raise SystemExit(
            "Smoke test completed with one or more errors. Inspect inference_records.jsonl."
        )


if __name__ == "__main__":
    main()
