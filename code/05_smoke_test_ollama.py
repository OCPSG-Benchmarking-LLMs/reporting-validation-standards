#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from reporting_validation_standards.experiment_specs import (
    load_model_config,
    load_prompt_config,
    render_prompt,
)
from reporting_validation_standards.local_compute import (
    CodeCarbonRunTracker,
    collect_nvidia_metadata,
    collect_platform_metadata,
)
from reporting_validation_standards.ollama_adapter import OllamaClassifier
from reporting_validation_standards.sample_io import load_frozen_sample, select_input_text


DEFAULT_MODEL_KEY = "ollama_qwen36_35b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test one local Ollama model on frozen UK cases using P0."
    )
    parser.add_argument("--n-cases", type=int, default=5)
    parser.add_argument("--model-key", default=DEFAULT_MODEL_KEY)
    parser.add_argument("--prompt-id", default="p0_capbench_canonical")
    parser.add_argument("--input-mode", choices=["text_en", "text"], default="text_en")
    parser.add_argument("--ollama-host", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--track-carbon", action="store_true")
    parser.add_argument(
        "--country-iso-code",
        default=None,
        help="Three-letter ISO code required with --track-carbon, e.g. NLD.",
    )
    parser.add_argument("--measure-power-secs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.n_cases <= 0:
        raise ValueError("--n-cases must be positive.")
    if args.track_carbon and not args.country_iso_code:
        raise ValueError(
            "--country-iso-code is required when --track-carbon is enabled "
            "(for example, NLD when the machine is in the Netherlands)."
        )

    model_config = load_model_config()
    prompt_config = load_prompt_config()
    official_labels = prompt_config["official_labels"]

    model_lookup = {model["model_key"]: model for model in model_config["models"]}
    if args.model_key not in model_lookup:
        raise KeyError(f"Unknown model key: {args.model_key}")

    model = model_lookup[args.model_key]
    if model["provider"] != "ollama":
        raise ValueError(f"{args.model_key} is not an Ollama model.")

    classifier = OllamaClassifier(
        host=args.ollama_host,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )

    local_model = classifier.resolve_local_model(model["model_id"])
    local_digest = str(local_model.get("digest") or "")
    if not local_digest:
        raise RuntimeError(
            f"Ollama did not return a digest for installed model {model['model_id']}."
        )

    show = classifier.show_model(model["model_id"])
    capabilities = show.get("capabilities") or []
    if not isinstance(capabilities, list):
        capabilities = []

    # The native Ollama API accepts think=false. We send it explicitly so that
    # any thinking-capable model is used as a classifier rather than a deliberative
    # reasoner. The adapter records whether the parameter was sent.
    send_think_false = True

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    sample, sample_metadata = load_frozen_sample(token=token)
    smoke_cases = sample.iloc[: args.n_cases].copy()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"ollama_smoke_{args.model_key}_{timestamp}"
    run_dir = Path("runs/smoke/ollama") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    platform_metadata = collect_platform_metadata()
    nvidia_metadata = collect_nvidia_metadata()
    ollama_version = classifier.version()

    carbon_tracker = None
    carbon_summary = None
    if args.track_carbon:
        carbon_tracker = CodeCarbonRunTracker(
            run_dir=run_dir,
            run_id=run_id,
            country_iso_code=args.country_iso_code,
            measure_power_secs=args.measure_power_secs,
        )

    records = []
    run_started = time.perf_counter()

    if carbon_tracker is not None:
        carbon_tracker.start()

    try:
        for position, (_, row) in enumerate(smoke_cases.iterrows(), start=1):
            text = select_input_text(row, input_mode=args.input_mode)
            rendered = render_prompt(prompt_id=args.prompt_id, text=text)

            # Keep the model resident between cases but unload it after the last
            # observation so sequential model runs do not compete for VRAM.
            keep_alive = "0" if position == len(smoke_cases) else "5m"

            record = classifier.classify(
                run_id=run_id,
                observation_id=str(row[sample_metadata["id_column"]]),
                input_text=text,
                model_key=args.model_key,
                model_id=model["model_id"],
                local_model_digest=local_digest,
                prompt=rendered,
                official_labels=official_labels,
                repetition=1,
                reasoning_setting=model["reasoning_setting"],
                temperature=0.0,
                max_output_tokens=40,
                structured_output_mode=model["structured_output_mode"],
                send_think_false=send_think_false,
                keep_alive=keep_alive,
            )
            records.append(record.to_dict())

            status = "OK" if record.parse_status == "ok" else "ERROR"
            meta = record.provider_metadata or {}
            speed = meta.get("eval_tokens_per_second")
            load_s = meta.get("load_duration_seconds")
            prompt_s = meta.get("prompt_eval_duration_seconds")
            generation_s = meta.get("eval_duration_seconds")
            timing_text = ""
            if all(isinstance(value, (float, int)) for value in (load_s, prompt_s, generation_s)):
                timing_text = (
                    f"  load={load_s:.3f}s"
                    f"  prompt={prompt_s:.3f}s"
                    f"  gen={generation_s:.3f}s"
                )
            speed_text = f"  tok_s={speed:.2f}" if isinstance(speed, (float, int)) else ""
            print(
                f"{status:5}  {args.model_key:24}  "
                f"id={record.observation_id}  "
                f"label={record.predicted_label}  "
                f"tokens={record.total_tokens}  "
                f"latency={record.latency_seconds:.3f}s"
                f"{timing_text}"
                f"{speed_text}"
            )
    finally:
        if carbon_tracker is not None:
            carbon_summary = carbon_tracker.stop()

    run_elapsed = time.perf_counter() - run_started

    jsonl_path = run_dir / "inference_records.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    timing_rows = [
        record.get("provider_metadata") or {}
        for record in records
        if record.get("parse_status") == "ok"
    ]

    def _sum_numeric(field: str) -> float | None:
        values = [
            row.get(field)
            for row in timing_rows
            if isinstance(row.get(field), (float, int))
        ]
        return float(sum(values)) if values else None

    timing_summary = {
        "cold_start_definition": (
            "First observation in a run includes model loading; subsequent observations "
            "reuse the resident model until the final keep_alive=0 request."
        ),
        "load_duration_seconds_total": _sum_numeric("load_duration_seconds"),
        "prompt_eval_duration_seconds_total": _sum_numeric("prompt_eval_duration_seconds"),
        "generation_duration_seconds_total": _sum_numeric("eval_duration_seconds"),
    }

    manifest = {
        "run_id": run_id,
        "purpose": "ollama_smoke_test",
        "sample_id_sha256": sample_metadata["sample_id_sha256"],
        "dataset_revision": sample_metadata["dataset_revision"],
        "n_cases": args.n_cases,
        "input_mode": args.input_mode,
        "prompt_id": args.prompt_id,
        "model_key": args.model_key,
        "requested_model_id": model["model_id"],
        "local_model_digest": local_digest,
        "ollama_version": ollama_version,
        "ollama_host": classifier.host,
        "local_model_registry_metadata": {
            "name": local_model.get("name"),
            "model": local_model.get("model"),
            "modified_at": local_model.get("modified_at"),
            "size_bytes": local_model.get("size"),
            "details": local_model.get("details"),
        },
        "ollama_capabilities": capabilities,
        "temperature": 0.0,
        "thinking_policy": "send think=false",
        "max_output_tokens": 40,
        "keep_alive_policy": "5m between cases; 0 on final case to unload model",
        "n_records": len(records),
        "n_errors": sum(record["parse_status"] != "ok" for record in records),
        "elapsed_seconds": run_elapsed,
        "timing_summary": timing_summary,
        "platform_metadata": platform_metadata,
        "nvidia_metadata": nvidia_metadata,
        "carbon_tracking_enabled": bool(args.track_carbon),
        "carbon_summary": asdict(carbon_summary) if carbon_summary is not None else None,
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"\nLocal model digest: {local_digest}")
    if local_model.get("details"):
        print(f"Model details: {json.dumps(local_model['details'], ensure_ascii=False)}")
    print(f"Ollama version: {ollama_version}")
    print(f"Run artefacts: {run_dir}")
    print(f"Successful records: {len(records) - manifest['n_errors']}/{len(records)}")

    if carbon_summary is not None:
        print(
            "CodeCarbon: "
            f"energy={carbon_summary.energy_consumed_kwh} kWh  "
            f"emissions={carbon_summary.emissions_kg_co2e} kgCO2e"
        )

    if manifest["n_errors"]:
        raise SystemExit(
            "Smoke test completed with one or more errors. Inspect inference_records.jsonl."
        )


if __name__ == "__main__":
    main()
