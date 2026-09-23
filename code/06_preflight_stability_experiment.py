#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np

from reporting_validation_standards.experiment_specs import load_prompt_config, render_prompt
from reporting_validation_standards.sample_io import load_frozen_sample, select_input_text


DEFAULT_SMOKE_ROOT = Path("runs/smoke/ollama")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight the frozen 1,000-case stability experiment before production "
            "inference. Reports sample/prompt length diagnostics and uses successful "
            "Ollama smoke records, when available, to calibrate an empirical token proxy."
        )
    )
    parser.add_argument("--input-mode", choices=["text_en", "text"], default="text_en")
    parser.add_argument("--smoke-root", type=Path, default=DEFAULT_SMOKE_ROOT)
    parser.add_argument("--output", type=Path, default=Path("outputs/preflight/stability_preflight.json"))
    return parser.parse_args()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _words(text: str) -> int:
    return len(text.split())


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def _load_smoke_records(root: Path) -> list[dict]:
    records: list[dict] = []
    if not root.exists():
        return records

    for path in root.glob("ollama_smoke_*/inference_records.jsonl"):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("parse_status") == "ok" and record.get("input_tokens"):
                records.append(record)
    return records


def _latest_runtime_contexts(root: Path) -> dict[str, int]:
    latest: dict[str, tuple[float, int]] = {}
    if not root.exists():
        return {}

    for path in root.glob("ollama_smoke_*/run_manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        model_key = manifest.get("model_key")
        residency = manifest.get("runtime_residency_metadata") or {}
        context = residency.get("context_length")
        if not model_key or not isinstance(context, int):
            continue
        mtime = path.stat().st_mtime
        if model_key not in latest or mtime > latest[model_key][0]:
            latest[model_key] = (mtime, context)

    return {key: value[1] for key, value in latest.items()}


def main() -> None:
    args = parse_args()
    sample, metadata = load_frozen_sample()
    prompt_config = load_prompt_config()
    prompt_ids = [prompt["prompt_id"] for prompt in prompt_config["prompts"]]

    rows = []
    text_by_hash: dict[str, str] = {}
    for _, row in sample.iterrows():
        text = select_input_text(row, input_mode=args.input_mode)
        observation_id = str(row[metadata["id_column"]])
        text_by_hash[_sha256(text)] = text
        rows.append(
            {
                "observation_id": observation_id,
                "text_chars": len(text),
                "text_words": _words(text),
            }
        )

    text_chars = [row["text_chars"] for row in rows]
    text_words = [row["text_words"] for row in rows]

    prompt_lengths: dict[str, dict] = {}
    prompt_words_by_observation: dict[tuple[str, str], int] = {}
    for prompt_id in prompt_ids:
        total_chars = []
        total_words = []
        for _, row in sample.iterrows():
            text = select_input_text(row, input_mode=args.input_mode)
            observation_id = str(row[metadata["id_column"]])
            rendered = render_prompt(prompt_id=prompt_id, text=text)
            combined = rendered.system_prompt + "\n" + rendered.user_prompt
            words = _words(combined)
            total_chars.append(len(combined))
            total_words.append(words)
            prompt_words_by_observation[(prompt_id, observation_id)] = words

        prompt_lengths[prompt_id] = {
            "characters": _percentiles(total_chars),
            "whitespace_words": _percentiles(total_words),
        }

    smoke_records = _load_smoke_records(args.smoke_root)
    calibration: dict[str, dict] = {}
    ratios_by_model: dict[str, list[float]] = defaultdict(list)

    for record in smoke_records:
        model_key = record.get("model_key")
        input_hash = record.get("input_sha256")
        prompt_id = record.get("prompt_id")
        observation_id = str(record.get("observation_id"))
        input_tokens = record.get("input_tokens")
        if (
            not model_key
            or input_hash not in text_by_hash
            or prompt_id not in prompt_ids
            or not isinstance(input_tokens, int)
            or input_tokens <= 0
        ):
            continue

        words = prompt_words_by_observation.get((prompt_id, observation_id))
        if not words:
            continue
        ratios_by_model[model_key].append(input_tokens / words)

    runtime_contexts = _latest_runtime_contexts(args.smoke_root)

    for model_key, ratios in sorted(ratios_by_model.items()):
        max_ratio = max(ratios)
        median_ratio = median(ratios)
        estimated_by_prompt = {}

        for prompt_id in prompt_ids:
            prompt_word_counts = [
                prompt_words_by_observation[(prompt_id, str(row[metadata["id_column"]]))]
                for _, row in sample.iterrows()
            ]
            estimated_tokens = [math.ceil(words * max_ratio) for words in prompt_word_counts]
            context = runtime_contexts.get(model_key)
            estimated_by_prompt[prompt_id] = {
                "estimated_input_tokens_using_max_smoke_ratio": _percentiles(estimated_tokens),
                "n_estimated_ge_90pct_runtime_context": (
                    sum(value >= 0.9 * context for value in estimated_tokens)
                    if context is not None
                    else None
                ),
                "n_estimated_ge_runtime_context": (
                    sum(value >= context for value in estimated_tokens)
                    if context is not None
                    else None
                ),
            }

        calibration[model_key] = {
            "n_smoke_records": len(ratios),
            "median_input_tokens_per_whitespace_word": median_ratio,
            "max_input_tokens_per_whitespace_word": max_ratio,
            "runtime_context_length_latest_smoke": runtime_contexts.get(model_key),
            "prompt_estimates": estimated_by_prompt,
            "note": (
                "Token estimates are diagnostics calibrated from smoke-test prompt/token ratios; "
                "they are not tokenizer-exact counts and should not be treated as an upper bound."
            ),
        }

    longest = sorted(rows, key=lambda row: row["text_chars"], reverse=True)[:10]

    n_models = 6
    repetitions = 3
    decisions_per_model = len(sample) * len(prompt_ids) * repetitions

    report = {
        "sample": {
            "sample_id_sha256": metadata["sample_id_sha256"],
            "dataset_revision": metadata["dataset_revision"],
            "n_cases": len(sample),
            "input_mode": args.input_mode,
            "text_characters": _percentiles(text_chars),
            "text_whitespace_words": _percentiles(text_words),
            "longest_cases": longest,
        },
        "prompts": prompt_lengths,
        "experiment_scale": {
            "n_models_planned": n_models,
            "n_prompts": len(prompt_ids),
            "repetitions": repetitions,
            "decisions_per_model": decisions_per_model,
            "decisions_total": decisions_per_model * n_models,
        },
        "ollama_smoke_calibration": calibration,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Frozen stability experiment preflight")
    print(f"Sample cases: {len(sample)}")
    print(f"Prompts: {len(prompt_ids)}")
    print(f"Repetitions: {repetitions}")
    print(f"Decisions per model: {decisions_per_model:,}")
    print(f"Planned total decisions: {decisions_per_model * n_models:,}")

    print("\nText length (whitespace words)")
    for key, value in report["sample"]["text_whitespace_words"].items():
        print(f"  {key:>4}: {value:.1f}")

    print("\nLongest frozen cases")
    for row in longest:
        print(
            f"  {row['observation_id']}: "
            f"{row['text_words']:,} words / {row['text_chars']:,} chars"
        )

    if calibration:
        print("\nOllama context diagnostics (empirical proxy, not exact token counts)")
        for model_key, details in calibration.items():
            print(
                f"  {model_key}: smoke_n={details['n_smoke_records']}  "
                f"runtime_ctx={details['runtime_context_length_latest_smoke']}  "
                f"max_token/word_ratio={details['max_input_tokens_per_whitespace_word']:.3f}"
            )
            for prompt_id, estimate in details["prompt_estimates"].items():
                stats = estimate["estimated_input_tokens_using_max_smoke_ratio"]
                print(
                    f"    {prompt_id}: p99~{stats['p99']:.0f}  max~{stats['max']:.0f}  "
                    f">=90%ctx={estimate['n_estimated_ge_90pct_runtime_context']}  "
                    f">=ctx={estimate['n_estimated_ge_runtime_context']}"
                )
    else:
        print("\nNo successful Ollama smoke records found for empirical token calibration.")

    print(f"\nPreflight report: {args.output}")


if __name__ == "__main__":
    main()
