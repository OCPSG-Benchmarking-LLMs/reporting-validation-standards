from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_OPENAI_PRICING_PATH = Path("config/pricing/openai_2026-09-23.yaml")


@dataclass(frozen=True)
class CostEstimate:
    cost_usd: float
    pricing_config_id: str
    ordinary_input_tokens: int
    cached_input_tokens: int
    cache_write_tokens: int
    output_tokens: int


def load_pricing_config(path: str | Path = DEFAULT_OPENAI_PRICING_PATH) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict) or "models" not in payload:
        raise ValueError(f"Invalid pricing configuration: {path}")
    return payload


def estimate_openai_cost(
    *,
    model_id: str,
    input_tokens: int,
    cached_input_tokens: int = 0,
    cache_write_tokens: int = 0,
    output_tokens: int,
    path: str | Path = DEFAULT_OPENAI_PRICING_PATH,
) -> CostEstimate:
    config = load_pricing_config(path)
    models = config["models"]
    if model_id not in models:
        raise KeyError(f"No pricing entry for model_id={model_id}")

    rates = models[model_id]
    ordinary_input_tokens = input_tokens - cached_input_tokens - cache_write_tokens
    if ordinary_input_tokens < 0:
        raise ValueError(
            "input_tokens must be at least cached_input_tokens + cache_write_tokens."
        )

    cache_write_rate = rates.get("cache_write_per_million")
    if cache_write_tokens and cache_write_rate is None:
        raise ValueError(
            f"Observed cache-write tokens for {model_id}, but the pricing config "
            "does not define a cache-write rate."
        )

    input_multiplier = 1.0
    output_multiplier = 1.0
    threshold = rates.get("long_context_threshold_input_tokens")
    if threshold is not None and input_tokens > int(threshold):
        input_multiplier = float(rates.get("long_context_input_multiplier", 1.0))
        output_multiplier = float(rates.get("long_context_output_multiplier", 1.0))

    input_cost = (
        ordinary_input_tokens * float(rates["input_per_million"]) * input_multiplier
        + cached_input_tokens
        * float(rates["cached_input_per_million"])
        * input_multiplier
        + cache_write_tokens
        * float(cache_write_rate or 0.0)
        * input_multiplier
    ) / 1_000_000

    output_cost = (
        output_tokens * float(rates["output_per_million"]) * output_multiplier
    ) / 1_000_000

    return CostEstimate(
        cost_usd=input_cost + output_cost,
        pricing_config_id=str(config["pricing_config_id"]),
        ordinary_input_tokens=ordinary_input_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_tokens=cache_write_tokens,
        output_tokens=output_tokens,
    )
