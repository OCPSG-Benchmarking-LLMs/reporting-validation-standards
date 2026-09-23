from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI

from .experiment_specs import RenderedPrompt
from .inference import InferenceRecord
from .pricing import estimate_openai_cost


def prediction_json_schema(labels: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "predicted_label": {
                "type": "string",
                "enum": labels,
            }
        },
        "required": ["predicted_label"],
        "additionalProperties": False,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_nested(obj: Any, *attrs: str, default: Any = None) -> Any:
    current = obj
    for attr in attrs:
        if current is None:
            return default
        current = getattr(current, attr, None)
    return default if current is None else current


def _retryable_exception(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        name = type(exc).__name__.lower()
        return "connection" in name or "timeout" in name
    return int(status_code) in {408, 409, 429} or int(status_code) >= 500


class OpenAIClassifier:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")

        # Disable SDK retries so every retry is visible in the experiment record.
        self.client = OpenAI(
            api_key=key,
            timeout=timeout_seconds,
            max_retries=0,
        )
        self.max_retries = max_retries

    def classify(
        self,
        *,
        run_id: str,
        observation_id: str,
        input_text: str,
        model_key: str,
        model_id: str,
        prompt: RenderedPrompt,
        official_labels: list[str],
        repetition: int = 1,
        reasoning_setting: str = "none",
        temperature: float | None = 0.0,
        max_output_tokens: int = 40,
        structured_output_mode: str = "json_schema",
    ) -> InferenceRecord:
        input_hash = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
        started_utc = _utc_now()
        started = time.perf_counter()
        last_exc: Exception | None = None
        retries_used = 0

        for attempt in range(self.max_retries + 1):
            try:
                request: dict[str, Any] = {
                    "model": model_id,
                    "input": [
                        {
                            "role": "system",
                            "content": [
                                {"type": "input_text", "text": prompt.system_prompt}
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt.user_prompt}
                            ],
                        },
                    ],
                    "reasoning": {"effort": reasoning_setting},
                    "max_output_tokens": max_output_tokens,
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "policy_agenda_prediction",
                            "schema": prediction_json_schema(official_labels),
                            "strict": True,
                        }
                    },
                    "store": False,
                    "service_tier": "default",
                }
                if temperature is not None:
                    request["temperature"] = temperature

                response = self.client.responses.create(**request)
                received_utc = _utc_now()
                latency = time.perf_counter() - started

                raw_output = str(getattr(response, "output_text", "") or "").strip()
                payload = json.loads(raw_output)
                predicted_label = payload.get("predicted_label")
                if predicted_label not in official_labels:
                    raise ValueError(
                        f"Structured output returned invalid label: {predicted_label!r}"
                    )

                usage = getattr(response, "usage", None)
                input_tokens = _get_nested(usage, "input_tokens")
                cached_tokens = _get_nested(
                    usage, "input_tokens_details", "cached_tokens", default=0
                )
                cache_write_tokens = _get_nested(
                    usage, "input_tokens_details", "cache_write_tokens", default=0
                )
                output_tokens = _get_nested(usage, "output_tokens")
                reasoning_tokens = _get_nested(
                    usage, "output_tokens_details", "reasoning_tokens", default=0
                )
                total_tokens = _get_nested(usage, "total_tokens")

                cost_usd = None
                pricing_config_id = None
                pricing_error = None
                if input_tokens is not None and output_tokens is not None:
                    try:
                        estimate = estimate_openai_cost(
                            model_id=model_id,
                            input_tokens=int(input_tokens),
                            cached_input_tokens=int(cached_tokens or 0),
                            cache_write_tokens=int(cache_write_tokens or 0),
                            output_tokens=int(output_tokens),
                        )
                        cost_usd = estimate.cost_usd
                        pricing_config_id = estimate.pricing_config_id
                    except Exception as exc:
                        # Pricing metadata must never invalidate a successful
                        # classification. Preserve the inference and surface the
                        # accounting problem for inspection.
                        pricing_error = f"{type(exc).__name__}: {exc}"

                return InferenceRecord(
                    run_id=run_id,
                    observation_id=str(observation_id),
                    provider="openai",
                    model_key=model_key,
                    requested_model_id=model_id,
                    returned_model_id=str(getattr(response, "model", "") or "") or None,
                    provider_response_id=str(getattr(response, "id", "") or "") or None,
                    prompt_id=prompt.prompt_id,
                    prompt_spec_sha256=prompt.prompt_spec_sha256,
                    repetition=repetition,
                    predicted_label=str(predicted_label),
                    parse_status="ok",
                    raw_output=raw_output,
                    input_sha256=input_hash,
                    input_tokens=int(input_tokens) if input_tokens is not None else None,
                    cached_input_tokens=int(cached_tokens or 0),
                    cache_write_tokens=int(cache_write_tokens or 0),
                    output_tokens=int(output_tokens) if output_tokens is not None else None,
                    reasoning_tokens=int(reasoning_tokens or 0),
                    total_tokens=int(total_tokens) if total_tokens is not None else None,
                    latency_seconds=latency,
                    retry_count=retries_used,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    reasoning_setting=reasoning_setting,
                    structured_output_mode=structured_output_mode,
                    cost_usd=cost_usd,
                    pricing_config_id=pricing_config_id,
                    service_tier=str(getattr(response, "service_tier", "") or "") or None,
                    request_started_utc=started_utc,
                    response_received_utc=received_utc,
                    provider_metadata={
                        "status": getattr(response, "status", None),
                        "pricing_error": pricing_error,
                    },
                )

            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries or not _retryable_exception(exc):
                    break
                retries_used += 1
                time.sleep(2 ** attempt)

        received_utc = _utc_now()
        return InferenceRecord(
            run_id=run_id,
            observation_id=str(observation_id),
            provider="openai",
            model_key=model_key,
            requested_model_id=model_id,
            prompt_id=prompt.prompt_id,
            prompt_spec_sha256=prompt.prompt_spec_sha256,
            repetition=repetition,
            predicted_label=None,
            parse_status="error",
            input_sha256=input_hash,
            latency_seconds=time.perf_counter() - started,
            retry_count=retries_used,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            reasoning_setting=reasoning_setting,
            structured_output_mode=structured_output_mode,
            request_started_utc=started_utc,
            response_received_utc=received_utc,
            error_type=type(last_exc).__name__ if last_exc else "UnknownError",
            error_message=str(last_exc) if last_exc else "Unknown inference error",
        )
