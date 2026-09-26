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
from .pricing import estimate_mistral_cost


MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
MISTRAL_SERVICE_TIER = "standard_only"


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


class MistralClassifier:
    """Mistral Chat Completions adapter using the documented OpenAI-compatible API.

    The experiment already depends on the OpenAI Python client. Mistral documents
    https://api.mistral.ai/v1 as an OpenAI-compatible base URL, so this adapter
    avoids adding a second SDK while retaining provider-specific configuration,
    identity, pricing, and telemetry.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        key = api_key or os.getenv("MISTRAL_API_KEY")
        if not key:
            raise EnvironmentError("MISTRAL_API_KEY is not set.")

        self.client = OpenAI(
            api_key=key,
            base_url=MISTRAL_BASE_URL,
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
        reasoning_setting: str = "not_exposed_for_model",
        temperature: float | None = 0.0,
        max_output_tokens: int = 40,
        structured_output_mode: str = "json_schema",
    ) -> InferenceRecord:
        input_hash = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
        started_utc = _utc_now()
        started = time.perf_counter()
        retries_used = 0
        response: Any | None = None
        api_error: Exception | None = None
        retry_events: list[dict[str, Any]] = []
        successful_attempt_latency_seconds: float | None = None

        request: dict[str, Any] = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "policy_agenda_prediction",
                    "schema": prediction_json_schema(official_labels),
                },
            },
            "max_tokens": max_output_tokens,
            # Pass the provider-specific service-tier enum through extra_body so
            # it is not constrained by the OpenAI SDK's own type surface.
            "extra_body": {
                "service_tier": MISTRAL_SERVICE_TIER,
            },
        }
        # Mistral Large 3 does not document adjustable reasoning support.
        # Only send reasoning_effort for models whose frozen specification
        # explicitly exposes a supported value.
        if reasoning_setting not in {"not_exposed_for_model", "not_exposed"}:
            request["extra_body"]["reasoning_effort"] = reasoning_setting

        if temperature is not None:
            request["temperature"] = temperature

        for attempt in range(self.max_retries + 1):
            attempt_started = time.perf_counter()
            try:
                response = self.client.chat.completions.create(**request)
                successful_attempt_latency_seconds = (
                    time.perf_counter() - attempt_started
                )
                api_error = None
                break
            except Exception as exc:
                api_error = exc
                attempt_latency = time.perf_counter() - attempt_started
                retryable = _retryable_exception(exc)
                status_code = getattr(exc, "status_code", None)

                event: dict[str, Any] = {
                    "attempt": attempt + 1,
                    "exception_type": type(exc).__name__,
                    "status_code": (
                        int(status_code) if status_code is not None else None
                    ),
                    "retryable": bool(retryable),
                    "attempt_latency_seconds": attempt_latency,
                }

                if attempt >= self.max_retries or not retryable:
                    retry_events.append(event)
                    break

                backoff_seconds = float(2**attempt)
                event["backoff_seconds"] = backoff_seconds
                retry_events.append(event)
                retries_used += 1
                time.sleep(backoff_seconds)

        received_utc = _utc_now()
        latency = time.perf_counter() - started

        if response is None:
            return InferenceRecord(
                run_id=run_id,
                observation_id=str(observation_id),
                provider="mistral",
                model_key=model_key,
                requested_model_id=model_id,
                prompt_id=prompt.prompt_id,
                prompt_spec_sha256=prompt.prompt_spec_sha256,
                repetition=repetition,
                predicted_label=None,
                parse_status="error",
                input_sha256=input_hash,
                latency_seconds=latency,
                retry_count=retries_used,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                reasoning_setting=reasoning_setting,
                structured_output_mode=structured_output_mode,
                service_tier=MISTRAL_SERVICE_TIER,
                request_started_utc=started_utc,
                response_received_utc=received_utc,
                error_type=(
                    type(api_error).__name__ if api_error else "UnknownAPIError"
                ),
                error_message=(
                    str(api_error) if api_error else "Unknown API error"
                ),
                provider_metadata={
                    "attempt_count": len(retry_events),
                    "retry_events": retry_events,
                    "successful_attempt_latency_seconds": None,
                    "latency_definition": (
                        "end_to_end_including_retries_and_backoff"
                    ),
                    "api_compatibility_layer": "openai_sdk",
                    "base_url": MISTRAL_BASE_URL,
                },
            )

        choices = getattr(response, "choices", None) or []
        choice = choices[0] if choices else None
        message = (
            getattr(choice, "message", None) if choice is not None else None
        )
        raw_output = str(getattr(message, "content", "") or "").strip()

        usage = getattr(response, "usage", None)
        input_tokens = _get_nested(usage, "prompt_tokens")
        cached_tokens = _get_nested(
            usage, "prompt_tokens_details", "cached_tokens", default=0
        )
        output_tokens = _get_nested(usage, "completion_tokens")
        reasoning_tokens = _get_nested(
            usage,
            "completion_tokens_details",
            "reasoning_tokens",
            default=0,
        )
        total_tokens = _get_nested(usage, "total_tokens")

        cost_usd = None
        pricing_config_id = None
        pricing_error = None
        if input_tokens is not None and output_tokens is not None:
            try:
                estimate = estimate_mistral_cost(
                    model_id=model_id,
                    input_tokens=int(input_tokens),
                    cached_input_tokens=int(cached_tokens or 0),
                    cache_write_tokens=0,
                    output_tokens=int(output_tokens),
                )
                cost_usd = estimate.cost_usd
                pricing_config_id = estimate.pricing_config_id
            except Exception as exc:
                pricing_error = f"{type(exc).__name__}: {exc}"

        returned_service_tier = (
            str(getattr(response, "service_tier", "") or "") or None
        )

        common = {
            "run_id": run_id,
            "observation_id": str(observation_id),
            "provider": "mistral",
            "model_key": model_key,
            "requested_model_id": model_id,
            "returned_model_id": (
                str(getattr(response, "model", "") or "") or None
            ),
            "provider_response_id": (
                str(getattr(response, "id", "") or "") or None
            ),
            "prompt_id": prompt.prompt_id,
            "prompt_spec_sha256": prompt.prompt_spec_sha256,
            "repetition": repetition,
            "raw_output": raw_output,
            "input_sha256": input_hash,
            "input_tokens": (
                int(input_tokens) if input_tokens is not None else None
            ),
            "cached_input_tokens": int(cached_tokens or 0),
            "cache_write_tokens": 0,
            "output_tokens": (
                int(output_tokens) if output_tokens is not None else None
            ),
            "reasoning_tokens": int(reasoning_tokens or 0),
            "total_tokens": (
                int(total_tokens) if total_tokens is not None else None
            ),
            "latency_seconds": latency,
            "retry_count": retries_used,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "reasoning_setting": reasoning_setting,
            "structured_output_mode": structured_output_mode,
            "cost_usd": cost_usd,
            "pricing_config_id": pricing_config_id,
            "service_tier": returned_service_tier or MISTRAL_SERVICE_TIER,
            "request_started_utc": started_utc,
            "response_received_utc": received_utc,
            "provider_metadata": {
                "finish_reason": getattr(choice, "finish_reason", None),
                "pricing_error": pricing_error,
                "attempt_count": retries_used + 1,
                "retry_events": retry_events,
                "successful_attempt_latency_seconds": (
                    successful_attempt_latency_seconds
                ),
                "latency_definition": (
                    "end_to_end_including_retries_and_backoff"
                ),
                "api_compatibility_layer": "openai_sdk",
                "base_url": MISTRAL_BASE_URL,
                "requested_service_tier": MISTRAL_SERVICE_TIER,
                "returned_service_tier": returned_service_tier,
            },
        }

        try:
            payload = json.loads(raw_output)
            predicted_label = payload.get("predicted_label")
            if predicted_label not in official_labels:
                raise ValueError(
                    "Structured output returned invalid label: "
                    f"{predicted_label!r}"
                )
        except Exception as exc:
            return InferenceRecord(
                **common,
                predicted_label=None,
                parse_status="error",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

        return InferenceRecord(
            **common,
            predicted_label=str(predicted_label),
            parse_status="ok",
        )
