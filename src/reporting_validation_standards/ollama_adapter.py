from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

from .experiment_specs import RenderedPrompt
from .inference import InferenceRecord


DEFAULT_OLLAMA_HOST = "http://localhost:11434"


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


def _ns_to_seconds(value: Any) -> float | None:
    if value is None:
        return None
    return float(value) / 1_000_000_000


def _retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 409, 429} or exc.code >= 500
    return isinstance(exc, (urllib.error.URLError, TimeoutError))


class OllamaClassifier:
    def __init__(
        self,
        *,
        host: str | None = None,
        timeout_seconds: float = 600.0,
        max_retries: int = 1,
    ) -> None:
        self.host = (host or os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def _request_json(
        self,
        *,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.host}{path}"
        body = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            url=url,
            data=body,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            raw = response.read().decode("utf-8")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"Ollama returned a non-object JSON response from {path}.")
        return parsed

    def version(self) -> str | None:
        try:
            payload = self._request_json(method="GET", path="/api/version")
        except Exception:
            return None
        value = payload.get("version")
        return str(value) if value is not None else None

    def list_models(self) -> list[dict[str, Any]]:
        payload = self._request_json(method="GET", path="/api/tags")
        models = payload.get("models", [])
        if not isinstance(models, list):
            raise ValueError("Ollama /api/tags response does not contain a models list.")
        return [model for model in models if isinstance(model, dict)]

    def resolve_local_model(self, model_id: str) -> dict[str, Any]:
        candidates = []
        for model in self.list_models():
            names = {str(model.get("name", "")), str(model.get("model", ""))}
            if model_id in names:
                candidates.append(model)

        if not candidates:
            available = [
                str(model.get("name") or model.get("model"))
                for model in self.list_models()
            ]
            raise RuntimeError(
                f"Ollama model {model_id!r} is not installed locally. "
                f"Available models: {available}"
            )

        if len(candidates) > 1:
            raise RuntimeError(f"Multiple local Ollama entries matched {model_id!r}.")

        return candidates[0]

    def show_model(self, model_id: str) -> dict[str, Any]:
        return self._request_json(
            method="POST",
            path="/api/show",
            payload={"model": model_id},
        )


    def running_models(self) -> list[dict[str, Any]]:
        payload = self._request_json(method="GET", path="/api/ps")
        models = payload.get("models", [])
        if not isinstance(models, list):
            raise ValueError("Ollama /api/ps response does not contain a models list.")
        return [model for model in models if isinstance(model, dict)]

    def resolve_running_model(self, model_id: str) -> dict[str, Any] | None:
        for model in self.running_models():
            names = {str(model.get("name", "")), str(model.get("model", ""))}
            if model_id in names:
                return model
        return None

    def unload_model(self, model_id: str) -> None:
        self._request_json(
            method="POST",
            path="/api/generate",
            payload={"model": model_id, "keep_alive": 0},
        )

    def classify(
        self,
        *,
        run_id: str,
        observation_id: str,
        input_text: str,
        model_key: str,
        model_id: str,
        local_model_digest: str,
        prompt: RenderedPrompt,
        official_labels: list[str],
        repetition: int = 1,
        reasoning_setting: str = "thinking_false",
        temperature: float = 0.0,
        max_output_tokens: int = 40,
        structured_output_mode: str = "json_schema",
        send_think_false: bool = True,
        keep_alive: str = "5m",
    ) -> InferenceRecord:
        input_hash = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
        started_utc = _utc_now()
        started = time.perf_counter()

        request_payload: dict[str, Any] = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            "stream": False,
            "format": prediction_json_schema(official_labels),
            "options": {
                "temperature": temperature,
                "num_predict": max_output_tokens,
            },
            "keep_alive": keep_alive,
        }
        if send_think_false:
            request_payload["think"] = False

        retries_used = 0
        response: dict[str, Any] | None = None
        api_error: Exception | None = None
        retry_events: list[dict[str, Any]] = []
        successful_attempt_latency_seconds: float | None = None

        for attempt in range(self.max_retries + 1):
            attempt_started = time.perf_counter()
            try:
                response = self._request_json(
                    method="POST",
                    path="/api/chat",
                    payload=request_payload,
                )
                successful_attempt_latency_seconds = time.perf_counter() - attempt_started
                api_error = None
                break
            except Exception as exc:
                api_error = exc
                attempt_latency = time.perf_counter() - attempt_started
                retryable = _retryable_exception(exc)
                status_code = exc.code if isinstance(exc, urllib.error.HTTPError) else None
                event: dict[str, Any] = {
                    "attempt": attempt + 1,
                    "exception_type": type(exc).__name__,
                    "status_code": status_code,
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
                provider="ollama",
                model_key=model_key,
                requested_model_id=model_id,
                local_model_digest=local_model_digest,
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
                request_started_utc=started_utc,
                response_received_utc=received_utc,
                error_type=type(api_error).__name__ if api_error else "UnknownOllamaError",
                error_message=str(api_error) if api_error else "Unknown Ollama error",
                provider_metadata={
                    "attempt_count": len(retry_events),
                    "retry_events": retry_events,
                    "successful_attempt_latency_seconds": None,
                    "latency_definition": "end_to_end_including_retries_and_backoff",
                    "think_parameter_sent": send_think_false,
                },
            )

        message = response.get("message") or {}
        raw_output = str(message.get("content") or "").strip()
        thinking_output = str(message.get("thinking") or "")

        input_tokens = response.get("prompt_eval_count")
        cached_input_tokens = response.get("prompt_eval_cached_count")
        output_tokens = response.get("eval_count")
        total_tokens = None
        if input_tokens is not None and output_tokens is not None:
            total_tokens = int(input_tokens) + int(output_tokens)

        prompt_eval_duration_s = _ns_to_seconds(response.get("prompt_eval_duration"))
        eval_duration_s = _ns_to_seconds(response.get("eval_duration"))
        eval_tokens_per_second = None
        if output_tokens and eval_duration_s and eval_duration_s > 0:
            eval_tokens_per_second = float(output_tokens) / eval_duration_s

        common = {
            "run_id": run_id,
            "observation_id": str(observation_id),
            "provider": "ollama",
            "model_key": model_key,
            "requested_model_id": model_id,
            "returned_model_id": str(response.get("model") or "") or None,
            "local_model_digest": local_model_digest,
            "prompt_id": prompt.prompt_id,
            "prompt_spec_sha256": prompt.prompt_spec_sha256,
            "repetition": repetition,
            "raw_output": raw_output,
            "input_sha256": input_hash,
            "input_tokens": int(input_tokens) if input_tokens is not None else None,
            "cached_input_tokens": (
                int(cached_input_tokens) if cached_input_tokens is not None else None
            ),
            "cache_write_tokens": None,
            "output_tokens": int(output_tokens) if output_tokens is not None else None,
            "reasoning_tokens": 0 if send_think_false else None,
            "total_tokens": total_tokens,
            "latency_seconds": latency,
            "retry_count": retries_used,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "reasoning_setting": reasoning_setting,
            "structured_output_mode": structured_output_mode,
            "cost_usd": None,
            "pricing_config_id": None,
            "service_tier": None,
            "request_started_utc": started_utc,
            "response_received_utc": received_utc,
            "provider_metadata": {
                "created_at": response.get("created_at"),
                "done": response.get("done"),
                "done_reason": response.get("done_reason"),
                "total_duration_seconds": _ns_to_seconds(response.get("total_duration")),
                "load_duration_seconds": _ns_to_seconds(response.get("load_duration")),
                "prompt_eval_duration_seconds": prompt_eval_duration_s,
                "eval_duration_seconds": eval_duration_s,
                "eval_tokens_per_second": eval_tokens_per_second,
                "thinking_output_present": bool(thinking_output),
                "thinking_output_chars": len(thinking_output),
                "think_parameter_sent": send_think_false,
                "attempt_count": retries_used + 1,
                "retry_events": retry_events,
                "successful_attempt_latency_seconds": successful_attempt_latency_seconds,
                "latency_definition": "end_to_end_including_retries_and_backoff",
            },
        }

        try:
            payload = json.loads(raw_output)
            predicted_label = payload.get("predicted_label")
            if predicted_label not in official_labels:
                raise ValueError(
                    f"Structured output returned invalid label: {predicted_label!r}"
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
