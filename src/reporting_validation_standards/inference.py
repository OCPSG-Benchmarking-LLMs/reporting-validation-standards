from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


INFERENCE_SCHEMA_VERSION = "1.0"


@dataclass(kw_only=True)
class InferenceRecord:
    """Provider-neutral per-observation inference contract.

    Provider adapters should populate the fields they can observe directly and
    leave genuinely unavailable fields as None. Carbon/energy accounting belongs
    at run level rather than being invented per request.
    """

    run_id: str
    observation_id: str
    provider: str
    model_key: str
    requested_model_id: str
    prompt_id: str
    prompt_spec_sha256: str
    repetition: int

    predicted_label: str | None
    parse_status: str

    returned_model_id: str | None = None
    local_model_digest: str | None = None
    provider_response_id: str | None = None

    raw_output: str | None = None
    input_sha256: str | None = None

    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None

    latency_seconds: float | None = None
    retry_count: int = 0

    temperature: float | None = None
    max_output_tokens: int | None = None
    reasoning_setting: str | None = None
    structured_output_mode: str | None = None

    cost_usd: float | None = None
    pricing_config_id: str | None = None
    service_tier: str | None = None

    request_started_utc: str | None = None
    response_received_utc: str | None = None

    error_type: str | None = None
    error_message: str | None = None

    provider_metadata: dict[str, Any] | None = None
    schema_version: str = INFERENCE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
