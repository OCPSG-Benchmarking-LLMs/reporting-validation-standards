from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_MODELS_PATH = Path("config/models.yaml")
DEFAULT_PROMPTS_PATH = Path("config/prompts.yaml")


@dataclass(frozen=True)
class RenderedPrompt:
    prompt_id: str
    system_prompt: str
    user_prompt: str
    prompt_spec_sha256: str


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return payload


def load_model_config(path: str | Path = DEFAULT_MODELS_PATH) -> dict[str, Any]:
    payload = _load_yaml(path)
    models = payload.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("Model configuration must contain a non-empty 'models' list.")

    keys = [model.get("model_key") for model in models]
    if any(not key for key in keys):
        raise ValueError("Every model must define model_key.")
    if len(keys) != len(set(keys)):
        raise ValueError("model_key values must be unique.")

    return payload


def load_prompt_config(path: str | Path = DEFAULT_PROMPTS_PATH) -> dict[str, Any]:
    payload = _load_yaml(path)
    labels = payload.get("official_labels")
    prompts = payload.get("prompts")

    if not isinstance(labels, list) or not labels:
        raise ValueError("Prompt configuration must contain official_labels.")
    if len(labels) != len(set(labels)):
        raise ValueError("official_labels contains duplicates.")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Prompt configuration must contain a non-empty 'prompts' list.")

    prompt_ids = [prompt.get("prompt_id") for prompt in prompts]
    if any(not prompt_id for prompt_id in prompt_ids):
        raise ValueError("Every prompt must define prompt_id.")
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("prompt_id values must be unique.")

    official_set = set(labels)
    for prompt in prompts:
        order = prompt.get("label_order")
        if order == "canonical":
            resolved = labels
        elif isinstance(order, list):
            resolved = order
        else:
            raise ValueError(
                f"Prompt {prompt.get('prompt_id')} has an invalid label_order."
            )

        if len(resolved) != len(labels) or set(resolved) != official_set:
            raise ValueError(
                f"Prompt {prompt.get('prompt_id')} must contain exactly the official label set."
            )

        system_template = prompt.get("system_template", "")
        user_template = prompt.get("user_template", "")
        if "{labels_block}" not in system_template:
            raise ValueError(
                f"Prompt {prompt.get('prompt_id')} system_template lacks {{labels_block}}."
            )
        if "{text}" not in user_template:
            raise ValueError(
                f"Prompt {prompt.get('prompt_id')} user_template lacks {{text}}."
            )

    return payload


def _stable_prompt_spec_hash(prompt: dict[str, Any], labels: list[str]) -> str:
    material = {
        "prompt_id": prompt["prompt_id"],
        "label_order": labels,
        "system_template": prompt["system_template"],
        "user_template": prompt["user_template"],
    }
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def render_prompt(
    *,
    prompt_id: str,
    text: str,
    path: str | Path = DEFAULT_PROMPTS_PATH,
) -> RenderedPrompt:
    payload = load_prompt_config(path)
    official_labels = payload["official_labels"]

    matches = [prompt for prompt in payload["prompts"] if prompt["prompt_id"] == prompt_id]
    if not matches:
        raise KeyError(f"Unknown prompt_id: {prompt_id}")
    prompt = matches[0]

    order = prompt["label_order"]
    labels = official_labels if order == "canonical" else order
    labels_block = "\n".join(f"- {label}" for label in labels)

    system_prompt = prompt["system_template"].format(labels_block=labels_block)
    user_prompt = prompt["user_template"].format(text=text)

    return RenderedPrompt(
        prompt_id=prompt_id,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        prompt_spec_sha256=_stable_prompt_spec_hash(prompt, labels),
    )
