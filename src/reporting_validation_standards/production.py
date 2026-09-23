from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


SCHEDULE_ALGORITHM_VERSION = "randomized_cells_pcg64_v1"


@dataclass(frozen=True)
class DecisionCell:
    observation_id: str
    prompt_id: str
    repetition: int

    @property
    def key(self) -> str:
        return decision_key(
            observation_id=self.observation_id,
            prompt_id=self.prompt_id,
            repetition=self.repetition,
        )


def decision_key(*, observation_id: str, prompt_id: str, repetition: int) -> str:
    return f"{prompt_id}\t{int(repetition)}\t{observation_id}"


def record_decision_key(record: dict) -> str:
    return decision_key(
        observation_id=str(record["observation_id"]),
        prompt_id=str(record["prompt_id"]),
        repetition=int(record["repetition"]),
    )


def make_stability_schedule(
    *,
    observation_ids: Sequence[str],
    prompt_ids: Sequence[str],
    repetitions: int,
    seed: int,
) -> list[DecisionCell]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    if not observation_ids:
        raise ValueError("observation_ids must not be empty.")
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty.")

    base = [
        DecisionCell(
            observation_id=str(observation_id),
            prompt_id=str(prompt_id),
            repetition=repetition,
        )
        for prompt_id in prompt_ids
        for repetition in range(1, repetitions + 1)
        for observation_id in observation_ids
    ]

    rng = np.random.Generator(np.random.PCG64(seed))
    permutation = rng.permutation(len(base))
    schedule = [base[int(index)] for index in permutation]

    keys = [cell.key for cell in schedule]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Generated production schedule contains duplicate decision cells.")

    return schedule


def schedule_sha256(schedule: Sequence[DecisionCell]) -> str:
    payload = [
        {
            "observation_id": cell.observation_id,
            "prompt_id": cell.prompt_id,
            "repetition": cell.repetition,
        }
        for cell in schedule
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prompt_hashes(
    *,
    prompt_ids: Sequence[str],
    representative_text: str = "__PROMPT_HASH_SENTINEL__",
) -> dict[str, str]:
    # Imported lazily so this module remains lightweight for schedule-only tests.
    from .experiment_specs import render_prompt

    return {
        prompt_id: render_prompt(
            prompt_id=prompt_id,
            text=representative_text,
        ).prompt_spec_sha256
        for prompt_id in prompt_ids
    }


def validate_existing_records(
    records: Iterable[dict],
    *,
    expected_model_key: str,
    expected_schedule_keys: set[str],
) -> dict[str, dict]:
    by_key: dict[str, dict] = {}

    for record in records:
        if record.get("model_key") != expected_model_key:
            raise RuntimeError(
                "Existing production records contain a different model_key: "
                f"{record.get('model_key')!r}."
            )
        key = record_decision_key(record)
        if key not in expected_schedule_keys:
            raise RuntimeError(
                f"Existing production record is outside the frozen schedule: {key!r}."
            )
        if key in by_key:
            raise RuntimeError(
                f"Duplicate decision cell found in existing production records: {key!r}."
            )
        by_key[key] = record

    return by_key
