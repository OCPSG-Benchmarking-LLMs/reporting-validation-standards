#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from reporting_validation_standards.experiment_specs import (
    load_model_config,
    load_prompt_config,
    render_prompt,
)


def main() -> None:
    models = load_model_config(Path("config/models.yaml"))
    prompts = load_prompt_config(Path("config/prompts.yaml"))

    print("Frozen model panel")
    print("------------------")
    for model in models["models"]:
        pin = "pinned" if model["snapshot_pinned"] else "moving/provider-managed"
        print(
            f'{model["model_key"]}: {model["model_id"]} '
            f'[{model["provider"]}; {model["stage"]}; {pin}]'
        )

    print("\nFrozen prompt variants")
    print("----------------------")
    hashes: dict[str, str] = {}
    for prompt in prompts["prompts"]:
        rendered = render_prompt(
            prompt_id=prompt["prompt_id"],
            text="__VALIDATION_PLACEHOLDER__",
            path=Path("config/prompts.yaml"),
        )
        hashes[prompt["prompt_id"]] = rendered.prompt_spec_sha256
        print(
            f'{prompt["prompt_id"]}: {prompt["perturbation"]} '
            f'[{rendered.prompt_spec_sha256}]'
        )

    print("\nPrompt-spec hashes")
    print("------------------")
    print(json.dumps(hashes, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
