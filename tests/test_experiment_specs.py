from pathlib import Path

from reporting_validation_standards.experiment_specs import (
    load_model_config,
    load_prompt_config,
    render_prompt,
)


PROMPTS = Path("config/prompts.yaml")
MODELS = Path("config/models.yaml")


def test_model_panel_is_unique_and_has_expected_size():
    config = load_model_config(MODELS)
    models = config["models"]

    assert len(models) == 7
    assert len({model["model_key"] for model in models}) == 7
    assert sum(model["stage"] == "core" for model in models) == 5
    assert sum(model["stage"] == "secondary" for model in models) == 1
    assert sum(model["stage"] == "supplementary" for model in models) == 1


def test_all_prompt_variants_use_same_label_inventory():
    config = load_prompt_config(PROMPTS)
    official = set(config["official_labels"])

    assert len(config["prompts"]) == 4

    for prompt in config["prompts"]:
        order = prompt["label_order"]
        labels = config["official_labels"] if order == "canonical" else order
        assert set(labels) == official
        assert len(labels) == len(official)


def test_p0_renders_capbench_reference_wording():
    rendered = render_prompt(
        prompt_id="p0_capbench_canonical",
        text="Example speech.",
        path=PROMPTS,
    )

    assert rendered.system_prompt.startswith(
        "You are classifying parliamentary speeches into one major policy agenda label.\n"
    )
    assert 'The JSON must contain exactly one key: "predicted_label".' in rendered.system_prompt
    assert rendered.user_prompt == (
        "Classify the following parliamentary speech into exactly one official benchmark label.\n"
        "\n"
        "Speech:\n"
        "Example speech.\n"
    )


def test_p2_is_a_noncanonical_fixed_permutation():
    config = load_prompt_config(PROMPTS)
    p2 = next(
        prompt for prompt in config["prompts"]
        if prompt["prompt_id"] == "p2_label_order_permutation"
    )

    assert p2["label_order_seed"] == 20260923
    assert p2["label_order"] != config["official_labels"]
    assert set(p2["label_order"]) == set(config["official_labels"])


def test_fireworks_reasoning_is_explicitly_disabled():
    config = load_model_config(MODELS)
    model = next(
        model for model in config["models"]
        if model["model_key"] == "fireworks_deepseek_v41_flash"
    )

    assert model["reasoning_setting"] == "not_exposed_for_model"


def test_local_models_freeze_runtime_context():
    config = load_model_config(MODELS)
    local_models = [
        model for model in config["models"]
        if model["provider"] == "ollama"
    ]

    assert local_models
    assert all(model["runtime_context_length"] == 8192 for model in local_models)


def test_local_models_freeze_exact_digests():
    config = load_model_config(MODELS)
    lookup = {model["model_key"]: model for model in config["models"]}

    assert (
        lookup["ollama_qwen36_35b"]["expected_local_digest"]
        == "096fdbd02fe620fc10cbeb6537e080f8041aece851e5d696aed024d4f70f2e47"
    )
    assert (
        lookup["ollama_gemma4_31b"]["expected_local_digest"]
        == "6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7"
    )


def test_mistral_extension_is_snapshot_pinned():
    config = load_model_config(MODELS)
    model = next(
        model for model in config["models"]
        if model["model_key"] == "mistral_large3_2512"
    )

    assert model["provider"] == "mistral"
    assert model["model_id"] == "mistral-large-2512"
    assert model["stage"] == "supplementary"
    assert model["snapshot_pinned"] is True
    assert model["reasoning_setting"] == "none"
    assert model["extension_status"] == "post_production_robustness"
