import math

import pandas as pd

from reporting_validation_standards.stability_analysis import (
    build_observation_model_summary,
    build_prompt_comparisons,
    build_prompt_quads,
    build_repeat_triads,
    entropy_bits,
    modal_label_and_share,
    prepare_records,
    unweighted_ratio,
    weighted_mean,
    weighted_ratio,
)


def _records():
    rows = []
    prompts = ["p0_capbench_canonical", "p1_lexical_paraphrase"]
    predictions = {
        ("m1", "o1", "p0_capbench_canonical"): ["A", "A", "A"],
        ("m1", "o1", "p1_lexical_paraphrase"): ["A", "A", "B"],
        ("m1", "o2", "p0_capbench_canonical"): ["B", "B", "B"],
        ("m1", "o2", "p1_lexical_paraphrase"): ["B", None, "B"],
    }
    for (model, obs, prompt), labels in predictions.items():
        for repetition, label in enumerate(labels, start=1):
            rows.append(
                {
                    "observation_id": obs,
                    "model_key": model,
                    "prompt_id": prompt,
                    "repetition": repetition,
                    "predicted_label": label,
                    "parse_status": "ok" if label is not None else "error",
                    "is_valid": label is not None,
                    "analysis_label": label if label is not None else "__INVALID_OUTPUT__",
                }
            )
    return pd.DataFrame(rows)


def _design():
    return pd.DataFrame(
        {
            "id": ["o1", "o2"],
            "predicted_label": ["R1", "R2"],
            "sampling_weight": [1.0, 3.0],
            "inclusion_probability": [1.0, 1 / 3],
            "stratum_population_n": [1, 3],
            "stratum_sample_n": [1, 1],
        }
    )


def test_weighted_helpers():
    assert weighted_mean([0, 1], [1, 3]) == 0.75
    assert weighted_ratio([0, 1], [1, 1], [1, 3]) == 0.75
    assert unweighted_ratio([0, 1], [1, 1]) == 0.5


def test_entropy_and_modal_share():
    assert entropy_bits(["A", "A", "A"]) == 0.0
    assert math.isclose(entropy_bits(["A", "B"]), 1.0)
    label, share = modal_label_and_share(["A", "A", "B"])
    assert label == "A"
    assert math.isclose(share, 2 / 3)


def test_repeat_triads_preserve_failure_as_incomplete():
    records = prepare_records(_records(), _design())
    triads = build_repeat_triads(records, expected_repetitions=3)

    o1_p0 = triads.loc[
        triads["observation_id"].eq("o1")
        & triads["prompt_id"].eq("p0_capbench_canonical")
    ].iloc[0]
    assert o1_p0["complete"] == 1
    assert o1_p0["unanimous"] == 1
    assert o1_p0["pairwise_agreement"] == 1.0

    o2_p1 = triads.loc[
        triads["observation_id"].eq("o2")
        & triads["prompt_id"].eq("p1_lexical_paraphrase")
    ].iloc[0]
    assert o2_p1["complete"] == 0
    assert o2_p1["unanimous"] == 0
    assert pd.isna(o2_p1["pairwise_agreement"])


def test_prompt_quads_and_observation_summary():
    records = prepare_records(_records(), _design())
    quads = build_prompt_quads(records, expected_prompts=2)
    assert len(quads) == 6

    summary = build_observation_model_summary(
        records,
        expected_decisions_per_observation=6,
    )
    o1 = summary.loc[summary["observation_id"].eq("o1")].iloc[0]
    assert o1["all12_complete"] == 1
    assert o1["all12_unanimous"] == 0
    assert o1["unique_valid_labels"] == 2

    o2 = summary.loc[summary["observation_id"].eq("o2")].iloc[0]
    assert o2["all12_complete"] == 0
    assert o2["valid_decisions"] == 5


def test_prompt_comparison_reports_validity_and_match_separately():
    records = prepare_records(_records(), _design())
    comparisons = build_prompt_comparisons(
        records,
        canonical_prompt_id="p0_capbench_canonical",
        perturbation_prompt_ids=["p1_lexical_paraphrase"],
    )
    row = comparisons.iloc[0]

    assert row["n_decision_pairs"] == 6
    assert math.isclose(row["paired_valid_rate_unweighted"], 5 / 6)
    assert math.isclose(row["exact_match_unconditional_unweighted"], 4 / 6)
    assert math.isclose(row["exact_match_given_valid_unweighted"], 4 / 5)
