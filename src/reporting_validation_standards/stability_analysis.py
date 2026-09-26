from __future__ import annotations

import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


INVALID_LABEL = "__INVALID_OUTPUT__"


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_inference_jsonl(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Inference records not found: {path}")

    frame = pd.read_json(path, lines=True, dtype=False)
    required = {
        "observation_id",
        "model_key",
        "prompt_id",
        "repetition",
        "predicted_label",
        "parse_status",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    frame["observation_id"] = frame["observation_id"].astype(str)
    frame["model_key"] = frame["model_key"].astype(str)
    frame["prompt_id"] = frame["prompt_id"].astype(str)
    frame["repetition"] = frame["repetition"].astype(int)
    frame["is_valid"] = (
        frame["parse_status"].eq("ok")
        & frame["predicted_label"].notna()
    )
    frame["analysis_label"] = frame["predicted_label"].where(
        frame["is_valid"],
        INVALID_LABEL,
    )
    return frame


def weighted_ratio(
    numerator: Iterable[float] | pd.Series | np.ndarray,
    denominator: Iterable[float] | pd.Series | np.ndarray,
    weights: Iterable[float] | pd.Series | np.ndarray,
) -> float | None:
    numerator_array = np.asarray(numerator, dtype=float)
    denominator_array = np.asarray(denominator, dtype=float)
    weight_array = np.asarray(weights, dtype=float)

    if not (
        numerator_array.shape
        == denominator_array.shape
        == weight_array.shape
    ):
        raise ValueError("numerator, denominator, and weights must have equal shape.")

    mask = (
        np.isfinite(numerator_array)
        & np.isfinite(denominator_array)
        & np.isfinite(weight_array)
    )
    if not mask.any():
        return None

    weighted_denominator = np.sum(weight_array[mask] * denominator_array[mask])
    if weighted_denominator <= 0:
        return None

    return float(
        np.sum(weight_array[mask] * numerator_array[mask])
        / weighted_denominator
    )


def unweighted_ratio(
    numerator: Iterable[float] | pd.Series | np.ndarray,
    denominator: Iterable[float] | pd.Series | np.ndarray,
) -> float | None:
    numerator_array = np.asarray(numerator, dtype=float)
    denominator_array = np.asarray(denominator, dtype=float)
    if numerator_array.shape != denominator_array.shape:
        raise ValueError("numerator and denominator must have equal shape.")

    mask = np.isfinite(numerator_array) & np.isfinite(denominator_array)
    if not mask.any():
        return None

    denominator_total = np.sum(denominator_array[mask])
    if denominator_total <= 0:
        return None

    return float(np.sum(numerator_array[mask]) / denominator_total)


def weighted_mean(
    values: Iterable[float] | pd.Series | np.ndarray,
    weights: Iterable[float] | pd.Series | np.ndarray,
) -> float | None:
    values_array = np.asarray(values, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    if values_array.shape != weight_array.shape:
        raise ValueError("values and weights must have equal shape.")

    mask = np.isfinite(values_array) & np.isfinite(weight_array)
    if not mask.any():
        return None

    denominator = np.sum(weight_array[mask])
    if denominator <= 0:
        return None

    return float(np.sum(values_array[mask] * weight_array[mask]) / denominator)


def entropy_bits(labels: Iterable[str]) -> float | None:
    labels_list = [str(label) for label in labels if str(label) != INVALID_LABEL]
    if not labels_list:
        return None

    counts = np.asarray(list(Counter(labels_list).values()), dtype=float)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def modal_label_and_share(labels: Iterable[str]) -> tuple[str | None, float | None]:
    labels_list = [str(label) for label in labels if str(label) != INVALID_LABEL]
    if not labels_list:
        return None, None

    counts = Counter(labels_list)
    top_count = max(counts.values())
    top_labels = sorted(label for label, count in counts.items() if count == top_count)
    modal_label = top_labels[0] if len(top_labels) == 1 else None
    return modal_label, float(top_count / len(labels_list))


def prepare_records(
    records: pd.DataFrame,
    sample_design: pd.DataFrame,
) -> pd.DataFrame:
    design = sample_design.copy()
    design["id"] = design["id"].astype(str)
    required_design = {
        "id",
        "predicted_label",
        "sampling_weight",
        "inclusion_probability",
        "stratum_population_n",
        "stratum_sample_n",
    }
    missing_design = sorted(required_design - set(design.columns))
    if missing_design:
        raise ValueError(
            f"Sample design is missing required columns: {missing_design}"
        )

    merged = records.merge(
        design[
            [
                "id",
                "predicted_label",
                "sampling_weight",
                "inclusion_probability",
                "stratum_population_n",
                "stratum_sample_n",
            ]
        ].rename(
            columns={
                "id": "observation_id",
                "predicted_label": "reference_label",
            }
        ),
        on="observation_id",
        how="left",
        validate="many_to_one",
    )
    if merged["sampling_weight"].isna().any():
        missing_ids = (
            merged.loc[merged["sampling_weight"].isna(), "observation_id"]
            .drop_duplicates()
            .head(5)
            .tolist()
        )
        raise ValueError(
            "Inference records contain IDs outside the frozen sample design: "
            f"{missing_ids}"
        )
    return merged


def _pairwise_agreement_complete(labels: list[str]) -> float | None:
    if any(label == INVALID_LABEL for label in labels):
        return None
    if len(labels) < 2:
        return None

    equal = 0
    total = 0
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            total += 1
            equal += int(labels[left] == labels[right])
    return equal / total if total else None


def build_repeat_triads(
    records: pd.DataFrame,
    *,
    expected_repetitions: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["model_key", "prompt_id", "observation_id"]
    for keys, group in records.groupby(group_cols, sort=False):
        model_key, prompt_id, observation_id = keys
        group = group.sort_values("repetition")
        labels = group["analysis_label"].astype(str).tolist()
        valid_count = int(group["is_valid"].sum())
        complete = int(
            len(group) == expected_repetitions
            and valid_count == expected_repetitions
        )
        unanimous = int(complete and len(set(labels)) == 1)
        rows.append(
            {
                "model_key": model_key,
                "prompt_id": prompt_id,
                "observation_id": observation_id,
                "reference_label": str(group["reference_label"].iloc[0]),
                "sampling_weight": float(group["sampling_weight"].iloc[0]),
                "n_records": len(group),
                "valid_count": valid_count,
                "complete": complete,
                "unanimous": unanimous,
                "pairwise_agreement": (
                    _pairwise_agreement_complete(labels) if complete else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def build_prompt_quads(
    records: pd.DataFrame,
    *,
    expected_prompts: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["model_key", "repetition", "observation_id"]
    for keys, group in records.groupby(group_cols, sort=False):
        model_key, repetition, observation_id = keys
        labels = group["analysis_label"].astype(str).tolist()
        valid_count = int(group["is_valid"].sum())
        complete = int(
            len(group) == expected_prompts
            and valid_count == expected_prompts
        )
        unanimous = int(complete and len(set(labels)) == 1)
        rows.append(
            {
                "model_key": model_key,
                "repetition": int(repetition),
                "observation_id": observation_id,
                "reference_label": str(group["reference_label"].iloc[0]),
                "sampling_weight": float(group["sampling_weight"].iloc[0]),
                "n_records": len(group),
                "valid_count": valid_count,
                "complete": complete,
                "unanimous": unanimous,
            }
        )
    return pd.DataFrame(rows)


def build_observation_model_summary(
    records: pd.DataFrame,
    *,
    expected_decisions_per_observation: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["model_key", "observation_id"]
    for keys, group in records.groupby(group_cols, sort=False):
        model_key, observation_id = keys
        valid = group.loc[group["is_valid"], "analysis_label"].astype(str).tolist()
        complete = int(
            len(group) == expected_decisions_per_observation
            and len(valid) == expected_decisions_per_observation
        )
        unique_labels = len(set(valid)) if valid else 0
        modal_label, modal_share = modal_label_and_share(valid)
        rows.append(
            {
                "model_key": model_key,
                "observation_id": observation_id,
                "reference_label": str(group["reference_label"].iloc[0]),
                "sampling_weight": float(group["sampling_weight"].iloc[0]),
                "n_records": len(group),
                "valid_decisions": len(valid),
                "valid_rate": (
                    len(valid) / expected_decisions_per_observation
                    if expected_decisions_per_observation
                    else np.nan
                ),
                "all12_complete": complete,
                "all12_unanimous": int(complete and unique_labels == 1),
                "unique_valid_labels": unique_labels,
                "modal_label": modal_label,
                "modal_share_valid": modal_share,
                "entropy_bits_valid": entropy_bits(valid),
            }
        )
    return pd.DataFrame(rows)


def summarize_model(
    *,
    model_key: str,
    decisions: pd.DataFrame,
    repeat_triads: pd.DataFrame,
    prompt_quads: pd.DataFrame,
    observation_summary: pd.DataFrame,
) -> dict[str, Any]:
    d = decisions.loc[decisions["model_key"].eq(model_key)]
    r = repeat_triads.loc[repeat_triads["model_key"].eq(model_key)]
    p = prompt_quads.loc[prompt_quads["model_key"].eq(model_key)]
    o = observation_summary.loc[observation_summary["model_key"].eq(model_key)]

    valid_num = d["is_valid"].astype(int)
    valid_den = np.ones(len(d), dtype=float)
    repeat_complete = r["complete"].astype(int)
    repeat_den = np.ones(len(r), dtype=float)
    prompt_complete = p["complete"].astype(int)
    prompt_den = np.ones(len(p), dtype=float)

    return {
        "model_key": model_key,
        "n_decisions": len(d),
        "n_valid_decisions": int(d["is_valid"].sum()),
        "valid_decision_rate_unweighted": unweighted_ratio(valid_num, valid_den),
        "valid_decision_rate_weighted": weighted_ratio(
            valid_num,
            valid_den,
            d["sampling_weight"],
        ),
        "repeat_complete_rate_unweighted": unweighted_ratio(
            repeat_complete,
            repeat_den,
        ),
        "repeat_complete_rate_weighted": weighted_ratio(
            repeat_complete,
            repeat_den,
            r["sampling_weight"],
        ),
        "repeat_unanimity_unconditional_unweighted": unweighted_ratio(
            r["unanimous"],
            repeat_den,
        ),
        "repeat_unanimity_unconditional_weighted": weighted_ratio(
            r["unanimous"],
            repeat_den,
            r["sampling_weight"],
        ),
        "repeat_unanimity_given_complete_unweighted": unweighted_ratio(
            r["unanimous"],
            repeat_complete,
        ),
        "repeat_unanimity_given_complete_weighted": weighted_ratio(
            r["unanimous"],
            repeat_complete,
            r["sampling_weight"],
        ),
        "repeat_pairwise_agreement_given_complete_unweighted": (
            float(r["pairwise_agreement"].mean())
            if r["pairwise_agreement"].notna().any()
            else None
        ),
        "repeat_pairwise_agreement_given_complete_weighted": weighted_mean(
            r["pairwise_agreement"],
            r["sampling_weight"],
        ),
        "prompt_complete_rate_unweighted": unweighted_ratio(
            prompt_complete,
            prompt_den,
        ),
        "prompt_complete_rate_weighted": weighted_ratio(
            prompt_complete,
            prompt_den,
            p["sampling_weight"],
        ),
        "four_prompt_unanimity_unconditional_unweighted": unweighted_ratio(
            p["unanimous"],
            prompt_den,
        ),
        "four_prompt_unanimity_unconditional_weighted": weighted_ratio(
            p["unanimous"],
            prompt_den,
            p["sampling_weight"],
        ),
        "four_prompt_unanimity_given_complete_unweighted": unweighted_ratio(
            p["unanimous"],
            prompt_complete,
        ),
        "four_prompt_unanimity_given_complete_weighted": weighted_ratio(
            p["unanimous"],
            prompt_complete,
            p["sampling_weight"],
        ),
        "all12_complete_rate_unweighted": float(o["all12_complete"].mean()),
        "all12_complete_rate_weighted": weighted_mean(
            o["all12_complete"],
            o["sampling_weight"],
        ),
        "all12_unanimity_unconditional_unweighted": float(
            o["all12_unanimous"].mean()
        ),
        "all12_unanimity_unconditional_weighted": weighted_mean(
            o["all12_unanimous"],
            o["sampling_weight"],
        ),
        "all12_unanimity_given_complete_unweighted": unweighted_ratio(
            o["all12_unanimous"],
            o["all12_complete"],
        ),
        "all12_unanimity_given_complete_weighted": weighted_ratio(
            o["all12_unanimous"],
            o["all12_complete"],
            o["sampling_weight"],
        ),
        "mean_unique_labels_if_complete_unweighted": (
            float(
                o.loc[o["all12_complete"].eq(1), "unique_valid_labels"].mean()
            )
            if o["all12_complete"].any()
            else None
        ),
        "mean_unique_labels_if_complete_weighted": weighted_mean(
            o["unique_valid_labels"].where(o["all12_complete"].eq(1)),
            o["sampling_weight"],
        ),
        "mean_modal_share_if_complete_unweighted": (
            float(
                o.loc[o["all12_complete"].eq(1), "modal_share_valid"].mean()
            )
            if o["all12_complete"].any()
            else None
        ),
        "mean_modal_share_if_complete_weighted": weighted_mean(
            o["modal_share_valid"].where(o["all12_complete"].eq(1)),
            o["sampling_weight"],
        ),
        "mean_entropy_bits_if_complete_unweighted": (
            float(
                o.loc[o["all12_complete"].eq(1), "entropy_bits_valid"].mean()
            )
            if o["all12_complete"].any()
            else None
        ),
        "mean_entropy_bits_if_complete_weighted": weighted_mean(
            o["entropy_bits_valid"].where(o["all12_complete"].eq(1)),
            o["sampling_weight"],
        ),
    }


def summarize_repeat_by_prompt(
    repeat_triads: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_key, prompt_id), group in repeat_triads.groupby(
        ["model_key", "prompt_id"],
        sort=False,
    ):
        denominator = np.ones(len(group), dtype=float)
        rows.append(
            {
                "model_key": model_key,
                "prompt_id": prompt_id,
                "n_observations": len(group),
                "complete_rate_unweighted": unweighted_ratio(
                    group["complete"],
                    denominator,
                ),
                "complete_rate_weighted": weighted_ratio(
                    group["complete"],
                    denominator,
                    group["sampling_weight"],
                ),
                "unanimity_unconditional_unweighted": unweighted_ratio(
                    group["unanimous"],
                    denominator,
                ),
                "unanimity_unconditional_weighted": weighted_ratio(
                    group["unanimous"],
                    denominator,
                    group["sampling_weight"],
                ),
                "unanimity_given_complete_unweighted": unweighted_ratio(
                    group["unanimous"],
                    group["complete"],
                ),
                "unanimity_given_complete_weighted": weighted_ratio(
                    group["unanimous"],
                    group["complete"],
                    group["sampling_weight"],
                ),
                "pairwise_agreement_given_complete_unweighted": (
                    float(group["pairwise_agreement"].mean())
                    if group["pairwise_agreement"].notna().any()
                    else None
                ),
                "pairwise_agreement_given_complete_weighted": weighted_mean(
                    group["pairwise_agreement"],
                    group["sampling_weight"],
                ),
            }
        )
    return pd.DataFrame(rows)


def build_prompt_comparisons(
    records: pd.DataFrame,
    *,
    canonical_prompt_id: str,
    perturbation_prompt_ids: list[str],
) -> pd.DataFrame:
    key = ["model_key", "observation_id", "repetition"]

    base = records.loc[
        records["prompt_id"].eq(canonical_prompt_id),
        key + ["analysis_label", "is_valid", "sampling_weight"],
    ].rename(
        columns={
            "analysis_label": "canonical_label",
            "is_valid": "canonical_valid",
        }
    )

    rows: list[dict[str, Any]] = []
    for prompt_id in perturbation_prompt_ids:
        other = records.loc[
            records["prompt_id"].eq(prompt_id),
            key + ["analysis_label", "is_valid"],
        ].rename(
            columns={
                "analysis_label": "perturbation_label",
                "is_valid": "perturbation_valid",
            }
        )
        merged = base.merge(other, on=key, how="inner", validate="one_to_one")
        merged["pair_valid"] = (
            merged["canonical_valid"] & merged["perturbation_valid"]
        ).astype(int)
        merged["match_and_valid"] = (
            merged["pair_valid"].eq(1)
            & merged["canonical_label"].eq(merged["perturbation_label"])
        ).astype(int)

        for model_key, group in merged.groupby("model_key", sort=False):
            denominator = np.ones(len(group), dtype=float)
            rows.append(
                {
                    "model_key": model_key,
                    "canonical_prompt_id": canonical_prompt_id,
                    "perturbation_prompt_id": prompt_id,
                    "n_decision_pairs": len(group),
                    "paired_valid_rate_unweighted": unweighted_ratio(
                        group["pair_valid"],
                        denominator,
                    ),
                    "paired_valid_rate_weighted": weighted_ratio(
                        group["pair_valid"],
                        denominator,
                        group["sampling_weight"],
                    ),
                    "exact_match_unconditional_unweighted": unweighted_ratio(
                        group["match_and_valid"],
                        denominator,
                    ),
                    "exact_match_unconditional_weighted": weighted_ratio(
                        group["match_and_valid"],
                        denominator,
                        group["sampling_weight"],
                    ),
                    "exact_match_given_valid_unweighted": unweighted_ratio(
                        group["match_and_valid"],
                        group["pair_valid"],
                    ),
                    "exact_match_given_valid_weighted": weighted_ratio(
                        group["match_and_valid"],
                        group["pair_valid"],
                        group["sampling_weight"],
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_model_pair_agreement(
    records: pd.DataFrame,
    *,
    model_keys: list[str],
    canonical_prompt_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, scoped in (
        (
            "canonical_prompt",
            records.loc[records["prompt_id"].eq(canonical_prompt_id)],
        ),
        ("all_prompts", records),
    ):
        key = ["observation_id", "prompt_id", "repetition"]
        lookup: dict[str, pd.DataFrame] = {}
        for model_key in model_keys:
            lookup[model_key] = scoped.loc[
                scoped["model_key"].eq(model_key),
                key + ["analysis_label", "is_valid", "sampling_weight"],
            ].rename(
                columns={
                    "analysis_label": f"label_{model_key}",
                    "is_valid": f"valid_{model_key}",
                    "sampling_weight": "sampling_weight",
                }
            )

        for left_index, left in enumerate(model_keys):
            for right in model_keys[left_index + 1 :]:
                merged = lookup[left].merge(
                    lookup[right].drop(columns="sampling_weight"),
                    on=key,
                    how="inner",
                    validate="one_to_one",
                )
                pair_valid = (
                    merged[f"valid_{left}"] & merged[f"valid_{right}"]
                ).astype(int)
                match_and_valid = (
                    pair_valid.eq(1)
                    & merged[f"label_{left}"].eq(merged[f"label_{right}"])
                ).astype(int)
                denominator = np.ones(len(merged), dtype=float)

                rows.append(
                    {
                        "scope": scope,
                        "model_key_left": left,
                        "model_key_right": right,
                        "n_decision_pairs": len(merged),
                        "paired_valid_rate_unweighted": unweighted_ratio(
                            pair_valid,
                            denominator,
                        ),
                        "paired_valid_rate_weighted": weighted_ratio(
                            pair_valid,
                            denominator,
                            merged["sampling_weight"],
                        ),
                        "agreement_unconditional_unweighted": unweighted_ratio(
                            match_and_valid,
                            denominator,
                        ),
                        "agreement_unconditional_weighted": weighted_ratio(
                            match_and_valid,
                            denominator,
                            merged["sampling_weight"],
                        ),
                        "agreement_given_valid_unweighted": unweighted_ratio(
                            match_and_valid,
                            pair_valid,
                        ),
                        "agreement_given_valid_weighted": weighted_ratio(
                            match_and_valid,
                            pair_valid,
                            merged["sampling_weight"],
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_panel_agreement(
    records: pd.DataFrame,
    *,
    panels: dict[str, list[str]],
    canonical_prompt_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for scope, scoped in (
        (
            "canonical_prompt",
            records.loc[records["prompt_id"].eq(canonical_prompt_id)],
        ),
        ("all_prompts", records),
    ):
        key = ["observation_id", "prompt_id", "repetition"]
        for panel_name, model_keys in panels.items():
            panel = scoped.loc[scoped["model_key"].isin(model_keys)].copy()
            grouped_rows: list[dict[str, Any]] = []

            for cell_key, group in panel.groupby(key, sort=False):
                observation_id, prompt_id, repetition = cell_key
                labels = group["analysis_label"].astype(str).tolist()
                valid = group["is_valid"].astype(bool).tolist()
                complete = int(
                    len(group) == len(model_keys)
                    and all(valid)
                )
                unanimous = int(
                    complete
                    and len(set(labels)) == 1
                )
                grouped_rows.append(
                    {
                        "observation_id": observation_id,
                        "prompt_id": prompt_id,
                        "repetition": int(repetition),
                        "sampling_weight": float(group["sampling_weight"].iloc[0]),
                        "complete": complete,
                        "unanimous": unanimous,
                    }
                )

            cell_frame = pd.DataFrame(grouped_rows)
            denominator = np.ones(len(cell_frame), dtype=float)
            rows.append(
                {
                    "panel": panel_name,
                    "scope": scope,
                    "n_models": len(model_keys),
                    "n_cells": len(cell_frame),
                    "complete_rate_unweighted": unweighted_ratio(
                        cell_frame["complete"],
                        denominator,
                    ),
                    "complete_rate_weighted": weighted_ratio(
                        cell_frame["complete"],
                        denominator,
                        cell_frame["sampling_weight"],
                    ),
                    "unanimity_unconditional_unweighted": unweighted_ratio(
                        cell_frame["unanimous"],
                        denominator,
                    ),
                    "unanimity_unconditional_weighted": weighted_ratio(
                        cell_frame["unanimous"],
                        denominator,
                        cell_frame["sampling_weight"],
                    ),
                    "unanimity_given_complete_unweighted": unweighted_ratio(
                        cell_frame["unanimous"],
                        cell_frame["complete"],
                    ),
                    "unanimity_given_complete_weighted": weighted_ratio(
                        cell_frame["unanimous"],
                        cell_frame["complete"],
                        cell_frame["sampling_weight"],
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_by_reference_label(
    *,
    records: pd.DataFrame,
    observation_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model_keys = records["model_key"].drop_duplicates().tolist()

    for model_key in model_keys:
        decision_model = records.loc[records["model_key"].eq(model_key)]
        obs_model = observation_summary.loc[
            observation_summary["model_key"].eq(model_key)
        ]

        labels = sorted(decision_model["reference_label"].drop_duplicates())
        for label in labels:
            d = decision_model.loc[decision_model["reference_label"].eq(label)]
            o = obs_model.loc[obs_model["reference_label"].eq(label)]
            denominator = np.ones(len(d), dtype=float)
            rows.append(
                {
                    "model_key": model_key,
                    "reference_label": label,
                    "sample_n": int(o["observation_id"].nunique()),
                    "decision_n": len(d),
                    "valid_decision_rate": unweighted_ratio(
                        d["is_valid"].astype(int),
                        denominator,
                    ),
                    "all12_complete_rate": (
                        float(o["all12_complete"].mean()) if len(o) else None
                    ),
                    "all12_unanimity_unconditional": (
                        float(o["all12_unanimous"].mean()) if len(o) else None
                    ),
                    "mean_unique_labels_if_complete": (
                        float(
                            o.loc[
                                o["all12_complete"].eq(1),
                                "unique_valid_labels",
                            ].mean()
                        )
                        if o["all12_complete"].any()
                        else None
                    ),
                    "mean_entropy_bits_if_complete": (
                        float(
                            o.loc[
                                o["all12_complete"].eq(1),
                                "entropy_bits_valid",
                            ].mean()
                        )
                        if o["all12_complete"].any()
                        else None
                    ),
                }
            )

    return pd.DataFrame(rows)
