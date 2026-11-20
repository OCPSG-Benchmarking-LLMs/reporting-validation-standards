"""Validation performance across prompt specifications and repeated inference.

Reads the committed outputs/validation/condition_performance.csv and draws a 2 x 2 figure
(Accuracy, Macro-Precision, Macro-Recall, Macro-F1) with one row per model x prompt and
the three single-repetition estimates as separate points.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from reporting_validation_standards import plot_style as ps

INPUT_FILE = Path("outputs/validation/condition_performance.csv")
OUTPUT_STEM = Path("outputs/figures/validation_prompt_repetition_profile")

MODEL_NAMES = {
    "openai_gpt54": "GPT-5.4",
    "openai_gpt54_mini": "GPT-5.4-mini",
    "fireworks_deepseek_v41_flash": "DeepSeek V4.1 Flash",
    "ollama_qwen36_35b": "Qwen3.6 (35B)",
    "ollama_gemma4_31b": "Gemma 4 (31B)",
    "openai_gpt6_luna": "GPT-6 Luna",
    "mistral_large3_2512": "Mistral Large 3",
}
MODEL_ORDER = list(MODEL_NAMES)

PROMPT_NAMES = {
    "p0_capbench_canonical": "P0 Canonical",
    "p1_lexical_paraphrase": "P1 Lexical paraphrase",
    "p2_label_order_permutation": "P2 Label-order permutation",
    "p3_structural_format": "P3 Structural format",
}
PROMPT_ORDER = list(PROMPT_NAMES)
PROMPT_COLOURS = dict(zip(PROMPT_ORDER, ps.SERIES[:4], strict=True))

# The four prompt rows of a model sit around its integer position; the three
# repetitions of a prompt share one row.
PROMPT_OFFSETS = dict(zip(PROMPT_ORDER, (-0.3, -0.1, 0.1, 0.3), strict=True))

METRICS = {
    "Accuracy": "weighted_operational_accuracy",
    "Macro-Precision": "weighted_operational_macro_precision",
    "Macro-Recall": "weighted_operational_macro_recall",
    "Macro-F1": "weighted_operational_macro_f1",
}

X_LIMITS = (0.7, 1.0)
X_TICKS = (0.7, 0.8, 0.9, 1.0)
FIGSIZE = (ps.PRINT_WIDTH_IN, 7.0)
POINT_AREA = 22

CAPTION = (
    "Validation Performance across Prompt Specifications and Repeated Inference\n\n"
    "Note. Each panel reports a design-weighted operational validation metric against the "
    "OCPSG silver reference: Accuracy, Macro-Precision, Macro-Recall and Macro-F1. Within "
    "each model group, the four frozen prompt conditions occupy separate horizontal lines "
    "and are distinguished by colour. The three points on each prompt line report the three "
    "single-repetition estimates. Models are shown in the fixed experimental order rather "
    "than sorted by performance, and the x-axis keeps the original 0–1 metric scale. The "
    "figure complements, rather than replaces, the full stability and validation tables."
)


def prepare(data: pd.DataFrame, source: Path = INPUT_FILE) -> pd.DataFrame:
    """Keep the 12 single-repetition estimates per model and check completeness.

    condition_performance.csv holds four rows per model x prompt: repetitions 1-3 are
    single-repetition estimates, while repetition 0 is the expected-over-repetitions
    aggregate and must not be drawn as a fourth repetition.
    """
    required = {"model_key", "prompt_id", "repetition", "aggregation", *METRICS.values()}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")

    data = data.loc[
        data["aggregation"].astype(str).eq("single_repetition")
        & data["repetition"].isin([1, 2, 3])
    ].copy()

    models = set(data["model_key"].astype(str))
    if models != set(MODEL_ORDER):
        raise ValueError(
            f"Expected models {sorted(MODEL_ORDER)}; found {sorted(models)}."
        )
    prompts = set(data["prompt_id"].astype(str))
    if prompts != set(PROMPT_ORDER):
        raise ValueError(
            "Expected exactly the four frozen prompt conditions; "
            f"found {sorted(prompts)}."
        )
    data["repetition"] = pd.to_numeric(data["repetition"], errors="raise").astype(int)
    if sorted(data["repetition"].unique().tolist()) != [1, 2, 3]:
        raise ValueError("Expected single-repetition rows 1, 2 and 3 after filtering.")

    for column in METRICS.values():
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[column].isna().any():
            raise ValueError(f"{source} contains missing values in {column}.")
        outside = data[column].lt(X_LIMITS[0]) | data[column].gt(X_LIMITS[1])
        if outside.any():
            raise ValueError(
                f"{column} has values outside the plotted range {X_LIMITS}; widen X_LIMITS."
            )

    expected_rows = len(MODEL_ORDER) * len(PROMPT_ORDER) * 3
    if len(data) != expected_rows:
        raise ValueError(
            "Single-repetition validation output is incomplete. "
            f"Expected {expected_rows} rows, found {len(data)}."
        )
    if data.duplicated(subset=["model_key", "prompt_id", "repetition"]).any():
        raise ValueError(f"Duplicate model x prompt x repetition rows found in {source}.")

    model_to_y = {model: index for index, model in enumerate(MODEL_ORDER)}
    data["y"] = [
        model_to_y[model] + PROMPT_OFFSETS[prompt]
        for model, prompt in zip(data["model_key"], data["prompt_id"], strict=True)
    ]
    return data


def build_figure(data: pd.DataFrame):
    ps.use_style()
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=True)

    for ax, (title, column) in zip(axes.flat, METRICS.items(), strict=True):
        for prompt in PROMPT_ORDER:
            rows = data.loc[data["prompt_id"].eq(prompt)]
            ax.scatter(
                rows[column], rows["y"], s=POINT_AREA, color=PROMPT_COLOURS[prompt],
                alpha=0.85, edgecolors=ps.EDGE, linewidths=0.45, zorder=2,
            )
        ax.set_title(title)
        ax.set_xlim(X_LIMITS[0] - 0.01, X_LIMITS[1] + 0.01)
        ax.set_xticks(X_TICKS, [f"{tick:.1f}" for tick in X_TICKS])
        ax.set_yticks(range(len(MODEL_ORDER)), [MODEL_NAMES[m] for m in MODEL_ORDER])
        ax.set_ylim(len(MODEL_ORDER) - 0.5, -0.5)   # GPT-5.4 at the top
        ps.clean_axes(ax)
        ax.tick_params(axis="x", labelbottom=True)

    handles = [
        Line2D(
            [], [], marker="o", linestyle="", markersize=5.5,
            markerfacecolor=PROMPT_COLOURS[prompt], markeredgecolor=ps.EDGE,
            markeredgewidth=0.45, label=PROMPT_NAMES[prompt],
        )
        for prompt in PROMPT_ORDER
    ]
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=1.6, w_pad=3.0)
    ps.bottom_legend(fig, handles, ncol=2, y=0.055)
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output-stem", type=Path, default=OUTPUT_STEM)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = prepare(pd.read_csv(args.input, encoding="utf-8-sig"), args.input)
    fig = build_figure(data)
    paths = ps.save(fig, args.output_stem, CAPTION)
    plt.close(fig)
    print(
        f"Single-repetition rows: {len(data):,} "
        f"({len(MODEL_ORDER)} models x {len(PROMPT_ORDER)} prompts x 3 repetitions)"
    )
    for path in paths:
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
