"""Validation performance across prompt specifications and repeated inference.

Reads the committed outputs/validation/condition_performance.csv (single-repetition
estimates) and outputs/validation/prompt_performance_intervals.csv (repetition-averaged
estimates with 95% Rao-Wu bootstrap intervals, from script 08b) and draws a 2 x 2 figure
(Accuracy, Macro-Precision, Macro-Recall, Macro-F1). Each model x prompt row shows the
interval as a thin line, the range of the three single-repetition estimates as a thicker
bar and the averaged estimate as a circle.
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
INTERVAL_FILE = Path("outputs/validation/prompt_performance_intervals.csv")
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
# oxford-soft with sky aqua in place of light sage: even visual weight across the four
# prompts and a worst colour-vision-deficiency separation of 12.1 (CIEDE2000).
PROMPT_COLOURS = dict(zip(
    PROMPT_ORDER, (ps.BLUE, ps.TERRACOTTA, ps.SKY_AQUA, ps.PLUM), strict=True,
))

# The four prompt rows of a model sit around its integer position, 0.2 apart, which
# leaves a wider gap between models.
PROMPT_OFFSETS = dict(zip(PROMPT_ORDER, (-0.3, -0.1, 0.1, 0.3), strict=True))

METRICS = {
    "Accuracy": "weighted_operational_accuracy",
    "Macro-Precision": "weighted_operational_macro_precision",
    "Macro-Recall": "weighted_operational_macro_recall",
    "Macro-F1": "weighted_operational_macro_f1",
}

X_LIMITS = (0.7, 1.0)
X_TICKS = (0.7, 0.8, 0.9, 1.0)
LINEWIDTH_FRACTION = 0.8     # the manuscript sets this figure at 0.8\linewidth
FIGSIZE = (ps.print_width(LINEWIDTH_FRACTION), 6.0)
POINT_AREA = 20
CI_WIDTH = 1.0
RANGE_WIDTH = 4.5
RANGE_ALPHA = 0.45

CAPTION = (
    "Validation Performance across Prompt Specifications and Repeated Inference\n\n"
    "Note. Each panel reports a design-weighted operational validation metric against the "
    "OCPSG silver reference: Accuracy, Macro-Precision, Macro-Recall and Macro-F1. Within "
    "each model group, the four frozen prompt conditions occupy separate lines and are "
    "distinguished by colour. Each circle is the prompt-level estimate, in which the three "
    "repetitions of a speech are averaged so that the speech carries its sampling weight "
    "once, and the line through it is the pointwise 95% percentile interval from 2,000 "
    "stratified Rao–Wu rescaled bootstrap replicates (the same replicates as the primary P0 "
    "analysis). The thicker bar on each line spans the lowest to the highest of the three "
    "single-repetition estimates. "
    "Models are shown in the fixed experimental order rather than sorted by performance, "
    "and the x-axis keeps the original 0–1 metric scale."
)


def _check_range(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    for column in columns:
        outside = frame[column].lt(X_LIMITS[0]) | frame[column].gt(X_LIMITS[1])
        if outside.any():
            raise ValueError(
                f"{source}: {column} has values outside the plotted range {X_LIMITS}; "
                "widen X_LIMITS."
            )


def _y_positions(frame: pd.DataFrame) -> list[float]:
    model_to_y = {model: index for index, model in enumerate(MODEL_ORDER)}
    return [
        model_to_y[model] + PROMPT_OFFSETS[prompt]
        for model, prompt in zip(frame["model_key"], frame["prompt_id"], strict=True)
    ]


def prepare(data: pd.DataFrame, source: Path = INPUT_FILE) -> pd.DataFrame:
    """Keep the 12 single-repetition estimates per model and check completeness.

    condition_performance.csv holds four rows per model x prompt: repetitions 1-3 are
    single-repetition estimates, while repetition 0 is the expected-over-repetitions
    aggregate, which the interval table supplies instead.
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
        raise ValueError(f"Expected models {sorted(MODEL_ORDER)}; found {sorted(models)}.")
    prompts = set(data["prompt_id"].astype(str))
    if prompts != set(PROMPT_ORDER):
        raise ValueError(
            f"Expected exactly the four frozen prompt conditions; found {sorted(prompts)}."
        )
    data["repetition"] = pd.to_numeric(data["repetition"], errors="raise").astype(int)
    if sorted(data["repetition"].unique().tolist()) != [1, 2, 3]:
        raise ValueError("Expected single-repetition rows 1, 2 and 3 after filtering.")

    for column in METRICS.values():
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[column].isna().any():
            raise ValueError(f"{source} contains missing values in {column}.")
    _check_range(data, list(METRICS.values()), source)

    expected_rows = len(MODEL_ORDER) * len(PROMPT_ORDER) * 3
    if len(data) != expected_rows:
        raise ValueError(
            "Single-repetition validation output is incomplete. "
            f"Expected {expected_rows} rows, found {len(data)}."
        )
    if data.duplicated(subset=["model_key", "prompt_id", "repetition"]).any():
        raise ValueError(f"Duplicate model x prompt x repetition rows found in {source}.")
    data["y"] = _y_positions(data)
    return data


def prepare_intervals(intervals: pd.DataFrame, source: Path = INTERVAL_FILE) -> pd.DataFrame:
    """One repetition-averaged estimate and 95% interval per model x prompt."""
    columns = [
        f"{column}{suffix}" for column in METRICS.values()
        for suffix in ("", "_ci_low", "_ci_high")
    ]
    missing = {"model_key", "prompt_id", *columns} - set(intervals.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")
    intervals = intervals.copy()
    cells = set(zip(intervals["model_key"], intervals["prompt_id"], strict=True))
    expected = {(model, prompt) for model in MODEL_ORDER for prompt in PROMPT_ORDER}
    if cells != expected or len(intervals) != len(expected):
        raise ValueError(f"{source} must hold exactly one row per model x prompt.")
    for column in columns:
        intervals[column] = pd.to_numeric(intervals[column], errors="raise")
    for column in METRICS.values():
        low, high = intervals[f"{column}_ci_low"], intervals[f"{column}_ci_high"]
        if (low.gt(intervals[column]) | high.lt(intervals[column])).any():
            raise ValueError(f"{source}: {column} lies outside its interval.")
    _check_range(intervals, columns, source)
    intervals["y"] = _y_positions(intervals)
    return intervals


def repetition_ranges(data: pd.DataFrame) -> pd.DataFrame:
    """Lowest and highest single-repetition estimate per model x prompt and metric."""
    grouped = data.groupby(["model_key", "prompt_id"], sort=False)[list(METRICS.values())]
    ranges = grouped.min().add_suffix("_rep_min").join(grouped.max().add_suffix("_rep_max"))
    if (grouped.size() != 3).any():
        raise ValueError("Each model x prompt needs exactly three single-repetition estimates.")
    ranges = ranges.reset_index()
    ranges["y"] = _y_positions(ranges)
    return ranges


def build_figure(data: pd.DataFrame, intervals: pd.DataFrame):
    ps.use_style()
    ranges = repetition_ranges(data)
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=True)

    for ax, (title, column) in zip(axes.flat, METRICS.items(), strict=True):
        for prompt in PROMPT_ORDER:
            colour = PROMPT_COLOURS[prompt]
            cell = intervals.loc[intervals["prompt_id"].eq(prompt)]
            spread = ranges.loc[ranges["prompt_id"].eq(prompt)]
            ax.hlines(
                cell["y"], cell[f"{column}_ci_low"], cell[f"{column}_ci_high"],
                color=colour, linewidth=CI_WIDTH, zorder=2,
            )
            ax.hlines(
                spread["y"], spread[f"{column}_rep_min"], spread[f"{column}_rep_max"],
                color=colour, linewidth=RANGE_WIDTH, alpha=RANGE_ALPHA, zorder=3,
                capstyle="butt",
            )
            ax.scatter(
                cell[column], cell["y"], s=POINT_AREA, color=colour,
                edgecolors=ps.EDGE, linewidths=0.5, zorder=4,
            )
        ax.set_title(title)
        ax.set_xlim(X_LIMITS[0] - 0.01, X_LIMITS[1] + 0.01)
        ax.set_xticks(X_TICKS, [f"{tick:.1f}" for tick in X_TICKS])
        ax.set_yticks(range(len(MODEL_ORDER)), [MODEL_NAMES[m] for m in MODEL_ORDER])
        ax.set_ylim(len(MODEL_ORDER) - 0.5, -0.5)   # GPT-5.4 at the top
        ps.clean_axes(ax)
        ax.tick_params(axis="x", labelbottom=True)

    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=5.5,
               markerfacecolor=PROMPT_COLOURS[prompt], markeredgecolor=ps.EDGE,
               markeredgewidth=0.5, label=PROMPT_NAMES[prompt])
        for prompt in PROMPT_ORDER
    ]
    handles += [
        Line2D([], [], color=ps.EDGE, linewidth=CI_WIDTH, label="95% CI"),
        Line2D([], [], color=ps.EDGE, linewidth=RANGE_WIDTH, alpha=RANGE_ALPHA,
               solid_capstyle="butt", label="Range of 3 repetitions"),
    ]
    fig.tight_layout(rect=(0, 0.11, 1, 1), h_pad=1.2, w_pad=2.0)
    ps.bottom_legend(fig, handles, ncol=2, y=0.105)
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--intervals", type=Path, default=INTERVAL_FILE)
    parser.add_argument("--output-stem", type=Path, default=OUTPUT_STEM)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = prepare(pd.read_csv(args.input, encoding="utf-8-sig"), args.input)
    intervals = prepare_intervals(pd.read_csv(args.intervals), args.intervals)
    fig = build_figure(data, intervals)
    paths = ps.save(fig, args.output_stem, CAPTION)
    plt.close(fig)
    print(
        f"Single-repetition rows: {len(data):,}; interval rows: {len(intervals):,} "
        f"({len(MODEL_ORDER)} models x {len(PROMPT_ORDER)} prompts)"
    )
    for path in paths:
        print(f"Wrote: {path.as_posix()}")


if __name__ == "__main__":
    main()
