"""Validation performance across prompt specifications and repeated inference.

Reads outputs/validation/prompt_performance_intervals.csv (repetition-averaged estimates
with 95% Rao-Wu bootstrap intervals, from script 08b), condition_performance.csv
(single-repetition estimates, summarised in the caption) and
primary_model_performance.csv (the P0 macro-F1 that orders the models), and draws a
2 x 2 figure (Accuracy, Macro-Precision, Macro-Recall, Macro-F1). Each model x prompt
row shows the averaged estimate as a circle and its interval as a line.
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
PRIMARY_FILE = Path("outputs/validation/primary_model_performance.csv")
OUTPUT_STEM = Path("outputs/figures/validation_prompt_repetition_profile")
ORDER_COLUMN = "weighted_operational_macro_f1"

MODEL_NAMES = {
    "openai_gpt54": "GPT-5.4",
    "openai_gpt54_mini": "GPT-5.4-mini",
    "fireworks_deepseek_v41_flash": "DeepSeek V4.1 Flash",
    "ollama_qwen36_35b": "Qwen3.6 (35B)",
    "ollama_gemma4_31b": "Gemma 4 (31B)",
    "openai_gpt6_luna": "GPT-6 Luna",
    "mistral_large3_2512": "Mistral Large 3",
}

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
FIGURE_WIDTH = ps.print_width(LINEWIDTH_FRACTION)
FIGSIZE = (FIGURE_WIDTH, 6.0)
POINT_AREA = 22
CI_WIDTH = 1.1

TITLE = "Validation Performance across Prompt Specifications and Repeated Inference"


def model_order(primary: pd.DataFrame, source: Path = PRIMARY_FILE) -> list[str]:
    """Models by descending primary P0 macro-F1 (ties broken by model key)."""
    if {"model_key", ORDER_COLUMN} - set(primary.columns):
        raise ValueError(f"{source} needs model_key and {ORDER_COLUMN}.")
    if set(primary["model_key"]) != set(MODEL_NAMES) or len(primary) != len(MODEL_NAMES):
        raise ValueError(f"{source} must hold exactly one row for each of the seven models.")
    ordered = primary.assign(_score=pd.to_numeric(primary[ORDER_COLUMN], errors="raise"))
    ordered = ordered.sort_values(["_score", "model_key"], ascending=[False, True])
    return ordered["model_key"].astype(str).tolist()


def _check_range(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    for column in columns:
        outside = frame[column].lt(X_LIMITS[0]) | frame[column].gt(X_LIMITS[1])
        if outside.any():
            raise ValueError(
                f"{source}: {column} has values outside the plotted range {X_LIMITS}; "
                "widen X_LIMITS."
            )


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
    if models != set(MODEL_NAMES):
        raise ValueError(f"Expected models {sorted(MODEL_NAMES)}; found {sorted(models)}.")
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

    expected_rows = len(MODEL_NAMES) * len(PROMPT_ORDER) * 3
    if len(data) != expected_rows:
        raise ValueError(
            "Single-repetition validation output is incomplete. "
            f"Expected {expected_rows} rows, found {len(data)}."
        )
    if data.duplicated(subset=["model_key", "prompt_id", "repetition"]).any():
        raise ValueError(f"Duplicate model x prompt x repetition rows found in {source}.")
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
    expected = {(model, prompt) for model in MODEL_NAMES for prompt in PROMPT_ORDER}
    if cells != expected or len(intervals) != len(expected):
        raise ValueError(f"{source} must hold exactly one row per model x prompt.")
    for column in columns:
        intervals[column] = pd.to_numeric(intervals[column], errors="raise")
    for column in METRICS.values():
        low, high = intervals[f"{column}_ci_low"], intervals[f"{column}_ci_high"]
        if (low.gt(intervals[column]) | high.lt(intervals[column])).any():
            raise ValueError(f"{source}: {column} lies outside its interval.")
    _check_range(intervals, columns, source)
    return intervals


def repetition_spread(data: pd.DataFrame) -> dict:
    """Median and largest gap between the three repetitions, over all cells and metrics."""
    grouped = data.groupby(["model_key", "prompt_id"])[list(METRICS.values())]
    spread = (grouped.max() - grouped.min()).stack()
    (model, prompt, column), largest = spread.idxmax(), float(spread.max())
    metric = {value: key for key, value in METRICS.items()}[column]
    return {
        "median": float(spread.median()), "max": largest,
        "model": MODEL_NAMES[model], "prompt": PROMPT_NAMES[prompt].split()[0],
        "metric": metric.lower().replace("macro-f1", "macro-F1"),
    }


def caption(data: pd.DataFrame) -> str:
    spread = repetition_spread(data)
    return (
        f"{TITLE}\n\n"
        "Note. Each panel reports a design-weighted operational validation metric against "
        "the OCPSG silver reference: Accuracy, Macro-Precision, Macro-Recall and Macro-F1. "
        "Within each model group, the four frozen prompt conditions occupy separate lines "
        "and are distinguished by colour. Each circle is the prompt-level estimate, in which "
        "the three repetitions of a speech are averaged so that the speech carries its "
        "sampling weight once, and the line through it is the pointwise 95% percentile "
        "interval from 2,000 stratified Rao–Wu rescaled bootstrap replicates (the same "
        "replicates as the primary P0 analysis). The three single-repetition estimates of a "
        f"prompt differ by a median of {spread['median']:.3f} and at most {spread['max']:.3f} "
        f"({spread['metric']} for {spread['model']} under {spread['prompt']}). Models are "
        "ordered by their primary P0 macro-F1 estimate; because many intervals overlap, the "
        "order is not a ranking. The x-axis keeps the original 0–1 metric scale."
    )


def build_figure(intervals: pd.DataFrame, order: list[str]):
    ps.use_style()
    position = {model: index for index, model in enumerate(order)}
    intervals = intervals.assign(y=[
        position[model] + PROMPT_OFFSETS[prompt]
        for model, prompt in zip(intervals["model_key"], intervals["prompt_id"], strict=True)
    ])
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=True)

    for ax, (title, column) in zip(axes.flat, METRICS.items(), strict=True):
        for prompt in PROMPT_ORDER:
            colour = PROMPT_COLOURS[prompt]
            cell = intervals.loc[intervals["prompt_id"].eq(prompt)]
            ax.hlines(
                cell["y"], cell[f"{column}_ci_low"], cell[f"{column}_ci_high"],
                color=colour, linewidth=CI_WIDTH, zorder=2,
            )
            ax.scatter(
                cell[column], cell["y"], s=POINT_AREA, color=colour,
                edgecolors=ps.EDGE, linewidths=0.5, zorder=3,
            )
        ax.set_title(title)
        ax.set_xlim(X_LIMITS[0] - 0.01, X_LIMITS[1] + 0.01)
        ax.set_xticks(X_TICKS, [f"{tick:.1f}" for tick in X_TICKS])
        ax.set_yticks(range(len(order)), [MODEL_NAMES[m] for m in order])
        ax.set_ylim(len(order) - 0.5, -0.5)   # highest P0 macro-F1 at the top
        ps.clean_axes(ax)
        ax.tick_params(axis="x", labelbottom=True)

    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=5.5,
               markerfacecolor=PROMPT_COLOURS[prompt], markeredgecolor=ps.EDGE,
               markeredgewidth=0.5, label=PROMPT_NAMES[prompt])
        for prompt in PROMPT_ORDER
    ]
    handles.append(Line2D([], [], color=ps.EDGE, linewidth=CI_WIDTH, label="95% CI"))
    fig.tight_layout(rect=(0, 0.09, 1, 1), h_pad=1.2, w_pad=2.0)
    ps.bottom_legend(fig, handles, ncol=3, y=0.085)
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--intervals", type=Path, default=INTERVAL_FILE)
    parser.add_argument("--primary", type=Path, default=PRIMARY_FILE)
    parser.add_argument("--output-stem", type=Path, default=OUTPUT_STEM)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = prepare(pd.read_csv(args.input, encoding="utf-8-sig"), args.input)
    intervals = prepare_intervals(pd.read_csv(args.intervals), args.intervals)
    order = model_order(pd.read_csv(args.primary), args.primary)
    fig = build_figure(intervals, order)
    paths = ps.save(fig, args.output_stem, caption(data), target_width=FIGURE_WIDTH)
    plt.close(fig)
    print("Model order (P0 macro-F1): " + ", ".join(MODEL_NAMES[m] for m in order))
    for path in paths:
        print(f"Wrote: {path.as_posix()}")


if __name__ == "__main__":
    main()
