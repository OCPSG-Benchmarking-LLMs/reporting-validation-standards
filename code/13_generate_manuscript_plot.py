from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


##########################################################################
## 1. Paths
##########################################################################

INPUT_FILE = Path("outputs/validation/condition_performance.csv")

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_NAME = OUTPUT_DIR / "validation_prompt_repetition_profile"


##########################################################################
## 2. Model display names and fixed experimental order
##########################################################################

MODEL_NAMES = {
    "openai_gpt54": "GPT-5.4",
    "openai_gpt54_mini": "GPT-5.4 Mini",
    "fireworks_deepseek_v41_flash": "DeepSeek V4.1 Flash",
    "ollama_qwen36_35b": "Qwen3.6 35B",
    "ollama_gemma4_31b": "Gemma 4 31B",
    "openai_gpt6_luna": "GPT-6 Luna",
    "mistral_large3_2512": "Mistral Large 3",
}

MODEL_ORDER = [
    "openai_gpt54",
    "openai_gpt54_mini",
    "fireworks_deepseek_v41_flash",
    "ollama_qwen36_35b",
    "ollama_gemma4_31b",
    "openai_gpt6_luna",
    "mistral_large3_2512",
]


##########################################################################
## 3. Prompt display names and Oxford-inspired palette
##########################################################################

PROMPT_ORDER = [
    "p0_capbench_canonical",
    "p1_lexical_paraphrase",
    "p2_label_order_permutation",
    "p3_structural_format",
]

PROMPT_NAMES = {
    "p0_capbench_canonical": "P0 Canonical",
    "p1_lexical_paraphrase": "P1 Lexical paraphrase",
    "p2_label_order_permutation": "P2 Label-order permutation",
    "p3_structural_format": "P3 Structural format",
}

## Softened Oxford-style plotting tones, following the related Eval4SD scripts.
PROMPT_COLOURS = {
    "p0_capbench_canonical": "#5F7A9D",       # softened Oxford blue
    "p1_lexical_paraphrase": "#E6A57E",       # soft terracotta
    "p2_label_order_permutation": "#A8C5A1",  # sage
    "p3_structural_format": "#D9B8D0",        # muted mauve
}

DARK_GREY = "#4D4D4D"
LIGHT_GREY = "#D9D9D9"


##########################################################################
## 4. Metrics
##########################################################################

METRICS = [
    {
        "title": "Accuracy",
        "column": "weighted_operational_accuracy",
    },
    {
        "title": "Macro-Precision",
        "column": "weighted_operational_macro_precision",
    },
    {
        "title": "Macro-Recall",
        "column": "weighted_operational_macro_recall",
    },
    {
        "title": "Macro-F1",
        "column": "weighted_operational_macro_f1",
    },
]


##########################################################################
## 5. Plot geometry
##########################################################################

## Four prompt bands inside each model row.
PROMPT_OFFSETS = {
    "p0_capbench_canonical": -0.24,
    "p1_lexical_paraphrase": -0.08,
    "p2_label_order_permutation": 0.08,
    "p3_structural_format": 0.24,
}

## Three tiny vertical offsets inside each prompt band.
REPETITION_OFFSETS = {
    1: -0.035,
    2: 0.000,
    3: 0.035,
}

REPETITION_POINT_SIZE = 25
PROMPT_MEAN_POINT_SIZE = 88


##########################################################################
## 6. Helpers
##########################################################################


def require_columns(
    df: pd.DataFrame,
    required: set[str],
    source: Path,
) -> None:
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{source} is missing columns: {sorted(missing)}"
        )


def save_figure(
    fig,
    output_name: Path,
    caption: str,
) -> None:
    Path(f"{output_name}.txt").write_text(
        caption.strip() + "\n",
        encoding="utf-8",
    )

    fig.savefig(
        f"{output_name}.png",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )

    fig.savefig(
        f"{output_name}.pdf",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )

    plt.close(fig)


def decimal_ticks(lower: float) -> list[float]:
    start = math.ceil(lower * 10 - 1e-9) / 10
    ticks = list(np.arange(start, 1.0001, 0.1))

    if 1.0 not in ticks:
        ticks.append(1.0)

    return [round(value, 1) for value in ticks]


##########################################################################
## 7. Load and prepare data
##########################################################################

data = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig",
)

required_cols = {
    "model_key",
    "prompt_id",
    "repetition",
    "aggregation",
    *[metric["column"] for metric in METRICS],
}

require_columns(
    data,
    required_cols,
    INPUT_FILE,
)

## condition_performance.csv contains four rows per model × prompt:
## repetitions 1–3 are the actual single-repetition estimates, while
## repetition 0 is the pre-computed expected-over-repetitions aggregate.
## The figure needs exactly the 12 underlying prompt × repetition points,
## so repetition 0 must not be treated as a fourth repetition.
data = data.loc[
    data["aggregation"].astype(str).eq("single_repetition")
    & data["repetition"].isin([1, 2, 3])
].copy()

observed_models = set(data["model_key"].astype(str))
expected_models = set(MODEL_ORDER)

missing_models = expected_models - observed_models
unexpected_models = observed_models - expected_models

if missing_models:
    raise ValueError(
        f"Missing expected models: {sorted(missing_models)}"
    )

if unexpected_models:
    raise ValueError(
        f"Unexpected models: {sorted(unexpected_models)}"
    )

observed_prompts = set(data["prompt_id"].astype(str))
expected_prompts = set(PROMPT_ORDER)

if observed_prompts != expected_prompts:
    raise ValueError(
        "Expected exactly the four frozen prompt conditions; "
        f"found {sorted(observed_prompts)}."
    )

observed_repetitions = sorted(
    pd.to_numeric(
        data["repetition"],
        errors="raise",
    )
    .astype(int)
    .unique()
    .tolist()
)

if observed_repetitions != [1, 2, 3]:
    raise ValueError(
        "Expected single-repetition rows 1, 2, and 3 after filtering; "
        f"found {observed_repetitions}."
    )

data["repetition"] = data["repetition"].astype(int)

for metric in METRICS:
    column = metric["column"]

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce",
    )

    if data[column].isna().any():
        raise ValueError(
            f"{INPUT_FILE} contains missing values in {column}."
        )

expected_rows = (
    len(MODEL_ORDER)
    * len(PROMPT_ORDER)
    * len(REPETITION_OFFSETS)
)

if len(data) != expected_rows:
    raise ValueError(
        "Single-repetition validation output is incomplete. "
        f"Expected {expected_rows} rows, found {len(data)}."
    )

duplicate_cells = data.duplicated(
    subset=[
        "model_key",
        "prompt_id",
        "repetition",
    ],
    keep=False,
)

if duplicate_cells.any():
    raise ValueError(
        "Duplicate model × prompt × repetition rows found in "
        f"{INPUT_FILE}."
    )

model_to_y = {
    model: index
    for index, model in enumerate(MODEL_ORDER)
}

data["y_base"] = data["model_key"].map(model_to_y)

data["y"] = [
    base
    + PROMPT_OFFSETS[prompt]
    + REPETITION_OFFSETS[repetition]
    for base, prompt, repetition in zip(
        data["y_base"],
        data["prompt_id"],
        data["repetition"],
        strict=True,
    )
]

## Arithmetic prompt means are calculated directly from the three plotted
## repetition-level metric values. The repetition-0 aggregate is deliberately
## excluded because some non-linear metrics are not identical to the arithmetic
## mean of the three single-repetition scores.
prompt_means = (
    data.groupby(
        ["model_key", "prompt_id"],
        as_index=False,
    )[[metric["column"] for metric in METRICS]]
    .mean()
)

prompt_means["y_base"] = prompt_means["model_key"].map(model_to_y)

prompt_means["y"] = [
    base + PROMPT_OFFSETS[prompt]
    for base, prompt in zip(
        prompt_means["y_base"],
        prompt_means["prompt_id"],
        strict=True,
    )
]

## Use one common x scale across all four panels.
all_values = pd.concat(
    [
        data[metric["column"]]
        for metric in METRICS
    ],
    ignore_index=True,
)

global_min = float(all_values.min())

x_lower = max(
    0.0,
    math.floor((global_min - 0.025) / 0.05) * 0.05,
)

x_upper = 1.01
x_ticks = decimal_ticks(x_lower)


##########################################################################
## 8. Plot
##########################################################################

fig, axes = plt.subplots(
    1,
    4,
    figsize=(14.5, 8.0),
    sharey=True,
)

for i, metric in enumerate(METRICS):
    ax = axes[i]
    metric_column = metric["column"]

    for prompt in PROMPT_ORDER:
        prompt_data = data.loc[
            data["prompt_id"].eq(prompt)
        ]

        ## Three small repetition-level points per prompt and model.
        ax.scatter(
            prompt_data[metric_column],
            prompt_data["y"],
            color=PROMPT_COLOURS[prompt],
            s=REPETITION_POINT_SIZE,
            alpha=0.78,
            edgecolors="none",
            zorder=2,
        )

        ## One larger point at the arithmetic mean of the three repetitions.
        mean_data = prompt_means.loc[
            prompt_means["prompt_id"].eq(prompt)
        ]

        ax.scatter(
            mean_data[metric_column],
            mean_data["y"],
            color=PROMPT_COLOURS[prompt],
            s=PROMPT_MEAN_POINT_SIZE,
            alpha=0.98,
            edgecolors=DARK_GREY,
            linewidths=0.55,
            zorder=4,
        )

    ax.set_title(
        metric["title"],
        fontsize=14,
    )

    ax.set_xlim(
        x_lower,
        x_upper,
    )

    ax.set_xticks(
        x_ticks
    )

    ax.set_xticklabels(
        [
            f"{value:.1f}"
            for value in x_ticks
        ],
        fontsize=9,
    )

    ax.tick_params(
        axis="x",
        length=0,
    )

    ax.set_yticks(
        range(len(MODEL_ORDER))
    )

    if i == 0:
        ax.set_yticklabels(
            [
                MODEL_NAMES[model]
                for model in MODEL_ORDER
            ],
            fontsize=9.5,
        )

        ax.tick_params(
            axis="y",
            length=0,
            labelleft=True,
        )
    else:
        ax.tick_params(
            axis="y",
            length=0,
            labelleft=False,
        )

    for spine in [
        "top",
        "right",
        "left",
        "bottom",
    ]:
        ax.spines[spine].set_visible(False)

## Preserve the fixed experimental order with GPT-5.4 at the top.
axes[0].invert_yaxis()


##########################################################################
## 9. Legend and caption
##########################################################################

prompt_handles = [
    Line2D(
        [],
        [],
        marker="o",
        linestyle="",
        markersize=7,
        markerfacecolor=PROMPT_COLOURS[prompt],
        markeredgecolor="none",
        label=PROMPT_NAMES[prompt],
    )
    for prompt in PROMPT_ORDER
]

repetition_handle = Line2D(
    [],
    [],
    marker="o",
    linestyle="",
    markersize=4.5,
    markerfacecolor=LIGHT_GREY,
    markeredgecolor="none",
    label="Single repetition",
)

mean_handle = Line2D(
    [],
    [],
    marker="o",
    linestyle="",
    markersize=8.5,
    markerfacecolor="white",
    markeredgecolor=DARK_GREY,
    markeredgewidth=0.7,
    label="Prompt mean",
)

fig.legend(
    handles=prompt_handles + [
        repetition_handle,
        mean_handle,
    ],
    loc="lower center",
    ncol=3,
    frameon=False,
    fontsize=9,
    bbox_to_anchor=(0.5, 0.01),
)

plt.subplots_adjust(
    bottom=0.16,
)

plt.tight_layout(
    rect=[0, 0.08, 1, 1]
)

caption = (
    "Validation performance across prompt specifications and repeated inference.\n\n"
    "Note. Each panel reports a design-weighted operational validation metric "
    "against the OCPSG silver reference: Accuracy, Macro-Precision, Macro-Recall, "
    "and Macro-F1. Within each model row, colour identifies the four frozen prompt "
    "conditions. For each prompt, three small vertically offset points report the "
    "three single-repetition estimates and the larger outlined point reports their "
    "arithmetic mean. Models are shown in the fixed experimental order rather than "
    "sorted by performance. The x-axis is reported on the original 0–1 metric scale. "
    "The figure complements, rather than replaces, the full stability and validation "
    "tables."
)


##########################################################################
## 10. Save
##########################################################################

save_figure(
    fig,
    OUTPUT_NAME,
    caption,
)

print(
    "Single-repetition rows: "
    f"{len(data):,} "
    f"({len(MODEL_ORDER)} models × {len(PROMPT_ORDER)} prompts × 3 repetitions)"
)
print(f"Wrote: {OUTPUT_NAME}.png")
print(f"Wrote: {OUTPUT_NAME}.pdf")
print(f"Wrote: {OUTPUT_NAME}.txt")
