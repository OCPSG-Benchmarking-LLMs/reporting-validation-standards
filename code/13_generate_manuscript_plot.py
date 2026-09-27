from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import pandas as pd


##########################################################################
## 1. Paths
##########################################################################

INPUT_FILE = Path("outputs/validation/primary_model_performance.csv")

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_NAME = OUTPUT_DIR / "validation_weighted_unweighted_comparison"


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
## 3. Oxford-inspired palette
##########################################################################

## These colours follow the softened Oxford-style palette used in the
## related Eval4SD plotting scripts.

OXFORD_BLUE = "#5F7A9D"
SOFT_TERRACOTTA = "#E6A57E"
DARK_GREY = "#4D4D4D"
CONNECTOR_GREY = "#D9D9D9"


##########################################################################
## 4. Metrics
##########################################################################

## Each panel compares the design-weighted population estimate with the
## corresponding naïve unweighted validation-sample estimate. The weighted
## estimate also carries its design-based 95% Rao-Wu bootstrap interval.

METRICS = [
    {
        "title": "Accuracy",
        "weighted": "weighted_operational_accuracy",
        "unweighted": "unweighted_operational_accuracy",
        "ci_low": "weighted_operational_accuracy_ci_low",
        "ci_high": "weighted_operational_accuracy_ci_high",
    },
    {
        "title": "Macro-Precision",
        "weighted": "weighted_operational_macro_precision",
        "unweighted": "unweighted_operational_macro_precision",
        "ci_low": "weighted_operational_macro_precision_ci_low",
        "ci_high": "weighted_operational_macro_precision_ci_high",
    },
    {
        "title": "Macro-Recall",
        "weighted": "weighted_operational_macro_recall",
        "unweighted": "unweighted_operational_macro_recall",
        "ci_low": "weighted_operational_macro_recall_ci_low",
        "ci_high": "weighted_operational_macro_recall_ci_high",
    },
    {
        "title": "Macro-F1",
        "weighted": "weighted_operational_macro_f1",
        "unweighted": "unweighted_operational_macro_f1",
        "ci_low": "weighted_operational_macro_f1_ci_low",
        "ci_high": "weighted_operational_macro_f1_ci_high",
    },
]


##########################################################################
## 5. Helpers
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


##########################################################################
## 6. Load and prepare data
##########################################################################

data = pd.read_csv(
    INPUT_FILE,
    encoding="utf-8-sig",
)

required_cols = {
    "model_key",
    "prompt_id",
}

for metric in METRICS:
    required_cols.update(
        {
            metric["weighted"],
            metric["unweighted"],
            metric["ci_low"],
            metric["ci_high"],
        }
    )

require_columns(
    data,
    required_cols,
    INPUT_FILE,
)

if not (
    data["prompt_id"].astype(str) == "p0_capbench_canonical"
).all():
    raise ValueError(
        "Primary validation output contains a non-canonical prompt."
    )

observed_models = set(
    data["model_key"].astype(str)
)

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

data = (
    data.set_index("model_key")
    .reindex(MODEL_ORDER)
    .reset_index()
)

for metric in METRICS:
    for column in (
        metric["weighted"],
        metric["unweighted"],
        metric["ci_low"],
        metric["ci_high"],
    ):
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        if data[column].isna().any():
            raise ValueError(
                f"{INPUT_FILE} contains missing values in {column}."
            )

data["model_name"] = (
    data["model_key"]
    .map(MODEL_NAMES)
)

y_vals = list(
    range(len(data))
)


##########################################################################
## 7. Plot
##########################################################################

fig, axes = plt.subplots(
    1,
    4,
    figsize=(14, 7.8),
    sharey=True,
)

for i, metric in enumerate(METRICS):
    ax = axes[i]

    weighted = data[metric["weighted"]]
    unweighted = data[metric["unweighted"]]
    ci_low = data[metric["ci_low"]]
    ci_high = data[metric["ci_high"]]

    ## Light connectors make the design effect visible for each model.
    for y, weighted_value, unweighted_value in zip(
        y_vals,
        weighted,
        unweighted,
        strict=True,
    ):
        ax.plot(
            [weighted_value, unweighted_value],
            [y, y],
            color=CONNECTOR_GREY,
            linewidth=1.2,
            zorder=1,
        )

    ## Design-weighted population estimates with 95% design-based intervals.
    for y, estimate, low, high in zip(
        y_vals,
        weighted,
        ci_low,
        ci_high,
        strict=True,
    ):
        ax.errorbar(
            estimate,
            y,
            xerr=[
                [estimate - low],
                [high - estimate],
            ],
            fmt="o",
            markersize=7.2,
            color=OXFORD_BLUE,
            ecolor=OXFORD_BLUE,
            markeredgecolor="none",
            elinewidth=1.35,
            capsize=2.8,
            alpha=0.95,
            zorder=3,
        )

    ## Naïve unweighted sample estimates.
    ax.scatter(
        unweighted,
        y_vals,
        color=SOFT_TERRACOTTA,
        s=58,
        alpha=0.92,
        edgecolors="none",
        zorder=4,
    )

    ax.set_title(
        metric["title"],
        fontsize=13,
    )

    ax.set_xlim(
        0.70,
        1.01,
    )

    ax.set_xticks(
        [0.70, 0.80, 0.90, 1.00]
    )

    ax.xaxis.set_major_formatter(
        PercentFormatter(
            xmax=1.0,
            decimals=0,
        )
    )

    ax.tick_params(
        axis="x",
        length=0,
        labelsize=9,
    )

    ax.set_yticks(
        y_vals
    )

    if i == 0:
        ax.set_yticklabels(
            data["model_name"],
            fontsize=9,
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

## Put the first model at the top while preserving the fixed experimental order.
axes[0].invert_yaxis()


##########################################################################
## 8. Legend and caption
##########################################################################

weighted_handle = Line2D(
    [],
    [],
    marker="o",
    linestyle="-",
    markersize=7,
    color=OXFORD_BLUE,
    markerfacecolor=OXFORD_BLUE,
    markeredgecolor="none",
    linewidth=1.35,
    label="Design-weighted estimate (95% CI)",
)

unweighted_handle = Line2D(
    [],
    [],
    marker="o",
    linestyle="",
    markersize=7,
    markerfacecolor=SOFT_TERRACOTTA,
    markeredgecolor="none",
    label="Unweighted sample estimate",
)

fig.legend(
    handles=[
        weighted_handle,
        unweighted_handle,
    ],
    loc="lower center",
    ncol=2,
    frameon=False,
    fontsize=9.5,
    bbox_to_anchor=(0.5, 0.015),
)

plt.subplots_adjust(
    bottom=0.14,
)

plt.tight_layout(
    rect=[0, 0.065, 1, 1]
)

caption = (
    "Design-weighted and unweighted validation performance across four "
    "goodness-of-prediction indicators.\n\n"
    "Note. Each panel compares the design-weighted population estimate with "
    "the corresponding naïve unweighted estimate from the deliberately "
    "label-stratified 1,000-speech validation sample. Blue points report "
    "inverse-inclusion-weighted operational estimates; horizontal intervals "
    "are 95% confidence intervals from 2,000 paired Rao-Wu rescaled bootstrap "
    "replicates. Terracotta points report the corresponding unweighted sample "
    "estimates. Models are shown in the fixed experimental order rather than "
    "ordered by performance. Performance is measured against the OCPSG silver "
    "reference, not an independent human gold standard."
)


##########################################################################
## 9. Save
##########################################################################

save_figure(
    fig,
    OUTPUT_NAME,
    caption,
)

print(f"Wrote: {OUTPUT_NAME}.png")
print(f"Wrote: {OUTPUT_NAME}.pdf")
print(f"Wrote: {OUTPUT_NAME}.txt")
