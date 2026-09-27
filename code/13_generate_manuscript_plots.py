from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import pandas as pd

##########################################################################
## 1. Paths
##########################################################################

STABILITY_FILE = Path("outputs/stability/model_summary.csv")
VALIDATION_FILE = Path("outputs/validation/primary_model_performance.csv")
PROPAGATION_FILE = Path("outputs/propagation/prevalence_comparison.csv")

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STABILITY_OUTPUT = OUTPUT_DIR / "stability_profile"
VALIDATION_OUTPUT = OUTPUT_DIR / "validation_macro_f1"
PROPAGATION_OUTPUT = OUTPUT_DIR / "prevalence_propagation"

##########################################################################
## 2. Display names and plotting palette
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

## Inspired by the Oxford-style softened palette used in the related
## Eval4SD plotting scripts. Colours are deliberately restrained so the
## figures remain legible in print and do not imply model rankings.
OXFORD_BLUE = "#5F7A9D"
SOFT_TERRACOTTA = "#E6A57E"
SAGE = "#A8C5A1"
MAUVE = "#D9B8D0"
PALE_AQUA = "#A8DADC"
MUTED_GOLD = "#E9C46A"
CORAL = "#FE615A"
DARK_GREY = "#4D4D4D"
MID_GREY = "#777777"
LIGHT_GREY = "#D9D9D9"
VERY_LIGHT_GREY = "#E6E6E6"

##########################################################################
## 3. Helpers
##########################################################################


def require_columns(df: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")


def pretty_label(value: str) -> str:
    return str(value).strip().replace("_", " ")


def clean_axes(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.tick_params(axis="both", length=0)


def write_caption(save_path_prefix: Path, caption: str) -> None:
    Path(f"{save_path_prefix}.txt").write_text(
        caption.strip() + "\n",
        encoding="utf-8",
    )


def save_current_figure(save_path_prefix: Path) -> None:
    for ext in ("png", "pdf"):
        plt.savefig(
            f"{save_path_prefix}.{ext}",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
    plt.close()


def order_models(df: pd.DataFrame) -> pd.DataFrame:
    observed = set(df["model_key"].astype(str))
    expected = set(MODEL_ORDER)

    missing = expected - observed
    unexpected = observed - expected

    if missing:
        raise ValueError(f"Missing expected models: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"Unexpected models: {sorted(unexpected)}")

    ordered = df.copy()
    ordered["model_key"] = pd.Categorical(
        ordered["model_key"],
        categories=MODEL_ORDER,
        ordered=True,
    )
    ordered = ordered.sort_values("model_key").reset_index(drop=True)
    ordered["model_name"] = ordered["model_key"].astype(str).map(MODEL_NAMES)
    return ordered


##########################################################################
## 4. Plot 1: Stability profile
##########################################################################


def plot_stability_profile() -> None:
    data = pd.read_csv(STABILITY_FILE)

    required = {
        "model_key",
        "repeat_unanimity_unconditional_weighted",
        "four_prompt_unanimity_unconditional_weighted",
        "all12_unanimity_unconditional_weighted",
    }
    require_columns(data, required, STABILITY_FILE)

    data = order_models(data)

    metric_specs = [
        (
            "repeat_unanimity_unconditional_weighted",
            "Repeated inference",
            OXFORD_BLUE,
            "o",
            -0.16,
        ),
        (
            "four_prompt_unanimity_unconditional_weighted",
            "Prompt perturbation",
            SOFT_TERRACOTTA,
            "s",
            0.00,
        ),
        (
            "all12_unanimity_unconditional_weighted",
            "All 12 decisions",
            SAGE,
            "D",
            0.16,
        ),
    ]

    ypos = list(range(len(data)))
    fig, ax = plt.subplots(figsize=(10.5, 6.8))

    for column, label, colour, marker, offset in metric_specs:
        values = pd.to_numeric(data[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"{STABILITY_FILE} contains missing values in {column}.")

        ax.scatter(
            values,
            [y + offset for y in ypos],
            s=85,
            color=colour,
            marker=marker,
            alpha=0.92,
            edgecolors=DARK_GREY,
            linewidths=0.55,
            label=label,
            zorder=3,
        )

    ax.set_xlabel("Weighted unanimity", fontsize=12)
    ax.set_ylabel("")
    ax.set_xlim(0.70, 1.005)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_yticks(ypos)
    ax.set_yticklabels(data["model_name"], fontsize=10)
    ax.invert_yaxis()

    clean_axes(ax)

    caption = (
        "Stability across repeated inference and prompt perturbation.\n\n"
        "Note. Population-weighted unconditional unanimity in the frozen GB probability sample. "
        "Repeated inference summarises agreement across three repetitions within a prompt; "
        "prompt perturbation summarises unanimity across the four prompt conditions at matched "
        "observation × repetition cells; all 12 decisions require unanimity across all four "
        "prompts and three repetitions. Invalid outputs therefore reduce unconditional stability."
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16, left=0.22)

    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=3,
        frameon=False,
    )

    write_caption(STABILITY_OUTPUT, caption)
    save_current_figure(STABILITY_OUTPUT)


##########################################################################
## 5. Plot 2: Design-based validation
##########################################################################


def plot_validation_macro_f1() -> None:
    data = pd.read_csv(VALIDATION_FILE)

    required = {
        "model_key",
        "prompt_id",
        "weighted_operational_macro_f1",
        "weighted_operational_macro_f1_ci_low",
        "weighted_operational_macro_f1_ci_high",
        "unweighted_operational_macro_f1",
    }
    require_columns(data, required, VALIDATION_FILE)

    if not (data["prompt_id"].astype(str) == "p0_capbench_canonical").all():
        raise ValueError("Primary validation input contains a non-P0 prompt.")

    data = order_models(data)

    weighted = pd.to_numeric(data["weighted_operational_macro_f1"], errors="coerce")
    ci_low = pd.to_numeric(
        data["weighted_operational_macro_f1_ci_low"],
        errors="coerce",
    )
    ci_high = pd.to_numeric(
        data["weighted_operational_macro_f1_ci_high"],
        errors="coerce",
    )
    unweighted = pd.to_numeric(
        data["unweighted_operational_macro_f1"],
        errors="coerce",
    )

    if pd.concat([weighted, ci_low, ci_high, unweighted], axis=1).isna().any().any():
        raise ValueError(f"{VALIDATION_FILE} contains missing macro-F1 values.")

    ypos = list(range(len(data)))
    fig, ax = plt.subplots(figsize=(10.5, 6.8))

    for y, w, lo, hi, uw in zip(
        ypos,
        weighted,
        ci_low,
        ci_high,
        unweighted,
        strict=True,
    ):
        ax.plot(
            [min(w, uw), max(w, uw)],
            [y, y],
            color=LIGHT_GREY,
            linewidth=1.6,
            alpha=0.95,
            zorder=1,
        )

        ax.errorbar(
            w,
            y,
            xerr=[[w - lo], [hi - w]],
            fmt="o",
            markersize=7.5,
            color=OXFORD_BLUE,
            ecolor=OXFORD_BLUE,
            markeredgecolor=DARK_GREY,
            markeredgewidth=0.55,
            elinewidth=1.8,
            capsize=3.5,
            alpha=0.95,
            zorder=3,
        )

        ax.scatter(
            uw,
            y,
            s=68,
            color=SOFT_TERRACOTTA,
            marker="D",
            alpha=0.92,
            edgecolors=DARK_GREY,
            linewidths=0.55,
            zorder=4,
        )

    ax.set_xlabel("Operational macro-F1", fontsize=12)
    ax.set_ylabel("")
    ax.set_xlim(0.70, 1.00)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_yticks(ypos)
    ax.set_yticklabels(data["model_name"], fontsize=10)
    ax.invert_yaxis()

    clean_axes(ax)

    legend_handles = [
        Line2D(
            [], [],
            marker="o",
            linestyle="-",
            markersize=7,
            color=OXFORD_BLUE,
            markerfacecolor=OXFORD_BLUE,
            markeredgecolor=DARK_GREY,
            markeredgewidth=0.55,
            label="Design-weighted estimate (95% CI)",
        ),
        Line2D(
            [], [],
            marker="D",
            linestyle="",
            markersize=7,
            markerfacecolor=SOFT_TERRACOTTA,
            markeredgecolor=DARK_GREY,
            markeredgewidth=0.55,
            label="Unweighted sample estimate",
        ),
    ]

    caption = (
        "Design-based silver-reference performance.\n\n"
        "Note. Canonical P0 prompt. Design-weighted operational macro-F1 uses inverse-inclusion "
        "weights and 2,000 paired Rao–Wu rescaled bootstrap replicates. Unweighted values are "
        "descriptive diagnostics for the deliberately stratified validation sample. Performance "
        "is measured against the OCPSG silver reference, not an independent human gold standard."
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.17, left=0.22)

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=2,
        frameon=False,
    )

    write_caption(VALIDATION_OUTPUT, caption)
    save_current_figure(VALIDATION_OUTPUT)


##########################################################################
## 6. Plot 3: Prevalence propagation
##########################################################################


def plot_prevalence_propagation() -> None:
    data = pd.read_csv(PROPAGATION_FILE)

    required = {
        "label",
        "present_in_silver_reference_population",
        "silver_true_prevalence",
        "naive_classifier_prevalence",
        "corrected_prevalence",
        "corrected_ci_low",
        "corrected_ci_high",
    }
    require_columns(data, required, PROPAGATION_FILE)

    present = (
        data["present_in_silver_reference_population"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )

    data = data.loc[present].copy()
    if len(data) != 17:
        raise ValueError(
            "Expected 17 silver-reference-supported GB labels, "
            f"found {len(data)}."
        )

    numeric_columns = [
        "silver_true_prevalence",
        "naive_classifier_prevalence",
        "corrected_prevalence",
        "corrected_ci_low",
        "corrected_ci_high",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data[numeric_columns].isna().any().any():
        raise ValueError(f"{PROPAGATION_FILE} contains missing prevalence values.")

    data["label_pretty"] = data["label"].map(pretty_label)
    data = data.sort_values(
        ["silver_true_prevalence", "label_pretty"],
        ascending=[True, True],
    ).reset_index(drop=True)

    ypos = list(range(len(data)))
    fig, ax = plt.subplots(figsize=(11.2, 9.0))

    for y, row in zip(ypos, data.itertuples(index=False), strict=True):
        ax.plot(
            [
                min(row.naive_classifier_prevalence, row.corrected_prevalence),
                max(row.naive_classifier_prevalence, row.corrected_prevalence),
            ],
            [y, y],
            color=LIGHT_GREY,
            linewidth=1.5,
            alpha=0.95,
            zorder=1,
        )

        ax.errorbar(
            row.corrected_prevalence,
            y,
            xerr=[
                [row.corrected_prevalence - row.corrected_ci_low],
                [row.corrected_ci_high - row.corrected_prevalence],
            ],
            fmt="s",
            markersize=6.8,
            color=OXFORD_BLUE,
            ecolor=OXFORD_BLUE,
            markeredgecolor=DARK_GREY,
            markeredgewidth=0.5,
            elinewidth=1.5,
            capsize=3.0,
            alpha=0.95,
            zorder=3,
        )

        ax.scatter(
            row.naive_classifier_prevalence,
            y,
            s=58,
            color=SOFT_TERRACOTTA,
            marker="o",
            alpha=0.92,
            edgecolors=DARK_GREY,
            linewidths=0.5,
            zorder=4,
        )

        ax.scatter(
            row.silver_true_prevalence,
            y,
            s=62,
            color=DARK_GREY,
            marker="D",
            alpha=0.92,
            edgecolors=DARK_GREY,
            linewidths=0.5,
            zorder=5,
        )

    maximum = max(
        data["silver_true_prevalence"].max(),
        data["naive_classifier_prevalence"].max(),
        data["corrected_ci_high"].max(),
    )
    upper = min(1.0, max(0.20, (int(maximum * 100 / 5) + 2) * 0.05))

    ax.set_xlabel("Share of GB test speeches", fontsize=12)
    ax.set_ylabel("")
    ax.set_xlim(0, upper)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_yticks(ypos)
    ax.set_yticklabels(data["label_pretty"], fontsize=9.5)

    clean_axes(ax)

    legend_handles = [
        Line2D(
            [], [],
            marker="D",
            linestyle="",
            markersize=7,
            markerfacecolor=DARK_GREY,
            markeredgecolor=DARK_GREY,
            label="Silver-reference prevalence",
        ),
        Line2D(
            [], [],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=SOFT_TERRACOTTA,
            markeredgecolor=DARK_GREY,
            markeredgewidth=0.5,
            label="Naïve GPT-5.4 Mini prevalence",
        ),
        Line2D(
            [], [],
            marker="s",
            linestyle="-",
            markersize=7,
            color=OXFORD_BLUE,
            markerfacecolor=OXFORD_BLUE,
            markeredgecolor=DARK_GREY,
            markeredgewidth=0.5,
            label="Corrected prevalence (95% CI)",
        ),
    ]

    caption = (
        "Propagation of classification error to policy-agenda prevalence.\n\n"
        "Note. Seventeen policy categories with positive support in the frozen GB silver-reference "
        "population are shown. The corrected prevalence uses the misclassification mechanism "
        "estimated from the 1,000-speech probability validation sample; its 95% intervals propagate "
        "validation-sample uncertainty through 2,000 Rao–Wu bootstrap replicates. Full-population "
        "silver prevalence is used only for ex-post comparison, not to fit the correction."
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16, left=0.25)

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=3,
        frameon=False,
    )

    write_caption(PROPAGATION_OUTPUT, caption)
    save_current_figure(PROPAGATION_OUTPUT)


##########################################################################
## 7. Main
##########################################################################


def main() -> None:
    plot_stability_profile()
    plot_validation_macro_f1()
    plot_prevalence_propagation()

    print("Wrote manuscript figures:")
    for output in [STABILITY_OUTPUT, VALIDATION_OUTPUT, PROPAGATION_OUTPUT]:
        print(f"  {output}.png")
        print(f"  {output}.pdf")
        print(f"  {output}.txt")


if __name__ == "__main__":
    main()
