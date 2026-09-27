from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


##########################################################################
## 1. Paths
##########################################################################

MODEL_AGREEMENT_FILE = Path("outputs/stability/model_pair_agreement.csv")
PROMPT_COMPARISON_FILE = Path("outputs/stability/prompt_comparisons.csv")
MISCLASSIFICATION_FILE = Path("outputs/propagation/misclassification_matrix.csv")
OBSERVABILITY_FILE = Path(
    "outputs/compute_environment/environmental_reporting_scope.csv"
)

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_AGREEMENT_OUTPUT = OUTPUT_DIR / "inter_model_agreement_heatmap"
PROMPT_PERTURBATION_OUTPUT = OUTPUT_DIR / "prompt_perturbation_heatmap"
MISCLASSIFICATION_OUTPUT = OUTPUT_DIR / "misclassification_matrix_heatmap"
OBSERVABILITY_OUTPUT = OUTPUT_DIR / "observability_matrix"


##########################################################################
## 2. Display names
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

PROMPT_NAMES = {
    "p1_lexical_paraphrase": "Lexical paraphrase",
    "p2_label_order_permutation": "Label-order permutation",
    "p3_structural_format": "Structural format",
}

PROMPT_ORDER = [
    "p1_lexical_paraphrase",
    "p2_label_order_permutation",
    "p3_structural_format",
]

OBSERVABILITY_COLUMNS = {
    "request_latency_reportable": "Request\nlatency",
    "token_usage_reportable": "Token\nusage",
    "retry_diagnostics_reportable": "Retry\ndiagnostics",
    "monetary_cost_reportable": "Monetary\ncost",
    "machine_energy_reportable": "Machine\nenergy",
    "machine_carbon_estimate_reportable": "Machine carbon\nestimate",
    "provider_side_energy_reportable": "Provider-side\nenergy",
    "provider_side_carbon_reportable": "Provider-side\ncarbon",
}


##########################################################################
## 3. Oxford-inspired palette
##########################################################################

## The plotting palette follows the softened Oxford-style tones used in the
## related Eval4SD scripts.

OXFORD_BLUE = "#5F7A9D"
SOFT_TERRACOTTA = "#E6A57E"
SAGE = "#A8C5A1"
MAUVE = "#D9B8D0"
PALE_AQUA = "#A8DADC"
MUTED_GOLD = "#E9C46A"

DARK_GREY = "#4D4D4D"
MID_GREY = "#999999"
LIGHT_GREY = "#D9D9D9"
VERY_LIGHT_GREY = "#F2F2F2"
OFF_WHITE = "#FAFAFA"

OXFORD_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "oxford_sequential",
    [
        "#FFFFFF",
        "#E8EEF4",
        "#C8D5E3",
        "#9FB4CA",
        "#7895B2",
        OXFORD_BLUE,
    ],
)

BINARY_OBSERVABILITY = ListedColormap(
    [
        VERY_LIGHT_GREY,
        OXFORD_BLUE,
    ]
)


##########################################################################
## 4. General helpers
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


def pretty_label(value: str) -> str:
    return str(value).strip().replace("_", " ")


def clean_axes(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(
        axis="both",
        length=0,
    )


def write_caption(
    save_path_prefix: Path,
    caption: str,
) -> None:
    Path(f"{save_path_prefix}.txt").write_text(
        caption.strip() + "\n",
        encoding="utf-8",
    )


def save_figure(
    fig,
    save_path_prefix: Path,
    caption: str,
) -> None:
    write_caption(
        save_path_prefix,
        caption,
    )

    for ext in ("png", "pdf"):
        fig.savefig(
            f"{save_path_prefix}.{ext}",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )

    plt.close(fig)


def validate_model_keys(
    observed: set[str],
    source: Path,
) -> None:
    expected = set(MODEL_ORDER)

    missing = expected - observed
    unexpected = observed - expected

    if missing:
        raise ValueError(
            f"{source} is missing expected models: {sorted(missing)}"
        )

    if unexpected:
        raise ValueError(
            f"{source} contains unexpected models: {sorted(unexpected)}"
        )


def annotate_heatmap(
    ax,
    matrix: np.ndarray,
    *,
    fmt: str = ".1%",
    threshold: float | None = None,
    text_size: float = 9,
) -> None:
    finite = matrix[np.isfinite(matrix)]

    if finite.size == 0:
        return

    if threshold is None:
        threshold = float(np.nanmin(matrix) + np.nanmax(matrix)) / 2

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]

            if not np.isfinite(value):
                continue

            colour = "white" if value >= threshold else DARK_GREY

            ax.text(
                col,
                row,
                format(value, fmt),
                ha="center",
                va="center",
                fontsize=text_size,
                color=colour,
            )


##########################################################################
## 5. Inter-model agreement heatmap
##########################################################################


def plot_inter_model_agreement() -> None:
    data = pd.read_csv(
        MODEL_AGREEMENT_FILE,
        encoding="utf-8-sig",
    )

    required_cols = {
        "scope",
        "model_key_left",
        "model_key_right",
        "agreement_given_valid_weighted",
    }

    require_columns(
        data,
        required_cols,
        MODEL_AGREEMENT_FILE,
    )

    ## Use all prompt conditions because this represents the complete
    ## stability experiment rather than the canonical prompt alone.
    data = data[
        data["scope"].astype(str) == "all_prompts"
    ].copy()

    if data.empty:
        raise ValueError(
            "No all_prompts rows found in model_pair_agreement.csv."
        )

    observed_models = (
        set(data["model_key_left"].astype(str))
        | set(data["model_key_right"].astype(str))
    )

    validate_model_keys(
        observed_models,
        MODEL_AGREEMENT_FILE,
    )

    matrix = pd.DataFrame(
        np.nan,
        index=MODEL_ORDER,
        columns=MODEL_ORDER,
        dtype=float,
    )

    np.fill_diagonal(
        matrix.values,
        1.0,
    )

    for row in data.itertuples(index=False):
        left = str(row.model_key_left)
        right = str(row.model_key_right)
        value = float(row.agreement_given_valid_weighted)

        matrix.loc[left, right] = value
        matrix.loc[right, left] = value

    if matrix.isna().any().any():
        raise ValueError(
            "Inter-model agreement matrix is incomplete."
        )

    display_names = [
        MODEL_NAMES[model]
        for model in MODEL_ORDER
    ]

    fig, ax = plt.subplots(
        figsize=(9.5, 8.5),
    )

    image = ax.imshow(
        matrix.to_numpy(),
        cmap=OXFORD_SEQUENTIAL,
        vmin=0.70,
        vmax=1.00,
        aspect="equal",
    )

    ax.set_xticks(
        range(len(display_names))
    )
    ax.set_yticks(
        range(len(display_names))
    )

    ax.set_xticklabels(
        display_names,
        rotation=45,
        ha="right",
        fontsize=9.5,
    )

    ax.set_yticklabels(
        display_names,
        fontsize=9.5,
    )

    clean_axes(ax)

    annotate_heatmap(
        ax,
        matrix.to_numpy(),
        fmt=".1%",
        threshold=0.90,
        text_size=8.5,
    )

    colourbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.045,
        pad=0.04,
    )

    colourbar.ax.yaxis.set_major_formatter(
        PercentFormatter(
            xmax=1.0,
            decimals=0,
        )
    )

    colourbar.set_label(
        "Weighted agreement",
        fontsize=10,
    )

    colourbar.outline.set_visible(False)
    colourbar.ax.tick_params(length=0)

    fig.tight_layout()

    caption = (
        "Inter-model agreement across the complete prompt-stability experiment.\n\n"
        "Note. Cells report population-weighted exact agreement conditional on both "
        "models returning valid classifications, using all four prompt conditions. "
        "The diagonal is one by construction. The figure complements the stability "
        "table by visualising model-family dependence: high within-model stability "
        "does not imply that different classifiers instantiate the same measurement "
        "instrument."
    )

    save_figure(
        fig,
        MODEL_AGREEMENT_OUTPUT,
        caption,
    )


##########################################################################
## 6. Prompt-perturbation heatmap
##########################################################################


def plot_prompt_perturbation() -> None:
    data = pd.read_csv(
        PROMPT_COMPARISON_FILE,
        encoding="utf-8-sig",
    )

    required_cols = {
        "model_key",
        "canonical_prompt_id",
        "perturbation_prompt_id",
        "exact_match_given_valid_weighted",
    }

    require_columns(
        data,
        required_cols,
        PROMPT_COMPARISON_FILE,
    )

    observed_models = set(
        data["model_key"].astype(str)
    )

    validate_model_keys(
        observed_models,
        PROMPT_COMPARISON_FILE,
    )

    observed_prompts = set(
        data["perturbation_prompt_id"].astype(str)
    )

    expected_prompts = set(PROMPT_ORDER)

    if observed_prompts != expected_prompts:
        raise ValueError(
            "Prompt-comparison output does not contain exactly the "
            "three expected perturbation conditions."
        )

    matrix = (
        data.pivot(
            index="model_key",
            columns="perturbation_prompt_id",
            values="exact_match_given_valid_weighted",
        )
        .reindex(
            index=MODEL_ORDER,
            columns=PROMPT_ORDER,
        )
    )

    if matrix.isna().any().any():
        raise ValueError(
            "Prompt-perturbation matrix is incomplete."
        )

    model_names = [
        MODEL_NAMES[model]
        for model in MODEL_ORDER
    ]

    prompt_names = [
        PROMPT_NAMES[prompt]
        for prompt in PROMPT_ORDER
    ]

    fig, ax = plt.subplots(
        figsize=(8.5, 6.8),
    )

    image = ax.imshow(
        matrix.to_numpy(dtype=float),
        cmap=OXFORD_SEQUENTIAL,
        vmin=0.90,
        vmax=1.00,
        aspect="auto",
    )

    ax.set_xticks(
        range(len(prompt_names))
    )
    ax.set_yticks(
        range(len(model_names))
    )

    ax.set_xticklabels(
        prompt_names,
        fontsize=10,
    )

    ax.set_yticklabels(
        model_names,
        fontsize=10,
    )

    ax.set_xlabel(
        "Prompt perturbation relative to canonical P0",
        fontsize=11,
    )
    ax.set_ylabel("")

    clean_axes(ax)

    annotate_heatmap(
        ax,
        matrix.to_numpy(dtype=float),
        fmt=".1%",
        threshold=0.965,
        text_size=9,
    )

    colourbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.05,
        pad=0.04,
    )

    colourbar.ax.yaxis.set_major_formatter(
        PercentFormatter(
            xmax=1.0,
            decimals=0,
        )
    )

    colourbar.set_label(
        "Weighted exact agreement",
        fontsize=10,
    )

    colourbar.outline.set_visible(False)
    colourbar.ax.tick_params(length=0)

    fig.tight_layout()

    caption = (
        "Prompt sensitivity relative to the canonical classification prompt.\n\n"
        "Note. Cells report population-weighted exact agreement between the "
        "canonical P0 prompt and each pre-specified perturbation, conditional on "
        "both classifications being valid. The three perturbations alter lexical "
        "wording, official-label ordering, and structural formatting while "
        "preserving the substantive classification task. The figure therefore "
        "shows which aspects of prompt specification produce the greatest "
        "within-model measurement sensitivity."
    )

    save_figure(
        fig,
        PROMPT_PERTURBATION_OUTPUT,
        caption,
    )


##########################################################################
## 7. Misclassification matrix heatmap
##########################################################################


def plot_misclassification_matrix() -> None:
    data = pd.read_csv(
        MISCLASSIFICATION_FILE,
        encoding="utf-8-sig",
    )

    required_cols = {
        "reference_label",
        "prediction_label",
        "estimated_probability",
    }

    require_columns(
        data,
        required_cols,
        MISCLASSIFICATION_FILE,
    )

    data = data.copy()

    data["estimated_probability"] = pd.to_numeric(
        data["estimated_probability"],
        errors="coerce",
    )

    if data["estimated_probability"].isna().any():
        raise ValueError(
            "Misclassification matrix contains missing probabilities."
        )

    reference_order = list(
        dict.fromkeys(
            data["reference_label"]
            .astype(str)
            .tolist()
        )
    )

    prediction_order = list(
        dict.fromkeys(
            data["prediction_label"]
            .astype(str)
            .tolist()
        )
    )

    if len(reference_order) != 17:
        raise ValueError(
            "Expected 17 reference labels in the GB "
            f"misclassification matrix, found {len(reference_order)}."
        )

    if len(prediction_order) != 23:
        raise ValueError(
            "Expected 23 prediction labels in the benchmark inventory, "
            f"found {len(prediction_order)}."
        )

    matrix = (
        data.pivot(
            index="reference_label",
            columns="prediction_label",
            values="estimated_probability",
        )
        .reindex(
            index=reference_order,
            columns=prediction_order,
        )
    )

    if matrix.isna().any().any():
        raise ValueError(
            "Misclassification matrix is incomplete."
        )

    unsupported_predictions = (
        set(prediction_order)
        - set(reference_order)
    )

    reference_labels = [
        pretty_label(label)
        for label in reference_order
    ]

    prediction_labels = [
        (
            f"{pretty_label(label)} †"
            if label in unsupported_predictions
            else pretty_label(label)
        )
        for label in prediction_order
    ]

    fig, ax = plt.subplots(
        figsize=(16.0, 11.0),
    )

    image = ax.imshow(
        matrix.to_numpy(dtype=float),
        cmap=OXFORD_SEQUENTIAL,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )

    ax.set_xticks(
        range(len(prediction_labels))
    )
    ax.set_yticks(
        range(len(reference_labels))
    )

    ax.set_xticklabels(
        prediction_labels,
        rotation=55,
        ha="right",
        fontsize=8.3,
    )

    ax.set_yticklabels(
        reference_labels,
        fontsize=8.7,
    )

    ax.set_xlabel(
        "GPT-5.4 Mini predicted label",
        fontsize=11,
    )

    ax.set_ylabel(
        "Silver-reference label",
        fontsize=11,
    )

    clean_axes(ax)

    ## Annotate only substantively visible pathways. The complete numerical
    ## matrix remains available in the accompanying CSV/table.
    values = matrix.to_numpy(dtype=float)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]

            if value < 0.05:
                continue

            colour = (
                "white"
                if value >= 0.55
                else DARK_GREY
            )

            ax.text(
                col,
                row,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=7.4,
                color=colour,
            )

    colourbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.025,
        pad=0.02,
    )

    colourbar.ax.yaxis.set_major_formatter(
        PercentFormatter(
            xmax=1.0,
            decimals=0,
        )
    )

    colourbar.set_label(
        "Estimated classification probability",
        fontsize=10,
    )

    colourbar.outline.set_visible(False)
    colourbar.ax.tick_params(length=0)

    fig.tight_layout()

    caption = (
        "Estimated GPT-5.4 Mini misclassification mechanism used for prevalence "
        "correction.\n\n"
        "Note. Rows are the 17 silver-reference labels with positive support in "
        "the frozen GB population; columns are the complete 23-label benchmark "
        "prediction inventory. Cell shading reports the estimated design-weighted "
        "probability P(predicted label | silver-reference label) under the canonical "
        "P0 prompt. Numerical annotations are shown for probabilities of at least "
        "5%; the complete values remain available in the accompanying matrix table. "
        "A dagger marks prediction labels that do not occur as silver-reference "
        "categories in the GB population. The strong diagonal represents correct "
        "classification, while off-diagonal cells identify the error pathways "
        "propagated into the downstream prevalence adjustment."
    )

    save_figure(
        fig,
        MISCLASSIFICATION_OUTPUT,
        caption,
    )


##########################################################################
## 8. Observability matrix
##########################################################################


def plot_observability_matrix() -> None:
    data = pd.read_csv(
        OBSERVABILITY_FILE,
        encoding="utf-8-sig",
    )

    required_cols = {
        "model_key",
        *OBSERVABILITY_COLUMNS.keys(),
    }

    require_columns(
        data,
        required_cols,
        OBSERVABILITY_FILE,
    )

    observed_models = set(
        data["model_key"].astype(str)
    )

    validate_model_keys(
        observed_models,
        OBSERVABILITY_FILE,
    )

    data = (
        data.set_index("model_key")
        .reindex(MODEL_ORDER)
        .reset_index()
    )

    def parse_bool(value) -> bool:
        if isinstance(value, bool):
            return value

        normalised = str(value).strip().lower()

        if normalised in {"true", "1", "yes"}:
            return True

        if normalised in {"false", "0", "no"}:
            return False

        raise ValueError(
            f"Cannot interpret observability value as boolean: {value}"
        )

    matrix = np.array(
        [
            [
                int(parse_bool(row[column]))
                for column in OBSERVABILITY_COLUMNS
            ]
            for _, row in data.iterrows()
        ],
        dtype=int,
    )

    model_names = [
        MODEL_NAMES[model]
        for model in MODEL_ORDER
    ]

    column_names = list(
        OBSERVABILITY_COLUMNS.values()
    )

    fig, ax = plt.subplots(
        figsize=(12.5, 7.0),
    )

    ax.imshow(
        matrix,
        cmap=BINARY_OBSERVABILITY,
        vmin=0,
        vmax=1,
        aspect="auto",
    )

    ax.set_xticks(
        range(len(column_names))
    )
    ax.set_yticks(
        range(len(model_names))
    )

    ax.set_xticklabels(
        column_names,
        fontsize=9,
    )

    ax.set_yticklabels(
        model_names,
        fontsize=10,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    clean_axes(ax)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            available = bool(
                matrix[row, col]
            )

            ax.text(
                col,
                row,
                "Yes" if available else "No",
                ha="center",
                va="center",
                fontsize=9,
                color=(
                    "white"
                    if available
                    else MID_GREY
                ),
                fontweight=(
                    "semibold"
                    if available
                    else "normal"
                ),
            )

    ## Separate request-level accounting, local infrastructure measurement,
    ## and provider-side environmental observability.
    ax.axvline(
        3.5,
        color="white",
        linewidth=3.0,
    )

    ax.axvline(
        5.5,
        color="white",
        linewidth=3.0,
    )

    fig.tight_layout()

    caption = (
        "Observability boundaries for computational and environmental reporting.\n\n"
        "Note. Blue cells indicate quantities directly reportable from the frozen "
        "measurement pipeline; grey cells indicate quantities that are not observed "
        "and should not be inferred. Request-level latency, token use, and retry "
        "diagnostics are available across all seven evaluated models. Monetary cost "
        "is reportable for hosted APIs with dated pricing but not for local Ollama "
        "runs. Machine-level energy and carbon estimates are available only for the "
        "two locally executed models measured with CodeCarbon. No hosted provider "
        "exposes the infrastructure telemetry required for provider-side energy or "
        "carbon estimation. The figure operationalises the principle that "
        "environmental precision should not exceed infrastructure knowledge."
    )

    save_figure(
        fig,
        OBSERVABILITY_OUTPUT,
        caption,
    )


##########################################################################
## 9. Main
##########################################################################


def main() -> None:
    plot_inter_model_agreement()
    plot_prompt_perturbation()
    plot_misclassification_matrix()
    plot_observability_matrix()

    outputs = [
        MODEL_AGREEMENT_OUTPUT,
        PROMPT_PERTURBATION_OUTPUT,
        MISCLASSIFICATION_OUTPUT,
        OBSERVABILITY_OUTPUT,
    ]

    print("Wrote manuscript figures:")

    for output in outputs:
        print(f"  {output}.png")
        print(f"  {output}.pdf")
        print(f"  {output}.txt")


if __name__ == "__main__":
    main()
