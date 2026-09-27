from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import pandas as pd

##########################################################################
## 1. Paths
##########################################################################

INPUT_FILE = Path("outputs/propagation/prevalence_comparison.csv")
OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_NAME = OUTPUT_DIR / "prevalence_propagation"

##########################################################################
## 2. Plotting palette
##########################################################################

OXFORD_BLUE = "#5F7A9D"
SOFT_TERRACOTTA = "#E6A57E"
DARK_GREY = "#4D4D4D"
LIGHT_GREY = "#D9D9D9"

##########################################################################
## 3. Helpers
##########################################################################


def require_columns(df: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")


def pretty_label(value: str) -> str:
    return str(value).strip().replace("_", " ")


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


def clean_axes(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.tick_params(axis="both", length=0)


##########################################################################
## 4. Load and prepare data
##########################################################################

data = pd.read_csv(INPUT_FILE)

required_cols = {
    "label",
    "present_in_silver_reference_population",
    "silver_true_prevalence",
    "naive_classifier_prevalence",
    "corrected_prevalence",
    "corrected_ci_low",
    "corrected_ci_high",
}
require_columns(data, required_cols, INPUT_FILE)

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

numeric_cols = [
    "silver_true_prevalence",
    "naive_classifier_prevalence",
    "corrected_prevalence",
    "corrected_ci_low",
    "corrected_ci_high",
]

for column in numeric_cols:
    data[column] = pd.to_numeric(data[column], errors="coerce")

if data[numeric_cols].isna().any().any():
    raise ValueError(f"{INPUT_FILE} contains missing prevalence values.")

data["label_pretty"] = data["label"].map(pretty_label)

## Preserve the earlier visual ordering: categories are ordered by
## silver-reference prevalence rather than by error magnitude.
data = data.sort_values(
    ["silver_true_prevalence", "label_pretty"],
    ascending=[True, True],
).reset_index(drop=True)

data["ypos"] = range(len(data))

##########################################################################
## 5. Plot
##########################################################################

fig, ax = plt.subplots(figsize=(11.2, 9.0))

for row in data.itertuples(index=False):
    ## Light connector between the naive and corrected prevalence estimates.
    ax.plot(
        [
            min(row.naive_classifier_prevalence, row.corrected_prevalence),
            max(row.naive_classifier_prevalence, row.corrected_prevalence),
        ],
        [row.ypos, row.ypos],
        color=LIGHT_GREY,
        linewidth=1.5,
        alpha=0.95,
        zorder=1,
    )

    ## Corrected prevalence and propagated uncertainty.
    ax.errorbar(
        row.corrected_prevalence,
        row.ypos,
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

    ## Naive full-population classifier prevalence.
    ax.scatter(
        row.naive_classifier_prevalence,
        row.ypos,
        s=58,
        color=SOFT_TERRACOTTA,
        marker="o",
        alpha=0.92,
        edgecolors=DARK_GREY,
        linewidths=0.5,
        zorder=4,
    )

    ## Known silver-reference population prevalence.
    ax.scatter(
        row.silver_true_prevalence,
        row.ypos,
        s=62,
        color=DARK_GREY,
        marker="D",
        alpha=0.92,
        edgecolors=DARK_GREY,
        linewidths=0.5,
        zorder=5,
    )

##########################################################################
## 6. Axes
##########################################################################

maximum = max(
    data["silver_true_prevalence"].max(),
    data["naive_classifier_prevalence"].max(),
    data["corrected_ci_high"].max(),
)

upper = min(
    1.0,
    max(
        0.20,
        (int(maximum * 100 / 5) + 2) * 0.05,
    ),
)

ax.set_xlabel("Share of GB test speeches", fontsize=12)
ax.set_ylabel("")
ax.set_xlim(0, upper)
ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

ax.set_yticks(data["ypos"])
ax.set_yticklabels(data["label_pretty"], fontsize=9.5)

clean_axes(ax)

##########################################################################
## 7. Legend and caption
##########################################################################

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
        label="Naive GPT-5.4 Mini prevalence",
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

plt.tight_layout()
plt.subplots_adjust(bottom=0.16, left=0.25)

fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.025),
    ncol=3,
    frameon=False,
)

caption = (
    "Propagation of classification error to policy-agenda prevalence.\n\n"
    "Note. Seventeen policy categories with positive support in the frozen GB silver-reference "
    "population are shown. Grey diamonds show the known silver-reference prevalence; terracotta "
    "circles show the naive GPT-5.4 Mini full-population prevalence; blue squares show the "
    "support-conditioned corrected prevalence with propagated 95% bootstrap intervals. "
    "Categories are ordered by silver-reference prevalence. Full-population silver prevalence "
    "is used only for ex-post comparison, not to fit the correction."
)

write_caption(OUTPUT_NAME, caption)

##########################################################################
## 8. Save
##########################################################################

save_current_figure(OUTPUT_NAME)

print(f"Wrote: {OUTPUT_NAME}.png")
print(f"Wrote: {OUTPUT_NAME}.pdf")
print(f"Wrote: {OUTPUT_NAME}.txt")
