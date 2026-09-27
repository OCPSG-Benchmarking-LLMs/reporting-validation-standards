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

## Inspired by the Oxford-style softened palette used in the related
## Eval4SD plotting scripts. The palette is intentionally restrained:
## reference prevalence is neutral, the naive classifier estimate is warm,
## and the corrected estimate uses softened Oxford blue.

OXFORD_BLUE = "#5F7A9D"
SOFT_TERRACOTTA = "#E6A57E"
DARK_GREY = "#4D4D4D"
MID_GREY = "#999999"
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
    "naive_minus_true",
    "corrected_minus_true",
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
    "naive_minus_true",
    "corrected_minus_true",
]

for column in numeric_cols:
    data[column] = pd.to_numeric(data[column], errors="coerce")

if data[numeric_cols].isna().any().any():
    raise ValueError(f"{INPUT_FILE} contains missing prevalence values.")

data["label_pretty"] = data["label"].map(pretty_label)
data["absolute_naive_error"] = data["naive_minus_true"].abs()
data["absolute_corrected_error"] = data["corrected_minus_true"].abs()

## Order by the magnitude of the naive aggregate error. This makes the plot
## diagnostic rather than merely descriptive: the categories for which
## correction matters most appear first.
data = data.sort_values(
    ["absolute_naive_error", "label_pretty"],
    ascending=[False, True],
).reset_index(drop=True)

data["ypos"] = range(len(data))

##########################################################################
## 5. Plot
##########################################################################

fig, ax = plt.subplots(figsize=(12.5, 9.5))

## Direction of correction: naive estimate -> corrected estimate.
for row in data.itertuples(index=False):
    ax.annotate(
        "",
        xy=(row.corrected_prevalence, row.ypos),
        xytext=(row.naive_classifier_prevalence, row.ypos),
        arrowprops={
            "arrowstyle": "-|>",
            "color": LIGHT_GREY,
            "linewidth": 1.6,
            "shrinkA": 5,
            "shrinkB": 7,
            "mutation_scale": 10,
        },
        zorder=1,
    )

## Corrected prevalence and propagated uncertainty.
corrected_xerr = [
    data["corrected_prevalence"] - data["corrected_ci_low"],
    data["corrected_ci_high"] - data["corrected_prevalence"],
]

ax.errorbar(
    data["corrected_prevalence"],
    data["ypos"],
    xerr=corrected_xerr,
    fmt="s",
    markersize=7.5,
    color=OXFORD_BLUE,
    ecolor=OXFORD_BLUE,
    markeredgecolor=DARK_GREY,
    markeredgewidth=0.55,
    elinewidth=1.5,
    capsize=3.0,
    alpha=0.95,
    zorder=3,
)

## Naive full-population classifier prevalence.
ax.scatter(
    data["naive_classifier_prevalence"],
    data["ypos"],
    s=72,
    color=SOFT_TERRACOTTA,
    marker="o",
    alpha=0.90,
    edgecolors=DARK_GREY,
    linewidths=0.55,
    zorder=4,
)

## Known silver-reference population prevalence. This is the diagnostic target
## and is not used to fit the correction.
ax.scatter(
    data["silver_true_prevalence"],
    data["ypos"],
    s=76,
    color=DARK_GREY,
    marker="D",
    alpha=0.90,
    edgecolors=DARK_GREY,
    linewidths=0.50,
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

## Round the upper limit to the next five-percentage-point increment.
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
ax.invert_yaxis()

## Keep only a light x-axis guide, following the clean comparison style of
## stage_4_f1_comparison.py without adding a full grid.
ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.9, alpha=0.75, zorder=0)
ax.grid(axis="y", visible=False)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax.spines["left"].set_color(MID_GREY)
ax.spines["bottom"].set_color(MID_GREY)

ax.tick_params(axis="both", length=0)

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
        markeredgewidth=0.55,
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
        markeredgewidth=0.55,
        label="Corrected prevalence (95% CI)",
    ),
]

plt.tight_layout()
plt.subplots_adjust(bottom=0.15, left=0.24)

fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.035),
    ncol=3,
    frameon=False,
)

caption = (
    "Propagation of classification error to policy-agenda prevalence.\n\n"
    "Note. The figure shows the 17 policy categories with positive support in the frozen "
    "GB silver-reference population, ordered by the absolute error of the naive GPT-5.4 Mini "
    "prevalence estimate. Grey diamonds show the known silver-reference prevalence; terracotta "
    "circles show the naive full-population classifier prevalence; blue squares show the "
    "support-conditioned corrected prevalence with 95% intervals from 2,000 Rao-Wu bootstrap "
    "replicates. Light arrows indicate the direction of correction from the naive to the corrected "
    "estimate. Full-population silver prevalence is used only for ex-post comparison, not to fit "
    "the correction."
)

write_caption(OUTPUT_NAME, caption)

##########################################################################
## 8. Save
##########################################################################

save_current_figure(OUTPUT_NAME)

print(f"Wrote: {OUTPUT_NAME}.png")
print(f"Wrote: {OUTPUT_NAME}.pdf")
print(f"Wrote: {OUTPUT_NAME}.txt")
