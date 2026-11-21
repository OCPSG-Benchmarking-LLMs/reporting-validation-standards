"""Shared matplotlib style for the manuscript figures.

Every figure is drawn at the width at which the manuscript prints it
(``print_width(fraction)`` for ``\\includegraphics[width=<fraction>\\linewidth]``), so the
font sizes set here are the sizes a reader sees on the page. The conventions are:

- no spines, tick marks or gridlines, and one frameless legend below the panels;
- light fills with dark-grey edges, in a softened Oxford palette that remains
  distinguishable under the common forms of colour-vision deficiency;
- dashed coral reference lines, always named in the legend;
- no figure title and no text on the canvas: the title and the explanatory note go in a
  same-basename ``.txt`` caption file ("Title", a blank line, then "Note. ...");
- transparent PNG at 300 dpi plus vector PDF and SVG with editable text.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

TEXT_WIDTH_IN = 6.5       # the manuscript's text block (US Letter, 1-inch margins)
MIN_FONT = 7.0


def print_width(fraction: float) -> float:
    """Width in inches of a figure set at ``fraction`` of the text width."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction is the share of \\linewidth, in (0, 1]")
    return round(TEXT_WIDTH_IN * fraction, 4)


EDGE = "#4D4D4D"          # marker and bar edges
TEXT = "#211D1C"
MUTED = "#61615F"
LIGHT_GREY = "#D9D9D9"    # uncertainty ribbons and context series
BORDER = "#9A9A9A"        # map borders
NO_DATA = "#F2F2F2"       # maps: missing data, drawn with /// hatching
REFERENCE = "#FE615A"     # reference lines: dashed and labelled
FOCAL = "#002147"

SERIES = ["#5F7A9D", "#E6A57E", "#ACE0B8", "#5E4161", "#4FB3D1"]
SERIES_NAMES = ["blue", "terracotta", "light sage", "deep plum", "sky aqua"]
BLUE, TERRACOTTA, SAGE, PLUM, SKY_AQUA = SERIES

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "rvs_sequential", ["#EDF2F7", "#B9D6F2", "#5F7A9D", "#002147"], N=256
)


def use_style() -> None:
    """Set the house rcParams (sizes are print sizes, given figures drawn at print width)."""
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8.5,
        "text.color": TEXT, "axes.labelcolor": TEXT, "axes.titlecolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.spines.bottom": False, "xtick.major.size": 0, "ytick.major.size": 0,
        "xtick.minor.size": 0, "ytick.minor.size": 0, "axes.grid": False,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "patch.edgecolor": EDGE, "lines.markeredgecolor": EDGE,
        "hatch.color": BORDER, "hatch.linewidth": 0.5,
    })


def clean_axes(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.grid(False)


def bottom_legend(fig, handles, labels=None, ncol=None, y=0.0):
    """One frameless legend centred below all panels."""
    if labels is None:
        labels = [h.get_label() for h in handles]
    return fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, y),
        ncol=ncol or min(len(handles), 4), frameon=False,
    )


def reference_line(ax, value, label, axis="x", **kwargs):
    """A dashed coral reference line that always carries a legend label."""
    if not label:
        raise ValueError("a reference line needs a label")
    style = {"color": REFERENCE, "linestyle": "--", "linewidth": 1.3, "zorder": 1,
             "label": label}
    style.update(kwargs)
    return ax.axvline(value, **style) if axis == "x" else ax.axhline(value, **style)


def min_font_size(fig) -> float | None:
    fig.canvas.draw()
    sizes = [
        t.get_fontsize() for t in fig.findobj(mpl.text.Text)
        if t.get_visible() and t.get_text().strip()
    ]
    return min(sizes) if sizes else None


def paper_problems(fig) -> list[str]:
    """What a manuscript figure must not carry."""
    out = []
    if fig.texts:
        out.append(f"{len(fig.texts)} text(s) on the canvas: the note goes in the caption")
    if getattr(fig, "_suptitle", None) is not None and fig._suptitle.get_text():
        out.append("a figure title: it goes in the caption")
    if any(ax.texts for ax in fig.axes):
        out.append("text inside an axes: the note goes in the caption")
    panels = [a for a in fig.axes if a.get_label() != "<colorbar>"]
    if len(panels) == 1 and panels[0].get_title():
        out.append("a title on a single-panel figure: it goes in the caption")
    for ax in fig.axes:
        if any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines()):
            out.append("visible gridlines")
            break
    small = min_font_size(fig)
    if small is not None and small < MIN_FONT:
        out.append(f"text at {small:g} pt (minimum {MIN_FONT:g})")
    return out


def save(fig, stem, caption: str, svg: bool = True) -> list[Path]:
    """Write transparent PNG, PDF (and SVG) plus the ``.txt`` caption; lay out first."""
    problems = paper_problems(fig)
    if problems:
        raise ValueError("manuscript figure: " + "; ".join(problems))
    parts = caption.strip().split("\n\n", 1)
    if len(parts) != 2 or not parts[1].startswith("Note."):
        raise ValueError("caption must be 'Title', a blank line, then 'Note. ...'")
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf", "svg") if svg else ("png", "pdf"):
        path = stem.with_suffix("." + ext)
        fig.savefig(path, transparent=True, metadata=_metadata(ext))
        paths.append(path)
    txt = stem.with_suffix(".txt")
    txt.write_text(caption.strip() + "\n", encoding="utf-8")
    paths.append(txt)
    return paths


def _metadata(ext: str) -> dict:
    # Stable vector output: no creation dates, so regenerated files differ only when the
    # figure does.
    if ext == "pdf":
        return {"CreationDate": None}
    if ext == "svg":
        return {"Date": None}
    return {}
