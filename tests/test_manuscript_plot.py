"""The validation profile figure is complete and readable at its printed size."""
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
import pytest

from reporting_validation_standards import plot_style as ps

ROOT = Path(__file__).resolve().parents[1]
RUNNER = run_path(str(ROOT / "code" / "13_generate_manuscript_plot.py"))
INTERVALS = run_path(str(ROOT / "code" / "08b_analyze_prompt_intervals.py"))
P0 = "p0_capbench_canonical"
MODELS = list(RUNNER["MODEL_NAMES"])


def synthetic_conditions(drop_last: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for model in MODELS:
        for prompt in RUNNER["PROMPT_ORDER"]:
            for repetition in (0, 1, 2, 3):
                row = {
                    "model_key": model, "prompt_id": prompt, "repetition": repetition,
                    "aggregation": "expected_over_repetitions" if repetition == 0
                    else "single_repetition",
                }
                row.update({column: rng.uniform(.8, .9)
                            for column in RUNNER["METRICS"].values()})
                rows.append(row)
    data = pd.DataFrame(rows)
    return data.iloc[:-1] if drop_last else data


def synthetic_intervals(conditions: pd.DataFrame) -> pd.DataFrame:
    pooled = conditions.loc[conditions["repetition"].eq(0)].copy()
    for column in RUNNER["METRICS"].values():
        pooled[f"{column}_ci_low"] = pooled[column] - .03
        pooled[f"{column}_ci_high"] = pooled[column] + .04
    keep = ["model_key", "prompt_id"] + [
        c for c in pooled if c.startswith("weighted_operational_")
    ]
    return pooled[keep].reset_index(drop=True)


def synthetic_primary() -> pd.DataFrame:
    scores = np.linspace(.9, .7, len(MODELS))
    return pd.DataFrame({"model_key": MODELS[::-1], "weighted_operational_macro_f1": scores})


def test_figure_has_four_panels_at_print_size(tmp_path):
    conditions = synthetic_conditions()
    intervals = RUNNER["prepare_intervals"](synthetic_intervals(conditions))
    order = RUNNER["model_order"](synthetic_primary())
    fig = RUNNER["build_figure"](intervals, order)
    panels = [axes for axes in fig.axes if axes.get_title()]
    assert [axes.get_title() for axes in panels] == list(RUNNER["METRICS"])
    assert fig.get_figwidth() == ps.print_width(0.8)
    assert ps.paper_problems(fig) == []
    assert ps.min_font_size(fig) >= ps.MIN_FONT
    labels = [text.get_text() for text in fig.legends[0].get_texts()]
    assert labels[-1] == "95% CI" and len(labels) == 5
    tick_labels = [t.get_text() for t in panels[0].get_yticklabels()]
    assert tick_labels == [RUNNER["MODEL_NAMES"][m] for m in order]
    caption = RUNNER["caption"](RUNNER["prepare"](conditions))
    paths = ps.save(fig, tmp_path / "profile", caption)
    assert {path.suffix for path in paths} == {".png", ".pdf", ".svg", ".txt"}


def test_models_follow_primary_macro_f1():
    order = RUNNER["model_order"](synthetic_primary())
    assert order == MODELS[::-1]
    with pytest.raises(ValueError, match="seven models"):
        RUNNER["model_order"](synthetic_primary().iloc[1:])


def test_caption_reports_repetition_spread_and_ordering():
    data = RUNNER["prepare"](synthetic_conditions())
    spread = RUNNER["repetition_spread"](data)
    assert 0 <= spread["median"] <= spread["max"]
    text = RUNNER["caption"](data)
    title, note = text.split("\n\n", 1)
    assert title == RUNNER["TITLE"] and note.startswith("Note.")
    assert f"at most {spread['max']:.3f}" in note
    assert "ordered by their primary P0 macro-F1" in note
    assert "fixed experimental order" not in note


def test_incomplete_conditions_fail_closed():
    with pytest.raises(ValueError, match="incomplete|Expected"):
        RUNNER["prepare"](synthetic_conditions(drop_last=True))


def test_intervals_must_cover_every_cell_and_stay_in_range():
    intervals = synthetic_intervals(synthetic_conditions())
    with pytest.raises(ValueError, match="one row per model"):
        RUNNER["prepare_intervals"](intervals.iloc[1:])
    wide = intervals.copy()
    wide.loc[0, "weighted_operational_macro_f1_ci_low"] = .5
    with pytest.raises(ValueError, match="outside the plotted range"):
        RUNNER["prepare_intervals"](wide)


def test_prompt_intervals_must_reproduce_committed_estimates():
    conditions = synthetic_conditions()
    intervals = synthetic_intervals(conditions)
    primary = intervals.loc[intervals["prompt_id"].eq(P0)].drop(columns="prompt_id")
    INTERVALS["check_against_committed"](intervals, conditions, primary)

    shifted = primary.copy()
    shifted["weighted_operational_accuracy_ci_high"] += 1e-6
    with pytest.raises(ValueError, match="P0"):
        INTERVALS["check_against_committed"](intervals, conditions, shifted)

    moved = intervals.copy()
    moved.loc[3, "weighted_operational_macro_recall"] += 1e-6
    with pytest.raises(ValueError, match="condition_performance"):
        INTERVALS["check_against_committed"](moved, conditions, primary)
