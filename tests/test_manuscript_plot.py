"""The validation profile figure is complete and readable at its printed size."""
from pathlib import Path
from runpy import run_path

import numpy as np
import pandas as pd
import pytest

from reporting_validation_standards import plot_style as ps

SCRIPT = Path(__file__).resolve().parents[1] / "code" / "13_generate_manuscript_plot.py"
RUNNER = run_path(str(SCRIPT))


def synthetic_conditions(drop_last: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for model in RUNNER["MODEL_ORDER"]:
        for prompt in RUNNER["PROMPT_ORDER"]:
            for repetition in (0, 1, 2, 3):
                row = {
                    "model_key": model, "prompt_id": prompt, "repetition": repetition,
                    "aggregation": "expected_over_repetitions" if repetition == 0
                    else "single_repetition",
                }
                row.update({column: rng.uniform(.75, .95)
                            for column in RUNNER["METRICS"].values()})
                rows.append(row)
    data = pd.DataFrame(rows)
    return data.iloc[:-1] if drop_last else data


def test_figure_has_four_panels_at_print_size(tmp_path):
    data = RUNNER["prepare"](synthetic_conditions())
    assert len(data) == 7 * 4 * 3
    fig = RUNNER["build_figure"](data)
    panels = [axes for axes in fig.axes if axes.get_title()]
    assert [axes.get_title() for axes in panels] == list(RUNNER["METRICS"])
    assert fig.get_figwidth() == ps.PRINT_WIDTH_IN
    assert ps.paper_problems(fig) == []
    assert ps.min_font_size(fig) >= ps.MIN_FONT
    assert len(fig.legends) == 1 and len(fig.legends[0].get_texts()) == 4
    paths = ps.save(fig, tmp_path / "profile", RUNNER["CAPTION"])
    assert {path.suffix for path in paths} == {".png", ".pdf", ".svg", ".txt"}


def test_incomplete_conditions_fail_closed():
    with pytest.raises(ValueError, match="incomplete|Expected"):
        RUNNER["prepare"](synthetic_conditions(drop_last=True))
