"""Regression tests for the offline, fixed-energy country counterfactual."""
from pathlib import Path
from runpy import run_path

import matplotlib.image as mpimg
import numpy as np
import pytest

from reporting_validation_standards import plot_style as ps

ROOT = Path(__file__).resolve().parents[1]
RUNNER = run_path(str(ROOT / "code" / "17_simulate_country_carbon.py"))
FACTORS = ROOT / "config/geography/codecarbon_country_intensity_v3.3.1.csv"
SUMMARY = ROOT / "outputs/compute_environment/local_environment_summary.csv"
BOUNDARIES = ROOT / "config/geography/natural_earth_110m_countries_simplified.geojson"


def originals():
    factors = RUNNER["load_factors"](FACTORS)
    models = RUNNER["load_local_runs"](SUMMARY, list(RUNNER["LOCAL_MODELS"]))
    return factors, models


def test_counterfactuals_use_frozen_energy_without_retracking():
    factors, models = originals()
    result = RUNNER["build_counterfactuals"](models, factors)
    assert len(factors) >= 200
    assert len(result) == len(factors) * 2
    assert set(result["model_key"]) == set(RUNNER["LOCAL_MODELS"])
    for model_key in RUNNER["LOCAL_MODELS"]:
        data = result[result["model_key"] == model_key].set_index("country_iso3")
        assert data["measured_energy_kwh"].nunique() == 1
        assert data["n_decisions"].nunique() == 1
        assert data["n_decisions"].iloc[0] == 12000
        nld = data.loc["NLD"]
        fra = data.loc["FRA"]
        ind = data.loc["IND"]
        assert np.isclose(
            fra["counterfactual_g_co2e_per_1000_decisions"]
            / nld["counterfactual_g_co2e_per_1000_decisions"],
            fra["grid_intensity_g_co2e_per_kwh"]
            / nld["grid_intensity_g_co2e_per_kwh"],
        )
        assert ind["counterfactual_g_co2e_per_1000_decisions"] > nld[
            "counterfactual_g_co2e_per_1000_decisions"
        ] > fra["counterfactual_g_co2e_per_1000_decisions"]
    qwen = result[result["model_key"] == "ollama_qwen36_35b"].set_index("country_iso3")
    gemma = result[result["model_key"] == "ollama_gemma4_31b"].set_index("country_iso3")
    assert qwen.loc["NLD", "counterfactual_g_co2e_per_1000_decisions"] == pytest.approx(
        38.468643443582, abs=1e-8
    )
    assert gemma.loc["NLD", "counterfactual_g_co2e_per_1000_decisions"] == pytest.approx(
        72.06938605102717, abs=1e-8
    )
    assert qwen.loc["FRA", "counterfactual_g_co2e_per_1000_decisions"] == pytest.approx(
        8.055, abs=0.01
    )


def test_rejects_zero_or_unknown_country_factors(tmp_path):
    factors, models = originals()
    factors.loc[factors["iso3"] == "FRA", "intensity_g_co2e_per_kwh"] = 0.0
    bad_factors = tmp_path / "factors.csv"
    factors.to_csv(bad_factors, index=False)
    with pytest.raises(ValueError, match="positive"):
        RUNNER["load_factors"](bad_factors)
    factors = RUNNER["load_factors"](FACTORS)
    factors = factors[factors["iso3"] != "NLD"]
    with pytest.raises(ValueError, match="No country factor"):
        RUNNER["build_counterfactuals"](models, factors)


def test_rejects_mismatched_country_intensity_and_invalid_scope():
    factors, models = originals()
    mutated = models.copy()
    mutated.loc[mutated["model_key"] == "ollama_qwen36_35b", "emissions_kg_co2e"] = 1.0
    with pytest.raises(ValueError, match="baseline emissions"):
        RUNNER["build_counterfactuals"](mutated, factors)
    with pytest.raises(ValueError, match="Only instrumented local models"):
        RUNNER["load_local_runs"](SUMMARY, ["openai_gpt54"])


def test_equal_earth_projection_reference_points():
    x, y = RUNNER["equal_earth"](np.array([0.0, 180.0, 0.0]), np.array([0.0, 0.0, 90.0]))
    assert x[0] == pytest.approx(0.0) and y[0] == pytest.approx(0.0)
    assert x[1] == pytest.approx(2.70663, abs=1e-4)
    assert y[2] == pytest.approx(1.3173, abs=1e-4)


def test_combined_map_is_transparent_grid_free_and_exports(tmp_path):
    factors, models = originals()
    results = RUNNER["build_counterfactuals"](models, factors)
    boundaries = RUNNER["read_boundaries"](BOUNDARIES)
    keys = list(RUNNER["LOCAL_MODELS"])
    fig, matched = RUNNER["make_maps"](results, boundaries, model_keys=keys)
    assert set(matched) == set(keys)
    assert all(count >= 160 for count in matched.values())
    assert ps.paper_problems(fig) == []
    assert fig.get_figwidth() == RUNNER["FIGURE_WIDTH"]
    maps = [axes for axes in fig.axes if axes.get_title(loc="left")]
    assert [axes.get_title(loc="left") for axes in maps] == [RUNNER["PANEL_TITLES"][k] for k in keys]
    for axes in maps:
        assert all(not spine.get_visible() for spine in axes.spines.values())
        assert not axes.axison
    assert [t.get_text() for t in fig.legends[0].get_texts()] == ["No data"]
    caption = RUNNER["caption_for"](results, max(matched.values()))
    title, note = caption.split("\n\n", 1)
    assert "counterfactual" in title.lower()
    assert note.startswith("Note.")
    assert "per 1,000" in note
    stem = tmp_path / "local_carbon_country_ollama"
    RUNNER["write_maps"](fig, stem, caption)
    for suffix in (".png", ".svg", ".pdf", ".txt"):
        assert stem.with_suffix(suffix).exists()
        assert stem.with_suffix(suffix).stat().st_size > 0
    assert b"/Subtype /Image" not in stem.with_suffix(".pdf").read_bytes()
    width, _ = ps.pdf_size(stem.with_suffix(".pdf").read_bytes())
    assert width == pytest.approx(RUNNER["FIGURE_WIDTH"], abs=0.01)
    png = mpimg.imread(stem.with_suffix(".png"))
    assert png.shape[-1] == 4
    assert png[:, :, 3].min() == 0.0
    assert png[:, :, 3].max() > 0.0
