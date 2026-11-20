"""Regression tests for the offline, fixed-energy country counterfactual."""
from pathlib import Path
from runpy import run_path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pytest


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


def test_map_is_transparent_grid_free_and_exports(tmp_path):
    factors, models = originals()
    results = RUNNER["build_counterfactuals"](models, factors)
    boundaries = RUNNER["read_boundaries"](BOUNDARIES)
    fig, matched = RUNNER["make_map"](
        results, boundaries, model_key="ollama_qwen36_35b"
    )
    assert len(matched) >= 160
    assert not fig.axes[0].get_title()
    assert all(not spine.get_visible() for spine in fig.axes[0].spines.values())
    assert not any(line.get_visible() for line in fig.axes[0].get_xgridlines())
    assert not any(line.get_visible() for line in fig.axes[0].get_ygridlines())
    caption = RUNNER["caption_for"](
        results[results["model_key"] == "ollama_qwen36_35b"],
        len(matched),
    )
    assert "counterfactual" in caption.lower()
    assert "per 1,000" in caption
    stem = tmp_path / "local_carbon_country_ollama_qwen36_35b"
    RUNNER["write_map"](fig, stem, caption)
    for suffix in (".png", ".svg", ".pdf", ".txt"):
        assert stem.with_suffix(suffix).exists()
        assert stem.with_suffix(suffix).stat().st_size > 0
    png = mpimg.imread(stem.with_suffix(".png"))
    assert png.shape[-1] == 4
    assert png[:, :, 3].min() == 0.0
    assert png[:, :, 3].max() > 0.0
