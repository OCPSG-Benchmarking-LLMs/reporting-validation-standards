#!/usr/bin/env python3
"""Map country-counterfactual emissions using frozen local CodeCarbon energy.

No model inference, API call, CodeCarbon tracker, or online map lookup is made.
Only measured local Ollama run energy is held fixed across national grid factors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Polygon

from reporting_validation_standards import plot_style as ps

SUMMARY = Path("outputs/compute_environment/local_environment_summary.csv")
FACTORS = Path("config/geography/codecarbon_country_intensity_v3.3.1.csv")
BOUNDARIES = Path("config/geography/natural_earth_110m_countries_simplified.geojson")
TABLE = Path("outputs/compute_environment/local_country_emissions_counterfactuals.csv")
FIGURES = Path("outputs/figures")

LOCAL_MODELS = {
    "ollama_qwen36_35b": "Qwen 3.6 35B",
    "ollama_gemma4_31b": "Gemma 4 31B",
}
SOURCE_CODECARBON_VERSION = "3.3.1"
INTENSITY_SOURCE = (
    "https://github.com/mlco2/codecarbon/blob/v3.3.1/"
    "codecarbon/data/private_infra/global_energy_mix.json"
)
GEOGRAPHY_SOURCE = (
    "https://github.com/nvkelso/natural-earth-vector/blob/master/"
    "geojson/ne_110m_admin_0_countries.geojson"
)
# Display names in the manuscript; LOCAL_MODELS keeps the names in the CSV.
PANEL_TITLES = {
    "ollama_qwen36_35b": "Qwen3.6 (35B)",
    "ollama_gemma4_31b": "Gemma 4 (31B)",
}
FIGURE_STEM = "local_carbon_country_ollama"

def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_factors(path: Path) -> pd.DataFrame:
    factors = pd.read_csv(path, dtype={"iso3": str})
    expected = {"iso3", "country", "source_year", "intensity_g_co2e_per_kwh"}
    if not expected.issubset(factors.columns):
        raise ValueError(f"Missing intensity columns: {sorted(expected - set(factors))}")
    if factors.empty or factors["iso3"].isna().any() or factors["iso3"].duplicated().any():
        raise ValueError("Intensity table must contain unique, non-null countries")
    if not factors["iso3"].str.fullmatch("[A-Z]{3}").all():
        raise ValueError("Intensity country codes must be ISO-alpha-3 uppercase")
    if factors["country"].isna().any():
        raise ValueError("Country names must be present")
    values = pd.to_numeric(factors["intensity_g_co2e_per_kwh"], errors="raise")
    years = pd.to_numeric(factors["source_year"], errors="raise")
    if not values.map(lambda v: math.isfinite(v) and v > 0).all():
        raise ValueError("Country carbon intensities must be finite and positive")
    if not years.map(lambda v: math.isfinite(v) and v == int(v) and v >= 2000).all():
        raise ValueError("Country source years must be valid")
    factors = factors.copy()
    factors["intensity_g_co2e_per_kwh"] = values.astype(float)
    factors["source_year"] = years.astype(int)
    return factors.sort_values("iso3").reset_index(drop=True)


def load_local_runs(path: Path, model_keys: list[str]) -> pd.DataFrame:
    models = pd.read_csv(path)
    needed = {
        "model_key", "decisions", "energy_consumed_kwh",
        "emissions_kg_co2e", "country_iso_code", "codecarbon_version",
        "tracking_mode",
    }
    if not needed.issubset(models.columns):
        raise ValueError(f"Missing local run columns: {sorted(needed - set(models))}")
    if models["model_key"].duplicated().any():
        raise ValueError("Duplicate local model keys")
    unknown = set(model_keys) - set(LOCAL_MODELS)
    if unknown:
        raise ValueError(f"Only instrumented local models are supported: {sorted(unknown)}")
    subset = models.loc[models["model_key"].isin(model_keys)].copy()
    if len(subset) != len(model_keys):
        raise ValueError("Missing one or more requested measured local models")
    for row in subset.itertuples(index=False):
        if row.codecarbon_version != SOURCE_CODECARBON_VERSION:
            raise ValueError(
                f"{row.model_key} measured with CodeCarbon {row.codecarbon_version}, "
                f"but these country factors are from {SOURCE_CODECARBON_VERSION}"
            )
        if row.tracking_mode != "machine":
            raise ValueError("Only measured machine-level local energy is supported")
        if (
            not math.isfinite(float(row.energy_consumed_kwh))
            or float(row.energy_consumed_kwh) <= 0
            or not math.isfinite(float(row.emissions_kg_co2e))
            or float(row.emissions_kg_co2e) < 0
            or int(row.decisions) != float(row.decisions)
            or int(row.decisions) <= 0
        ):
            raise ValueError(f"Invalid measured totals for {row.model_key}")
    return subset.sort_values("model_key").reset_index(drop=True)


def build_counterfactuals(models: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Use E_measured * I_country / 1000; never estimate hosted-provider energy."""
    by_iso = factors.set_index("iso3")
    rows = []
    for model in models.itertuples(index=False):
        baseline = str(model.country_iso_code).upper()
        if baseline not in by_iso.index:
            raise ValueError(f"No country factor for measured baseline {baseline}")
        measured_kwh = float(model.energy_consumed_kwh)
        recorded_emissions_kg = float(model.emissions_kg_co2e)
        baseline_intensity = float(by_iso.loc[baseline, "intensity_g_co2e_per_kwh"])
        expected_recorded_kg = measured_kwh * baseline_intensity / 1000.0
        if not math.isclose(
            expected_recorded_kg, recorded_emissions_kg, rel_tol=0, abs_tol=1e-7
        ):
            raise ValueError(
                f"{model.model_key}: baseline emissions do not match pinned CodeCarbon "
                "country intensity; do not mix factor versions or overwrite measured energy"
            )
        for c in factors.itertuples(index=False):
            mass_g = measured_kwh * float(c.intensity_g_co2e_per_kwh)
            rows.append({
                "model_key": model.model_key,
                "model": LOCAL_MODELS[model.model_key],
                "country_iso3": c.iso3,
                "country": c.country,
                "intensity_source_year": int(c.source_year),
                "grid_intensity_g_co2e_per_kwh": float(c.intensity_g_co2e_per_kwh),
                "measured_baseline_country_iso3": baseline,
                "measured_energy_kwh": measured_kwh,
                "n_decisions": int(model.decisions),
                "counterfactual_emissions_kg_co2e": mass_g / 1000.0,
                "counterfactual_g_co2e_per_1000_decisions": (
                    mass_g * 1000.0 / int(model.decisions)
                ),
                "ratio_to_measured_country": float(c.intensity_g_co2e_per_kwh)
                / baseline_intensity,
            })
    return pd.DataFrame(rows).sort_values(
        ["model_key", "country_iso3"]
    ).reset_index(drop=True)


def read_boundaries(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") != "FeatureCollection":
        raise ValueError("Expected a pinned Natural Earth FeatureCollection")
    result = data.get("features", [])
    if not isinstance(result, list) or not result:
        raise ValueError("Missing Natural Earth country boundaries")
    for item in result:
        if item.get("geometry", {}).get("type") not in ("Polygon", "MultiPolygon"):
            raise ValueError("Unsupported boundary geometry type")
        if not item.get("properties", {}).get("iso3"):
            raise ValueError("Boundary country ISO-alpha-3 missing")
    return result


def polygon_outlines(geometry: dict) -> list[list[list[float]]]:
    """Return outer rings only for Natural Earth country polygons."""
    coordinates = geometry["coordinates"]
    if geometry["type"] == "Polygon":
        return [coordinates[0]]
    return [polygon[0] for polygon in coordinates]


def equal_earth(lon_deg: np.ndarray, lat_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equal Earth projection (Šavrič, Patterson and Jenny 2018) on the unit sphere."""
    lon = np.radians(lon_deg)
    theta = np.arcsin(np.sqrt(3) / 2 * np.sin(np.radians(lat_deg)))
    a1, a2, a3, a4 = 1.340264, -0.081106, 0.000893, 0.003796
    t2 = theta ** 2
    t6 = t2 ** 3
    x = (2 * np.sqrt(3) * lon * np.cos(theta)
         / (3 * (9 * a4 * t6 * t2 + 7 * a3 * t6 + 3 * a2 * t2 + a1)))
    y = theta * (a1 + a2 * t2 + t6 * (a3 + a4 * t2))
    return x, y


def _projected_rings(feature: dict) -> list[np.ndarray]:
    rings = []
    for ring in polygon_outlines(feature["geometry"]):
        if len(ring) < 3:
            continue
        coordinates = np.asarray(ring, dtype=float)
        x, y = equal_earth(coordinates[:, 0], coordinates[:, 1])
        rings.append(np.column_stack([x, y]))
    return rings


def make_maps(
    counterfactuals: pd.DataFrame, boundaries: list[dict], *,
    model_keys: list[str],
):
    """One panel per model on a shared absolute colour scale, with missing data hatched."""
    ps.use_style()
    cmap = ps.SEQUENTIAL
    # Keep the absolute-emissions colour scale common across the model panels.
    norm = Normalize(
        vmin=0,
        vmax=float(counterfactuals["counterfactual_g_co2e_per_1000_decisions"].max()),
    )
    x_edge, _ = equal_earth(np.array([180.0]), np.array([0.0]))
    _, y_bounds = equal_earth(np.zeros(2), np.array([-58.0, 84.0]))

    fig, axes = plt.subplots(
        len(model_keys), 1, figsize=(ps.PRINT_WIDTH_IN, 1.0 + 2.6 * len(model_keys)),
        squeeze=False,
    )
    mapped: dict[str, int] = {}
    for ax, model_key in zip(axes[:, 0], model_keys, strict=True):
        data = counterfactuals.loc[counterfactuals["model_key"] == model_key]
        if data.empty:
            raise ValueError(f"No counterfactual rows for {model_key}")
        lookup = data.set_index("country_iso3")[
            "counterfactual_g_co2e_per_1000_decisions"
        ].to_dict()
        filled, fills, missing = [], [], []
        matched = set()
        for feature in boundaries:
            code = feature["properties"]["iso3"]
            value = lookup.get(code)
            rings = [Polygon(ring, closed=True) for ring in _projected_rings(feature)]
            if value is None:
                missing.extend(rings)
                continue
            matched.add(code)
            filled.extend(rings)
            fills.extend([cmap(norm(value))] * len(rings))
        ax.add_collection(PatchCollection(
            filled, facecolor=fills, edgecolor=ps.BORDER, linewidth=0.25,
        ))
        ax.add_collection(PatchCollection(
            missing, facecolor=ps.NO_DATA, edgecolor=ps.BORDER, linewidth=0.25,
            hatch="//////",
        ))
        ax.set_xlim(-x_edge[0] * 1.01, x_edge[0] * 1.01)
        ax.set_ylim(y_bounds[0], y_bounds[1])
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(PANEL_TITLES[model_key])
        mapped[model_key] = len(matched)

    fig.tight_layout(rect=(0, 0.13, 1, 1), h_pad=0.8)
    colorbar_ax = fig.add_axes((0.2, 0.085, 0.48, 0.022))
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_ax, orientation="horizontal"
    )
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0, pad=3)
    colorbar.set_label("Counterfactual g CO₂e per 1,000 classifications", labelpad=5)
    no_data = Patch(facecolor=ps.NO_DATA, edgecolor=ps.BORDER, linewidth=0.25,
                    hatch="//////", label="No data")
    fig.legend(handles=[no_data], loc="center left", bbox_to_anchor=(0.71, 0.096),
               frameon=False, handlelength=1.6, handleheight=1.1)
    return fig, mapped


def write_maps(fig, stem: Path, caption: str) -> list[Path]:
    paths = ps.save(fig, stem, caption)
    plt.close(fig)
    return paths


def caption_for(data: pd.DataFrame, n_mapped: int) -> str:
    models = data.drop_duplicates("model_key").set_index("model_key")
    keys = [key for key in LOCAL_MODELS if key in models.index]
    names = [PANEL_TITLES[key] for key in keys]
    energies = "; ".join(
        f"{PANEL_TITLES[key]}: {models.loc[key, 'measured_energy_kwh']:.6f} kWh" for key in keys
    )
    n_decisions = int(models["n_decisions"].iloc[0])
    title_models = " and ".join(names)
    return (
        f"Counterfactual Electricity-Related Emissions for {title_models} across National "
        "Electricity Grids\n\n"
        "Note. Measured machine-level CodeCarbon energy for "
        f"{n_decisions:,} classifications in the Netherlands ({energies}) is held "
        "constant across countries and multiplied by the annual national-average "
        "lifecycle electricity carbon intensity reported in CodeCarbon v3.3.1 (country "
        "data years vary). "
        + ("Both panels share one colour scale in " if len(keys) > 1 else "The map shows ")
        + "g CO2e per 1,000 classifications, a functional unit rather than a directly "
        f"observed per-request footprint. In total, {n_mapped} countries with matched "
        "Natural Earth 1:110m boundaries are coloured; hatched grey areas have no "
        "matching country factor, and some microstates are not drawn at this resolution. "
        "Country differences are counterfactual location assumptions, not measurements "
        "in each country. The analysis excludes hosted API models and does not estimate "
        "energy or emissions from their token usage or latency. Hardware embodied "
        "emissions, transport, location-dependent cooling, grid time-of-day variation "
        "and differences in hardware are excluded. Source: measured production summary, "
        "CodeCarbon country factors v3.3.1 and Natural Earth 1:110m public-domain "
        "boundaries."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--factors", type=Path, default=FACTORS)
    parser.add_argument("--boundaries", type=Path, default=BOUNDARIES)
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument("--figures", type=Path, default=FIGURES)
    parser.add_argument(
        "--model-key", action="append", choices=list(LOCAL_MODELS),
        help="Select one local model; repeat for both (default: both).",
    )
    parser.add_argument(
        "--table-only", action="store_true",
        help="Calculate counterfactuals without rendering maps.",
    )
    args = parser.parse_args()
    selected = list(dict.fromkeys(args.model_key or list(LOCAL_MODELS)))
    factors = load_factors(args.factors)
    models = load_local_runs(args.summary, selected)
    simulated = build_counterfactuals(models, factors)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    simulated.to_csv(args.table, index=False, float_format="%.12g")

    figure_files = []
    n_mapped = {}
    if not args.table_only:
        boundaries = read_boundaries(args.boundaries)
        ordered = [key for key in LOCAL_MODELS if key in selected]
        fig, n_mapped = make_maps(simulated, boundaries, model_keys=ordered)
        stem = args.figures / FIGURE_STEM
        caption = caption_for(
            simulated.loc[simulated["model_key"].isin(ordered)],
            max(n_mapped.values()),
        )
        figure_files = [path.as_posix() for path in write_maps(fig, stem, caption)]
    manifest = {
        "analysis": "fixed_local_energy_country_carbon_counterfactual",
        "schema_version": 1,
        "model_keys": selected,
        "measurement_scope": "machine_level_CodeCarbon_local_Ollama_only",
        "functional_unit": "g CO2e per 1000 classifications",
        "original_measurement_country": "NLD",
        "pue": "Use CodeCarbon energy_consumed as recorded; no second PUE multiplier",
        "codecarbon_country_factor_version": SOURCE_CODECARBON_VERSION,
        "codecarbon_country_factor_source": INTENSITY_SOURCE,
        "country_source_years": sorted(factors["source_year"].unique().tolist()),
        "natural_earth_source": GEOGRAPHY_SOURCE,
        "source_hashes_sha256": {
            "measured_energy_summary": file_sha256(args.summary),
            "country_intensity_snapshot": file_sha256(args.factors),
            "map_geometry_snapshot": file_sha256(args.boundaries),
        },
        "formula": "emissions_kg = measured_energy_kwh * country_gCO2e_per_kwh / 1000",
        "scope_limitations": (
            "Counterfactual emissions only. Location-specific hardware energy, "
            "data-centre PUE, time-varying grid mix, embodied emissions and "
            "hosted API provider-side energy are not measured or estimated."
        ),
        "countries_with_factors": len(factors),
        "mapped_countries": n_mapped,
        "table": args.table.as_posix(),
        "figure_files": figure_files,
    }
    manifest_path = args.table.with_name("local_country_emissions_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Counterfactual table: {args.table} ({len(simulated):,} rows)")
    print(f"Provenance manifest: {manifest_path}")
    for model_key in selected:
        sample = simulated.loc[
            (simulated["model_key"] == model_key)
            & simulated["country_iso3"].isin(["NLD", "FRA", "SWE", "POL", "IND"])
        ]
        print(f"\n{LOCAL_MODELS[model_key]} — g CO2e per 1,000 decisions")
        print(sample[[
            "country", "intensity_source_year",
            "counterfactual_g_co2e_per_1000_decisions",
        ]].to_string(index=False, float_format=lambda value: f"{value:.2f}"))
        if not args.table_only:
            print(f"  map panel in: {(args.figures / FIGURE_STEM).as_posix()}.*")


if __name__ == "__main__":
    main()
