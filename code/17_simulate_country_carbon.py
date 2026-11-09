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
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon
import pandas as pd


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
# Same restrained Oxford-inspired family used by the manuscript plots;
# the sequential ramp moves from pale aqua to muted blue and terracotta.
MAP_COLORS = ["#EBF4F3", "#A8DADC", "#3A8995", "#5F7A9D", "#B5837B", "#A35C55"]
INK = "#394B56"
NO_DATA = "#E4E7E8"


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


def make_map(
    counterfactuals: pd.DataFrame, boundaries: list[dict], *,
    model_key: str,
):
    """Return a title-free, grid-free, transparent cartographic figure."""
    data = counterfactuals.loc[counterfactuals["model_key"] == model_key]
    if data.empty:
        raise ValueError(f"No counterfactual rows for {model_key}")
    lookup = data.set_index("country_iso3")[
        "counterfactual_g_co2e_per_1000_decisions"
    ].to_dict()
    cmap = LinearSegmentedColormap.from_list("oxford_carbon", MAP_COLORS, N=256)
    # Keep the absolute-emissions colour scale common across both model maps.
    norm = Normalize(
        vmin=0,
        vmax=float(counterfactuals["counterfactual_g_co2e_per_1000_decisions"].max()),
    )

    patches = []
    fills = []
    mapped = set()
    for feature in boundaries:
        code = feature["properties"]["iso3"]
        value = lookup.get(code)
        if value is not None:
            mapped.add(code)
        for ring in polygon_outlines(feature["geometry"]):
            if len(ring) < 3:
                continue
            patches.append(Polygon(ring, closed=True))
            fills.append(cmap(norm(value)) if value is not None else NO_DATA)

    fig, ax = plt.subplots(figsize=(13.6, 6.1), dpi=300)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    fig.subplots_adjust(left=0.012, right=0.988, top=0.985, bottom=0.15)
    collection = PatchCollection(
        patches, facecolor=fills, edgecolor="#FFFFFF", linewidth=0.26,
        antialiased=True, rasterized=False,
    )
    ax.add_collection(collection)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-59, 84)
    ax.set_aspect("equal")
    ax.set_axis_off()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False, which="both")

    colorbar_ax = fig.add_axes((0.29, 0.108, 0.42, 0.027))
    colorbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_ax, orientation="horizontal"
    )
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(
        labelsize=9, labelcolor=INK, length=0, pad=6, bottom=False, top=False
    )
    colorbar.set_label(
        "Counterfactual g CO₂e per 1,000 classifications",
        fontsize=10, color=INK, labelpad=10,
    )
    colorbar.ax.grid(False, which="both")
    return fig, mapped


def write_map(fig, stem: Path, caption: str) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in (".png", ".pdf", ".svg"):
        fig.savefig(
            stem.with_suffix(extension),
            dpi=300, transparent=True, facecolor="none",
            bbox_inches="tight", pad_inches=0.12,
        )
    stem.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")
    plt.close(fig)


def caption_for(data: pd.DataFrame, n_mapped: int) -> str:
    first = data.iloc[0]
    model_key = str(first["model_key"])
    return (
        "Counterfactual electricity-related emissions for "
        f"{LOCAL_MODELS[model_key]} across national electricity grids. "
        f"Measured machine-level CodeCarbon energy ({first['measured_energy_kwh']:.6f} kWh) "
        f"for {int(first['n_decisions']):,} classifications in the Netherlands is "
        "held constant across countries; it is multiplied by the annual "
        "national-average lifecycle electricity carbon intensity reported in "
        "CodeCarbon v3.3.1 (country data years vary). "
        "The mapped functional unit is g CO2e per 1,000 classifications, not "
        "a directly observed per-request footprint. "
        f"{n_mapped} countries with matched Natural Earth 110m boundaries are coloured. "
        "Grey denotes missing matching country factors or omitted source coverage; "
        "some microstates are not drawn at this resolution. "
        "Country differences are counterfactual location assumptions, not "
        "measurements in each country. The analysis excludes hosted API models "
        "and does not estimate energy or emissions from their token usage or latency. "
        "Hardware embodied emissions, transport, location-dependent cooling, "
        "grid time-of-day variation and differences in hardware are excluded. "
        "Source: pinned measured production summary, CodeCarbon country factors "
        "v3.3.1 and Natural Earth 1:110m public-domain boundaries."
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
        for model_key in selected:
            fig, mapped = make_map(simulated, boundaries, model_key=model_key)
            stem = args.figures / f"local_carbon_country_{model_key}"
            caption = caption_for(
                simulated.loc[simulated["model_key"] == model_key], len(mapped)
            )
            write_map(fig, stem, caption)
            n_mapped[model_key] = len(mapped)
            figure_files.extend(str(stem.with_suffix(ext)) for ext in
                                (".png", ".pdf", ".svg", ".txt"))
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
        "table": str(args.table),
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
            print(f"  map: {args.figures}/local_carbon_country_{model_key}.*")


if __name__ == "__main__":
    main()
