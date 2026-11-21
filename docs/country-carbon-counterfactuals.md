# Country-counterfactual CodeCarbon simulation for local LLM inference

## Purpose and measurement boundary

This **offline** analysis is a geographical sensitivity exercise, not a fresh
CodeCarbon measurement. It holds each model's **observed, machine-level
electricity consumption in the Netherlands** fixed and multiplies it by
country-specific, annual-average electricity carbon-intensity factors. A new
classification run is **not** required for each country. Different models,
quantisation settings, hardware, or substantial workload changes, by contrast,
require separate energy measurements.

The two eligible local models completed the same frozen 12,000-decision schedule
(1,000 GB speeches × four prompts × three repetitions), tracked with CodeCarbon
3.3.1 in **machine** mode across all production sessions. Their run-aggregated
energy includes tracked CPU, GPU and RAM consumption, as reported in
`outputs/compute_environment/local_environment_summary.csv`. This is not
request-level energy metering and is **not** the server-side electricity use of
OpenAI, Fireworks, Mistral or TypeSafe JEV.

| Frozen local run | Decisions | Energy (kWh) | Netherlands emissions (kg CO₂e) |
|---|---:|---:|---:|
| Qwen 3.6 35B (`ollama_qwen36_35b`) | 12,000 | 1.724909 | 0.461624 |
| Gemma 4 31B (`ollama_gemma4_31b`) | 12,000 | 3.231545 | 0.864833 |

## Reproducibility and inputs

- **Country factors:** `config/geography/codecarbon_country_intensity_v3.3.1.csv`.
  A frozen extraction of CodeCarbon **v3.3.1**'s
  [`global_energy_mix.json`](https://github.com/mlco2/codecarbon/blob/v3.3.1/codecarbon/data/private_infra/global_energy_mix.json),
  retaining country, ISO-alpha-3 code, source year and positive
  `carbon_intensity` values (g CO₂e/kWh). The original CodeCarbon country
  dataset derives from Our World in Data / Ember, using annual-average
  electricity *lifecycle* intensities. Most source years are 2022–2023, but
  some territories have 2021 values and Western Sahara has a 2009 value.
  **These are not 2026 real-time readings**; consult the source-year column
  before interpreting an individual country's estimate.
- **No silent fallback:** 212 positive country intensities are retained.
  Central African Republic's zero sentinel is excluded rather than treated
  as genuinely zero-carbon electricity; the script does **not** substitute
  CodeCarbon's world-average or regional factors without disclosure.
- **Geography:** a simplified, four-decimal-point snapshot of
  [Natural Earth 1:110m administrative country boundaries](https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_110m_admin_0_countries.geojson)
  in `config/geography/natural_earth_110m_countries_simplified.geojson`.
  The snapshot excludes Antarctica. Not all microstates appear at 1:110m,
  and country boundaries/recognition may differ across datasets; unmatched
  regions are hatched grey rather than assigned a guessed factor.
- **Observed energy:** the finalised, committed
  `outputs/compute_environment/local_environment_summary.csv`. The simulation
  checks that its measured Netherlands emissions equal measured energy times
  the **pinned CodeCarbon v3.3.1 Netherlands intensity (267.622 g CO₂e/kWh)**.
  If the version or measurement boundary differs, it stops instead of
  silently combining incompatible sources.
- The output manifest stores SHA-256 hashes of each input and identifies the
  baseline, formula, factor version, original scope and figure paths. The
  renderer is Matplotlib only: no shapefile installation, mapping API,
  online tiles, model inference, or new pip dependency is required.

For model \(m\) and hypothetical country \(c\):

\[
  \mathrm{CO_{2}e}_{m,c}\;(\mathrm{kg}) =
  E_{m,\mathrm{measured}}\;(\mathrm{kWh})
  \frac{I_c\;(\mathrm{g\,CO_{2}e/kWh})}{1000}.
\]

The mapped value divides total grams by the 12,000 completed
classifications and multiplies by 1,000. It is therefore a **run-average
normalisation per 1,000 classifications**, not a measurement per request.

### Running the simulation

From the root of the repository:

```bash
pipenv run pytest
pipenv run python code/17_simulate_country_carbon.py
```

By default, both local models are mapped. To map a single model:

```bash
pipenv run python code/17_simulate_country_carbon.py --model-key ollama_qwen36_35b
```

For the computation and its checks without rendering the figure:

```bash
pipenv run python code/17_simulate_country_carbon.py --table-only
```

The script creates:

- `outputs/compute_environment/local_country_emissions_counterfactuals.csv`
  (full country × model table, including factor source year, grid intensity,
  simulated emissions per run and per 1,000 classifications, and multiplier
  relative to the Netherlands).
- `outputs/compute_environment/local_country_emissions_manifest.json`
  (source hashes and scope).
- `outputs/figures/local_carbon_country_ollama.{png,pdf,svg,txt}`
  (vector PDF and SVG at exactly the print width, a transparent 1200-dpi PNG, and a separate caption).

The figure stacks one Equal Earth map per model (Qwen above Gemma), drawn at
its printed width (5.7 in, as the paper sets it at 0.88 of the text width) with the shared style in
`src/reporting_validation_standards/plot_style.py`: a warm sequential scale
(pale terracotta, terracotta `#E6A57E`, Oxford coral `#FE615A`, deep plum `#5E4161`), grey country borders, no gridlines or embedded title, and one
horizontal colour bar below both panels. Countries in the boundary snapshot
without a matching factor are hatched grey and named "No data" in the legend;
the CSV retains the full factor table. The colour scale refers to **absolute
emissions per 1,000 classifications** and is **shared across both models**,
so the difference in measured energy is visible. When a single model is
selected, the scale runs to that model's simulated maximum.

## Interpretation

Holding the measured computation constant reveals substantial geographical
variation in its electricity-related carbon footprint. For Qwen 3.6 35B,
the observed Dutch baseline corresponds to **38.47 g CO₂e per 1,000
classifications**. Applying the pinned national-average electricity factors
instead produces counterfactual estimates of **5.85 g** in Sweden,
**8.06 g** in France, **54.76 g** in Germany, **95.15 g** in Poland and
**102.55 g** in India for the same 1,000-classification functional unit.
These differences follow directly from electricity carbon intensities,
rather than changes in classification accuracy or model computational work.

The second local model shows the same geographical pattern but a higher
absolute estimated footprint because its recorded energy consumption was
greater. Across 12,000 classifications, Gemma 4 31B consumed **3.232 kWh**,
compared with **1.725 kWh** for Qwen 3.6 35B on the recorded local setup.
Accordingly, its counterfactual estimates range, for example, from
**10.96 g CO₂e per 1,000 classifications in Sweden** to **192.13 g in India**,
against a Dutch baseline of **72.07 g**. For either model, relocating a
fixed quantity of measured electricity from the Swedish to the Indian
national-average mix increases estimated emissions by about **17.5 times**;
this is a grid-intensity ratio, not an experimental relocation effect.

These results should be interpreted as **country-specific counterfactual
estimates**, not measured emissions from deployments physically situated
across the world. They use annual national factors from different reference
years, omit subnational and time-of-day variation, and assume identical
machine energy requirements, hardware, workload and cooling overhead in
every location. Furthermore, the system boundary concerns CPU/GPU/RAM
energy recorded by CodeCarbon and associated electricity carbon intensity;
it does not capture embodied hardware emissions, other infrastructure,
or provider-side energy for hosted API models. The exercise illustrates
why country assumptions and environmental measurement boundaries should
accompany emissions estimates in computational social science.

## Provenance and sources

- CodeCarbon v3.3.1 country factor snapshot: https://github.com/mlco2/codecarbon/blob/v3.3.1/codecarbon/data/private_infra/global_energy_mix.json
- CodeCarbon methodology: https://docs.codecarbon.io/latest/explanation/methodology/
- CodeCarbon output definitions (PUE-scaled measured energy): https://docs.codecarbon.io/latest/reference/output/
- Natural Earth public-domain world boundaries: https://www.naturalearthdata.com/about/terms-of-use/
- Original source geometry: https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_110m_admin_0_countries.geojson
