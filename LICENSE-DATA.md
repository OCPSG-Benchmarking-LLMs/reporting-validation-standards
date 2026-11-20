# Licence for documentation, configuration, results and outputs

The code in `src/`, `code/` and `tests/` is released under the MIT Licence (see `LICENSE`).

Unless stated otherwise below, everything else in this repository is released under the
[Creative Commons Attribution 4.0 International licence (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/):
the documentation (`README.md`, `docs/`), configuration (`config/`, `schemas/`), sample
manifests (`data/manifests/`), finalised provenance (`results/`) and analysis outputs and
figures (`outputs/`). Please attribute it by citing the working paper (see `CITATION.cff`).

## Third-party material

| Material | Where | Source and terms |
|---|---|---|
| Speech identifiers, silver-standard labels and every result derived from them | `data/manifests/`, `results/`, `outputs/` | OCPSG Silver Standard of parliamentary speeches, `OCPSG-Benchmarking-LLMs/ocpsg-silver-standard-parliamentary-speeches` on Hugging Face, CC BY 4.0 (gated access). The speeches come from the ParlaMint corpus of parliamentary debates (ParlaMint-GB); the identifiers follow ParlaMint's utterance IDs. No speech text is included in this repository. |
| Canonical classification prompt | `config/prompts.yaml` (`p0_capbench_canonical`) | Reference prompt wording of the OCPSG CapBench benchmark, reproduced unchanged so that results are comparable; P1–P3 are perturbations of it. |
| Country electricity carbon-intensity factors | `config/geography/codecarbon_country_intensity_v3.3.1.csv` | Extracted from CodeCarbon v3.3.1 (`codecarbon/data/private_infra/global_energy_mix.json`), MIT Licence. CodeCarbon derives these values from Our World in Data and Ember electricity data, both published under CC BY 4.0. |
| World country boundaries | `config/geography/natural_earth_110m_countries_simplified.geojson` | Simplified from Natural Earth 1:110m Admin 0 countries, public domain. |
| Model and provider names | throughout | Names of models and services (OpenAI, Fireworks, DeepSeek, Mistral, Qwen, Gemma, Ollama, TypeSafe JEV) are used only to identify what was evaluated; they belong to their owners. |

Model outputs recorded in the raw run files are not distributed in this repository; only
their SHA-256 hashes and derived aggregates are.
