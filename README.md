# Reporting and Validation Standards

Code, frozen specifications and compact provenance for the working paper:

**Report, Diagnose, Propagate: Reporting and Validation Standards for Machine-Learning and LLM Classifiers in Political Science**

> **Language convention.** Documentation uses British English. Source-code identifiers, command-line flags and machine-readable provenance keep their original American-English spellings (for example `--finalize` and `07_analyze_stability.py`) for compatibility and reproducibility.

The repository uses the OCPSG Silver Standard of parliamentary speeches as a controlled demonstration environment for:

1. a frozen, label-stratified probability sample of 1,000 Great Britain test-set speeches;
2. prompt and inference stability across seven API and local LLMs;
3. design-based performance estimation with sampling weights and uncertainty;
4. propagation of classification error into aggregate policy-agenda prevalence;
5. computational reporting, including API token and cost accounting and local CodeCarbon energy measurement; and
6. two post hoc extensions: native probabilistic classification with JEV, and an offline country-counterfactual carbon simulation for the local models.

All performance estimates are measured against the automated **silver** reference labels, not against independent human gold labels.

## Repository structure

```text
code/       numbered research scripts (run from the repository root)
src/        reusable Python package (reporting_validation_standards)
tests/      automated tests (offline; no API calls)
config/     frozen model, prompt, deployment, pricing and geography inputs
schemas/    provider-neutral inference-record contract
data/       frozen sample manifests (generated observations are not committed)
results/    compact, immutable provenance of finalised production runs
outputs/    analysis tables and manuscript figures
docs/       decision log and methodology notes for the extensions
```

The directory is called `code/` rather than `scripts/` on purpose: this is a research-companion repository, and `code/` makes the role of the numbered analysis programs explicit.

Raw inference records and CodeCarbon files are written to `runs/`, which is not distributed with this repository. The SHA-256 hash of every raw file is committed under `results/`, and every analysis script verifies the local raw files against those hashes before computing anything.

## Setup

The project uses a `src/` package layout, so the package itself must be installed in the environment; installing only its third-party dependencies is not enough. With Pipenv, from the repository root:

```bash
pipenv install --dev
pipenv run hf auth login
pipenv run pytest
```

The `Pipfile` installs this repository as an editable package, so imports such as `reporting_validation_standards.sampling` resolve. Alternatively, with venv and pip:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
hf auth login
```

The OCPSG Silver Standard dataset is gated on Hugging Face: accept its access conditions before running the sampling code. Scripts that call providers read `OPENAI_API_KEY`, `FIREWORKS_API_KEY`, `MISTRAL_API_KEY` or `TYPESAFE_API_KEY` from the environment, and the local models need a running Ollama server.

## Pipeline

| Stage | Scripts | Main outputs |
|---|---|---|
| Freeze the sample | `01_freeze_uk_sample.py` | `data/manifests/` |
| Validate specifications | `02_validate_experiment_specs.py` | (checks only) |
| Provider smoke tests | `03a`–`03d` | `runs/smoke/` |
| Context preflight | `04_preflight_stability_experiment.py` | `outputs/preflight/` |
| Stability production runs | `05_run_stability_experiment.py`, `06_check_and_finalize_production.py` | `runs/production/`, `results/production/` |
| Stability analysis | `07_analyze_stability.py` | `outputs/stability/` |
| Design-based validation | `08_analyze_validation.py`, `08b_analyze_prompt_intervals.py` | `outputs/validation/` |
| Full-population deployment | `09_run_population_deployment.py`, `10_check_and_finalize_deployment.py` | `runs/deployment/`, `results/deployment/` |
| Prevalence error propagation | `11_analyze_error_propagation.py` | `outputs/propagation/` |
| Compute and environment | `12_analyze_compute_environment.py` | `outputs/compute_environment/` |
| Validation figure | `13_generate_manuscript_plot.py` | `outputs/figures/` |
| JEV extension | `14_run_jev_calibration.py`, `14b_audit_jev_probability_failures.py`, `15_analyze_jev_calibration.py`, `16_generate_jev_figures.py` | `runs/jev/`, `outputs/jev/`, `outputs/figures/` |
| Country-carbon extension | `17_simulate_country_carbon.py` | `outputs/compute_environment/`, `outputs/figures/` |

Run every script from the repository root, for example `pipenv run python code/07_analyze_stability.py`. Scripts 13 and 17 need only committed files. The analysis scripts 07, 08, 11, 12, 15 and 16 also read the local raw files in `runs/`, after checking them against the committed hashes; none of them makes an API call.

## 1. Frozen UK sample

Script 01 downloads the canonical `GB/test.parquet` split from the Hugging Face dataset repository, resolves the dataset revision to an immutable commit SHA, draws a reproducible label-stratified probability sample and stores the information needed for design-based estimation:

```bash
pipenv run python code/01_freeze_uk_sample.py
```

The approved design uses:

- target sample size: 1,000;
- strata: silver-standard policy label;
- minimum allocation: 25 observations per available label, subject to stratum capacity;
- remainder: allocated approximately proportional to population stratum size;
- random seed: 20260923.

The script writes:

- `data/manifests/uk_test_1000_ids.csv`: frozen IDs, inclusion probabilities and inverse-probability weights;
- `data/manifests/uk_test_1000_strata.csv`: stratum population and sample counts;
- `data/manifests/uk_test_1000_metadata.json`: dataset revision, seed, design, allocation-rule version and sample-membership hash;
- `data/generated/uk_test_1000.parquet`: full sampled observations for local analysis (not committed).

The sample is frozen and must not be redrawn. A change to the design should create a new, explicitly versioned sample rather than overwrite the registered one. With `--overwrite`, the script keeps the immutable Hugging Face revision recorded in the existing metadata unless a different `--revision` is supplied, so the same revision, seed and design reproduce the same membership.

## 2. Frozen prompt-stability specification

The experiment specification is stored separately from provider code:

- `config/models.yaml`: the seven evaluated models; machine-readable stage fields preserve the chronology and reproducibility status of the original experiment;
- `config/prompts.yaml`: the canonical CapBench prompt and three semantically equivalent perturbations;
- `schemas/inference_record.schema.json`: the provider-neutral per-observation inference contract;
- `src/reporting_validation_standards/experiment_specs.py`: configuration validation and prompt rendering;
- `src/reporting_validation_standards/inference.py`: the Python representation of the inference record.

The four frozen prompt conditions are:

1. `p0_capbench_canonical`: the exact CapBench reference wording;
2. `p1_lexical_paraphrase`: a surface-wording perturbation;
3. `p2_label_order_permutation`: a fixed permutation of the official label order;
4. `p3_structural_format`: a layout and format perturbation with unchanged substantive instructions.

Seven models are evaluated: GPT-5.4, GPT-5.4 Mini, DeepSeek V4.1 Flash, Qwen3.6 35B, Gemma 4 31B, GPT-6 Luna and Mistral Large 3. The machine-readable stage labels (`core`, `secondary`, `supplementary`) record experimental chronology rather than scientific importance: GPT-6 Luna was part of the original six-model panel but used a moving alias, while Mistral Large 3 was added after that panel had been completed to broaden provider and model-family coverage. The working paper presents all seven together as evaluated models.

Validate the frozen specifications with:

```bash
pipenv run python code/02_validate_experiment_specs.py
pipenv run pytest
```

The provider adapters do not redefine prompt wording. They translate the frozen prompt and the common `predicted_label` output contract into each provider's API while preserving raw outputs, usage metadata, latency, model identity and failures.

## 3. Provider adapters and smoke tests

Each smoke test runs the first five frozen cases once per model by default (`--n-cases 1` gives a one-case connectivity check) and writes its artefacts under `runs/`.

**OpenAI** (`03a_smoke_test_openai.py`) uses the Responses API with the two pinned GPT-5.4 snapshots. The adapter:

- requests `reasoning.effort="none"` and `temperature=0`;
- uses strict JSON Schema structured output for `predicted_label`;
- sets `store=false` and requests the `default` service tier, so the dated standard-pricing configuration applies;
- disables hidden SDK retries and records explicit retry counts, failed-attempt status and type, backoff time and successful-attempt latency;
- records returned model identity, response ID, token usage, cache reads and writes, reasoning tokens, end-to-end latency, timestamps, raw structured output and failures;
- calculates cost from `config/pricing/openai_2026-09-23.yaml`.

**Fireworks** (`03b_smoke_test_fireworks.py`) calls DeepSeek V4.1 Flash through Fireworks' OpenAI-compatible Chat Completions endpoint with JSON-schema structured output. Because DeepSeek V4.1 reasons by default when the control is omitted, the adapter sends `reasoning_effort="none"`. Cost comes from `config/pricing/fireworks_2026-09-23.yaml`, using separate ordinary-input, cached-input and output prices.

**Mistral** (`03d_smoke_test_mistral.py`) uses Mistral's OpenAI-compatible endpoint (`https://api.mistral.ai/v1`) with the exact snapshot `mistral-large-2512`, the same frozen prompts, `temperature=0`, a 40-token output cap, JSON-schema structured output and the standard service tier. Mistral Large 3 does not document adjustable reasoning, so the adapter sends no `reasoning_effort` and the manifest records `not_exposed_for_model`. Pricing is in `config/pricing/mistral_2026-09-26.yaml`.

**Ollama** (`03c_smoke_test_ollama.py`) uses Ollama's native `/api/chat` endpoint to capture local telemetry that the OpenAI-compatible surface does not expose. It sends the frozen system and user prompt, a JSON schema through `format`, `temperature=0` and `think=false`, and records the model tag and local digest, token counts, cache reads, model-load time, prompt-evaluation time, generation time, tokens per second, end-to-end latency, retries and raw structured output. Local timing is reported as three components (cold model loading, prompt evaluation and generation) rather than one latency statistic. The two frozen local tags are `qwen3.6:35b` and `gemma4:31b`:

```bash
ollama pull qwen3.6:35b
ollama pull gemma4:31b
pipenv run python code/03c_smoke_test_ollama.py --n-cases 1 --model-key ollama_qwen36_35b
pipenv run python code/03c_smoke_test_ollama.py --model-key ollama_gemma4_31b
```

The script keeps the model resident during the run, queries Ollama's running-model state after the first successful request to record total and VRAM-resident size, and unloads the model at the end, so the manifest documents CPU and GPU residency rather than inferring it from nominal model size.

### Local energy and carbon measurement

CodeCarbon is applied at the **run level**, not once per speech. Because Ollama executes inference in a separate server process, CodeCarbon uses `tracking_mode="machine"` so that the server's GPU and CPU energy is not excluded by process-level tracking. Pass an explicit three-letter code for the machine's physical location:

```bash
pipenv run python code/03c_smoke_test_ollama.py \
  --model-key ollama_qwen36_35b \
  --track-carbon \
  --country-iso-code NLD
```

The run directory then contains `codecarbon_emissions.csv` and a compact `carbon_summary.json` alongside the inference records and run manifest. The manifest also records platform metadata and, when available, the NVIDIA GPU model, UUID, driver, VRAM and power limit.

## 4. Production stability runs

Before the full 1,000-case × four-prompt × three-repetition experiment, a length and context preflight reports text and rendered-prompt length distributions, lists the longest sampled speeches, confirms the planned 12,000 decisions per model, and uses successful Ollama smoke records to build an empirical prompt-token proxy:

```bash
pipenv run python code/04_preflight_stability_experiment.py
```

The token proxy is diagnostic rather than tokeniser-exact. The preflight showed that the longest rendered prompts came too close to the 4,096-token smoke-test context to leave a defensible margin, so production freezes `num_ctx=8192` for both local models. Its result is committed as `outputs/preflight/stability_preflight.json`. The longest frozen speech can be used for a targeted context test:

```bash
pipenv run python code/03c_smoke_test_ollama.py --observation-id ParlaMint-GB_2016-06-08-lords.u126 --model-key ollama_qwen36_35b
```

Each model is then run on its own:

```bash
pipenv run python code/05_run_stability_experiment.py --model-key <MODEL_KEY>
```

The runner uses the same deterministic randomised 12,000-cell schedule for every model (1,000 observations × four prompts × three repetitions), appends each completed inference record immediately, validates the frozen experiment manifest on resume and skips every cell already present. `--plan-only` inspects a run without sending requests, and `--max-decisions N` stages a launch; rerunning the same command resumes the identical schedule. Local Ollama production runs require energy tracking (`--track-carbon --country-iso-code NLD`). Each invocation creates a session manifest, so interrupted local runs resume while keeping separate CodeCarbon segments. Local models, Ollama, the GPU driver and the Python environment were not changed between production sessions; any such change would have to be documented and the affected run restarted.

Mistral Large 3 was added after the original six-model panel had been completed. It was chosen for additional provider and model-family coverage and feasible API access, not after observing its classifications. The original panel contributes 72,000 decisions and Mistral adds 12,000, for 84,000 in total.

### Check and finalise a production run

```bash
pipenv run python code/06_check_and_finalize_production.py --model-key <MODEL_KEY>
pipenv run python code/06_check_and_finalize_production.py --model-key <MODEL_KEY> --finalize
```

Check-only mode has no archival side effects and can be used on staged or incomplete runs. It validates the frozen schedule, prompt hashes, model and provider identity, manifest consistency, duplicate or unexpected cells, prompt × repetition coverage, parse outcomes, retries, latency, token usage, returned model identity and recorded API cost.

Finalisation is fail-closed: it proceeds only when structural and completion QA both pass. It leaves the raw inference and CodeCarbon files under `runs/`, hashes every raw run file and writes only compact provenance to the version-controlled path:

```text
results/production/<MODEL_KEY>/
├── experiment_manifest.json
├── qa_summary.json
├── raw_files_sha256.json
└── sessions/<SESSION_ID>/session_manifest.json
```

Finalised artefacts are immutable: rerunning `--finalize` on unchanged raw data is a no-op, and the script refuses to overwrite an artefact that would change.

## 5. Stability analysis

```bash
pipenv run python code/07_analyze_stability.py
```

Before calculating any statistic, the script checks that every local raw `inference_records.jsonl` matches the SHA-256 in the committed `results/production/<MODEL_KEY>/qa_summary.json`. The analysis keeps two concepts separate: **operational reliability** (whether a scheduled decision produced a valid structured classification) and **classification stability** (whether valid classifications stay the same under repetition, prompt perturbation or a change of model family). Observed API and parse failures are preserved rather than silently rerun.

Primary aggregates use the inverse-inclusion weights of the frozen sample, so they describe the GB test population represented by the design; unweighted quantities are kept as sample diagnostics. Outputs in `outputs/stability/`:

- `model_summary.csv`: model-level operational reliability and stability;
- `repetition_by_prompt.csv`: repeated-inference consistency within each prompt;
- `prompt_comparisons.csv`: canonical P0 against each prompt perturbation;
- `model_pair_agreement.csv`: pairwise model agreement under P0 and across all prompts;
- `panel_agreement.csv`: unanimity for the original six-model panel and for all seven models;
- `stability_by_reference_label.csv`: diagnostics by silver-standard reference label;
- `failure_diagnostics.csv`: production failures by model, error type and final status code;
- `observation_instability.csv`: observation-level instability for case inspection;
- `analysis_manifest.json`: source hashes, panel definitions and analysis rules.

## 6. Design-based validation

```bash
pipenv run python code/08_analyze_validation.py
```

The primary validation condition is the canonical P0 prompt. The three repeated P0 decisions for each sampled speech are averaged, so each observation contributes its inverse-inclusion weight exactly once to the confusion matrix. The resulting totals are Horvitz–Thompson-style population estimates under the frozen design. Macro-averaged metrics are calculated across the 17 silver-reference labels present in the GB target population; predictions into the six absent benchmark labels still count as errors for the true class.

Operational metrics count API and parse failures as measurement failures; conditional metrics describe performance among valid outputs only. Sampling uncertainty comes from 2,000 paired Rao–Wu rescaled bootstrap replicates by default, resampling sampled speeches within the 17 strata and keeping all repeated inferences of a resampled speech together. The rescaling includes the stratum sampling fraction, so census strata contribute no sampling variance. Because the same replicate weights are used for every model, model differences are evaluated from paired replicate differences rather than from overlapping marginal intervals.

Outputs in `outputs/validation/`:

- `primary_model_performance.csv`: weighted P0 performance, unweighted diagnostics and 95% bootstrap intervals;
- `condition_performance.csv`: every prompt × repetition condition plus repetition-averaged prompt estimates;
- `per_class_performance.csv`: weighted class-specific precision, recall, F1-score, support and F1 intervals;
- `confusion_matrix_population.csv`: weighted confusion-matrix cells as estimated GB test-population counts;
- `bootstrap_metrics.csv`: replicate-level primary metrics;
- `paired_model_differences.csv`: paired-bootstrap differences in accuracy and macro-F1;
- `analysis_manifest.json`: estimands, source hashes, bootstrap specification and uncertainty scope.

These are **silver-reference performance estimates**, not human-gold validation.

### Per-prompt intervals

```bash
pipenv run python code/08b_analyze_prompt_intervals.py
```

Script 08b applies the same estimator to each model × prompt: the three repetitions of a speech are averaged so that it carries its sampling weight once, and 95% percentile intervals come from the same 2,000 Rao–Wu replicates (seed and replicate count are read from `analysis_manifest.json`). It fails closed unless every point estimate equals the repetition-averaged row of `condition_performance.csv` and every P0 interval equals `primary_model_performance.csv`, and it rewrites no existing output. It writes `prompt_performance_intervals.csv` and `prompt_intervals_manifest.json` to `outputs/validation/`; the P1–P3 intervals are descriptive and change no committed estimate.

## 7. Full-population deployment

The prevalence-propagation demonstration uses one frozen deployment over the complete GB silver-reference test population, specified in `config/deployment.yaml`: the pinned GPT-5.4 Mini snapshot (`gpt-5.4-mini-2026-03-17`), the canonical P0 prompt, the same `text_en` input rule and one classification per speech. The model was selected after the validation stage because its API cost is substantially lower than GPT-5.4's while the paired bootstrap interval for their macro-F1 difference included zero; this is not an equivalence claim.

```bash
pipenv run python code/09_run_population_deployment.py --plan-only
pipenv run python code/09_run_population_deployment.py --max-decisions 5
pipenv run python code/10_check_and_finalize_deployment.py
pipenv run python code/09_run_population_deployment.py
pipenv run python code/10_check_and_finalize_deployment.py --finalize
```

The runner is append-only and resumable. It loads the population from the same immutable Hugging Face revision as the validation sample, records a population-ID SHA-256 and uses a deterministic randomised PCG64 schedule, so elapsed API time is not aligned with source-data order. Persistent failures are kept after the frozen adapter retry policy. Finalisation over all 28,978 decisions writes compact provenance to `results/deployment/gb_test_p0_gpt54_mini_v1/`.

## 8. Prevalence error propagation

```bash
pipenv run python code/11_analyze_error_propagation.py
```

The analysis combines the one-pass GPT-5.4 Mini deployment over all 28,978 GB test speeches with the 1,000-speech validation sample and its three repeated P0 classifications; both raw inputs are checked against their committed hashes first. Let `pi` be the unknown 17-label prevalence vector over the GB silver-reference support, `M` the estimated 17 × 23 misclassification matrix with entries `P(predicted=j | reference=i)` and `q` the observed 23-label prevalence vector from the deployment. The measurement model is `q ≈ M^T pi`, and the corrected estimator solves

```text
min ||M^T pi - q||²   subject to   pi >= 0 and sum(pi) = 1
```

with a dependency-free projected FISTA solver and exact Euclidean projection onto the simplex. Direct inversion is not applicable because the forward operator is 23 × 17. The unconstrained least-squares solution is kept only as a diagnostic.

The point matrix uses inverse-inclusion weights and averages the three P0 classifications per speech. Uncertainty is propagated by rebuilding `M` in each of 2,000 Rao–Wu replicates and solving again; the deployment is treated as the observed census output of one frozen model realisation, so the intervals reflect validation-sample uncertainty in the error mechanism, not provider drift or benchmark-construction uncertainty. The correction conditions on the 17-label support known from the sampling frame, so the six unsupported labels are constrained to zero by design; the numerical prevalence of the 17 labels is used only afterwards, to compare naive and corrected estimates.

Outputs in `outputs/propagation/`: `prevalence_comparison.csv`, `propagation_summary.csv`, `misclassification_matrix.csv`, `bootstrap_propagation.csv`, `solver_diagnostics.json` and `analysis_manifest.json`.

## 9. Computational performance and environmental transparency

```bash
pipenv run python code/12_analyze_compute_environment.py
```

After verifying every production and deployment raw file against its finalised hash, the script reports request-level end-to-end latency, token accounting, valid-output rate, explicit retries and monetary cost where dated pricing and usage were available, with latency summarised for all, valid and invalid decisions.

For hosted APIs, server-side hardware, utilisation, power draw, data-centre overhead and electricity mix are not exposed, so the analysis **does not estimate provider-side energy or emissions from latency, tokens or price**. For the two local Ollama models, all production sessions were tracked with machine-level CodeCarbon; the script aggregates session totals into CPU, GPU, RAM and total energy and estimated emissions. Energy and emissions per 1,000 decisions are **run-average normalisations of session-level totals**, not direct per-request measurements. Local timing is decomposed into session-start model loading, warm load overhead, prompt evaluation, generation time and throughput; the first valid request of each session is a **session-start load**, not a perfect cold start, because operating-system and driver caches may persist.

Outputs in `outputs/compute_environment/`: `production_compute_summary.csv`, `deployment_compute_summary.csv`, `local_inference_timing.csv`, `local_environment_summary.csv`, `local_session_environment.csv`, `environmental_reporting_scope.csv` and `analysis_manifest.json`. The governing reporting principle is: **environmental precision should not exceed infrastructure knowledge**.

## 10. Manuscript figures

All figures are drawn at the width at which the working paper prints them (5.2 in for the figures set at 0.8 of the text width and 4.55 in for the map at 0.7), so their 8–9 pt text keeps that size on the page. They use the shared style in `src/reporting_validation_standards/plot_style.py`: no titles, gridlines, spines or text on the canvas, one frameless legend below the panels, a palette that stays distinguishable under the common forms of colour-vision deficiency, and dashed coral reference lines named in the legend. Each figure is saved as a transparent 300-dpi PNG plus vector PDF and SVG, with a same-basename `.txt` caption (a title line, a blank line, then "Note. …").

```bash
pipenv run python code/13_generate_manuscript_plot.py
```

`validation_prompt_repetition_profile` shows design-weighted operational Accuracy, Macro-Precision, Macro-Recall and Macro-F1 in a 2 × 2 layout. For each of the seven models, each panel has four prompt rows coloured by prompt (P0 blue `#5F7A9D`, P1 terracotta `#E6A57E`, P2 sky aqua `#4FB3D1`, P3 deep plum `#5E4161`). On each row, the circle is the repetition-averaged estimate and the thin line its 95% interval (both from script 08b), and a thicker translucent bar spans the lowest to the highest of the three single-repetition estimates in `condition_performance.csv`. Models keep the fixed experimental order, and the figure complements the full numerical tables rather than replacing them.

Figures for the extensions are described in their notes: `jev_calibration` and `jev_risk_coverage` (script 16) and `local_carbon_country_ollama` (script 17).

## 11. Extensions

**JEV native probability demonstration** (scripts 14–16): a separate, post hoc probability-based validation on the unchanged frozen GB sample, covering design-weighted scoring, calibration, risk–coverage analysis and an opt-in probability-sum audit of rejected provider responses. JEV is not an eighth model in the seven-model P0 panel. See [`docs/jev-demonstration.md`](docs/jev-demonstration.md).

**Offline country-carbon counterfactuals** (script 17): holds the measured machine-level CodeCarbon energy of the two local models fixed across 212 countries using pinned CodeCarbon v3.3.1 national electricity-intensity factors, and maps the result. It makes no inference or new measurement and does not simulate provider-side energy for hosted APIs. See [`docs/country-carbon-counterfactuals.md`](docs/country-carbon-counterfactuals.md).

Methodological decisions and environment notes for the whole project are logged in [`docs/experiment-decisions.md`](docs/experiment-decisions.md).

## Data and licences

- **Code** (`src/`, `code/`, `tests/`) is released under the MIT Licence ([`LICENSE`](LICENSE)).
- **Documentation, configuration, results, outputs and manifests** are released under CC BY 4.0, subject to the third-party terms listed in [`LICENSE-DATA.md`](LICENSE-DATA.md).
- The source speeches and silver-standard labels come from the OCPSG Silver Standard dataset on Hugging Face (`OCPSG-Benchmarking-LLMs/ocpsg-silver-standard-parliamentary-speeches`, CC BY 4.0, gated access). This repository contains speech identifiers and derived results, not speech texts.

## Citation

If you use this code or its outputs, please cite the working paper. Citation metadata is in [`CITATION.cff`](CITATION.cff).
