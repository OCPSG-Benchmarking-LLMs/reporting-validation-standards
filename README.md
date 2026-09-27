# Reporting and Validation Standards

Code and reproducible demonstrations for the working paper:


> **Language convention.** Documentation uses British English. Existing source-code identifiers and machine-readable provenance retain their original American-English spellings for compatibility and reproducibility.
**Report, Diagnose, Propagate: Reporting and Validation Standards for Machine-Learning and LLM Classifiers in Political Science**

The repository is intentionally separate from `capbench`. It uses the OCPSG Silver Standard as a controlled demonstration environment for:

1. a frozen, stratified sample of 1,000 Great Britain test-set speeches;
2. prompt and inference stability across API and local models;
3. design-based performance estimation with sampling weights and uncertainty;
4. propagation of classification error into aggregate policy-agenda prevalence; and
5. computational reporting, including API token/cost accounting and local CodeCarbon measurement.

## Planned workflow

```text
OCPSG Silver Standard (Hugging Face)
        |
        +-- GB/test.parquet
                |
                +-- 01_freeze_uk_sample.py
                |       -> frozen 1,000-case manifest
                |       -> inclusion probabilities
                |       -> inverse-probability weights
                |
                +-- prompt/inference stability
                +-- weighted performance + uncertainty
                +-- prevalence error propagation
                +-- compute/cost/carbon summaries
```

## Current stage

The UK sampling stage is frozen, and the prompt/inference-stability experiment specification is now versioned in the repository. The first implemented component is the frozen UK sampling pipeline. It downloads the canonical `GB/test.parquet` split directly from the Hugging Face dataset repository, resolves the current dataset revision to an immutable commit SHA, draws a reproducible label-stratified probability sample, and stores the information needed for design-based estimation.

The default design currently uses:

- target sample size: 1,000;
- strata: silver-standard policy label;
- minimum allocation: 25 observations per available label, subject to stratum capacity;
- remainder: allocated approximately proportional to population stratum size;
- random seed: 20260923.

The GB test-set distribution has now been reviewed and this allocation rule is the approved sampling design for the 1,000-case demonstration. The metadata manifest records the full test-population size, number of represented labels, immutable Hugging Face revision, allocation-rule version, random generator, seed, and sample-membership hash.

## Setup with Pipenv

The project uses a `src/` package layout. Therefore the local project itself must be installed in the environment; installing only its third-party dependencies is not enough.

From the repository root:

```bash
pipenv install --dev
pipenv run hf auth login
```

The included `Pipfile` installs this repository as an editable local package, so imports such as `reporting_validation_standards.sampling` resolve correctly.

If you already created a local Pipfile before pulling the repository version, remove or reconcile that local file first.

### Alternative setup with venv/pip

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
hf auth login
```

If the OCPSG dataset is gated, make sure access has been accepted on Hugging Face before running the sampling code.

## Freeze the UK sample

```bash
pipenv run python code/01_freeze_uk_sample.py
```

or, in an activated venv:

```bash
python code/01_freeze_uk_sample.py
```

The script writes:

- `data/manifests/uk_test_1000_ids.csv` — frozen IDs and design variables;
- `data/manifests/uk_test_1000_strata.csv` — stratum population/sample counts;
- `data/manifests/uk_test_1000_metadata.json` — dataset revision, seed, design, and sample hash;
- `data/generated/uk_test_1000.parquet` — full sampled observations for local analysis (gitignored).

Do not redraw the sample after it is formally frozen. Changes to the design should create a new explicitly versioned sample rather than silently overwrite the registered one.

If provisional manifests already exist from an earlier run, pull the latest code and regenerate them once with:

```bash
pipenv run python code/01_freeze_uk_sample.py --overwrite
```

When `--overwrite` is used and an existing metadata manifest is present, the script preserves its recorded immutable Hugging Face revision unless a different `--revision` is explicitly supplied. With the same revision, seed, and design parameters, the sample membership should therefore remain unchanged while the richer metadata fields are added.


## Frozen prompt-stability specification

The experiment specification is stored separately from provider code:

- `config/models.yaml` — seven evaluated models; machine-readable stage fields preserve the chronology and reproducibility status of the original experiment;
- `config/prompts.yaml` — the canonical CapBench prompt and three semantically equivalent perturbations;
- `schemas/inference_record.schema.json` — provider-neutral per-observation inference contract;
- `src/reporting_validation_standards/experiment_specs.py` — configuration validation and prompt rendering;
- `src/reporting_validation_standards/inference.py` — Python representation of the inference record.

The four frozen prompt conditions are:

1. `p0_capbench_canonical` — exact CapBench reference wording;
2. `p1_lexical_paraphrase` — surface-wording perturbation;
3. `p2_label_order_permutation` — fixed permutation of the official label order;
4. `p3_structural_format` — layout/format perturbation with unchanged substantive instructions.

Seven models are evaluated in the completed demonstrations: GPT-5.4, GPT-5.4 Mini, DeepSeek V4.1 Flash, Qwen3.6 35B, Gemma 4 31B, GPT-6 Luna, and Mistral Large 3. The machine-readable provenance retains the original stage labels because they record experimental chronology rather than scientific importance: GPT-6 Luna was part of the original six-model panel but used a moving alias, while Mistral Large 3 was added after that panel had been completed to broaden provider and model-family coverage. In the working paper, all seven are presented together as evaluated models; the provenance labels do not imply a hierarchy of evidential status or substantive importance.

Validate the frozen specifications locally with:

```bash
pipenv run python code/02_validate_experiment_specs.py
pipenv run pytest
```

The provider adapters should not redefine prompt wording. Their job is to translate the frozen prompt and the common `predicted_label` output contract into the provider-specific API mechanism while preserving raw outputs, usage metadata, latency, model identity, and failures.

## OpenAI smoke test

The first provider adapter uses the OpenAI Responses API with the two pinned GPT-5.4 snapshots and the canonical P0 prompt. The smoke test is intentionally small: by default it runs the first five frozen cases once on each model, for ten API calls total.

The adapter:

- requests `reasoning.effort="none"` and `temperature=0`;
- uses strict JSON Schema structured output for `predicted_label`;
- sets `store=false`;
- requests the `default` service tier so the dated standard-pricing configuration applies;
- disables hidden SDK retries and records explicit retry counts, failed-attempt status/type, backoff time, and successful-attempt latency;
- records returned model identity, response ID, token usage, cache reads/writes, reasoning tokens, end-to-end latency, successful-attempt latency, timestamps, raw structured output, and failures;
- calculates cost from `config/pricing/openai_2026-09-23.yaml`; and
- writes smoke-test artefacts under `runs/`, which is gitignored.

After pulling this stage, refresh the environment and tests:

```bash
pipenv install --dev
pipenv lock
pipenv run pytest
```

Set `OPENAI_API_KEY` in the environment and then run:

```bash
pipenv run python code/03a_smoke_test_openai.py
```

For an even smaller connectivity check:

```bash
pipenv run python code/03a_smoke_test_openai.py --n-cases 1
```

Do not commit API keys or the raw `runs/` directory.

## Fireworks smoke test

The Fireworks adapter uses the same frozen prompt text and the same provider-neutral inference contract as the OpenAI adapter. It calls DeepSeek V4.1 Flash through Fireworks' OpenAI-compatible Chat Completions endpoint.

For comparability with the OpenAI runs, it uses JSON-schema structured output rather than the older CapBench function-call wrapper. Structured output constrains the final answer format but does not by itself disable model reasoning. DeepSeek V4.1 defaults to reasoning when the control is omitted, so the adapter explicitly sends `reasoning_effort="none"` for this classifier-style experiment.

The dated Fireworks pricing configuration is stored in:

```text
config/pricing/fireworks_2026-09-23.yaml
```

The current serverless rates recorded for `accounts/fireworks/models/deepseek-v4p1-flash` are separate ordinary-input, cached-input, and output prices. Cost is calculated from provider-returned usage and the dated configuration rather than hard-coded into the adapter.

After setting `FIREWORKS_API_KEY`, begin with one frozen case:

```bash
pipenv run python code/03b_smoke_test_fireworks.py --n-cases 1
```

Then run the same five-case smoke panel used for OpenAI:

```bash
pipenv run python code/03b_smoke_test_fireworks.py
```

The Fireworks adapter records returned model identity, response ID, token usage, cached tokens when exposed, finish reason, system fingerprint when exposed, retries, end-to-end latency, successful-attempt latency, timestamps, raw structured output, and calculated cost.

## Ollama smoke test

The local-inference adapter uses Ollama's native `/api/chat` endpoint so the experiment can capture local-specific telemetry that is not exposed through the OpenAI-compatible surface. The adapter sends the frozen system/user prompt, a JSON schema through `format`, `temperature=0`, and `think=false`. It records the Ollama model tag and local digest, token counts, cache reads, model-load time, prompt-evaluation time, generation time, tokens per second, end-to-end latency, retries, and raw structured output. Local timing is reported explicitly as three components—cold model loading, prompt evaluation, and generation—rather than collapsed into a single latency statistic.

Before running the local smoke tests, install the CodeCarbon dependency and make sure Ollama is running:

```bash
pipenv install --dev
pipenv run pytest
ollama list
```

The two frozen local model tags are:

```text
qwen3.6:35b
gemma4:31b
```

If either is not installed locally:

```bash
ollama pull qwen3.6:35b
ollama pull gemma4:31b
```

Begin with one Qwen case:

```bash
pipenv run python code/03c_smoke_test_ollama.py --n-cases 1 --model-key ollama_qwen36_35b
```

Then test Gemma:

```bash
pipenv run python code/03c_smoke_test_ollama.py --n-cases 1 --model-key ollama_gemma4_31b
```

If both succeed, run the same five frozen cases for each model:

```bash
pipenv run python code/03c_smoke_test_ollama.py --model-key ollama_qwen36_35b
pipenv run python code/03c_smoke_test_ollama.py --model-key ollama_gemma4_31b
```

The script keeps a model resident throughout the run, queries Ollama's running-model state after the first successful request to record total resident size and VRAM-resident size, and then explicitly unloads the model after the run. This allows the manifest to document CPU/GPU residency rather than inferring it from nominal model size or GPU capacity.

### Local energy and carbon measurement

CodeCarbon is optional during smoke testing and is applied at the **run level**, not once per speech. Because Ollama executes inference in a separate server process, CodeCarbon uses `tracking_mode="machine"` so GPU/CPU energy associated with the server is not excluded by process-level tracking.

Use an explicit three-letter country code for the machine's physical location. For example, when the machine is physically in the Netherlands:

```bash
pipenv run python code/03c_smoke_test_ollama.py \
  --model-key ollama_qwen36_35b \
  --track-carbon \
  --country-iso-code NLD
```

The run directory then contains `codecarbon_emissions.csv` and a compact `carbon_summary.json` alongside the inference records and run manifest. The manifest also records platform metadata and, when available, NVIDIA GPU model, UUID, driver, VRAM, and power limit.

## Production stability preflight

Before launching the full 1,000-case × four-prompt × three-repetition stability experiment, run a length/context preflight:

```bash
pipenv run python code/04_preflight_stability_experiment.py
```

The script verifies the frozen sample, reports text and rendered-prompt length distributions, lists the longest sampled speeches, confirms the planned 12,000 decisions per model / 72,000 decisions overall, and uses successful Ollama smoke records to construct an empirical prompt-token proxy. When recent Ollama residency manifests are available, it compares those estimates with the runtime context length.

The token proxy is diagnostic rather than tokenizer-exact. The completed preflight showed that the longest rendered prompts approach the 4,096-token smoke-test context too closely to leave a defensible safety margin. The production configuration therefore freezes `num_ctx=8192` for both local models. The generated preflight JSON is written below `outputs/` and is therefore not committed.

The longest frozen speech can be used for a targeted context test before production:

```bash
pipenv run python code/03c_smoke_test_ollama.py --observation-id ParlaMint-GB_2016-06-08-lords.u126 --model-key ollama_qwen36_35b
pipenv run python code/03c_smoke_test_ollama.py --observation-id ParlaMint-GB_2016-06-08-lords.u126 --model-key ollama_gemma4_31b
```

## Repository structure

```text
code/       executable research scripts
src/        reusable Python package
tests/      automated tests
data/       manifests and locally generated data
outputs/    generated analytical outputs (gitignored)
runs/       raw smoke/production run artefacts (gitignored)
results/    compact finalized production provenance (version controlled)
```

Using `code/` rather than `scripts/` is intentional: this repository is primarily a research-companion repository, and `code/` makes the role of the numbered analysis programs explicit.

## Related repositories

- OCPSG `capbench`: benchmarking infrastructure and provider templates.
- OCPSG Silver Standard: source benchmark and benchmark-construction workflow.


## Production stability runner

After smoke testing and the full-sample context preflight, run the frozen stability experiment one model at a time with:

```bash
pipenv run python code/05_run_stability_experiment.py --model-key <MODEL_KEY>
```

The runner uses the same deterministic randomised 12,000-cell schedule for every model (1,000 observations × four prompts × three repetitions), appends each completed inference record immediately, validates the frozen experiment manifest on resume, and skips every decision cell already present. Use `--plan-only` to inspect a model run without sending requests. Use `--max-decisions N` for a staged launch; rerunning the same command later resumes the identical schedule.

For local Ollama production runs, energy tracking is required:

```bash
pipenv run python code/05_run_stability_experiment.py \
  --model-key ollama_qwen36_35b \
  --track-carbon \
  --country-iso-code NLD

pipenv run python code/05_run_stability_experiment.py \
  --model-key ollama_gemma4_31b \
  --track-carbon \
  --country-iso-code NLD
```

Each invocation creates a session manifest. This permits interrupted local runs to be resumed while preserving separate CodeCarbon segments. The classification records remain model-level and append-only. Do not update local model files, Ollama, the GPU driver, or the Python environment between production sessions unless the resulting infrastructure change is intentionally documented and the affected run is restarted.


## Check and finalize a production run

Production runs remain under the gitignored `runs/` directory. Use the combined QA/finalisation script to inspect a partial or completed model run:

```bash
pipenv run python code/06_check_and_finalize_production.py \
  --model-key <MODEL_KEY>
```

Check-only mode has no archival side effects and can be used during staged or incomplete runs. It validates the frozen schedule, prompt hashes, model/provider identity, manifest consistency, duplicate or unexpected cells, prompt × repetition coverage, parse outcomes, retries, latency, token usage, returned model identity, and recorded API cost when available.

After all 12,000 scheduled cells for a model are complete, finalize the run explicitly:

```bash
pipenv run python code/06_check_and_finalize_production.py \
  --model-key <MODEL_KEY> \
  --finalize
```

Finalisation is fail-closed: it proceeds only when structural QA and completion QA both pass. It leaves the raw inference and CodeCarbon files under `runs/`, computes SHA-256 hashes for every raw run file, and writes only compact provenance artefacts to the version-controlled path:

```text
results/
└── production/
    └── <MODEL_KEY>/
        ├── experiment_manifest.json
        ├── qa_summary.json
        ├── raw_files_sha256.json
        └── sessions/
            └── <SESSION_ID>/
                └── session_manifest.json
```

The finalised artefacts are immutable by default. Re-running `--finalize` against unchanged raw data is a no-op; if an already-finalised artefact would change, the script refuses to overwrite it. This keeps source-code provenance in Git while allowing the larger raw production outputs to be released separately as a research-data archive.


## Mistral Large 3 post-production addition

Mistral Large 3 was added after the original six-model panel had been completed. The exact API snapshot is `mistral-large-2512`. The selection was made on the basis of additional provider/model-family coverage and feasible API access, not after observing Mistral classification outcomes. The existing machine-readable `supplementary`/extension labels are retained to preserve this chronology; they should not be read as assigning Mistral lower scientific status. The working paper treats Mistral alongside the other six models in the main comparative results.

The adapter uses Mistral's documented OpenAI-compatible Chat Completions endpoint at `https://api.mistral.ai/v1`, allowing the existing pinned OpenAI SDK dependency to be reused without changing the Python environment. It requests the same frozen prompts, `temperature=0`, a 40-token output cap, JSON-schema structured output, and the standard service tier. Mistral Large 3 does not document adjustable reasoning support, so the adapter does not send `reasoning_effort`; the run manifest records this as `not_exposed_for_model`. Pricing is recorded separately in `config/pricing/mistral_2026-09-26.yaml`.

Set `MISTRAL_API_KEY`, then validate the provider-specific configuration before touching the production schedule:

```bash
pipenv run python code/03d_smoke_test_mistral.py --n-cases 1
pipenv run python code/03d_smoke_test_mistral.py
```

If the five-case smoke test succeeds and the returned model identity is exactly `mistral-large-2512`, stage the first five legitimate production cells with:

```bash
pipenv run python code/05_run_stability_experiment.py \
  --model-key mistral_large3_2512 \
  --max-decisions 5
```

Check the partial run with:

```bash
pipenv run python code/06_check_and_finalize_production.py \
  --model-key mistral_large3_2512
```

If structural QA passes, resume all remaining cells by rerunning `05_run_stability_experiment.py` without `--max-decisions`. Finalize the completed 12,000-cell extension using the same `06_check_and_finalize_production.py --finalize` workflow used for the original panel.

The original six-model design contributes 72,000 decisions. The supplementary Mistral run adds 12,000 more without changing the original panel definition, for 84,000 decisions across all completed and supplementary runs.


## Stability analysis

After all model runs are finalised, analyse repeated-inference, prompt, and model-family stability with:

```bash
pipenv run python code/07_analyze_stability.py
```

Before calculating any statistic, the script verifies that every local raw `inference_records.jsonl` file has the SHA-256 recorded in the corresponding committed `results/production/<MODEL_KEY>/qa_summary.json`. Analysis therefore fails closed if the local raw data no longer match the finalised provenance.

The stability analysis keeps two concepts separate. **Operational reliability** concerns whether a scheduled decision produced a valid structured classification. **Classification stability** concerns whether valid classifications remain the same under repetition, prompt perturbation, or model-family changes. This matters for the supplementary Mistral run because observed API/parse failures are preserved rather than silently rerun.

Primary aggregate quantities use the inverse-inclusion weights from the frozen label-stratified probability sample so they describe the GB test population represented by the sampling design. Parallel unweighted quantities are retained as sample diagnostics. The script does not attach uncertainty intervals to these diagnostics; design-based uncertainty is handled in the subsequent validation analysis.

Generated files are written to `outputs/stability/`:

- `model_summary.csv` — model-level operational reliability and stability;
- `repetition_by_prompt.csv` — repeated-inference consistency within each prompt;
- `prompt_comparisons.csv` — canonical P0 versus each prompt perturbation;
- `model_pair_agreement.csv` — pairwise model agreement under P0 and across all prompts;
- `panel_agreement.csv` — unanimity for the original six-model panel and all seven models;
- `stability_by_reference_label.csv` — diagnostics by silver-standard reference label;
- `failure_diagnostics.csv` — observed production failures by model, error type, and final status code;
- `observation_instability.csv` — observation-level instability for case inspection; and
- `analysis_manifest.json` — source hashes, panel definitions, and analysis rules.

The original six-model panel remains distinguishable from the later Mistral addition in metadata and panel-level summaries so that experimental chronology is recoverable. This provenance distinction is not used as a substantive hierarchy: the working paper treats all seven models as evaluated models and reports their main results together.


## Design-based validation analysis

The next stage estimates classifier performance relative to the frozen OCPSG silver reference using the probability-sample design:

```bash
pipenv run python code/08_analyze_validation.py
```

The primary validation condition is the canonical P0 prompt. For each sampled speech, its three repeated P0 decisions are averaged so that the observation contributes its inverse-inclusion sampling weight exactly once in the primary confusion matrix. This avoids treating repeated calls on the same speech as independent test-set observations.

The resulting confusion-matrix totals are Horvitz-Thompson-style population estimates under the frozen label-stratified sampling design. Macro-averaged metrics are calculated across the 17 silver-reference labels that occur in the GB target population. The full 23-label benchmark inventory remains available to the classifiers: predictions into benchmark labels absent from the GB silver-reference population still count as errors for the true reference class.

Operational and conditional performance are reported separately. Operational metrics retain API/parse failures as measurement failures; they therefore reduce recall and F1 for the affected reference classes. Conditional metrics describe performance among valid model outputs only. This distinction matters for the supplementary Mistral run, while it is algebraically irrelevant for models with 100% valid output.

Sampling uncertainty is estimated using 2,000 paired Rao-Wu rescaled bootstrap replicates by default. Resampling occurs at the sampled-speech level within the 17 original sampling strata, preserving all repeated inferences for a resampled observation. The rescaling incorporates the stratum sampling fraction, so census strata contribute no sampling variance. The same bootstrap replicate design is reused across models. Percentile 95% intervals therefore quantify uncertainty induced by the frozen probability sample, not every possible source of model or benchmark uncertainty. Because the same bootstrap replicate weights are used for every model, model differences are evaluated from paired replicate differences rather than by comparing overlap of separate marginal confidence intervals.

Outputs are written under `outputs/validation/`:

- `primary_model_performance.csv` — weighted P0 performance, unweighted diagnostics, and 95% bootstrap intervals;
- `condition_performance.csv` — performance for every prompt × repetition condition plus repetition-averaged prompt estimates;
- `per_class_performance.csv` — weighted class-specific precision, recall, F1, support, and F1 intervals;
- `confusion_matrix_population.csv` — weighted confusion-matrix cells interpreted as estimated GB test-population counts;
- `bootstrap_metrics.csv` — replicate-level primary metrics for reproducibility;
- `paired_model_differences.csv` — paired-bootstrap differences in accuracy and macro-F1 across model pairs; and
- `analysis_manifest.json` — frozen estimands, source hashes, bootstrap specification, and uncertainty scope.

These are **silver-reference performance estimates**, not human-gold validation. The distinction should be retained in the manuscript and all tables.


## Full-population deployment for prevalence propagation

After the stability and design-based validation analyses, the prevalence-propagation demonstration uses one frozen deployment over the complete GB silver-reference test population. The deployment specification is stored in `config/deployment.yaml`.

The deployment uses the pinned GPT-5.4 Mini snapshot (`gpt-5.4-mini-2026-03-17`), the canonical P0 prompt, the same `text_en` fallback rule, and one classification per speech. This is a deliberately separate deployment stage rather than an additional stability-model run. The model was selected for the downstream demonstration after the seven-model validation stage because it offers substantially lower API cost than GPT-5.4 while the paired bootstrap comparison of their silver-reference macro-F1 difference included zero. This selection is not an equivalence claim and does not redefine the original model panel.

Inspect the frozen deployment plan without sending requests:

```bash
pipenv run python code/09_run_population_deployment.py --plan-only
```

Then begin with five legitimate population decisions:

```bash
pipenv run python code/09_run_population_deployment.py --max-decisions 5
```

Check the staged deployment with:

```bash
pipenv run python code/10_check_and_finalize_deployment.py
```

If structural QA passes, resume the remaining population:

```bash
pipenv run python code/09_run_population_deployment.py
```

The runner is append-only and resumable. The population is loaded from the exact immutable Hugging Face revision used to freeze the 1,000-case validation sample. It records a population-ID SHA-256 and uses a deterministic randomised PCG64 schedule so elapsed API time is not mechanically aligned with the source-data order. Persistent inference failures are retained after the frozen adapter retry policy rather than silently replaced.

After all 28,978 population decisions are present, finalize with:

```bash
pipenv run python code/10_check_and_finalize_deployment.py --finalize
```

Raw deployment records remain under the gitignored `runs/deployment/` tree. Compact immutable provenance is written under `results/deployment/gb_test_p0_gpt54_mini_v1/` and should be committed before running the subsequent error-propagation analysis.


## Prevalence error propagation

After the finalised full-population deployment is committed, estimate and propagate measurement error with:

```bash
pipenv run python code/11_analyze_error_propagation.py
```

The analysis combines two frozen inputs: the one-pass GPT-5.4 Mini P0 deployment over all 28,978 GB test speeches and the design-based 1,000-speech validation sample with three repeated P0 classifications. Both raw JSONL inputs are checked against the SHA-256 hashes in their committed finalised provenance before analysis begins.

Let `pi` denote the unknown 17-label prevalence vector over the GB silver-reference support, `M` the estimated 17 × 23 misclassification matrix with entries `P(predicted=j | reference=i)`, and `q` the observed 23-label prevalence vector from the full deployment. The measurement model is

```text
q ≈ M^T pi
```

and the corrected prevalence estimator solves

```text
min ||M^T pi - q||²
subject to pi >= 0 and sum(pi) = 1.
```

The implementation uses a dependency-free projected FISTA solver with exact Euclidean projection onto the probability simplex. Direct matrix inversion is not applicable because the forward operator is 23 × 17 rather than square. An unconstrained least-squares solution is retained only as a diagnostic for negative mass, unit-sum violations, and residual fit.

The point misclassification matrix uses inverse-inclusion validation weights and averages the three repeated P0 classifications so each sampled speech contributes its population weight once. Uncertainty is propagated by rebuilding the misclassification matrix in each of 2,000 Rao-Wu rescaled stratified bootstrap replicates and resolving the constrained prevalence problem. The full-population deployment distribution is treated as the observed census output of one frozen model realization; these intervals therefore quantify validation-sample uncertainty in the estimated error mechanism, not provider drift, benchmark-construction uncertainty, or the full stochastic distribution of the model.

The correction is conditional on the 17-label support already known from the frozen silver benchmark sampling frame; the six official labels with zero silver-reference support are therefore constrained to zero true prevalence. The **numerical prevalence values** of those 17 labels are not used to fit the corrected estimator. They are brought in only afterward to demonstrate whether correction moves the measured policy-agenda distribution toward the known benchmark distribution. Consequently, zero corrected mass on the six support-excluded labels is a design constraint, not an empirical success criterion. In applications where true class support is unknown, additional validation information would be required before applying the same correction.

Generated version-controlled outputs are written to `outputs/propagation/`:

- `prevalence_comparison.csv` — silver truth, naive deployment prevalence, corrected prevalence, and 95% propagated intervals for all 23 benchmark labels;
- `propagation_summary.csv` — naive versus corrected total-variation distance, MAE, RMSE, and improvement;
- `misclassification_matrix.csv` — the estimated 17 × 23 P0 error mechanism;
- `bootstrap_propagation.csv` — replicate-level corrected prevalence and error diagnostics;
- `solver_diagnostics.json` — conditioning, convergence, and unconstrained-solution diagnostics; and
- `analysis_manifest.json` — source hashes, estimand, estimator, and uncertainty scope.


## Computational performance and environmental transparency

The final substantive demonstration for Section 4.5 is:

```bash
pipenv run python code/12_analyze_compute_environment.py
```

The script verifies every production raw-inference file against the SHA-256 recorded in finalised provenance before summarising computational performance. It also verifies the finalised full-population GPT-5.4 Mini deployment.

For every model, the analysis reports request-level end-to-end latency, token accounting, valid-output rate, explicit adapter retries, and monetary cost where dated provider pricing and usage metadata were available. Latency is summarised both across all scheduled decisions and separately for valid and invalid outputs, so operational failures are not conflated with ordinary successful-request latency. Hosted and local inference are then treated differently for environmental reporting.

For hosted API models, server-side hardware, utilisation, power draw, data-centre overhead, and electricity mix are not exposed in the recorded API telemetry. The analysis therefore **does not estimate provider-side energy use or carbon emissions from latency, token counts, or price**.

For the two local Ollama models, all production sessions were run with machine-level CodeCarbon tracking. The analysis aggregates the finalised session summaries across every session contributing to the 12,000 decisions and reports total tracked duration, total CPU/GPU/RAM energy, total machine energy, and estimated emissions. Energy and emissions per 1,000 decisions are included only as **run-average normalisations of session-level totals**; they are not presented as direct per-request carbon measurements.

The local inference records also permit timing decomposition into session-start model loading, warm load overhead, prompt evaluation, generation time, and generation throughput. The first valid request in each production session is labelled a **session-start load**, not a perfect cold start, because operating-system, filesystem, driver, and other caches may persist across sessions.

Outputs are written to `outputs/compute_environment/`:

- `production_compute_summary.csv` — computational accounting for all seven production models;
- `deployment_compute_summary.csv` — accounting for the 28,978-case GPT-5.4 Mini deployment;
- `local_inference_timing.csv` — Ollama timing decomposition and generation throughput;
- `local_environment_summary.csv` — aggregated local machine energy/emissions and runtime metadata;
- `local_session_environment.csv` — session-level CodeCarbon summaries;
- `environmental_reporting_scope.csv` — explicit statement of which environmental claims are and are not supported for each model; and
- `analysis_manifest.json` — source hashes, measurement scope, aggregation rules, and interpretive limits.

The governing reporting principle is: **environmental precision should not exceed infrastructure knowledge**.


## Manuscript plot

The working paper retains the detailed stability, validation, and computational results as tables. A single companion figure is generated for the prevalence-propagation demonstration, where the category-level movement from naïve classifier prevalence to corrected prevalence is more informative visually than in an aggregate table alone.

Generate it from the committed analytical outputs with:

```bash
pipenv run python code/13_generate_manuscript_plots.py
```

The figure follows the comparison-oriented style used in the related Eval4SD plotting scripts: a horizontal category layout, softened Oxford-inspired colours, restrained markers, a light x-axis guide, and minimal axis furniture. The caption is not embedded in the graphic.

The script creates:

- `outputs/figures/prevalence_propagation.png`
- `outputs/figures/prevalence_propagation.pdf`
- `outputs/figures/prevalence_propagation.txt`

The 17 policy categories with positive silver-reference support are ordered by silver-reference prevalence. Grey diamonds mark known silver-reference prevalence, terracotta circles mark naïve classifier prevalence, blue squares and horizontal intervals mark corrected prevalence and its propagated 95% uncertainty, and a light connector links the naïve and corrected estimates. The aggregate propagation metrics remain in `outputs/propagation/propagation_summary.csv` and should be reported separately in the manuscript table.

The plotting script requires Matplotlib, which is declared in `pyproject.toml` and `Pipfile`.
