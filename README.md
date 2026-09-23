# Reporting and Validation Standards

Code and reproducible demonstrations for the working paper:

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

- `config/models.yaml` — five core models plus the secondary GPT-6 Luna efficiency model;
- `config/prompts.yaml` — the canonical CapBench prompt and three semantically equivalent perturbations;
- `schemas/inference_record.schema.json` — provider-neutral per-observation inference contract;
- `src/reporting_validation_standards/experiment_specs.py` — configuration validation and prompt rendering;
- `src/reporting_validation_standards/inference.py` — Python representation of the inference record.

The four frozen prompt conditions are:

1. `p0_capbench_canonical` — exact CapBench reference wording;
2. `p1_lexical_paraphrase` — surface-wording perturbation;
3. `p2_label_order_permutation` — fixed permutation of the official label order;
4. `p3_structural_format` — layout/format perturbation with unchanged substantive instructions.

The model panel contains the two pinned GPT-5.4 snapshots, Fireworks DeepSeek V4.1 Flash, local Qwen3.6 35B and Gemma 4 31B, plus `gpt-6-luna` as a secondary moving-alias run after the OpenAI smoke tests succeed.

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
pipenv run python code/03_smoke_test_openai.py
```

For an even smaller connectivity check:

```bash
pipenv run python code/03_smoke_test_openai.py --n-cases 1
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
pipenv run python code/04_smoke_test_fireworks.py --n-cases 1
```

Then run the same five-case smoke panel used for OpenAI:

```bash
pipenv run python code/04_smoke_test_fireworks.py
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
pipenv run python code/05_smoke_test_ollama.py --n-cases 1 --model-key ollama_qwen36_35b
```

Then test Gemma:

```bash
pipenv run python code/05_smoke_test_ollama.py --n-cases 1 --model-key ollama_gemma4_31b
```

If both succeed, run the same five frozen cases for each model:

```bash
pipenv run python code/05_smoke_test_ollama.py --model-key ollama_qwen36_35b
pipenv run python code/05_smoke_test_ollama.py --model-key ollama_gemma4_31b
```

The script keeps a model resident throughout the run, queries Ollama's running-model state after the first successful request to record total resident size and VRAM-resident size, and then explicitly unloads the model after the run. This allows the manifest to document CPU/GPU residency rather than inferring it from nominal model size or GPU capacity.

### Local energy and carbon measurement

CodeCarbon is optional during smoke testing and is applied at the **run level**, not once per speech. Because Ollama executes inference in a separate server process, CodeCarbon uses `tracking_mode="machine"` so GPU/CPU energy associated with the server is not excluded by process-level tracking.

Use an explicit three-letter country code for the machine's physical location. For example, when the machine is physically in the Netherlands:

```bash
pipenv run python code/05_smoke_test_ollama.py \
  --model-key ollama_qwen36_35b \
  --track-carbon \
  --country-iso-code NLD
```

The run directory then contains `codecarbon_emissions.csv` and a compact `carbon_summary.json` alongside the inference records and run manifest. The manifest also records platform metadata and, when available, NVIDIA GPU model, UUID, driver, VRAM, and power limit.

## Production stability preflight

Before launching the full 1,000-case × four-prompt × three-repetition stability experiment, run a length/context preflight:

```bash
pipenv run python code/06_preflight_stability_experiment.py
```

The script verifies the frozen sample, reports text and rendered-prompt length distributions, lists the longest sampled speeches, confirms the planned 12,000 decisions per model / 72,000 decisions overall, and uses successful Ollama smoke records to construct an empirical prompt-token proxy. When recent Ollama residency manifests are available, it compares those estimates with the runtime context length.

The token proxy is diagnostic rather than tokenizer-exact. The completed preflight showed that the longest rendered prompts approach the 4,096-token smoke-test context too closely to leave a defensible safety margin. The production configuration therefore freezes `num_ctx=8192` for both local models. The generated preflight JSON is written below `outputs/` and is therefore not committed.

The longest frozen speech can be used for a targeted context test before production:

```bash
pipenv run python code/05_smoke_test_ollama.py --observation-id ParlaMint-GB_2016-06-08-lords.u126 --model-key ollama_qwen36_35b
pipenv run python code/05_smoke_test_ollama.py --observation-id ParlaMint-GB_2016-06-08-lords.u126 --model-key ollama_gemma4_31b
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
pipenv run python code/07_run_stability_experiment.py --model-key <MODEL_KEY>
```

The runner uses the same deterministic randomized 12,000-cell schedule for every model (1,000 observations × four prompts × three repetitions), appends each completed inference record immediately, validates the frozen experiment manifest on resume, and skips every decision cell already present. Use `--plan-only` to inspect a model run without sending requests. Use `--max-decisions N` for a staged launch; rerunning the same command later resumes the identical schedule.

For local Ollama production runs, energy tracking is required:

```bash
pipenv run python code/07_run_stability_experiment.py \
  --model-key ollama_qwen36_35b \
  --track-carbon \
  --country-iso-code NLD

pipenv run python code/07_run_stability_experiment.py \
  --model-key ollama_gemma4_31b \
  --track-carbon \
  --country-iso-code NLD
```

Each invocation creates a session manifest. This permits interrupted local runs to be resumed while preserving separate CodeCarbon segments. The classification records remain model-level and append-only. Do not update local model files, Ollama, the GPU driver, or the Python environment between production sessions unless the resulting infrastructure change is intentionally documented and the affected run is restarted.


## Check and finalize a production run

Production runs remain under the gitignored `runs/` directory. Use the combined QA/finalization script to inspect a partial or completed model run:

```bash
pipenv run python code/08_check_and_finalize_production.py \
  --model-key <MODEL_KEY>
```

Check-only mode has no archival side effects and can be used during staged or incomplete runs. It validates the frozen schedule, prompt hashes, model/provider identity, manifest consistency, duplicate or unexpected cells, prompt × repetition coverage, parse outcomes, retries, latency, token usage, returned model identity, and recorded API cost when available.

After all 12,000 scheduled cells for a model are complete, finalize the run explicitly:

```bash
pipenv run python code/08_check_and_finalize_production.py \
  --model-key <MODEL_KEY> \
  --finalize
```

Finalization is fail-closed: it proceeds only when structural QA and completion QA both pass. It leaves the raw inference and CodeCarbon files under `runs/`, computes SHA-256 hashes for every raw run file, and writes only compact provenance artefacts to the version-controlled path:

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

The finalized artefacts are immutable by default. Re-running `--finalize` against unchanged raw data is a no-op; if an already-finalized artefact would change, the script refuses to overwrite it. This keeps source-code provenance in Git while allowing the larger raw production outputs to be released separately as a research-data archive.
