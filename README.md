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

## Repository structure

```text
code/       executable research scripts
src/        reusable Python package
tests/      automated tests
data/       manifests and locally generated data
outputs/    generated analytical outputs
```

Using `code/` rather than `scripts/` is intentional: this repository is primarily a research-companion repository, and `code/` makes the role of the numbered analysis programs explicit.

## Related repositories

- OCPSG `capbench`: benchmarking infrastructure and provider templates.
- OCPSG Silver Standard: source benchmark and benchmark-construction workflow.
