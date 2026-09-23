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

The first implemented component is the frozen UK sampling pipeline. It downloads the canonical `GB/test.parquet` split directly from the Hugging Face dataset repository, resolves the current dataset revision to an immutable commit SHA, draws a reproducible label-stratified probability sample, and stores the information needed for design-based estimation.

The default design uses:

- target sample size: 1,000;
- strata: silver-standard policy label;
- minimum allocation: 25 observations per available label, subject to stratum capacity;
- remainder: allocated approximately proportional to population stratum size;
- random seed: 20260923.

These defaults are explicit parameters rather than hard-coded methodological commitments and can be revised before the sample is frozen.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
hf auth login
```

If the OCPSG dataset is gated, make sure access has been accepted on Hugging Face before running the sampling script.

## Freeze the UK sample

```bash
python scripts/01_freeze_uk_sample.py
```

The script writes:

- `data/manifests/uk_test_1000_ids.csv` — frozen IDs and design variables;
- `data/manifests/uk_test_1000_strata.csv` — stratum population/sample counts;
- `data/manifests/uk_test_1000_metadata.json` — dataset revision, seed, design, and sample hash;
- `data/generated/uk_test_1000.parquet` — full sampled observations for local analysis (gitignored).

Do not redraw the sample after it is frozen. Changes to the design should create a new explicitly versioned sample rather than silently overwrite the registered one.

## Related repositories

- OCPSG `capbench`: benchmarking infrastructure and provider templates.
- OCPSG Silver Standard: source benchmark and benchmark-construction workflow.

