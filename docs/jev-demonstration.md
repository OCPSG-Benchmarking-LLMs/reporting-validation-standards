# JEV native probability demonstration (post hoc extension)

This experiment is **separate from the frozen seven-model stability panel**.
Its purpose is to examine native probabilistic measurement, calibration,
selective classification and downstream aggregation. It is not an eighth
directly comparable P0 model. In particular, TypeSafe's native `choice`
contract differs from the CapBench JSON-only prompt; retain this distinction
in every manuscript table.

## Preflight

The official service is `POST https://api.typesafe.ai/v1/systemone`.
The `model` value must be an available **versioned** model returned by
authenticated `GET /v1/models`; do not silently use `jev-latest`.
The runner requires `TYPESAFE_API_KEY` in the local environment.
The repository does not send any network request in `--plan-only` mode.

```bash
pipenv run python code/14_run_jev_calibration.py --model <PINNED_MODEL> --plan-only
pipenv run python code/14_run_jev_calibration.py --model <PINNED_MODEL> --max-decisions 1
pipenv run python code/14_run_jev_calibration.py --model <PINNED_MODEL>
pipenv run python code/15_analyze_jev_calibration.py
```

The frozen source remains the 1,000-case label-stratified GB sample,
with `text_en` as input and 23 official prediction labels. Scripts do not
alter the sampling manifest or the seven-model experiment. The single-choice
instructions and criteria are hashed in `runs/jev/calibration/experiment_manifest.json`.
The single-pass runner writes an append-only record for each attempted speech
including failures, actual returned model, usage, latency and probability
vector. A resumed run does not silently retry failed cells.

Raw run records are gitignored and must not be committed. Before reporting,
archive them externally with their SHA-256; `outputs/jev/metrics.json`
records the raw file hash. No token-cost or energy claim is made without
an independently dated provider tariff or energy measurement.

## Publication figures (script 16)

The plotting step reads the **original** 1,000-row inference file and the
frozen probability-sample design. It verifies the original inference SHA-256
and experiment-specification hash against `outputs/jev/metrics.json` before
producing any figure. There are **no** JEV API calls, retries or data repairs.

Run scripts 15 and 16 in order:

```cmd
pipenv run python code/15_analyze_jev_calibration.py
pipenv run python code/16_generate_jev_figures.py
```

Script 16 defaults to the **same 2,000 Rao–Wu rescaled bootstrap draws and
seed** recorded by script 15. It writes publication-ready, white-background
300-dpi PNG, vector PDF and editable-text SVG, with standalone `.txt`
captions, under `outputs/figures/`:

- `jev_calibration.[png|pdf|svg|txt]` — observed bin accuracy and design-based
  95% bootstrap intervals against mean native top-class probability, plus
  a weighted-mass histogram and raw bin counts.
- `jev_risk_coverage.[png|pdf|svg|txt]` — weighted error versus automated
  coverage, pointwise 95% bootstrap interval, and the expected constant error
  benchmark under random selection.

The underlying tables are `outputs/jev/calibration_figure_data.csv`
and `outputs/jev/risk_coverage_figure_data.csv`.
`outputs/jev/figure_manifest.json` records the source hashes, bootstrap
specification and confidence-tie policy.

**Coverage estimand:** The new curve **averages over random selection within
groups of exactly equal top-class probabilities**. This makes risk
independent of arbitrary observation order, even if many probabilities
are exactly 1.0. Earlier point estimates under `risk_75` / `risk_90`
in script 15 rely on stable record-order tie breaking; the figure's
tie-averaged estimates are an explicitly different, more defensible
publication estimand and must not silently be identified with those earlier
numbers. At 100% valid-output coverage the estimates coincide.

Both plots **condition on the original 983 valid first-pass outputs**,
exclude the 17 original failures (and all diagnostic reruns), and assess
agreement only with the automated silver reference. The risk curve describes
*which cases would remain automated*, not achieved human-review outcomes.
The random-review line is a hypothetical expectation, not an observed
treatment arm. The bootstrap interval is pointwise, not a simultaneous band.
Neither figure estimates deployment performance in a new domain.

## Estimands

The target is agreement with **automated silver reference labels**, not
independent human gold labels. Report both design-weighted valid-output rate
and the proper scores *conditional on valid probability vectors*.
A failed response is not assigned fabricated uniform probabilities.

- Multiclass Brier score: sum of 23 squared probability-minus-indicator errors.
- Log loss: negative log probability of the silver reference label.
- Calibration gap: top-choice probability minus observed correctness.
- Reliability bins: weighted 0.1-wide confidence bins (descriptive).
- Selective prediction: weighted error rate at 75%, 90%, and 100%
  target coverage, ranking by highest class probability.
- Descriptive prevalence: weighted mean probability, hard argmax shares,
  and weighted silver-reference proportions on the **validation sample**.
  These estimates are not a full-population JEV deployment.

The Rao–Wu stratified rescaled bootstrap reuses the frozen probability-sample
design and keeps each observation's probability vector intact. Its intervals
capture sampling uncertainty conditional on a fixed silver benchmark and
one JEV inference realisation; they do not measure reference-label error,
model-version drift, or stochastic reruns.

## Interpretation controls

Inspect whether confidence ranks errors sensibly before proposing human-review
thresholds. Do not claim that Brier/log-loss are externally calibrated merely
because the provider calls probabilities 'calibrated'. Treat post hoc experiment
selection and the difference between API-native criteria and the P0 prompt
as explicit limitations. Store the API pricing date separately if a cost
comparison is subsequently added.

## 2026-10-08 — Probability-sum audit

The completed first-pass run has 983 valid and 17 invalid outputs. All 17
failures are recorded as `ValueError: Probabilities do not sum to one` under
the original strict `np.isclose(sum, 1, atol=1e-4)` acceptance rule.
The official TypeSafe API specifies that choice probabilities sum to
**approximately** one; it does not prescribe the repository's chosen tolerance.

The original runner recorded the failure reason but **did not preserve the
17 rejected probability vectors**, so we cannot infer the magnitude of the
original deviations retrospectively. Do not silently mark these outputs valid,
normalise them, raise the tolerance, or overwrite the first-pass records.

An optional, separate diagnostic runner repeats only the original 17
probability-sum failures using the exact saved request specification. These
are explicitly **new inference realisations**, not recovered original responses:

```cmd
pipenv run python code/14b_audit_jev_probability_failures.py --plan-only
pipenv run python code/14b_audit_jev_probability_failures.py --max-decisions 1
pipenv run python code/14b_audit_jev_probability_failures.py --max-decisions 0
```

It records the returned probability sum, signed deviation from one, class
probabilities, selected label, provider model, usage and original-parser status
under `runs/jev/probability_sum_audit.jsonl` (gitignored). It never changes
`runs/jev/calibration/inference_records.jsonl`. The audit is opt-in and each
observation is audited at most once by this runner. It is not a scientific basis
for selecting a favourable acceptance tolerance after observing accuracy.

Script 15 now additionally reports **operational accuracy**, counting an
invalid response as an incorrect measurement, with a separate design-based
bootstrap interval. Its pre-existing Brier, log-loss, calibration and
risk–coverage estimates remain **conditional on valid probability vectors**.

### Diagnostic audit outcome (8 October 2026)

The reported secondary audit completed all 17 selected reruns: **17/17**
passed the unchanged original parser, with each returned probability sum
exactly `1.0` (maximum absolute deviation `0.0`). This documents
**non-reproduction on a subsequent request**, not the original sums or
the cause of their initial rejection. Do not overwrite the first pass.

For future first-pass experiments, script 14 now retains raw class
probabilities, totals, signed sum deviation, returned choice, confidence,
model and usage when an API response is received but parsing rejects it.
This change does not alter the original 1,000-record run.
