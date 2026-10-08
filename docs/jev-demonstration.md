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
