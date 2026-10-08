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
seed** recorded by script 15. It writes publication-ready, **transparent-background**
300-dpi PNG, vector PDF and text-preserving SVG, with standalone `.txt`
captions, under `outputs/figures/`:

- `jev_calibration.[png|pdf|svg|txt]` — observed bin accuracy and design-based
  95% bootstrap intervals against mean native top-class probability, plus
  a weighted-mass histogram. The standalone caption now includes the
  unweighted `n` for every nonempty confidence bin; raw bin counts also
  remain in the CSV, not as annotations crowding the figure.
- `jev_risk_coverage.[png|pdf|svg|txt]` — weighted error versus automated
  coverage, pointwise 95% bootstrap interval, and the expected constant error
  benchmark under random selection. A small annotation identifies any
  initial error-free coverage region where the empirical bootstrap interval
  collapses to zero. This **cannot** establish that the underlying true
  error probability is zero.

### Established manuscript plotting standard

These plots follow the visual conventions in `code/13_generate_manuscript_plot.py`
and the supplied Eval4SD stage 2/4 plotting scripts. The figure style uses:

- softened Oxford blue (`#5F7A9D`) for the empirical series, pastel terracotta
  (`#E6A57E`) for reference lines, sage (`#A8C5A1`) for the
  histogram, and pastel mauve (`#D9B8D0`) for the risk interval;
- **no embedded chart titles**, frameless legends below the axes, **no
  gridlines**, no visible spines or tick marks, and decimal [0, 1]
  probability axes;
- **transparent** PNG/PDF/SVG exports with 300-dpi raster output, alongside
  full source notes in separate text captions (not overlaid on the chart).

The statistical calculations, confidence estimands and raw records are
unchanged. These revisions remove the gridlines, describe bin support in the
caption, and flag a statistical limitation of the zero-error region; they do
not create an unsupported upper confidence bound.

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

## How to read the two JEV figures

JEV is TypeSafe's focused, structured decision model. Here we send one GB
parliamentary speech and a closed inventory of 23 official major CAP policy
categories. In `choice` mode, JEV returns one selected class **and a probability
for each class**, rather than generating an unconstrained text response.
The figure scripts compute a **top-class probability** as the maximum of the
23 returned class probabilities. **This is not the provider's separate
`confidence` field**, which is a distinct summary of the distribution.

As a deliberately hypothetical example, if JEV assigns Health 0.80,
Social Welfare 0.15 and Education 0.05, it selects Health with a top-class
probability of 0.80. The class probabilities are informative only if their
relationship to actual outcomes is empirically evaluated; a probability of
0.80 is not by itself a guarantee of 80% accuracy.

**Calibration plot (diagnosis):** Each point groups speeches into
prespecified top-class probability bands. The horizontal position is
**design-weighted average predicted probability**, and the vertical position
is the **design-weighted share whose top-choice matches the automated silver
reference**. The diagonal is the hypothetical situation in which the two
quantities agree exactly. A point **below** the diagonal indicates
overconfidence relative to that reference, while a point **above** it indicates
underconfidence. Vertical whiskers reflect bootstrap sampling uncertainty.
The lower histogram reports each bin's share of the *weighted mass among
valid responses*, not its unweighted case count. The latter is reported in
the accompanying caption. The high-probability bin contains most of the
weight; low-confidence bins are relatively sparsely observed. These are
top-label reliability results, not a demonstration of full multiclass
calibration or agreement with human gold labels.

**Risk–coverage plot (selective classification):** Sort valid speeches
from highest to lowest **top-class probability**. At 75% coverage, keep
the highest-confidence 75% of the *design-weighted valid-output mass* in
automated classification, and set the remaining 25% aside for **potential**
human review. The horizontal axis is this **conditional automated
coverage**, not the share of all originally attempted speeches. The
vertical axis is the design-weighted disagreement rate within the
retained subset, measured against the automated silver reference.
The blue line is the error observed under this ranking; the orange line
is the expected error if an equally large subset were selected at random.
No actual reviewer outcomes were observed, and the plot does not assume
that human adjudication would be perfect.

**Zero-error caveat:** At the left of the risk–coverage figure, JEV may
have no disagreements with silver among the highest-ranked cases. The
resampled bootstrap then also finds zero errors and its percentile interval
can collapse to exactly zero. This is **a limitation of the empirical
bootstrap with no observed failures**, not a scientific claim that
unobserved failures are impossible. The figure marks this region and
its standalone caption states the limitation. We do not add an ad hoc
binomial or pseudo-count upper bound because it would not automatically
respect the original label-stratified sampling design. The existing
bootstrap intervals are pointwise (not simultaneous) and conditional
on this single JEV realisation and the automated reference.

For reporting, distinguish **first-pass validity** (98.60% weighted,
983/1,000 unweighted), **conditional agreement** (92.84% among valid
outputs), and **operational agreement** (91.53% when original rejected
outputs count as errors). The 17 later successful diagnostic reruns do
not replace the original failed responses.

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
