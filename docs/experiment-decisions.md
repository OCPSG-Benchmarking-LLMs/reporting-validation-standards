# Experiment decisions and environment notes

Documentation uses British English. Existing source-code identifiers and machine-readable provenance retain their original American-English spellings for compatibility and reproducibility.

This file records methodological decisions that should later be reflected in the working paper and reproducibility documentation. Runtime manifests remain the authoritative source for the exact environment used in each completed run.

## 2026-09-23 — Local inference environment

- NVIDIA GeForce driver updated to version 617.14 (2026-09-22 release).
- Ollama updated from 0.12.3 to 0.34.3 before the local-model smoke tests.
- Run manifests capture the driver reported by `nvidia-smi`, the Ollama version, the local model tag, digest, parameter size, quantization, and other registry metadata.

## 2026-09-23 — Retaining Qwen3.6 rather than switching to Qwen3.8

The frozen local panel retains `qwen3.6:35b` (Qwen3.6-35B-A3B) rather than replacing it with the newer Qwen3.8 family before substantive evaluation.

The rationale is methodological rather than a claim that Qwen3.6 is generally superior. First, the model panel had already been frozen before substantive evaluation, so retaining the specified model avoids post-specification model chasing under mutable model infrastructure. Second, Qwen3.6-35B-A3B provides a useful sparse Mixture-of-Experts contrast with the 31B Gemma model in the local-compute demonstration. Third, nominal parameter count is not directly comparable across sparse and dense architectures, making the Qwen/Gemma comparison useful for showing why computational performance, energy use, quantization, and deployment details should be measured rather than inferred from model size alone.

The local Qwen artefact used in the first smoke tests was:

- tag: `qwen3.6:35b`
- digest: `096fdbd02fe620fc10cbeb6537e080f8041aece851e5d696aed024d4f70f2e47`
- family: `qwen35moe`
- parameter size: 35.5B
- quantization: Q4_K_M
- parent model: `qwen3.6-source:35b-a3b-mtp-q4_K_M-20260824`

The corresponding Gemma smoke-test artefact was:

- tag: `gemma4:31b`
- digest: `6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7`
- family: `gemma4`
- parameter size: 31.3B
- quantization: Q4_K_M

## 2026-09-23 — Local timing decomposition

For local Ollama models, a single wall-clock latency statistic is insufficient. The analysis distinguishes:

1. model-load time reported by Ollama;
2. prompt-evaluation time;
3. generation time and generation throughput;
4. client-side end-to-end latency and residual overhead.

The first request in a run is treated as a model-load/first-request observation, while subsequent requests measure resident-model behaviour. A first-request load time should not be interpreted as a fully controlled storage-level cold start because operating-system and filesystem caches may persist across nearby runs.

Run-level CodeCarbon measurement includes the complete run, including model loading, because loading is part of the computational cost of deploying the measurement instrument. Steady-state timing metrics are reported separately.


## 2026-09-23 — Runtime CPU/GPU residency

Ollama `/api/ps` was queried while each local model was resident. The runtime context was 4,096 tokens in the smoke tests.

Qwen3.6-35B-A3B:
- total resident model size: 22,483,965,047 bytes
- VRAM-resident size: 9,329,285,528 bytes
- VRAM fraction: 0.41493 (41.5%)
- implied non-VRAM resident portion: approximately 13.15 GB

Gemma 4 31B:
- total resident model size: 21,471,166,461 bytes
- VRAM-resident size: 9,063,251,311 bytes
- VRAM fraction: 0.42211 (42.2%)
- implied non-VRAM resident portion: approximately 12.41 GB

These measurements confirm substantial partial CPU/GPU residency on the 12 GB RTX 4070 SUPER and help explain why CPU energy exceeds GPU energy in the short CodeCarbon smoke runs. The residency split is recorded from Ollama at runtime rather than inferred from nominal parameter count.

The 4,096-token runtime context should be checked against the full frozen 1,000-case sample before production inference. A preflight script was added for this purpose; production context settings should be frozen only after that check.


## 2026-09-23 — Freezing the local runtime context at 8,192 tokens

The initial Ollama smoke tests used the runtime default of 4,096 tokens. A full-sample preflight over the frozen 1,000-speech sample showed that the longest rendered prompts were estimated at approximately 4,018–4,035 input tokens under a conservative empirical token/word ratio calibrated from the successful Ollama smoke records. No case was estimated to exceed 4,096 input tokens, but one or two cases per prompt/model condition reached at least 90% of that context.

Because the estimate is a diagnostic proxy rather than an exact tokenizer count, and because the context must also leave room for the 40-token structured output budget and implementation overhead, 4,096 provides insufficient safety margin. The production local-model configuration therefore explicitly freezes `num_ctx=8192` for both Qwen3.6-35B-A3B and Gemma 4 31B.

This is a pre-analysis infrastructure decision based on the frozen sample's length distribution, not on classification outcomes. The chosen context remains far below the models' advertised maximum context length while providing substantial headroom for the longest sampled parliamentary speech. A targeted longest-case smoke test is used to confirm the setting before production inference.


## 2026-09-23 — Longest-case validation of the 8,192-token context

The longest frozen GB speech, `ParlaMint-GB_2016-06-08-lords.u126` (2,422 whitespace-delimited words), was classified successfully by both local models after explicitly setting `num_ctx=8192`.

Qwen3.6-35B-A3B returned `Transportation` with 3,037 total provider-reported tokens; Gemma 4 31B also returned `Transportation` with 3,008 total tokens. Ollama `/api/ps` confirmed a runtime context of 8,192 for both models.

Increasing the runtime context from 4,096 to 8,192 had negligible observed effect on model residency. Qwen's VRAM-resident size increased from 9,329,285,528 to 9,337,674,136 bytes (8 MiB), while Gemma's recorded VRAM-resident size remained 9,063,251,311 bytes. The production context is therefore frozen at 8,192.

## 2026-09-23 — Production decision ordering and resumability

The production prompt-stability experiment uses the same deterministic randomised decision schedule for every model. The base design contains 1,000 frozen observations × four frozen prompts × three repetitions = 12,000 cells per model. Cells are permuted with NumPy PCG64 using seed 20260923 under schedule algorithm `randomized_cells_pcg64_v1`.

Randomizing cells rather than running complete prompt or repetition blocks reduces the alignment between prompt condition and elapsed run time, and separates repeated classifications of the same observation. The schedule seed, algorithm, and SHA-256 are recorded in every model's production manifest.

Production output is append-only and resumable. A completed decision cell is never silently rerun after interruption, regardless of whether its parse status is successful or an observed failure; provider-level retries remain governed by the frozen adapter retry policy. This preserves failures as measurement outcomes rather than replacing them through undocumented recovery attempts.

For local production runs, CodeCarbon tracking is mandatory for every session. If a local run is interrupted and resumed, energy is measured in separate session-level segments and can later be aggregated without inventing request-level carbon estimates.


## 2026-09-23 — Production data archival policy

Raw production inference records and local CodeCarbon artefacts remain under the gitignored `runs/` tree rather than being committed to ordinary Git history. Completed model runs are checked and finalised with a single script, `code/06_check_and_finalize_production.py`.

The script operates in check-only mode by default so it can be used safely on partial runs. Explicit `--finalize` is allowed only when both structural QA and completion QA pass. Finalisation records SHA-256 hashes for every raw run file and copies only lightweight provenance into `results/production/<model_key>/`: the frozen experiment manifest, a machine-readable QA summary, a raw-file hash inventory, and session manifests.

Finalised provenance is immutable: a repeated finalisation with identical source data is a no-op, while changed source data or changed archival content causes the script to refuse a silent overwrite. The eventual full raw inference corpus can therefore be deposited as a separate versioned research-data release while remaining cryptographically linked to the Git repository.


## 2026-09-26 — Mistral Large 3 post-production robustness extension

After the original six-model stability panel had been fully run and finalised, Mistral Large 3 was added as a post-production model-family/provider-coverage extension, using the exact API snapshot `mistral-large-2512`.

This addition does not retrospectively alter the chronology of the original panel. It remains labelled `stage: supplementary` and `extension_status: post_production_robustness` in the model configuration so that provenance is explicit. Those labels are chronological, not evaluative: in the working paper Mistral is treated alongside the other six completed models. The selection rationale was to broaden provider and model-family coverage while using available API access at feasible cost. The choice was made before inspecting any Mistral classification outcomes and therefore is not based on comparative performance against the completed six models.

Mistral documents Large 3 v25.12 as an open-weight, general-purpose Mixture-of-Experts model with 41B active parameters, 675B total parameters, and a 256k context window. The exact versioned model identifier `mistral-large-2512` is used instead of the moving `mistral-large-latest` alias.

For API inference, the project uses Mistral's documented OpenAI-compatible base URL (`https://api.mistral.ai/v1`) through the already-frozen OpenAI Python SDK dependency. This avoids changing the Python environment solely to add the Mistral SDK. The adapter preserves the common experiment contract: frozen system/user prompts, `temperature=0`, a 40-token output cap, JSON-schema structured output, explicit retries, returned model identity, token accounting, latency, and dated cost accounting. Mistral's current reasoning documentation identifies adjustable `reasoning_effort` support for Mistral Small and Mistral Medium 3.5, but not Mistral Large 3; therefore the Large 3 adapter omits that request parameter and records the model-level setting as `not_exposed_for_model` rather than falsely claiming that reasoning was explicitly disabled.

The dated standard-tier pricing configuration is `config/pricing/mistral_2026-09-26.yaml`, recording $0.50 per million ordinary input tokens, $0.05 per million cached input tokens, and $1.50 per million output tokens. As with other hosted providers, no provider-side energy or carbon estimate is inferred from API latency or token usage.

The extension reuses the identical frozen 1,000-observation × four-prompt × three-repetition randomised schedule. The original six-model experiment therefore remains 72,000 decisions; Mistral contributes a separate additional 12,000 decisions.


### Initial Mistral smoke-test correction

The first one-case connectivity test failed before returning token usage or a model response. Before any Mistral production cell was run, the provider-specific specification was rechecked against current Mistral documentation. The initial adapter had sent `reasoning_effort="none"` by analogy with other models, but adjustable reasoning is not documented for Mistral Large 3. The parameter was therefore removed and the configuration changed to `reasoning_setting: not_exposed_for_model`. This is an adapter/configuration correction made during smoke testing, before production inference; it does not alter any completed experimental observations.


## 2026-09-26 — Stability analysis estimands and failure handling

The first post-production analysis is implemented in `code/07_analyze_stability.py`. It verifies every local raw inference file against the SHA-256 stored in finalised provenance before reading classifications.

The analysis separates operational reliability from classification stability. API or parse failures remain observed outcomes and are not converted into substantive policy labels. Stability statistics are therefore reported in two forms where relevant: an unconditional quantity in which a missing/invalid classification prevents the cell from being counted as stable, and a conditional quantity restricted to cells for which all classifications required by that comparison are valid. This distinction is particularly important for Mistral Large 3, which completed the scheduled design but retained observed failed classifications under the frozen no-silent-rerun policy.

Repeated-inference stability is measured within model × prompt × observation triads across the three repetitions. Prompt stability compares P0 with each perturbation at matched observation × repetition cells and also examines unanimity across all four prompt conditions. Observation-level stability summarizes all 12 model-specific decisions (four prompts × three repetitions) using complete-case unanimity, the number of distinct valid labels, modal share, and Shannon entropy in bits. Model-family dependence is summarised through pairwise exact agreement for matched decisions, with P0 reported separately from all-prompt agreement. Panel-level unanimity is reported both for the original six-model panel and for all seven models including the post-production Mistral extension.

Because the frozen sample deliberately oversamples rare reference labels, primary aggregate stability quantities use inverse-inclusion sampling weights. Unweighted counterparts are retained as descriptive sample diagnostics. Confidence intervals are not mixed into this diagnostic stage; uncertainty estimation is reserved for the subsequent design-based validation analysis.


## 2026-09-26 — Design-based silver-reference validation estimand

The primary validation analysis is implemented in `code/08_analyze_validation.py`. Its target is agreement with the OCPSG automated silver-standard reference labels in the frozen GB test population; it must not be described as validation against an independent human gold standard.

The primary inference condition is the canonical P0 prompt. Each sampled speech has three repeated P0 classifications. Rather than treating those calls as independent test observations or arbitrarily selecting one repetition, the primary confusion matrix averages the three repeated decisions for each observation. Consequently, every sampled speech contributes its inverse-inclusion weight exactly once in expectation, while observed repeated-inference variability remains represented in its fractional confusion contribution.

Performance estimates are constructed from inverse-inclusion weighted confusion-matrix totals under the frozen label-stratified SRSWOR design. Macro precision, recall, and F1 average across the 17 reference labels with positive support in the GB silver-reference population. The complete 23-label benchmark inventory remains allowable on the prediction side; a prediction into an official label that is absent from the GB reference population is still an error for the observation's true reference class.

Operational and valid-output-conditional estimands are distinguished. Operational metrics treat an observed API/parse failure as a failure of the measurement instrument: it contributes to the denominator and acts as a false negative for the relevant reference class. Conditional metrics exclude invalid outputs and answer the narrower question of performance given that the system returned a valid classification. This separation prevents infrastructure reliability from being silently erased while also allowing substantive classification quality to be examined independently.

Sampling uncertainty is estimated with a Rao-Wu rescaled bootstrap appropriate to stratified sampling without replacement. Within each reference-label stratum, each replicate draws `n_h - 1` sampled observations with replacement and applies the rescaling factor
`1 + sqrt(1-f_h) * (n_h m_hi/(n_h-1) - 1)`, where `f_h=n_h/N_h`. Replicate weights preserve each stratum population total; a census stratum therefore contributes no sampling variance. Resampling occurs at the observation level, keeping all three repeated P0 classifications attached to the resampled speech. The default analysis uses 2,000 paired replicates generated with NumPy PCG64 seed 20260926 and percentile 95% confidence intervals.

The bootstrap intervals quantify uncertainty from the probability-sample design conditional on the observed silver benchmark and classifier runs. They should not be interpreted as integrating benchmark-construction uncertainty, provider/infrastructure drift, or the full stochastic distribution of the model beyond the three observed repeated inferences.

Paired model comparisons use the same bootstrap replicate weights for every model. Differences in accuracy and macro-F1 are therefore calculated within replicate before constructing percentile intervals. This is preferable to inferring model differences from overlap or non-overlap of separate marginal confidence intervals.


## 2026-09-26 — Full-population deployment model for prevalence propagation

The downstream prevalence-error demonstration is separated from the seven-model stability/validation panel. It uses one frozen deployment over all 28,978 speeches in the GB silver-reference test population, with one P0 classification per speech.

The selected deployment instrument is the pinned GPT-5.4 Mini snapshot, `gpt-5.4-mini-2026-03-17`. This choice was made after completion of the comparative validation analysis for a practical downstream demonstration: GPT-5.4 Mini has substantially lower observed API cost than GPT-5.4, while their paired design-based P0 macro-F1 difference was -0.11 percentage points with a 95% paired bootstrap interval of approximately [-1.68, +1.67] percentage points. The interval including zero is treated only as absence of evidence of a difference at this precision, not as a formal equivalence result or a claim that Mini is superior.

The population deployment is frozen in `config/deployment.yaml`: canonical P0 prompt, `text_en` input mode, one decision per observation, temperature 0, 40-token output cap, the model's no-reasoning configuration, deterministic randomised PCG64 ordering, and the existing explicit adapter retry policy. The source population is loaded from the same immutable Hugging Face dataset revision used to draw the validation sample. A SHA-256 of the complete population ID set and the exact randomised schedule hash are stored in the deployment manifest.

The full-population run is append-only and resumable. Persistent provider/API failures are retained as deployment outcomes rather than silently rerun. The completed deployment is checked and finalised separately under `results/deployment/` before any prevalence correction is estimated.


## 2026-09-27 — Constrained prevalence correction estimator

The prevalence-propagation demonstration uses the finalised GPT-5.4 Mini P0 census deployment together with the design-based P0 validation sample. The classifier can output all 23 benchmark labels, whereas the GB silver-reference population has positive support on 17 labels. Consequently, the estimated forward misclassification operator is 23 × 17 and is not suitable for direct matrix inversion.

The primary correction therefore estimates the 17-dimensional reference prevalence vector by simplex-constrained least squares: minimize `||M^T pi - q||_2^2` subject to non-negativity and unit sum. Here `M` is the 17 × 23 conditional misclassification matrix estimated from the validation sample and `q` is the observed 23-label full-deployment prevalence vector. The constrained estimator respects prevalence geometry and remains well-defined when unconstrained least squares would yield negative components or non-unit total mass. The unconstrained solution is retained only as a conditioning/diagnostic comparison.

The point misclassification matrix averages the three P0 validation repetitions within each sampled observation through one-third decision weights, preserving one inverse-inclusion population weight per speech. Sampling uncertainty is propagated with the same 2,000-replicate Rao-Wu rescaled bootstrap design used for Section 4.3: each bootstrap replicate produces a new misclassification matrix, the constrained prevalence problem is solved again, and percentile intervals are computed from the resulting prevalence distribution.

The correction does use one piece of information from the frozen benchmark sampling frame: the target population is known to have positive silver-reference support on 17 of the 23 official labels. The estimator is therefore support-conditioned and constrains the six unsupported labels to zero true prevalence. This is not equivalent to using the full target prevalence vector, but it means that zero corrected mass on those six labels is imposed by design and must not be presented as empirical evidence of successful correction. The numerical prevalence values of the 17 supported labels are not supplied to the correction algorithm; they are used only ex post to compare naive and corrected prevalence. In an application where true class support is unknown, additional validation information for the unsupported classes would be necessary before applying the same correction.


## 2026-09-27 — Computational and environmental reporting scope

Section 4.5 separates request-level computational accounting from infrastructure-level environmental measurement.

Across all seven production models, request-level records support reporting of end-to-end latency, provider/runtime token accounting, explicit adapter retries, structured-output failures, and monetary cost where a dated price table and provider token usage were available. These quantities can be summarised directly from the finalised inference records.

Environmental claims depend on infrastructure observability. Hosted API providers do not expose the server hardware, utilisation, power draw, PUE, regional execution location, or contemporaneous electricity mix required for a defensible run-level energy/carbon estimate. Consequently, the analysis records hosted-provider energy and carbon as unobserved and does not infer them from latency, tokens, model size, or monetary price.

The two local Ollama models were measured with CodeCarbon in tracking mode `machine` for every production session. Section 4.5 therefore reports session/run-level CPU, GPU, RAM, total energy, and estimated emissions for those models, alongside the recorded country assumption, CodeCarbon version, hardware, driver, Ollama version, exact model digest, quantisation, runtime context length, and observed CPU/GPU residency. Session summaries are summed because the 12,000-decision production run for each model was completed across multiple resumable sessions. The initial staged five-decision session is included because those five records are part of the finalised production schedule.

Energy or emissions per 1,000 decisions are permitted only as transparent normalisations of the complete measured session totals. They are not described as directly measured per-request carbon footprints. Likewise, the first request after a production session starts is reported as a session-start model-load observation rather than a strict cold start, because lower-level caches may survive between sessions.

The analysis is implemented in `code/12_analyze_compute_environment.py`. Its governing principle is that environmental precision should not exceed infrastructure knowledge.


## 2026-09-27 — Model-panel provenance versus manuscript presentation

The existing machine-readable stage labels are retained unchanged for reproducibility. They encode the chronology and technical status of the experiment rather than a ranking of scientific importance. In particular, GPT-6 Luna is recorded as `stage: secondary` because it was included in the original six-model panel through a moving alias rather than a pinned snapshot. Mistral Large 3 is recorded as `stage: supplementary` with `extension_status: post_production_robustness` because it was added only after the original six-model panel had been completed, in order to broaden provider and model-family coverage.

These provenance labels must not be interpreted as making GPT-6 Luna or Mistral Large 3 second-class comparison models. In the working paper, all seven completed models are treated as evaluated models and should normally appear together in the principal tables, figures, and substantive discussion. The chronology should be explained once in the research-design description and then separated from the interpretation of model results.

The original-six versus all-seven panel distinction remains useful for analyses that depend on the pre-existing experimental boundary, especially panel-level unanimity. It is therefore preserved in analytical metadata and outputs without being used as a substantive hierarchy.


## 2026-09-27 — Manuscript figure design

The first manuscript-figure script is `code/13_generate_manuscript_plots.py`. It generates three figures directly from the committed stability, validation, and prevalence-propagation outputs so that graphical results remain downstream of the frozen analytical pipeline rather than introducing a separate data-processing path.

The stability figure uses population-weighted **unconditional** unanimity for all three displayed quantities: repeated inference, four-prompt unanimity, and all 12 prompt × repetition decisions. Using unconditional quantities keeps the measurement scale consistent across models and ensures that retained operational failures reduce observed stability rather than disappearing through complete-case conditioning. Conditional stability remains available in the underlying tables for substantive interpretation.

The validation figure juxtaposes design-weighted operational macro-F1 and its 95% Rao–Wu bootstrap interval with the corresponding unweighted sample estimate. The visual objective is not to rank models but to show simultaneously that performance is estimated with uncertainty and that ignoring the deliberately stratified probability-sample design can materially change the reported quantity.

The prevalence-propagation figure is restricted to the 17 policy categories with positive silver-reference support in the frozen GB population. It shows the known silver-reference prevalence only as an ex-post diagnostic target, alongside the naïve GPT-5.4 Mini prevalence and the support-conditioned corrected prevalence with propagated bootstrap intervals. The six support-excluded categories remain documented in the analytical outputs but are not plotted as if their zero corrected prevalence were an empirical discovery.

All seven completed models are treated as evaluated models in the principal stability and validation figures. The machine-readable stage labels are deliberately ignored for visual grouping because they encode experimental chronology rather than substantive importance.


### Figure rendering convention

The manuscript figures use a deliberately clean background consistent with the related Eval4SD plotting scripts: transparent export, no visible axis spines, no background grid, and no caption text embedded inside the image. Each PNG/PDF pair is accompanied by a same-basename UTF-8 TXT file containing the proposed manuscript caption. This keeps the graphical object reusable across manuscript, slides, and supplementary material without hard-coding prose into the image.
