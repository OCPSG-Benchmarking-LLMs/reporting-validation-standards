# Experiment decisions and environment notes

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

The first request in a run is treated as a model-load/first-request observation, while subsequent requests measure resident-model behavior. A first-request load time should not be interpreted as a fully controlled storage-level cold start because operating-system and filesystem caches may persist across nearby runs.

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

The production prompt-stability experiment uses the same deterministic randomized decision schedule for every model. The base design contains 1,000 frozen observations × four frozen prompts × three repetitions = 12,000 cells per model. Cells are permuted with NumPy PCG64 using seed 20260923 under schedule algorithm `randomized_cells_pcg64_v1`.

Randomizing cells rather than running complete prompt or repetition blocks reduces the alignment between prompt condition and elapsed run time, and separates repeated classifications of the same observation. The schedule seed, algorithm, and SHA-256 are recorded in every model's production manifest.

Production output is append-only and resumable. A completed decision cell is never silently rerun after interruption, regardless of whether its parse status is successful or an observed failure; provider-level retries remain governed by the frozen adapter retry policy. This preserves failures as measurement outcomes rather than replacing them through undocumented recovery attempts.

For local production runs, CodeCarbon tracking is mandatory for every session. If a local run is interrupted and resumed, energy is measured in separate session-level segments and can later be aggregated without inventing request-level carbon estimates.


## 2026-09-23 — Production data archival policy

Raw production inference records and local CodeCarbon artefacts remain under the gitignored `runs/` tree rather than being committed to ordinary Git history. Completed model runs are checked and finalized with a single script, `code/08_check_and_finalize_production.py`.

The script operates in check-only mode by default so it can be used safely on partial runs. Explicit `--finalize` is allowed only when both structural QA and completion QA pass. Finalization records SHA-256 hashes for every raw run file and copies only lightweight provenance into `results/production/<model_key>/`: the frozen experiment manifest, a machine-readable QA summary, a raw-file hash inventory, and session manifests.

Finalized provenance is immutable: a repeated finalization with identical source data is a no-op, while changed source data or changed archival content causes the script to refuse a silent overwrite. The eventual full raw inference corpus can therefore be deposited as a separate versioned research-data release while remaining cryptographically linked to the Git repository.
