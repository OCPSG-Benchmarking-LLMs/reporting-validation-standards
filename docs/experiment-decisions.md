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
