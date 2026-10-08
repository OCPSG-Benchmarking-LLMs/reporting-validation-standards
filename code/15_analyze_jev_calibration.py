#!/usr/bin/env python3
"""Design-weighted JEV probabilistic validity and risk–coverage demonstration."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reporting_validation_standards.jev import digest, reliability_bins, weighted_metrics
from reporting_validation_standards.validation_analysis import rao_wu_rescaled_bootstrap_weights
from reporting_validation_standards.stability_analysis import sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=Path("runs/jev/calibration"))
    p.add_argument("--design", type=Path, default=Path("data/manifests/uk_test_1000_ids.csv"))
    p.add_argument("--output", type=Path, default=Path("outputs/jev"))
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261008)
    args = p.parse_args()
    if args.bootstrap_reps < 1:
        raise ValueError("Require at least one bootstrap replicate")
    manifest = json.loads((args.run_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    spec = manifest["specification"]
    if digest(spec) != manifest["sha256"]:
        raise ValueError("Frozen specification hash mismatch")
    labels = spec["label_inventory"]
    design = pd.read_csv(args.design)
    if design.id.duplicated().any():
        raise ValueError("Duplicate IDs in design")
    records_path = args.run_dir / "inference_records.jsonl"
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    by_id = {}
    for record in records:
        oid = record["observation_id"]
        if oid in by_id or oid not in set(design.id) or record["specification_sha256"] != manifest["sha256"]:
            raise ValueError("Unexpected/duplicate record or incorrect specification")
        by_id[oid] = record
    if len(by_id) != len(design):
        raise RuntimeError(f"Incomplete JEV run: {len(by_id)}/{len(design)}")
    complete = [by_id[oid] for oid in design.id]
    valid = np.array([r["status"] == "valid" for r in complete])
    weights_all = design.sampling_weight.to_numpy(dtype=float)
    valid_mass = float(weights_all[valid].sum() / weights_all.sum())
    if not valid.any():
        raise RuntimeError("No valid JEV distributions")
    for i, record in enumerate(complete):
        if record["reference_label"] != design.predicted_label.iloc[i]:
            raise ValueError("Reference label mismatch")
    good = [complete[i] for i in range(len(complete)) if valid[i]]
    prob = np.asarray([[r["probabilities"][label] for label in labels] for r in good])
    if any(set(r["probabilities"]) != set(labels) for r in good):
        raise ValueError("Probability inventory mismatch")
    y = np.asarray([labels.index(r["reference_label"]) for r in good])
    w = weights_all[valid]
    primary = weighted_metrics(prob, y, w)
    # Operational accuracy counts an invalid response as incorrect, retaining
    # its frozen probability-sampling weight.
    operational_correct = np.zeros(len(design), dtype=float)
    operational_correct[valid] = (prob.argmax(axis=1) == y).astype(float)
    operational_accuracy = float(np.dot(weights_all, operational_correct) / weights_all.sum())
    boot, ids = rao_wu_rescaled_bootstrap_weights(
        design.reset_index(drop=True), bootstrap_reps=args.bootstrap_reps, seed=args.seed)
    if list(ids) != design.id.tolist():
        raise ValueError("Bootstrap observation order mismatch")
    estimates = []
    for b in boot:
        bw = b[valid]
        if bw.sum() <= 0:
            continue
        estimates.append(weighted_metrics(prob, y, np.maximum(bw, 1e-15)))
    boot_totals = boot.sum(axis=1)
    valid_rate_replicates = np.dot(boot, valid.astype(float)) / boot_totals
    operational_replicates = np.dot(boot, operational_correct) / boot_totals
    operational_summary = {
        "estimate": operational_accuracy,
        "ci_low": float(np.quantile(operational_replicates, .025)),
        "ci_high": float(np.quantile(operational_replicates, .975)),
    }
    valid_rate_summary = {
        "estimate": valid_mass,
        "ci_low": float(np.quantile(valid_rate_replicates, .025)),
        "ci_high": float(np.quantile(valid_rate_replicates, .975)),
    }
    columns = [key for key in primary if key != "n"]
    uncertainty = {key: {"estimate": primary[key],
                         "ci_low": float(np.quantile([d[key] for d in estimates], .025)),
                         "ci_high": float(np.quantile([d[key] for d in estimates], .975))}
                   for key in columns}
    # Calibrated probability vectors can be aggregated, conditional on valid outputs.
    soft = np.average(prob, axis=0, weights=w)
    hard = np.bincount(prob.argmax(axis=1), weights=w, minlength=len(labels)) / w.sum()
    reference = np.bincount(y, weights=w, minlength=len(labels)) / w.sum()
    prevalence = pd.DataFrame({"label": labels, "soft_probability_prevalence": soft,
                               "hard_argmax_prevalence": hard,
                               "silver_reference_prevalence": reference})
    # Fixed-width descriptive reliability bins, with effective mass and outcomes.
    confidence = prob.max(axis=1)
    correct = (prob.argmax(axis=1) == y).astype(float)
    bins = reliability_bins(confidence, correct, w)
    args.output.mkdir(parents=True, exist_ok=True)
    prevalence.to_csv(args.output / "prevalence_comparison.csv", index=False)
    pd.DataFrame(bins).to_csv(args.output / "reliability_bins.csv", index=False)
    (args.output / "metrics.json").write_text(json.dumps({
        "validation_target": "automated_silver_reference_not_independent_human_gold",
        "probability_status": "native_JEV_choice",
        "reference_population": "GB frozen test sample",
        "valid_observations": int(valid.sum()), "attempted_observations": len(valid),
        "design_weighted_valid_rate": valid_mass,
        "design_weighted_valid_rate_ci": valid_rate_summary,
        "operational_accuracy_treating_invalid_as_errors": operational_summary,
        "probabilistic_scores_conditional_on_valid_outputs": uncertainty,
        "bootstrap": {"method": "Rao-Wu rescaled stratified bootstrap",
                      "replicates": len(estimates), "seed": args.seed},
        "manifest_sha256": manifest["sha256"],
        "raw_inference_sha256": sha256_file(records_path),
        "interpretation": ("Intervals reflect sampling uncertainty conditional on the silver "
                           "benchmark and the observed single JEV realisation; the soft "
                           "prevalence comparison is descriptive, not a census deployment.")
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Valid {sum(valid)}/{len(valid)}; weighted valid rate {valid_mass:.4f}")
    print(f"Conditional accuracy {primary['accuracy']:.4f}; "
          f"operational accuracy {operational_accuracy:.4f}; "
          f"Brier {primary['brier_multiclass']:.4f}")
    print("Saved to", args.output)


if __name__ == "__main__":
    main()
