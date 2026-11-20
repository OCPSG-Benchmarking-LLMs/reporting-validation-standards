#!/usr/bin/env python3
"""Generate manuscript calibration and risk–coverage figures from frozen JEV records.

No model calls, no retries, no mutation of the original inference data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reporting_validation_standards import plot_style as ps
from reporting_validation_standards.jev import digest
from reporting_validation_standards.jev_figure_analysis import plot_data_with_uncertainty
from reporting_validation_standards.stability_analysis import sha256_file
from reporting_validation_standards.validation_analysis import rao_wu_rescaled_bootstrap_weights


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/jev/calibration"))
    parser.add_argument("--design", type=Path, default=Path("data/manifests/uk_test_1000_ids.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("outputs/jev/metrics.json"))
    parser.add_argument("--table-dir", type=Path, default=Path("outputs/jev"))
    parser.add_argument("--figure-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--bootstrap-reps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_frozen_inputs(args):
    """Fail closed on changed records, incomplete cases and mismatched analyses."""
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    manifest = json.loads((args.run_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    spec = manifest["specification"]
    if digest(spec) != manifest["sha256"]:
        raise ValueError("Frozen JEV manifest specification has changed")
    if manifest["sha256"] != metrics["manifest_sha256"]:
        raise ValueError("Manuscript metrics do not match the original experiment manifest")
    records_path = args.run_dir / "inference_records.jsonl"
    raw_hash = sha256_file(records_path)
    if raw_hash != metrics["raw_inference_sha256"]:
        raise ValueError("Frozen records differ from the hash in metrics.json; rerun script 15 first")

    design = pd.read_csv(args.design).reset_index(drop=True)
    if design["id"].duplicated().any() or len(design) != metrics["attempted_observations"]:
        raise ValueError("Frozen sample design must have unique, complete IDs")
    data = {}
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        observation_id = record["observation_id"]
        if observation_id in data or record["specification_sha256"] != manifest["sha256"]:
            raise ValueError("Duplicate inference ID or inconsistent run specification")
        data[observation_id] = record
    if set(data) != set(design["id"]):
        raise ValueError("Inference record IDs differ from frozen sample IDs")

    ordered = [data[oid] for oid in design["id"]]
    if any(record["reference_label"] != reference for record, reference in
           zip(ordered, design["predicted_label"], strict=True)):
        raise ValueError("Inference reference labels differ from frozen sample")
    if any(record.get("status") not in ("valid", "failed") for record in ordered):
        raise ValueError("Unknown inference result status")
    valid = np.asarray([record["status"] == "valid" for record in ordered])
    if valid.sum() != metrics["valid_observations"]:
        raise ValueError("Valid-output count differs from metrics.json")
    labels = spec["label_inventory"]
    good = [record for record in ordered if record["status"] == "valid"]
    if any(set(record["probabilities"]) != set(labels) for record in good):
        raise ValueError("Valid record has incomplete or unexpected probability keys")
    prob = np.asarray([[record["probabilities"][label] for label in labels]
                       for record in good], dtype=float)
    if (not np.isfinite(prob).all() or (prob < 0).any() or
            not np.allclose(prob.sum(axis=1), 1.0, atol=1e-4)):
        raise ValueError("Stored valid record has an invalid probability vector")

    references = np.asarray([labels.index(row["reference_label"]) for row in good])
    correct = (prob.argmax(axis=1) == references).astype(float)
    confidence = prob.max(axis=1)
    weights_all = design["sampling_weight"].to_numpy(dtype=float)
    weights = weights_all[valid]
    accuracy = float(np.average(correct, weights=weights))
    expected = metrics["probabilistic_scores_conditional_on_valid_outputs"]["accuracy"]["estimate"]
    if not np.isclose(accuracy, expected, rtol=0, atol=1e-10):
        raise ValueError("Recomputed conditional accuracy disagrees with script 15")
    valid_rate = float(weights.sum() / weights_all.sum())
    if not np.isclose(valid_rate, metrics["design_weighted_valid_rate"], rtol=0, atol=1e-10):
        raise ValueError("Recomputed weighted validity disagrees with script 15")
    return design, valid, confidence, correct, weights, metrics, manifest, raw_hash


def save_with_caption(fig, stem: Path, caption: str):
    """Transparent PNG, PDF and SVG at 300 dpi plus the .txt caption."""
    ps.save(fig, stem, caption)
    plt.close(fig)


def calibration_figure(data: pd.DataFrame, n_valid: int, stem: Path, reps: int,
                       n_attempted: int = 1000):
    """Calibration points with the weighted mass of each confidence bin below."""
    ps.use_style()
    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(ps.PRINT_WIDTH_IN, 5.4), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.1]},
    )
    reference, = ax.plot(
        [0, 1], [0, 1], linestyle="--", color=ps.REFERENCE, lw=1.3,
        label="Perfect calibration", zorder=2,
    )
    errors = ax.errorbar(
        data["mean_confidence"], data["observed_accuracy"],
        yerr=np.vstack([
            np.maximum(0, data["observed_accuracy"] - data["ci_low"]),
            np.maximum(0, data["ci_high"] - data["observed_accuracy"]),
        ]),
        fmt="o", linestyle="none", markersize=5.5, color=ps.BLUE,
        ecolor=ps.BLUE, markeredgecolor=ps.EDGE, markeredgewidth=.5,
        elinewidth=1.1, capsize=2.5, label="Observed accuracy (95% CI)", zorder=4,
    )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.set_ylabel("Observed accuracy")

    centres = (data["lower"].to_numpy() + data["upper"].to_numpy()) / 2
    bx.bar(
        centres, data["estimated_population_mass"], width=.078,
        color=ps.LIGHT_GREY, edgecolor=ps.EDGE, linewidth=.5, zorder=3,
    )
    bx.set_ylim(0, max(.1, float(data["estimated_population_mass"].max()) * 1.17))
    bx.set_ylabel("Weighted share")
    bx.set_xlabel("Predicted top-class probability")
    bx.set_xticks(np.arange(0, 1.01, .2))
    for axes in (ax, bx):
        ps.clean_axes(axes)
    ax.tick_params(axis="x", labelbottom=False)
    fig.align_ylabels((ax, bx))
    fig.tight_layout(rect=(0, .07, 1, 1), h_pad=1.0)
    ps.bottom_legend(fig, [errors, reference],
                     ["Observed accuracy (95% CI)", "Perfect calibration"], ncol=2, y=.06)

    bin_counts = "; ".join(
        f"{row.lower:.1f}–{row.upper:.1f}: {int(row.n)}"
        for row in data.itertuples(index=False)
    )
    caption = (
        "JEV Top-Class Calibration against the Automated OCPSG Silver Reference\n\n"
        f"Note. All {n_valid:,} valid first-pass outputs are inverse-probability "
        "weighted. Observed accuracy and mean confidence are summarised in "
        "prespecified 0.1-wide probability bins; error bars are pointwise 95% "
        f"percentile intervals from {reps:,} stratified Rao–Wu rescaled bootstrap "
        "replicates. The coral dashed line marks perfect calibration, and the lower "
        "panel shows each bin's weighted share among valid outcomes. Unweighted "
        f"observations by confidence bin: {bin_counts}. The low-confidence bins have "
        "few cases, so their estimates and intervals are imprecise. The "
        f"{n_attempted - n_valid:,} invalid first-pass responses and diagnostic reruns "
        "are excluded; the figure does not evaluate calibration against independent "
        "human labels."
    )
    save_with_caption(fig, stem, caption)


def zero_error_coverage_end(data: pd.DataFrame) -> float | None:
    """Last coverage of the leading stretch with zero observed error and a zero interval.

    A percentile bootstrap cannot generate unobserved errors, so where every retained
    case agrees with the reference both bounds collapse to zero. The caption reports
    this limitation; the figure does not invent an upper bound.
    """
    coverage = data["coverage"].to_numpy(dtype=float)
    zero_band = (
        np.isclose(data["selective_error"].to_numpy(), 0, rtol=0, atol=1e-12)
        & np.isclose(data["ci_high"].to_numpy(), 0, rtol=0, atol=1e-12)
    )
    if not zero_band[0]:
        return None
    prefix = int(np.argmax(~zero_band)) if not zero_band.all() else len(zero_band)
    return float(coverage[prefix - 1])


def risk_coverage_figure(data: pd.DataFrame, baseline: dict, n_valid: int,
                         stem: Path, reps: int, n_attempted: int = 1000):
    """Selective error by automated coverage against a random-selection benchmark."""
    ps.use_style()
    fig, ax = plt.subplots(figsize=(ps.PRINT_WIDTH_IN, 3.9))
    coverage = data["coverage"].to_numpy(dtype=float)
    random_band = ax.axhspan(
        baseline["random_review_ci_low"], baseline["random_review_ci_high"],
        color=ps.REFERENCE, alpha=.10, lw=0, zorder=1,
    )
    random_line = ax.axhline(
        baseline["random_review_expected_error"], color=ps.REFERENCE,
        linestyle="--", lw=1.3, zorder=3,
    )
    ribbon = ax.fill_between(
        coverage, data["ci_low"], data["ci_high"],
        color=ps.LIGHT_GREY, alpha=.85, linewidth=0, zorder=2,
    )
    selected, = ax.plot(coverage, data["selective_error"], color=ps.BLUE, lw=1.8, zorder=4)
    ax.set_xlim(.1, 1.01)
    upper = max(float(data["ci_high"].max()), baseline["random_review_ci_high"]) * 1.1
    ax.set_ylim(0, min(1., max(upper, .02)))
    ax.set_xticks(np.arange(.1, 1.01, .1))
    ax.set_xlabel("Automated coverage among valid outputs")
    ax.set_ylabel("Classification error rate")
    ps.clean_axes(ax)
    fig.tight_layout(rect=(0, .13, 1, 1))
    ps.bottom_legend(
        fig, [selected, ribbon, random_line, random_band],
        ["Highest confidence first", "95% CI",
         "Random selection", "Random selection (95% CI)"],
        ncol=2, y=.12,
    )

    zero_coverage_end = zero_error_coverage_end(data)
    zero_interval_note = (
        f" No disagreements were observed through {zero_coverage_end:.0%} "
        "automated coverage on the plotted grid. In this region, the bootstrap "
        "interval can collapse to zero because resampling cannot produce errors "
        "absent from the retained observations; zero observed risk must not be "
        "interpreted as a zero true error probability."
        if zero_coverage_end is not None else ""
    )
    caption = (
        "JEV Selective-Classification Error against the Automated OCPSG Silver "
        "Reference\n\n"
        f"Note. Based on {n_valid:,} first-pass valid outputs and inverse-probability "
        "design weights. The blue line shows expected error at each coverage threshold "
        "when retaining the most confident observations; exactly tied confidences are "
        "fractionally averaged, independently of row order. The grey ribbon shows "
        f"pointwise 95% percentile intervals from {reps:,} stratified Rao–Wu rescaled "
        "bootstrap replicates. The coral dashed line denotes expected error under random "
        "selection, with its 95% interval as a faint coral band. The excluded "
        "low-confidence observations are not assumed to have been successfully "
        "human-reviewed. Coverage and error are conditional on the "
        f"{n_valid:,} valid first-pass responses, not on all {n_attempted:,} attempts, "
        f"and exclude the {n_attempted - n_valid:,} invalid outputs and all diagnostic "
        f"reruns.{zero_interval_note}"
    )
    save_with_caption(fig, stem, caption)


def main():
    args = parse_args()
    design, valid, conf, correct, weights, metrics, manifest, raw_hash = load_frozen_inputs(args)
    reps = args.bootstrap_reps or int(metrics["bootstrap"]["replicates"])
    seed = args.seed if args.seed is not None else int(metrics["bootstrap"]["seed"])
    if reps < 20:
        raise ValueError("At least 20 bootstrap replicates required for figure intervals")
    replicates, ids = rao_wu_rescaled_bootstrap_weights(
        design, bootstrap_reps=reps, seed=seed
    )
    if ids != design["id"].astype(str).tolist():
        raise ValueError("Bootstrap observation ordering mismatch")
    calibration, risk, baseline = plot_data_with_uncertainty(
        conf, correct, weights, replicates[:, valid]
    )
    args.table_dir.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(args.table_dir / "calibration_figure_data.csv", index=False)
    risk.to_csv(args.table_dir / "risk_coverage_figure_data.csv", index=False)
    calibration_figure(
        calibration, len(conf), args.figure_dir / "jev_calibration", reps, len(design)
    )
    risk_coverage_figure(
        risk, baseline, len(conf), args.figure_dir / "jev_risk_coverage", reps,
        len(design),
    )
    figure_manifest = {
        "source_raw_sha256": raw_hash,
        "source_experiment_spec_sha256": manifest["sha256"],
        "reference": "automated_silver",
        "population": "frozen_GB_test_design_conditional_on_valid_JEV_responses",
        "bootstrap": {"method": "Rao-Wu rescaled", "replicates": reps,
                      "seed": seed, "interval": "95% percentile"},
        "risk_coverage_tie_policy": baseline["tie_policy"],
        "random_selection_expected_error": baseline["random_review_expected_error"],
        "random_selection_ci": [baseline["random_review_ci_low"],
                                baseline["random_review_ci_high"]],
        "note": "Figures use original inference records only; no JEV API calls.",
    }
    (args.table_dir / "figure_manifest.json").write_text(
        json.dumps(figure_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Verified original {len(design)} cases ({len(conf)} valid); no API calls")
    print(f"Bootstrap: {reps} Rao-Wu replicates, seed {seed}")
    print(f"Figures: {args.figure_dir / 'jev_calibration'}.[pdf|svg|png|txt]")
    print(f"Figures: {args.figure_dir / 'jev_risk_coverage'}.[pdf|svg|png|txt]")
    print(f"Plot data and figure manifest: {args.table_dir}")


if __name__ == "__main__":
    main()
