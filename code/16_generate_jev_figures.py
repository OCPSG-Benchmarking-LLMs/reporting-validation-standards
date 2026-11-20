#!/usr/bin/env python3
"""Generate journal-ready calibration and risk–coverage figures from frozen JEV records.

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

from reporting_validation_standards.jev import digest
from reporting_validation_standards.jev_figure_analysis import plot_data_with_uncertainty
from reporting_validation_standards.stability_analysis import sha256_file
from reporting_validation_standards.validation_analysis import rao_wu_rescaled_bootstrap_weights


# Soft Oxford-inspired palette used by stage_4_f1_comparison.py and
# the repository's existing validation_prompt_repetition_profile figure.
BLUE = "#5F7A9D"         # softened Oxford blue
TERRACOTTA = "#E6A57E"   # reference / random-selection baseline
AQUA = "#3A8995"         # Oxford aqua; weighted-mass histogram
MAUVE = "#D9B8D0"        # pointwise confidence intervals
DARK_GREY = "#4D4D4D"


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


def clean_axes(ax):
    """Manuscript standard: no gridlines, axes spines or tick marks."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, labelsize=9, colors=DARK_GREY)
    ax.grid(False, axis="both", which="both")
    ax.set_axisbelow(True)


def save_with_caption(fig, stem: Path, caption: str):
    """Match stage_2 and script 13: transparent PNG/PDF, 300 dpi, tight crop."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            stem.with_suffix("." + extension),
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
    stem.with_suffix(".txt").write_text(caption.strip() + "\n", encoding="utf-8")
    plt.close(fig)


def calibration_figure(data: pd.DataFrame, n_valid: int, stem: Path, reps: int):
    """Minimal calibration points with weighted confidence-bin mass below."""
    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(8.5, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.1]},
    )
    clean_axes(ax)
    clean_axes(bx)

    reference, = ax.plot(
        [0, 1], [0, 1], linestyle="--", color=TERRACOTTA,
        lw=1.8, alpha=.95, label="Perfect calibration", zorder=2,
    )
    errors = ax.errorbar(
        data["mean_confidence"], data["observed_accuracy"],
        yerr=np.vstack([
            np.maximum(0, data["observed_accuracy"] - data["ci_low"]),
            np.maximum(0, data["ci_high"] - data["observed_accuracy"]),
        ]),
        fmt="o", linestyle="none", markersize=7, color=BLUE,
        ecolor=BLUE, alpha=.9, markeredgecolor=DARK_GREY,
        markeredgewidth=.5, elinewidth=1.25, capsize=3.5,
        label="Observed accuracy (95% CI)", zorder=4,
    )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.set_ylabel("Observed accuracy", fontsize=12)
    plt.setp(ax.get_xticklabels(), visible=False)

    centres = (data["lower"].to_numpy() + data["upper"].to_numpy()) / 2
    bx.bar(
        centres, data["estimated_population_mass"], width=.078,
        color=AQUA, edgecolor=DARK_GREY, linewidth=.5,
        alpha=.78, zorder=3,
    )
    bx.set_ylim(0, max(.1, float(data["estimated_population_mass"].max()) * 1.17))
    bx.set_ylabel("Weighted share", fontsize=11)
    bx.set_xlabel("Predicted top-class probability", fontsize=12)
    bx.set_xticks(np.arange(0, 1.01, .2))

    fig.legend(
        handles=[errors, reference],
        labels=["Observed accuracy (95% CI)", "Perfect calibration"],
        loc="lower center", bbox_to_anchor=(.5, .025),
        ncol=2, frameon=False, fontsize=9.5,
    )
    fig.subplots_adjust(left=.13, right=.98, top=.98, bottom=.19, hspace=.20)

    bin_counts = ", ".join(
        f"{row.lower:.1f}–{row.upper:.1f}: n={int(row.n)}"
        for row in data.itertuples(index=False)
    )
    caption = (
        "Note. JEV top-class calibration relative to the automated OCPSG "
        "silver reference. "
        f"All {n_valid} valid first-pass outputs are inverse-probability "
        "weighted. Observed accuracy and mean confidence are summarised in "
        "prespecified 0.1-wide probability bins; error bars are pointwise "
        f"95% percentile intervals from {reps:,} stratified Rao–Wu rescaled "
        "bootstrap replicates. The orange dashed line marks perfect "
        "calibration; the lower panel shows each bin's weighted share "
        "among valid outcomes. Unweighted observations by confidence bin: "
        f"{bin_counts}. Nonempty bootstrap replicate counts are provided in "
        "calibration_figure_data.csv. The low-confidence bins have few cases, "
        "so their estimates and bootstrap intervals are imprecise. The 17 invalid "
        "first-pass responses and diagnostic reruns are excluded; this "
        "does not evaluate calibration against independent human labels."
    )
    save_with_caption(fig, stem, caption)


def risk_coverage_figure(data: pd.DataFrame, baseline: dict, n_valid: int,
                         stem: Path, reps: int):
    """Soft-palette risk–coverage plot with below-figure legend and no title."""
    fig, ax = plt.subplots(figsize=(8.7, 5.6))
    clean_axes(ax)

    coverage = data["coverage"].to_numpy(dtype=float)
    ribbon = ax.fill_between(
        coverage, data["ci_low"], data["ci_high"],
        color=MAUVE, alpha=.48, linewidth=0,
        label="Pointwise 95% CI", zorder=2,
    )
    selected, = ax.plot(
        coverage, data["selective_error"],
        color=BLUE, lw=2.2, alpha=.95,
        label="Highest confidence first", zorder=4,
    )
    ax.axhspan(
        baseline["random_review_ci_low"], baseline["random_review_ci_high"],
        color=TERRACOTTA, alpha=.13, lw=0, zorder=1,
    )
    random_line = ax.axhline(
        baseline["random_review_expected_error"],
        color=TERRACOTTA, linestyle="--", lw=1.8,
        alpha=.95, label="Random-selection benchmark", zorder=3,
    )
    ax.set_xlim(.1, 1.02)
    upper = max(
        .12, float(data["ci_high"].max()),
        baseline["random_review_ci_high"],
    ) * 1.13
    ax.set_ylim(0, min(1., upper))
    # A percentile bootstrap cannot generate unobserved errors. When all
    # retained cases have no observed disagreements, both bounds may be zero.
    # Make that limitation visible without inventing an unsupported upper bound.
    zero_band = (
        np.isclose(data["selective_error"].to_numpy(), 0, rtol=0, atol=1e-12)
        & np.isclose(data["ci_high"].to_numpy(), 0, rtol=0, atol=1e-12)
    )
    zero_coverage_end = None
    if zero_band[0]:
        prefix = int(np.argmax(~zero_band)) if not zero_band.all() else len(zero_band)
        zero_coverage_end = float(coverage[prefix - 1])
        if zero_coverage_end >= .25:
            ax.text(
                .14, ax.get_ylim()[1] * .22,
                f"No observed errors through {zero_coverage_end:.0%} coverage\n"
                "Bootstrap interval collapses at zero",
                fontsize=8.2, color=DARK_GREY, va="center", ha="left",
            )
    ax.set_xticks(np.arange(.1, 1.01, .1))
    ax.set_xlabel("Automated coverage among valid outputs", fontsize=12)
    ax.set_ylabel("Classification error rate", fontsize=12)

    fig.legend(
        handles=[selected, ribbon, random_line],
        labels=["Highest confidence first", "95% CI", "Random selection"],
        loc="lower center", bbox_to_anchor=(.5, .045),
        ncol=3, frameon=False, fontsize=9.5,
    )
    fig.subplots_adjust(left=.12, right=.98, top=.97, bottom=.22)

    zero_interval_note = (
        f" No disagreements were observed through {zero_coverage_end:.1%} "
        "automated coverage on the plotted grid. In this region the "
        "bootstrap interval can collapse to zero because resampling cannot "
        "produce errors absent from the retained observations; zero observed "
        "risk must not be interpreted as a zero true error probability."
        if zero_coverage_end is not None else ""
    )
    caption = (
        "Note. JEV selective-classification error against the automated "
        f"OCPSG silver reference, based on {n_valid} first-pass valid outputs "
        "and inverse-probability design weights. The blue line shows "
        "expected error at each coverage threshold when retaining the "
        "most confident observations; exactly tied confidences are "
        "fractionally averaged, independently of row order. The mauve "
        f"ribbon shows pointwise 95% percentile intervals from {reps:,} "
        "stratified Rao–Wu rescaled bootstrap replicates. The orange "
        "dashed line denotes expected error under random selection, with "
        "a faint corresponding uncertainty band. The excluded low-confidence "
        "observations are not assumed to have been successfully human-reviewed. "
        "Coverage and error are conditional on the 983 valid first-pass "
        "responses, not on all 1,000 attempts, and exclude the 17 invalid "
        "outputs and all diagnostic reruns."
        f"{zero_interval_note}"
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
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.dpi": 300,
    }):
        calibration_figure(
            calibration, len(conf), args.figure_dir / "jev_calibration", reps
        )
        risk_coverage_figure(
            risk, baseline, len(conf), args.figure_dir / "jev_risk_coverage", reps
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
