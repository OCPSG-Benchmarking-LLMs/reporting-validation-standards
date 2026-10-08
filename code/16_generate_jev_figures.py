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
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from reporting_validation_standards.jev import digest
from reporting_validation_standards.jev_figure_analysis import plot_data_with_uncertainty
from reporting_validation_standards.stability_analysis import sha256_file
from reporting_validation_standards.validation_analysis import rao_wu_rescaled_bootstrap_weights


BLUE = "#002147"      # Oxford blue
AQUA = "#3A8995"      # muted teal, legible with the blue line
CORAL = "#C06E50"     # muted coral
GREY = "#666D78"
PALE = "#E5E9ED"


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


def setup_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALE)
    ax.spines["bottom"].set_color(PALE)
    ax.tick_params(colors=GREY, labelsize=9, length=3)
    ax.grid(axis="y", color=PALE, linewidth=.7, zorder=0)
    ax.set_axisbelow(True)


def save_with_caption(fig, stem: Path, caption: str):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(
            stem.with_suffix("." + extension),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    stem.with_suffix(".txt").write_text(caption.strip() + "\n", encoding="utf-8")
    plt.close(fig)


def calibration_figure(data: pd.DataFrame, n_valid: int, stem: Path, reps: int):
    fig = plt.figure(figsize=(7.3, 7.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[3.8, 1.2], hspace=.05)
    ax = fig.add_subplot(grid[0])
    bx = fig.add_subplot(grid[1], sharex=ax)
    setup_axes(ax)
    setup_axes(bx)
    ax.plot([0, 1], [0, 1], linestyle="--", color=GREY, lw=1.2,
            label="Perfect calibration", zorder=1)
    error_below = np.maximum(0, data["observed_accuracy"] - data["ci_low"])
    error_above = np.maximum(0, data["ci_high"] - data["observed_accuracy"])
    ax.errorbar(
        data["mean_confidence"], data["observed_accuracy"],
        yerr=np.vstack([error_below, error_above]),
        fmt="o", linestyle="none", markersize=6.5, color=BLUE, ecolor=AQUA,
        markeredgecolor="white", markeredgewidth=.8,
        elinewidth=1.5, capsize=3.5,
        label="Design-weighted bin accuracy (95% CI)", zorder=3,
    )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Agreement with silver reference", fontsize=10)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title("JEV | Reliability of top-class probabilities",
                 loc="left", color=BLUE, fontsize=13, weight="bold", pad=16)
    plt.setp(ax.get_xticklabels(), visible=False)

    centres = (data["lower"].to_numpy() + data["upper"].to_numpy()) / 2
    bx.bar(centres, data["estimated_population_mass"], width=.082,
           color=AQUA, alpha=.72, edgecolor=BLUE, linewidth=.5, zorder=3)
    for centre, mass, n in zip(
        centres, data["estimated_population_mass"], data["n"], strict=True
    ):
        bx.annotate(f"n={n}", xy=(centre, mass), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=7.5, color=BLUE)
    bx.set_ylim(0, max(.10, float(data["estimated_population_mass"].max()) * 1.25))
    bx.set_ylabel("Weighted mass", fontsize=10)
    bx.set_xlabel("Predicted probability of the selected category", fontsize=10)
    bx.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    bx.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    caption = (
        "JEV calibration against the automated OCPSG silver reference. "
        f"All {n_valid} first-pass valid outputs contribute, weighted by inverse "
        "sampling probability. Points compare the mean top-class probability with "
        "observed accuracy within prespecified 0.1-wide confidence bins. "
        f"Whiskers are percentile 95% intervals from {reps:,} stratified Rao–Wu "
        "rescaled bootstrap replicates; empty-bin replicates are excluded only "
        "from that bin's interval. The dashed line marks perfect calibration. "
        "The histogram shows each bin's proportion of valid weighted mass; "
        "annotations give unweighted case counts. Small bins have imprecise "
        "estimates. Results do not establish calibration to independent human truth, "
        "nor do they include the 17 rejected first-pass outputs."
    )
    save_with_caption(fig, stem, caption)


def risk_coverage_figure(data: pd.DataFrame, baseline: dict, n_valid: int,
                         stem: Path, reps: int):
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    setup_axes(ax)
    coverage = data["coverage"].to_numpy()
    ax.fill_between(coverage, data["ci_low"], data["ci_high"],
                    color=AQUA, alpha=.19, lw=0, label="Selective risk: 95% CI")
    ax.plot(coverage, data["selective_error"], lw=2.3, color=BLUE,
            label="Highest-confidence cases first", zorder=4)
    ax.axhspan(baseline["random_review_ci_low"], baseline["random_review_ci_high"],
               color=CORAL, alpha=.10, lw=0)
    ax.axhline(baseline["random_review_expected_error"], color=CORAL,
               ls="--", lw=1.8, label="Random-selection benchmark", zorder=3)
    for target in (.75, .90, 1.0):
        index = int(np.argmin(abs(coverage - target)))
        ax.scatter(coverage[index], data["selective_error"].iloc[index],
                   marker="o", s=37, color=BLUE, zorder=6,
                   edgecolors="white", linewidths=.8)
        if target < 1:
            ax.annotate(f"{data['selective_error'].iloc[index]:.1%}",
                        (coverage[index], data["selective_error"].iloc[index]),
                        xytext=(-5, 9), textcoords="offset points",
                        fontsize=8.5, color=BLUE, ha="right")
    ax.set_xlim(.10, 1.01)
    upper = max(.12, float(data["ci_high"].max()),
                baseline["random_review_ci_high"]) * 1.12
    ax.set_ylim(0, min(1, upper))
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xticks(np.arange(.1, 1.01, .1))
    ax.set_xlabel("Automated coverage among valid JEV outputs", fontsize=10)
    ax.set_ylabel("Error among retained classifications", fontsize=10)
    ax.set_title("JEV | Risk under confidence-based selective classification",
                 loc="left", color=BLUE, fontsize=12.5, weight="bold", pad=14)
    ax.legend(loc="upper left", frameon=False, fontsize=8.6)
    caption = (
        "Selective-classification risk–coverage curve for JEV's "
        f"{n_valid} first-pass valid outputs, evaluated against the automated "
        "silver reference with inverse-probability sampling weights. At each "
        "coverage value, observations with the largest native top-class "
        "probabilities are retained for automated classification; ties are "
        "fractionally averaged in expectation to avoid arbitrary sample-order "
        "effects. The shaded envelope gives percentile 95% intervals from "
        f"{reps:,} stratified Rao–Wu rescaled bootstrap replicates. The dashed "
        "horizontal benchmark shows the expected error rate under random "
        "selection of the same proportion of cases, with a faint bootstrap "
        "interval. Low-confidence cases omitted from the curve would require "
        "separate human adjudication; no human-review outcomes are observed "
        "or assumed here. Coverage is conditional on valid first-pass outputs "
        "and does not include the 17 rejected responses."
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
