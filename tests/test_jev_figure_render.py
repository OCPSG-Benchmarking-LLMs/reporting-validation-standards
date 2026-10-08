"""Exercise the actual vector/raster JEV figure writers without API access."""
from pathlib import Path
from runpy import run_path

import numpy as np

from reporting_validation_standards.jev_figure_analysis import plot_data_with_uncertainty


SCRIPT = Path(__file__).resolve().parents[1] / "code" / "16_generate_jev_figures.py"


def test_figures_export_pdf_svg_png_and_caption(tmp_path):
    namespace = run_path(str(SCRIPT))
    confidence = np.array([.38, .52, .61, .72, .83, .91, .97, 1.])
    correct = np.array([0., 0., 1., 0., 1., 1., 1., 1.])
    weights = np.array([1., 2., 3., 1., 2., 3., 5., 2.])
    # Positive artificial replicate weights: rendering smoke test only.
    boot = np.vstack([weights, weights * 1.1, weights * .9, weights * 1.2])
    calibration, risk, baseline = plot_data_with_uncertainty(
        confidence, correct, weights, boot
    )
    calibration_stem = tmp_path / "jev_calibration"
    risk_stem = tmp_path / "jev_risk_coverage"
    namespace["calibration_figure"](
        calibration, len(confidence), calibration_stem, len(boot)
    )
    namespace["risk_coverage_figure"](
        risk, baseline, len(confidence), risk_stem, len(boot)
    )
    for stem in (calibration_stem, risk_stem):
        for extension in ("pdf", "svg", "png", "txt"):
            exported = stem.with_suffix("." + extension)
            assert exported.exists()
            assert exported.stat().st_size > 30
