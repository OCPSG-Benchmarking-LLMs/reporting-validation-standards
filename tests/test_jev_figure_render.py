"""Exercise the actual vector/raster JEV figure writers without API access."""
from pathlib import Path
from runpy import run_path

import matplotlib.image as mpimg
import numpy as np

from reporting_validation_standards import plot_style as ps
from reporting_validation_standards.jev_figure_analysis import plot_data_with_uncertainty

SCRIPT = Path(__file__).resolve().parents[1] / "code" / "16_generate_jev_figures.py"


def test_figures_export_pdf_svg_png_and_caption(tmp_path):
    namespace = run_path(str(SCRIPT))

    original_save = namespace["save_with_caption"]
    inspected = []

    def inspect_before_save(fig, stem, caption):
        # Every manuscript plot is title-free, visually restrained and frameless,
        # and is drawn at the width at which it is printed.
        assert ps.paper_problems(fig) == []
        assert fig.get_figwidth() == ps.PRINT_WIDTH_IN
        assert all(not axes.get_title() for axes in fig.axes)
        assert all(not spine.get_visible() for axes in fig.axes
                   for spine in axes.spines.values())
        assert len(fig.legends) == 1
        assert not fig.legends[0].get_frame_on()
        assert not fig.texts  # Full notes belong in an external caption file.
        assert all(
            not line.get_visible()
            for axes in fig.axes
            for line in axes.get_xgridlines() + axes.get_ygridlines()
        ), "Manuscript figures must have no visible gridlines"
        if stem.name == "jev_calibration":
            from matplotlib.colors import to_rgba

            assert len(fig.axes) == 2
            bars = fig.axes[1].patches
            assert bars
            assert all(np.allclose(bar.get_facecolor(), to_rgba(ps.LIGHT_GREY))
                       for bar in bars), "Calibration bars use light grey, not a second hue"
        if stem.name == "jev_risk_coverage":
            labels = [text.get_text() for text in fig.legends[0].get_texts()]
            assert "Random selection (95% CI)" in labels, "Every band is named in the legend"
        inspected.append(stem)
        return original_save(fig, stem, caption)

    # run_path() returns a namespace copy; function globals hold the live namespace.
    namespace["calibration_figure"].__globals__["save_with_caption"] = inspect_before_save
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

    assert len(inspected) == 2
    for stem in (calibration_stem, risk_stem):
        png = mpimg.imread(stem.with_suffix(".png"))
        assert png.ndim == 3 and png.shape[2] == 4
        assert np.any(png[..., 3] < 1), "PNG must preserve transparency"
        caption = stem.with_suffix(".txt").read_text(encoding="utf-8")
        title, note = caption.split("\n\n", 1)
        assert title.startswith("JEV ") and "\n" not in title
        assert note.startswith("Note.")
        if stem.name == "jev_calibration":
            assert "Unweighted observations by confidence bin:" in caption
            assert "0.3–0.4: 1" in caption
        else:
            assert "zero true error probability" in caption
            assert "not on all 1,000 attempts" in caption
