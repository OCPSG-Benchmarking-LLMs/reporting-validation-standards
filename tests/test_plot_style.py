"""The shared figure style saves files at their exact print width."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from reporting_validation_standards import plot_style as ps

CAPTION = "A Title\n\nNote. A note."


def simple_figure(width: float = 5.2):
    ps.use_style()
    fig, ax = plt.subplots(figsize=(width, 3.0))
    ax.plot([0, 1], [0, 1], label="Series")
    ax.set_xlabel("Share of outputs")
    ax.set_ylabel("Error rate")
    ps.clean_axes(ax)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    ps.bottom_legend(fig, ax.get_lines(), y=0.08)
    return fig


def test_print_width_is_a_share_of_the_text_block():
    assert ps.print_width(0.8) == pytest.approx(5.2)
    with pytest.raises(ValueError):
        ps.print_width(1.2)


@pytest.mark.parametrize("target", [4.0, 5.2, 6.175])
def test_saved_pdf_is_exactly_the_target_width(tmp_path, target):
    fig = simple_figure()
    paths = ps.save(fig, tmp_path / "figure", CAPTION, target_width=target)
    pdf = next(path for path in paths if path.suffix == ".pdf")
    width, _ = ps.pdf_size(pdf.read_bytes())
    assert width == pytest.approx(target, abs=0.01)
    plt.close(fig)


def test_save_refuses_text_on_the_canvas(tmp_path):
    fig = simple_figure()
    fig.text(0.5, 0.5, "a note on the canvas")
    with pytest.raises(ValueError, match="canvas"):
        ps.save(fig, tmp_path / "figure", CAPTION)
    plt.close(fig)
