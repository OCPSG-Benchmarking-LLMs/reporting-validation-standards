"""Offline tests for JEV reliability and selective-risk publication figures."""
import numpy as np
import pytest

from reporting_validation_standards.jev_figure_analysis import (
    plot_data_with_uncertainty,
    selective_risk,
)


def test_tied_confidence_is_fractionally_averaged_not_input_order():
    # The two highest-confidence cases are tied but have opposite outcomes.
    confidence = np.array([.9, .9, .5])
    correct = np.array([1., 0., 0.])
    weights = np.array([1., 1., 2.])
    coverage = np.array([.25, .5, .75, 1.])
    actual = selective_risk(confidence, correct, weights, coverage)
    np.testing.assert_allclose(actual, [.5, .5, 2 / 3, .75])
    np.testing.assert_allclose(
        actual,
        selective_risk(confidence[::-1], correct[::-1], weights[::-1], coverage),
    )


def test_weighted_selection_differs_from_unweighted():
    confidence = np.array([.95, .65])
    correct = np.array([1., 0.])
    weights = np.array([3., 1.])
    risk = selective_risk(confidence, correct, weights, np.array([.75, 1.]))
    np.testing.assert_allclose(risk, [0, .25])


def test_bootstrap_zero_weight_cases_are_safe():
    risk = selective_risk(
        np.array([.95, .7, .5]),
        np.array([1., 0., 0.]),
        np.array([2., 0., 1.]),
        np.array([.5, 1.]),
    )
    np.testing.assert_allclose(risk, [0, 1 / 3])


@pytest.mark.parametrize("coverage", [
    np.array([0.]), np.array([1.01]), np.array([.5, .5]),
])
def test_invalid_coverage_rejected_for_plot_data(coverage):
    with pytest.raises(ValueError):
        plot_data_with_uncertainty(
            np.array([.6, .9]), np.array([0, 1]), np.array([1, 1]),
            np.ones((4, 2)), coverage_grid=coverage,
        )


def test_calibration_bootstrap_and_plot_data_fully_cover_cases():
    confidence = np.array([.6, .7, .9, 1.])
    correct = np.array([0., 1., 1., 1.])
    weights = np.array([2., 1., 3., 1.])
    boot = np.array([
        [2., 1., 3., 1.],
        [1., 2., 2., 2.],
        [3., 1., 2., 1.],
        [2., 3., 1., 1.],
    ])
    calibration, risk, baseline = plot_data_with_uncertainty(
        confidence, correct, weights, boot,
        coverage_grid=np.array([.5, .75, 1.]),
    )
    assert calibration["n"].sum() == len(confidence)
    assert calibration["estimated_population_mass"].sum() == pytest.approx(1)
    assert calibration["nonempty_bootstrap_replicates"].min() == len(boot)
    assert (calibration["ci_low"] >= 0).all()
    assert (calibration["ci_high"] <= 1).all()
    assert risk["coverage"].to_list() == [.5, .75, 1.]
    assert risk.iloc[-1]["selective_error"] == pytest.approx(2 / 7)
    assert baseline["random_review_expected_error"] == pytest.approx(2 / 7)
    assert risk.iloc[-1]["selective_error"] == pytest.approx(
        risk.iloc[-1]["random_review_expected_error"]
    )


def test_sparse_bin_reports_nonempty_bootstrap_count():
    confidence = np.array([.31, .90, .95])
    correct = np.array([0., 1., 1.])
    weights = np.ones(3)
    boot = np.array([
        [1., 1., 1.],
        [0., 1., 2.],
        [1., 0., 2.],
        [1., 2., 0.],
    ])
    calibration, _risk, _baseline = plot_data_with_uncertainty(
        confidence, correct, weights, boot, coverage_grid=np.array([1.])
    )
    low = calibration.loc[calibration["lower"] == .3].iloc[0]
    assert low["n"] == 1
    assert low["nonempty_bootstrap_replicates"] == 3


def test_invalid_or_zero_total_bootstrap_rejected():
    with pytest.raises(ValueError):
        plot_data_with_uncertainty(
            np.array([.8]), np.array([1.]), np.array([1.]),
            np.zeros((2, 1)), coverage_grid=np.array([1.]),
        )
