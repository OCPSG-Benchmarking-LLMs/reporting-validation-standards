import math

import numpy as np

from reporting_validation_standards.error_propagation import (
    matrix_diagnostics,
    prevalence_error_metrics,
    project_simplex,
    row_normalize,
    solve_simplex_least_squares,
)


def test_project_simplex_returns_probability_vector():
    projected = project_simplex(np.array([0.6, 0.6, -0.2]))
    assert np.all(projected >= 0)
    assert math.isclose(float(projected.sum()), 1.0, abs_tol=1e-12)
    assert np.allclose(projected, np.array([0.5, 0.5, 0.0]))


def test_simplex_least_squares_recovers_known_prevalence():
    mechanism = np.array(
        [
            [0.90, 0.10, 0.00],
            [0.05, 0.90, 0.05],
        ]
    )
    truth = np.array([0.7, 0.3])
    observed = mechanism.T @ truth

    fit = solve_simplex_least_squares(
        mechanism.T,
        observed,
        tolerance=1e-10,
    )

    assert fit.converged
    assert np.all(fit.solution >= 0)
    assert math.isclose(float(fit.solution.sum()), 1.0, abs_tol=1e-10)
    assert np.allclose(fit.solution, truth, atol=1e-7)


def test_simplex_least_squares_enforces_nonnegative_geometry():
    matrix = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )
    target = np.array([1.2, -0.2, 0.5])
    fit = solve_simplex_least_squares(matrix, target)

    assert fit.converged
    assert np.all(fit.solution >= 0)
    assert math.isclose(float(fit.solution.sum()), 1.0, abs_tol=1e-10)


def test_row_normalize_and_prevalence_error_metrics():
    normalized = row_normalize(
        np.array(
            [
                [8.0, 2.0],
                [1.0, 9.0],
            ]
        )
    )
    assert np.allclose(normalized.sum(axis=1), 1.0)

    truth = np.array([0.6, 0.4, 0.0])
    estimate = np.array([0.5, 0.4, 0.1])
    metrics = prevalence_error_metrics(
        estimate,
        truth,
        target_indices=np.array([0, 1]),
    )
    assert math.isclose(metrics["total_variation_distance"], 0.1)
    assert math.isclose(metrics["max_absolute_error"], 0.1)
    assert math.isclose(metrics["mae_target_labels"], 0.05)


def test_matrix_diagnostics_reports_rank_and_shape():
    diagnostics = matrix_diagnostics(np.eye(3))
    assert diagnostics["shape"] == [3, 3]
    assert diagnostics["rank"] == 3
    assert math.isclose(diagnostics["condition_number"], 1.0)
