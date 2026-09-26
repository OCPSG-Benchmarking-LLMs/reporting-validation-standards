from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SimplexLeastSquaresResult:
    solution: np.ndarray
    objective: float
    iterations: int
    converged: bool
    projected_gradient_norm: float


def project_simplex(vector: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {x >= 0, sum(x) = 1}."""
    values = np.asarray(vector, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("vector must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("vector contains non-finite values.")

    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, values.size + 1)
    positive = ordered - cumulative / indices > 0
    if not positive.any():
        raise RuntimeError("Simplex projection failed to find an active coordinate.")

    rho = int(np.nonzero(positive)[0][-1])
    theta = cumulative[rho] / float(rho + 1)
    projected = np.maximum(values - theta, 0.0)

    # Remove machine-precision drift without changing the active set materially.
    total = float(projected.sum())
    if total <= 0:
        raise RuntimeError("Simplex projection produced zero total mass.")
    projected /= total
    return projected


def _objective(matrix: np.ndarray, target: np.ndarray, x: np.ndarray) -> float:
    residual = matrix @ x - target
    return 0.5 * float(residual @ residual)


def solve_simplex_least_squares(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    initial: np.ndarray | None = None,
    tolerance: float = 1e-11,
    max_iterations: int = 20_000,
) -> SimplexLeastSquaresResult:
    """Solve min ||A x - b||^2 subject to x >= 0 and sum(x)=1.

    Uses FISTA with the exact Euclidean projection onto the probability simplex.
    The problem is convex; the gradient Lipschitz constant is ||A||_2^2.
    """
    A = np.asarray(matrix, dtype=float)
    b = np.asarray(target, dtype=float)

    if A.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")
    if b.ndim != 1 or b.shape[0] != A.shape[0]:
        raise ValueError("target length must equal the number of matrix rows.")
    if A.shape[1] == 0:
        raise ValueError("matrix must contain at least one column.")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("matrix/target contains non-finite values.")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    spectral_norm = float(np.linalg.norm(A, ord=2))
    lipschitz = spectral_norm * spectral_norm
    if lipschitz <= 0:
        raise ValueError("matrix has zero spectral norm.")
    step = 1.0 / lipschitz

    if initial is None:
        x = np.full(A.shape[1], 1.0 / A.shape[1], dtype=float)
    else:
        x = project_simplex(np.asarray(initial, dtype=float))
        if x.shape[0] != A.shape[1]:
            raise ValueError("initial length must equal the number of matrix columns.")

    y = x.copy()
    acceleration = 1.0
    converged = False
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        gradient = A.T @ (A @ y - b)
        next_x = project_simplex(y - step * gradient)

        # Projected-gradient mapping at the new iterate is a direct first-order
        # convergence diagnostic for the constrained problem.
        gradient_next = A.T @ (A @ next_x - b)
        projected_next = project_simplex(next_x - step * gradient_next)
        pg_norm = float(np.linalg.norm((next_x - projected_next) / step))

        iterations = iteration
        if pg_norm <= tolerance:
            x = next_x
            converged = True
            break

        next_acceleration = 0.5 * (
            1.0 + np.sqrt(1.0 + 4.0 * acceleration * acceleration)
        )
        y = next_x + (
            (acceleration - 1.0) / next_acceleration
        ) * (next_x - x)
        x = next_x
        acceleration = next_acceleration

    if not converged:
        gradient = A.T @ (A @ x - b)
        projected_next = project_simplex(x - step * gradient)
        pg_norm = float(np.linalg.norm((x - projected_next) / step))

    return SimplexLeastSquaresResult(
        solution=x,
        objective=_objective(A, b, x),
        iterations=iterations,
        converged=converged,
        projected_gradient_norm=pg_norm,
    )


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")
    row_totals = values.sum(axis=1)
    if np.any(row_totals <= 0):
        raise ValueError("Every row must have positive mass.")
    return values / row_totals[:, None]


def prevalence_error_metrics(
    estimate: np.ndarray,
    truth: np.ndarray,
    *,
    target_indices: np.ndarray | None = None,
) -> dict[str, float]:
    estimate_array = np.asarray(estimate, dtype=float)
    truth_array = np.asarray(truth, dtype=float)
    if estimate_array.shape != truth_array.shape:
        raise ValueError("estimate and truth must have equal shape.")
    if estimate_array.ndim != 1:
        raise ValueError("prevalence vectors must be one-dimensional.")

    difference = estimate_array - truth_array
    absolute = np.abs(difference)
    metrics = {
        "total_variation_distance": 0.5 * float(absolute.sum()),
        "max_absolute_error": float(absolute.max()),
        "rmse_all_labels": float(np.sqrt(np.mean(difference * difference))),
        "mae_all_labels": float(np.mean(absolute)),
    }

    if target_indices is not None:
        indices = np.asarray(target_indices, dtype=int)
        metrics["mae_target_labels"] = float(np.mean(absolute[indices]))
        metrics["rmse_target_labels"] = float(
            np.sqrt(np.mean(difference[indices] ** 2))
        )
    return metrics


def matrix_diagnostics(matrix: np.ndarray) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=float)
    singular_values = np.linalg.svd(values, compute_uv=False)
    rank = int(np.linalg.matrix_rank(values))
    smallest = float(singular_values[-1]) if singular_values.size else np.nan
    largest = float(singular_values[0]) if singular_values.size else np.nan
    condition = (
        float(largest / smallest)
        if singular_values.size and smallest > 0
        else float("inf")
    )
    return {
        "shape": list(values.shape),
        "rank": rank,
        "singular_values": [float(value) for value in singular_values],
        "condition_number": condition,
    }
