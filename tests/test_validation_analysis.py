import math

import numpy as np
import pandas as pd

from reporting_validation_standards.stability_analysis import INVALID_LABEL
from reporting_validation_standards.validation_analysis import (
    build_confusion_matrix,
    confusion_metrics,
    rao_wu_rescaled_bootstrap_weights,
)


def test_confusion_metrics_penalize_invalid_outputs_operationally():
    # Two target classes A/B, one extra official prediction C, one invalid column.
    matrix = np.array(
        [
            [8.0, 1.0, 0.0, 1.0],
            [1.0, 7.0, 1.0, 1.0],
        ]
    )
    bundle, classes = confusion_metrics(
        matrix,
        n_reference_labels=2,
        invalid_prediction_index=3,
    )

    assert math.isclose(bundle.valid_rate, 18 / 20)
    assert math.isclose(bundle.operational_accuracy, 15 / 20)
    assert math.isclose(bundle.conditional_accuracy, 15 / 18)
    assert (
        bundle.conditional_macro_recall
        > bundle.operational_macro_recall
    )
    assert len(classes) == 2


def test_build_confusion_matrix_accepts_absent_reference_prediction_labels():
    records = pd.DataFrame(
        {
            "reference_label": ["A", "A", "B"],
            "analysis_label": ["A", "C", INVALID_LABEL],
        }
    )
    matrix = build_confusion_matrix(
        records,
        reference_labels=["A", "B"],
        prediction_labels=["A", "B", "C", INVALID_LABEL],
        weights=np.array([2.0, 2.0, 3.0]),
    )

    assert matrix.shape == (2, 4)
    assert matrix[0, 0] == 2.0
    assert matrix[0, 2] == 2.0
    assert matrix[1, 3] == 3.0
    assert matrix.sum() == 7.0


def test_rao_wu_bootstrap_preserves_population_totals_and_census_strata():
    design = pd.DataFrame(
        {
            "id": ["a1", "a2", "b1", "b2"],
            "predicted_label": ["A", "A", "B", "B"],
            "stratum_population_n": [4, 4, 2, 2],
            "stratum_sample_n": [2, 2, 2, 2],
            "inclusion_probability": [0.5, 0.5, 1.0, 1.0],
            "sampling_weight": [2.0, 2.0, 1.0, 1.0],
        }
    )

    weights, ids = rao_wu_rescaled_bootstrap_weights(
        design,
        bootstrap_reps=100,
        seed=123,
    )

    assert ids == ["a1", "a2", "b1", "b2"]
    assert weights.shape == (100, 4)
    assert np.allclose(weights.sum(axis=1), 6.0)
    # Census stratum B contributes no sampling variance.
    assert np.allclose(weights[:, 2:], 1.0)
    # Non-census stratum A should vary across replicate draws.
    assert np.unique(weights[:, 0]).size > 1
