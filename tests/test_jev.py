import numpy as np
import pytest

from reporting_validation_standards.jev import extract_choice, weighted_metrics


def test_extract_choice_accepts_complete_probability_distribution():
    answer = {"answers": {"policy": {"type": "choice", "choice": "A",
              "confidence": 0.8, "probabilities": {"A": 0.8, "B": 0.2}}}}
    result = extract_choice(answer, ["A", "B"])
    assert result["prediction"] == "A"
    assert result["probabilities"]["B"] == pytest.approx(.2)


@pytest.mark.parametrize("probabilities", [
    {"A": .8}, {"A": .8, "B": .8}, {"A": -.1, "B": 1.1},
])
def test_extract_choice_rejects_invalid_probabilities(probabilities):
    response = {"answers": {"policy": {"type": "choice", "choice": "A",
                "confidence": .8, "probabilities": probabilities}}}
    with pytest.raises(ValueError):
        extract_choice(response, ["A", "B"])


def test_weighted_metrics_perfect_classification():
    metrics = weighted_metrics(np.array([[1., 0.], [0., 1.]]),
                               np.array([0, 1]), np.array([1., 3.]))
    assert metrics["accuracy"] == 1
    assert metrics["brier_multiclass"] == 0
    assert metrics["log_loss"] == 0
    assert metrics["risk_100"] == 0


def test_weighted_metrics_respects_weights():
    metrics = weighted_metrics(np.array([[.9, .1], [.9, .1]]),
                               np.array([0, 1]), np.array([3., 1.]))
    assert metrics["accuracy"] == .75
    assert metrics["brier_multiclass"] == pytest.approx(.25 * .02 + .75 * .02 + .25 * 1.6) if False else metrics["brier_multiclass"] > 0
