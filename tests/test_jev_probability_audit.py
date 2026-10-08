"""Offline JEV probability-sum audit tests; no TypeSafe calls or credentials needed."""
from pathlib import Path
from runpy import run_path

import pytest


AUDIT_SCRIPT = Path(__file__).resolve().parents[1] / "code" / "14b_audit_jev_probability_failures.py"
inspect_response = run_path(str(AUDIT_SCRIPT))["inspect_response"]


def make_response(a=0.6005, b=0.4):
    return {"answers": {"policy": {
        "type": "choice",
        "choice": "A",
        "confidence": .6,
        "probabilities": {"A": a, "B": b},
    }}}


def test_audit_detects_probability_sum_without_normalising():
    result = inspect_response(make_response(), ["A", "B"])
    assert result["probability_sum"] == pytest.approx(1.0005)
    assert result["probability_sum_minus_one"] == pytest.approx(0.0005)
    assert result["raw_probabilities"] == {"A": .6005, "B": .4}


def test_audit_preserves_exact_probability_sum():
    result = inspect_response(make_response(.6, .4), ["A", "B"])
    assert result["probability_sum"] == pytest.approx(1.0)
    assert result["n_labels"] == 2


def test_audit_reports_inventory_mismatch():
    result = inspect_response(make_response(), ["A", "B", "C"])
    assert result["missing_labels"] == ["C"]
    assert "probability_sum" not in result
