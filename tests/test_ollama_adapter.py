import urllib.error

from reporting_validation_standards.ollama_adapter import (
    OllamaClassifier,
    _ns_to_seconds,
    prediction_json_schema,
)


def test_ollama_prediction_schema_is_closed_enum():
    labels = ["A", "B", "C"]
    schema = prediction_json_schema(labels)

    assert schema["required"] == ["predicted_label"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["predicted_label"]["enum"] == labels


def test_ollama_nanoseconds_are_converted_to_seconds():
    assert _ns_to_seconds(1_500_000_000) == 1.5
    assert _ns_to_seconds(None) is None


def test_resolve_local_model_uses_exact_tag(monkeypatch):
    classifier = OllamaClassifier()

    models = [
        {"name": "qwen3.6:35b", "model": "qwen3.6:35b", "digest": "abc"},
        {"name": "gemma4:31b", "model": "gemma4:31b", "digest": "def"},
    ]
    monkeypatch.setattr(classifier, "list_models", lambda: models)

    result = classifier.resolve_local_model("qwen3.6:35b")
    assert result["digest"] == "abc"
