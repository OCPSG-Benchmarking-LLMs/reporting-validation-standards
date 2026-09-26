from reporting_validation_standards.mistral_adapter import (
    MISTRAL_BASE_URL,
    MISTRAL_SERVICE_TIER,
    prediction_json_schema,
)


def test_mistral_prediction_schema_is_closed_enum():
    labels = ["A", "B", "C"]
    schema = prediction_json_schema(labels)

    assert schema["required"] == ["predicted_label"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["predicted_label"]["enum"] == labels


def test_mistral_adapter_freezes_documented_endpoint_and_standard_tier():
    assert MISTRAL_BASE_URL == "https://api.mistral.ai/v1"
    assert MISTRAL_SERVICE_TIER == "standard_only"
