from reporting_validation_standards.openai_adapter import prediction_json_schema


def test_prediction_schema_is_closed_enum():
    labels = ["A", "B", "C"]
    schema = prediction_json_schema(labels)

    assert schema["required"] == ["predicted_label"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["predicted_label"]["enum"] == labels
