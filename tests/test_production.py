from reporting_validation_standards.production import (
    SCHEDULE_ALGORITHM_VERSION,
    decision_key,
    make_stability_schedule,
    schedule_sha256,
    validate_existing_records,
)


def test_stability_schedule_is_deterministic_and_complete():
    kwargs = {
        "observation_ids": ["a", "b", "c"],
        "prompt_ids": ["p0", "p1"],
        "repetitions": 3,
        "seed": 20260923,
    }

    first = make_stability_schedule(**kwargs)
    second = make_stability_schedule(**kwargs)

    assert SCHEDULE_ALGORITHM_VERSION == "randomized_cells_pcg64_v1"
    assert first == second
    assert schedule_sha256(first) == schedule_sha256(second)
    assert len(first) == 18
    assert len({cell.key for cell in first}) == 18


def test_schedule_changes_with_seed():
    common = {
        "observation_ids": ["a", "b", "c", "d"],
        "prompt_ids": ["p0", "p1"],
        "repetitions": 2,
    }

    first = make_stability_schedule(seed=1, **common)
    second = make_stability_schedule(seed=2, **common)

    assert [cell.key for cell in first] != [cell.key for cell in second]


def test_validate_existing_records_rejects_duplicates():
    expected = {decision_key(observation_id="a", prompt_id="p0", repetition=1)}
    records = [
        {
            "model_key": "m",
            "observation_id": "a",
            "prompt_id": "p0",
            "repetition": 1,
        },
        {
            "model_key": "m",
            "observation_id": "a",
            "prompt_id": "p0",
            "repetition": 1,
        },
    ]

    try:
        validate_existing_records(
            records,
            expected_model_key="m",
            expected_schedule_keys=expected,
        )
    except RuntimeError as exc:
        assert "Duplicate decision cell" in str(exc)
    else:
        raise AssertionError("Expected duplicate decision cell to be rejected.")
