from reporting_validation_standards.pricing import estimate_openai_cost


def test_gpt54_cost_with_cached_tokens():
    estimate = estimate_openai_cost(
        model_id="gpt-5.4-2026-03-05",
        input_tokens=2000,
        cached_input_tokens=1000,
        cache_write_tokens=0,
        output_tokens=100,
    )

    expected = (1000 * 2.50 + 1000 * 0.25 + 100 * 15.00) / 1_000_000
    assert round(estimate.cost_usd, 12) == round(expected, 12)


def test_gpt54_long_context_multiplier():
    estimate = estimate_openai_cost(
        model_id="gpt-5.4-2026-03-05",
        input_tokens=300000,
        cached_input_tokens=0,
        cache_write_tokens=0,
        output_tokens=1000,
    )

    expected = (300000 * 2.50 * 2.0 + 1000 * 15.00 * 1.5) / 1_000_000
    assert round(estimate.cost_usd, 12) == round(expected, 12)


def test_gpt6_luna_cache_write_cost():
    estimate = estimate_openai_cost(
        model_id="gpt-6-luna",
        input_tokens=3000,
        cached_input_tokens=1000,
        cache_write_tokens=1000,
        output_tokens=100,
    )

    expected = (
        1000 * 0.10
        + 1000 * 0.01
        + 1000 * 0.125
        + 100 * 0.50
    ) / 1_000_000
    assert round(estimate.cost_usd, 12) == round(expected, 12)


def test_fireworks_deepseek_cost_with_cached_tokens():
    from reporting_validation_standards.pricing import estimate_fireworks_cost

    estimate = estimate_fireworks_cost(
        model_id="accounts/fireworks/models/deepseek-v4p1-flash",
        input_tokens=2000,
        cached_input_tokens=1000,
        cache_write_tokens=0,
        output_tokens=100,
    )

    expected = (1000 * 0.22 + 1000 * 0.007 + 100 * 0.66) / 1_000_000
    assert round(estimate.cost_usd, 12) == round(expected, 12)


def test_mistral_large3_cost_with_cached_tokens():
    from reporting_validation_standards.pricing import estimate_mistral_cost

    estimate = estimate_mistral_cost(
        model_id="mistral-large-2512",
        input_tokens=2000,
        cached_input_tokens=1000,
        cache_write_tokens=0,
        output_tokens=100,
    )

    expected = (1000 * 0.50 + 1000 * 0.05 + 100 * 1.50) / 1_000_000
    assert round(estimate.cost_usd, 12) == round(expected, 12)
