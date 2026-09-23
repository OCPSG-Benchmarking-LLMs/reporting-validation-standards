import pandas as pd

from reporting_validation_standards.sampling import draw_stratified_sample


def test_stratified_sample_has_target_size_and_weights():
    df = pd.DataFrame(
        {
            "id": range(100),
            "predicted_label": ["A"] * 70 + ["B"] * 20 + ["C"] * 10,
            "text": ["x"] * 100,
        }
    )

    sample, strata = draw_stratified_sample(
        df,
        label_column="predicted_label",
        id_column="id",
        target_n=30,
        minimum_per_label=5,
        seed=123,
    )

    assert len(sample) == 30
    assert sample["id"].is_unique
    assert int(strata["stratum_sample_n"].sum()) == 30
    assert (strata["stratum_sample_n"] <= strata["stratum_population_n"]).all()

    expected = strata["stratum_population_n"] / strata["stratum_sample_n"]
    assert (strata["sampling_weight"].round(12) == expected.round(12)).all()


def test_sampling_is_reproducible():
    df = pd.DataFrame(
        {
            "id": range(60),
            "predicted_label": ["A"] * 30 + ["B"] * 20 + ["C"] * 10,
        }
    )

    s1, _ = draw_stratified_sample(
        df,
        label_column="predicted_label",
        id_column="id",
        target_n=24,
        minimum_per_label=4,
        seed=99,
    )
    s2, _ = draw_stratified_sample(
        df,
        label_column="predicted_label",
        id_column="id",
        target_n=24,
        minimum_per_label=4,
        seed=99,
    )

    assert sorted(s1["id"].tolist()) == sorted(s2["id"].tolist())
