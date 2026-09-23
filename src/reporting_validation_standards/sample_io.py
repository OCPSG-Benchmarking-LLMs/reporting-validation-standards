from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .sampling import download_country_split, hash_sample_ids


def load_frozen_sample(
    *,
    stem: str = "uk_test_1000",
    manifest_dir: str | Path = "data/manifests",
    generated_dir: str | Path = "data/generated",
    token: str | None = None,
    cache_dir: str | None = None,
    write_generated: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Load and verify the frozen sample, rehydrating it from Hugging Face if needed."""
    manifest_dir = Path(manifest_dir)
    generated_dir = Path(generated_dir)

    metadata_path = manifest_dir / f"{stem}_metadata.json"
    ids_path = manifest_dir / f"{stem}_ids.csv"
    generated_path = generated_dir / f"{stem}.parquet"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    id_column = metadata["id_column"]

    ids = pd.read_csv(ids_path, dtype={id_column: str})
    expected_ids = ids[id_column].astype(str).tolist()

    if generated_path.exists():
        sample = pd.read_parquet(generated_path)
    else:
        parquet_path = download_country_split(
            repo_id=metadata["dataset_repo"],
            country_code=metadata["country_code"],
            split=metadata["split"],
            revision=metadata["dataset_revision"],
            token=token,
            cache_dir=cache_dir,
        )
        population = pd.read_parquet(parquet_path)
        key = population[id_column].astype(str)
        sample = population.loc[key.isin(set(expected_ids))].copy()

        if len(sample) != len(expected_ids):
            raise RuntimeError(
                f"Expected {len(expected_ids)} frozen IDs but recovered {len(sample)} rows."
            )

        order = {value: i for i, value in enumerate(expected_ids)}
        sample["_frozen_order"] = sample[id_column].astype(str).map(order)
        sample = (
            sample.sort_values("_frozen_order")
            .drop(columns="_frozen_order")
            .reset_index(drop=True)
        )

        if write_generated:
            generated_dir.mkdir(parents=True, exist_ok=True)
            sample.to_parquet(generated_path, index=False)

    actual_ids = sample[id_column].astype(str).tolist()
    if set(actual_ids) != set(expected_ids) or len(actual_ids) != len(expected_ids):
        raise RuntimeError("Generated sample membership does not match the frozen ID manifest.")

    actual_hash = hash_sample_ids(actual_ids)
    if actual_hash != metadata["sample_id_sha256"]:
        raise RuntimeError(
            "Frozen sample hash mismatch: "
            f"expected {metadata['sample_id_sha256']}, observed {actual_hash}."
        )

    return sample, metadata


def select_input_text(row: pd.Series, input_mode: str = "text_en") -> str:
    """Apply the CapBench text_en fallback rule or select original-language text."""
    if input_mode == "text":
        value = row.get("text")
    elif input_mode == "text_en":
        value = row.get("text_en")
        if value is None or pd.isna(value) or str(value).strip() == "":
            value = row.get("text")
    else:
        raise ValueError(f"Unsupported input_mode: {input_mode}")

    if value is None or pd.isna(value):
        raise ValueError("Selected input text is missing.")
    text = str(value).strip()
    if not text:
        raise ValueError("Selected input text is empty.")
    return text
