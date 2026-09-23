#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from reporting_validation_standards.sampling import (
    SampleMetadata,
    download_country_split,
    draw_stratified_sample,
    hash_sample_ids,
    resolve_dataset_revision,
    write_sample_outputs,
)

DEFAULT_REPO = "OCPSG-Benchmarking-LLMs/ocpsg-silver-standard-parliamentary-speeches"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the stratified 1,000-case Great Britain demonstration sample."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--country-code", default="GB")
    parser.add_argument("--split", default="test")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--target-n", type=int, default=1000)
    parser.add_argument("--minimum-per-label", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--label-column", default="predicted_label")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--stem", default="uk_test_1000")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    revision = args.revision or resolve_dataset_revision(args.repo_id, token=token)

    parquet_path = download_country_split(
        repo_id=args.repo_id,
        country_code=args.country_code,
        split=args.split,
        revision=revision,
        token=token,
        cache_dir=args.cache_dir,
    )
    population = pd.read_parquet(parquet_path)

    sample, strata = draw_stratified_sample(
        population,
        label_column=args.label_column,
        id_column=args.id_column,
        target_n=args.target_n,
        minimum_per_label=args.minimum_per_label,
        seed=args.seed,
    )

    sample_hash = hash_sample_ids(sample[args.id_column])

    metadata = SampleMetadata(
        dataset_repo=args.repo_id,
        dataset_revision=revision,
        country_code=args.country_code.upper(),
        split=args.split,
        label_column=args.label_column,
        id_column=args.id_column,
        target_n=args.target_n,
        realised_n=len(sample),
        minimum_per_label=args.minimum_per_label,
        seed=args.seed,
        n_strata=len(strata),
        sample_id_sha256=sample_hash,
    )

    paths = write_sample_outputs(
        sample=sample,
        strata=strata,
        metadata=metadata,
        id_column=args.id_column,
        label_column=args.label_column,
        manifest_dir=Path("data/manifests"),
        generated_dir=Path("data/generated"),
        stem=args.stem,
        overwrite=args.overwrite,
    )

    print(f"Dataset revision: {revision}")
    print(f"Population N: {len(population):,}")
    print(f"Sample n: {len(sample):,}")
    print(f"Number of strata: {len(strata)}")
    print(f"Sample ID SHA-256: {sample_hash}")
    print("\nStratum allocation:")
    print(
        strata[
            [
                args.label_column,
                "stratum_population_n",
                "stratum_sample_n",
                "inclusion_probability",
                "sampling_weight",
            ]
        ].to_string(index=False)
    )
    print("\nWrote:")
    for key, path in paths.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
