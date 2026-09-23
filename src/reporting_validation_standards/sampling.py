from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download


ALLOCATION_RULE_VERSION = "label_stratified_minimum_plus_proportional_v1"
ALLOCATION_RULE = (
    "Stratify on the silver-standard reference label. Give each stratum "
    "min(minimum_per_label, N_h) observations; allocate all remaining sample slots "
    "approximately in proportion to N_h among strata with remaining capacity using "
    "iterative floor allocation followed by a largest-remainder step with a stable "
    "lexical tie-break; then draw n_h observations without replacement within each "
    "stratum using NumPy PCG64 with the recorded random seed."
)


@dataclass(frozen=True)
class SampleMetadata:
    dataset_repo: str
    dataset_revision: str
    country_code: str
    split: str
    label_column: str
    id_column: str
    population_n: int
    population_labels_n: int
    target_n: int
    realised_n: int
    minimum_per_label: int
    allocation_rule: str
    allocation_rule_version: str
    random_generator: str
    seed: int
    n_strata: int
    sample_id_sha256: str


def resolve_dataset_revision(
    repo_id: str,
    *,
    token: str | None = None,
) -> str:
    """Resolve the dataset HEAD to an immutable Hugging Face commit SHA."""
    info = HfApi(token=token).dataset_info(repo_id=repo_id)
    if not info.sha:
        raise RuntimeError(f"Could not resolve an immutable revision for {repo_id}.")
    return str(info.sha)


def download_country_split(
    *,
    repo_id: str,
    country_code: str,
    split: str,
    revision: str,
    token: str | None = None,
    cache_dir: str | None = None,
) -> Path:
    """Download one country-level parquet split from the OCPSG dataset repository."""
    filename = f"{country_code.upper()}/{split}.parquet"
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
    )
    return Path(path)


def _largest_remainder_allocation(
    *,
    counts: pd.Series,
    target_n: int,
    minimum_per_stratum: int,
) -> pd.Series:
    """
    Allocate a fixed sample across strata.

    Each stratum first receives min(minimum_per_stratum, N_h). Remaining slots are
    apportioned approximately in proportion to population stratum size, subject to
    remaining stratum capacity. The procedure is deterministic given the stratum counts.
    """
    if target_n <= 0:
        raise ValueError("target_n must be positive.")

    counts = counts.astype(int).sort_index()
    if (counts <= 0).any():
        raise ValueError("All strata must contain at least one observation.")

    population_n = int(counts.sum())
    if target_n > population_n:
        raise ValueError(
            f"target_n={target_n} exceeds population size N={population_n}."
        )

    allocation = counts.clip(upper=minimum_per_stratum).copy()
    allocated = int(allocation.sum())

    if allocated > target_n:
        raise ValueError(
            "The requested minimum allocation is incompatible with target_n: "
            f"minimum allocations require {allocated} observations, but target_n={target_n}."
        )

    remaining = target_n - allocated

    while remaining > 0:
        capacity = counts - allocation
        eligible = capacity[capacity > 0]
        if eligible.empty:
            raise RuntimeError("No remaining stratum capacity, but allocation is incomplete.")

        weights = counts.loc[eligible.index].astype(float)
        quotas = remaining * weights / weights.sum()

        floor_add = np.floor(quotas).astype(int)
        floor_add = pd.Series(
            np.minimum(floor_add.to_numpy(), eligible.to_numpy()),
            index=eligible.index,
            dtype=int,
        )

        floor_total = int(floor_add.sum())
        if floor_total > 0:
            allocation.loc[floor_add.index] += floor_add
            remaining -= floor_total
            if remaining == 0:
                break

        capacity = counts - allocation
        eligible = capacity[capacity > 0]
        if eligible.empty:
            continue

        # Largest-remainder step. Stable lexical tie-break keeps the allocation reproducible.
        weights = counts.loc[eligible.index].astype(float)
        quotas = max(remaining, 1) * weights / weights.sum()
        fractional = quotas - np.floor(quotas)
        ranked = sorted(
            eligible.index,
            key=lambda label: (-float(fractional.loc[label]), str(label)),
        )

        for label in ranked:
            if remaining == 0:
                break
            if allocation.loc[label] < counts.loc[label]:
                allocation.loc[label] += 1
                remaining -= 1

    if int(allocation.sum()) != target_n:
        raise RuntimeError("Allocation does not sum to target_n.")
    if (allocation > counts).any():
        raise RuntimeError("Allocation exceeds at least one stratum population.")

    return allocation.astype(int)


def draw_stratified_sample(
    df: pd.DataFrame,
    *,
    label_column: str,
    id_column: str,
    target_n: int,
    minimum_per_label: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Draw a label-stratified probability sample without replacement.

    Returns
    -------
    sample
        Full sampled rows plus stratum population/sample counts, inclusion
        probabilities, and inverse-probability sampling weights.
    strata
        One row per label with N_h, n_h, pi_h, and weight_h.
    """
    required = {label_column, id_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df[id_column].isna().any():
        raise ValueError(f"{id_column} contains missing values.")
    if df[id_column].duplicated().any():
        raise ValueError(f"{id_column} must be unique within the sampling population.")
    if df[label_column].isna().any():
        raise ValueError(f"{label_column} contains missing values.")

    counts = df[label_column].astype(str).value_counts().sort_index()
    allocation = _largest_remainder_allocation(
        counts=counts,
        target_n=target_n,
        minimum_per_stratum=minimum_per_label,
    )

    rng = np.random.Generator(np.random.PCG64(seed))
    sampled_frames: list[pd.DataFrame] = []

    for label in counts.index:
        stratum = df.loc[df[label_column].astype(str) == label]
        n_h = int(allocation.loc[label])
        chosen_positions = rng.choice(len(stratum), size=n_h, replace=False)
        sampled_frames.append(stratum.iloc[chosen_positions].copy())

    sample = pd.concat(sampled_frames, ignore_index=True)
    # Shuffle the final presentation order without changing membership.
    sample = sample.iloc[rng.permutation(len(sample))].reset_index(drop=True)

    strata = pd.DataFrame(
        {
            label_column: counts.index,
            "stratum_population_n": counts.to_numpy(dtype=int),
            "stratum_sample_n": allocation.loc[counts.index].to_numpy(dtype=int),
        }
    )
    strata["inclusion_probability"] = (
        strata["stratum_sample_n"] / strata["stratum_population_n"]
    )
    strata["sampling_weight"] = 1.0 / strata["inclusion_probability"]

    sample = sample.merge(strata, on=label_column, how="left", validate="many_to_one")

    if len(sample) != target_n:
        raise RuntimeError(f"Expected {target_n} sampled rows, obtained {len(sample)}.")

    return sample, strata


def hash_sample_ids(ids: Iterable[object]) -> str:
    """Return a stable SHA-256 hash of the sorted frozen sample IDs."""
    normalised = "\n".join(sorted(str(x) for x in ids))
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def write_sample_outputs(
    *,
    sample: pd.DataFrame,
    strata: pd.DataFrame,
    metadata: SampleMetadata,
    id_column: str,
    label_column: str,
    manifest_dir: Path,
    generated_dir: Path,
    stem: str,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write commit-friendly manifests plus a gitignored full parquet sample."""
    manifest_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "ids": manifest_dir / f"{stem}_ids.csv",
        "strata": manifest_dir / f"{stem}_strata.csv",
        "metadata": manifest_dir / f"{stem}_metadata.json",
        "sample": generated_dir / f"{stem}.parquet",
    }

    if not overwrite:
        existing = [str(p) for p in paths.values() if p.exists()]
        if existing:
            raise FileExistsError(
                "Frozen sample outputs already exist. Refusing to overwrite:\n"
                + "\n".join(existing)
                + "\nUse --overwrite only before the sample is formally frozen."
            )

    design_cols = [
        id_column,
        label_column,
        "stratum_population_n",
        "stratum_sample_n",
        "inclusion_probability",
        "sampling_weight",
    ]
    sample[design_cols].to_csv(paths["ids"], index=False)
    strata.to_csv(paths["strata"], index=False)
    paths["metadata"].write_text(
        json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sample.to_parquet(paths["sample"], index=False)

    return paths
