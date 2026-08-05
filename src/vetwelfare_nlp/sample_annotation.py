from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
from datasets import DatasetDict, load_from_disk


def _has_list_annotations(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) > 0


def _primary_icd(value: Any) -> str:
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    return "NO_ICD_LABEL"


def _length_bin(words: int) -> str:
    if words <= 40:
        return "short"
    if words <= 90:
        return "medium"
    return "long"


@click.command()
@click.option("--dataset-path", type=click.Path(path_type=Path), default=Path("data/raw/peteval"), show_default=True)
@click.option("--split", default="test", show_default=True)
@click.option("--output", type=click.Path(path_type=Path), default=Path("data/annotation/pilot_500.csv"), show_default=True)
@click.option("--n", default=500, show_default=True, type=int)
@click.option("--seed", default=42, show_default=True, type=int)
def main(dataset_path: Path, split: str, output: Path, n: int, seed: int) -> None:
    """Create a reproducible, stratified Five-Domains annotation pilot sample."""
    loaded = load_from_disk(str(dataset_path))
    if not isinstance(loaded, DatasetDict):
        loaded = DatasetDict({"data": loaded})
    if split not in loaded:
        raise click.ClickException(f"Split '{split}' not found. Available: {list(loaded.keys())}")

    df = loaded[split].to_pandas().copy()
    if "sentence" not in df.columns:
        raise click.ClickException("Expected text column 'sentence' was not found.")

    df["word_count"] = df["sentence"].fillna("").astype(str).str.split().str.len()
    df["length_bin"] = df["word_count"].map(_length_bin)
    df["primary_icd"] = df.get("icd_label", pd.Series([[]] * len(df))).map(_primary_icd)
    df["has_icd"] = df.get("icd_label", pd.Series([[]] * len(df))).map(_has_list_annotations)
    df["has_disease_entity"] = df.get("disease", pd.Series([[]] * len(df))).map(_has_list_annotations)

    # Collapse rare ICD strata to prevent one-off categories from dominating sampling.
    icd_counts = df["primary_icd"].value_counts()
    df["icd_stratum"] = df["primary_icd"].where(df["primary_icd"].map(icd_counts) >= 25, "OTHER_ICD")
    df["stratum"] = (
        df["icd_stratum"].astype(str)
        + "|"
        + df["length_bin"].astype(str)
        + "|disease="
        + df["has_disease_entity"].astype(str)
    )

    rng = np.random.default_rng(seed)
    target = min(n, len(df))

    # Proportional allocation with at least one record per observed stratum.
    counts = df["stratum"].value_counts()
    allocation = (counts / counts.sum() * target).round().astype(int).clip(lower=1)

    while allocation.sum() > target:
        idx = allocation.idxmax()
        if allocation.loc[idx] > 1:
            allocation.loc[idx] -= 1
        else:
            break
    while allocation.sum() < target:
        capacity = counts - allocation
        idx = capacity.idxmax()
        if capacity.loc[idx] <= 0:
            break
        allocation.loc[idx] += 1

    sampled_parts = []
    for stratum, take in allocation.items():
        group = df[df["stratum"] == stratum]
        take = min(int(take), len(group))
        sampled_parts.append(group.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))

    sample = pd.concat(sampled_parts, ignore_index=True)
    if len(sample) < target:
        remaining = df.loc[~df.index.isin(sample.index)]
        if not remaining.empty:
            extra = remaining.sample(n=min(target - len(sample), len(remaining)), random_state=seed)
            sample = pd.concat([sample, extra], ignore_index=True)

    sample = sample.sample(frac=1, random_state=seed).head(target).reset_index(drop=True)
    sample.insert(0, "annotation_id", [f"VW{idx:05d}" for idx in range(1, len(sample) + 1)])

    annotation_columns = {
        "nutrition_status": "unknown",
        "nutrition_evidence": "",
        "environment_status": "unknown",
        "environment_evidence": "",
        "health_status": "unknown",
        "health_evidence": "",
        "behaviour_status": "unknown",
        "behaviour_evidence": "",
        "mental_state": "unknown",
        "mental_state_evidence": "",
        "overall_valence": "unknown",
        "severity": "unknown",
        "annotator_confidence": "",
        "annotator_id": "",
        "review_status": "unreviewed",
        "notes": "",
    }
    for column, default in annotation_columns.items():
        sample[column] = default

    output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output, index=False)

    manifest = {
        "dataset_path": str(dataset_path),
        "split": split,
        "requested_n": n,
        "sampled_n": len(sample),
        "seed": seed,
        "strata": sample["stratum"].value_counts().to_dict(),
        "length_bins": sample["length_bin"].value_counts().to_dict(),
        "has_disease_entity": sample["has_disease_entity"].value_counts().to_dict(),
        "has_icd": sample["has_icd"].value_counts().to_dict(),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    click.echo(f"Wrote annotation sample: {output}")
    click.echo(f"Wrote sampling manifest: {manifest_path}")
    click.echo(f"Rows: {len(sample)}")


if __name__ == "__main__":
    main()
