from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
from datasets import DatasetDict, load_from_disk


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _flatten_labels(series: pd.Series) -> Counter:
    counts: Counter = Counter()
    for value in series.dropna():
        if isinstance(value, (list, tuple, set)):
            counts.update(str(x) for x in value)
        elif isinstance(value, dict):
            counts.update(str(k) for k, v in value.items() if v)
        else:
            text = str(value).strip()
            if text:
                counts[text] += 1
    return counts


@click.command()
@click.option("--dataset-path", type=click.Path(path_type=Path), default=Path("data/raw/peteval"), show_default=True)
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("outputs/dataset_audit"), show_default=True)
@click.option("--text-column", default="sentence", show_default=True)
@click.option("--top-n", default=30, show_default=True, type=int)
def main(dataset_path: Path, output_dir: Path, text_column: str, top_n: int) -> None:
    """Audit a locally downloaded PetEVAL DatasetDict without redistributing records."""
    if not dataset_path.exists():
        raise click.ClickException(f"Dataset path not found: {dataset_path}")

    loaded = load_from_disk(str(dataset_path))
    if not isinstance(loaded, DatasetDict):
        loaded = DatasetDict({"data": loaded})

    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"dataset_path": str(dataset_path), "splits": {}}

    for split_name, split in loaded.items():
        df = split.to_pandas()
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        split_report: dict[str, Any] = {
            "rows": len(df),
            "columns": list(df.columns),
            "missing_by_column": {c: int(df[c].isna().sum()) for c in df.columns},
            "unique_by_column": {c: int(df[c].astype(str).nunique(dropna=True)) for c in df.columns},
        }

        if text_column in df.columns:
            text = df[text_column].fillna("").astype(str)
            chars = text.str.len()
            words = text.str.split().str.len()
            split_report["text_statistics"] = {
                "empty_records": int((chars == 0).sum()),
                "characters": {
                    "mean": float(chars.mean()),
                    "median": float(chars.median()),
                    "p95": float(chars.quantile(0.95)),
                    "max": int(chars.max()),
                },
                "words": {
                    "mean": float(words.mean()),
                    "median": float(words.median()),
                    "p95": float(words.quantile(0.95)),
                    "max": int(words.max()),
                },
            }
            pd.DataFrame({"characters": chars, "words": words}).describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_csv(
                split_dir / "text_length_summary.csv"
            )

        for label_column in ("icd_label", "annonymisation", "disease"):
            if label_column in df.columns:
                counts = _flatten_labels(df[label_column])
                top = counts.most_common(top_n)
                split_report[f"top_{label_column}"] = top
                pd.DataFrame(top, columns=[label_column, "count"]).to_csv(
                    split_dir / f"top_{label_column}.csv", index=False
                )

        report["splits"][split_name] = split_report

    with (output_dir / "audit_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=_jsonable)

    click.echo(f"Audit complete: {output_dir / 'audit_report.json'}")
    for split_name, split_report in report["splits"].items():
        click.echo(f"{split_name}: {split_report['rows']} rows; columns={split_report['columns']}")


if __name__ == "__main__":
    main()
