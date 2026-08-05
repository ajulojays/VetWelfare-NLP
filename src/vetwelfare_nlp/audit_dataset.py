from __future__ import annotations

import json
from ast import literal_eval
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import click
import numpy as np
import pandas as pd
from datasets import DatasetDict, load_from_disk


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _coerce(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("[", "{")):
            try:
                return literal_eval(text)
            except (ValueError, SyntaxError):
                return value
    return value


def _iter_icd_labels(value: Any) -> Iterable[str]:
    value = _coerce(value)
    if isinstance(value, (list, tuple, set)):
        for item in value:
            label = str(item).strip()
            if label:
                yield label
    elif value not in (None, "", "[]"):
        yield str(value).strip()


def _iter_disease_entities(value: Any) -> Iterable[str]:
    value = _coerce(value)
    if isinstance(value, dict):
        value = [value]
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, dict):
                entity = str(item.get("entity", "")).strip()
                if entity:
                    yield entity.lower()
            elif str(item).strip():
                yield str(item).strip().lower()


def _iter_anonymisation_labels(value: Any) -> Iterable[str]:
    value = _coerce(value)
    if isinstance(value, dict):
        value = [value]
    if isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, dict):
                label = str(item.get("label", item.get("entity", ""))).strip()
                if label:
                    yield label
            elif str(item).strip():
                yield str(item).strip()
    elif value not in (None, "", "[]"):
        yield str(value).strip()


def _count(series: pd.Series, extractor) -> Counter:
    counts: Counter = Counter()
    for value in series.dropna():
        counts.update(extractor(value))
    return counts


@click.command()
@click.option("--dataset-path", type=click.Path(path_type=Path), default=Path("data/raw/peteval"), show_default=True)
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("outputs/dataset_audit"), show_default=True)
@click.option("--text-column", default="sentence", show_default=True)
@click.option("--top-n", default=50, show_default=True, type=int)
def main(dataset_path: Path, output_dir: Path, text_column: str, top_n: int) -> None:
    """Audit PetEVAL and correctly flatten multi-label and span annotations."""
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
        }

        if text_column in df.columns:
            text = df[text_column].fillna("").astype(str)
            chars = text.str.len()
            words = text.str.split().str.len()
            split_report["text_statistics"] = {
                "empty_records": int((chars == 0).sum()),
                "characters": {"mean": float(chars.mean()), "median": float(chars.median()), "p95": float(chars.quantile(0.95)), "max": int(chars.max())},
                "words": {"mean": float(words.mean()), "median": float(words.median()), "p95": float(words.quantile(0.95)), "max": int(words.max())},
            }
            pd.DataFrame({"characters": chars, "words": words}).describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_csv(split_dir / "text_length_summary.csv")

        specs = {
            "icd_label": _iter_icd_labels,
            "disease": _iter_disease_entities,
            "annonymisation": _iter_anonymisation_labels,
        }
        for column, extractor in specs.items():
            if column not in df.columns:
                continue
            counts = _count(df[column], extractor)
            empty = int(sum(1 for value in df[column] if not list(extractor(value))))
            top = counts.most_common(top_n)
            split_report[f"{column}_records_without_annotations"] = empty
            split_report[f"top_{column}"] = top
            pd.DataFrame(top, columns=[column, "count"]).to_csv(split_dir / f"top_{column}.csv", index=False)

        report["splits"][split_name] = split_report

    with (output_dir / "audit_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=_jsonable)

    click.echo(f"Audit complete: {output_dir / 'audit_report.json'}")


if __name__ == "__main__":
    main()
