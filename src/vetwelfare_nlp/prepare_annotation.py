"""Create a Five-Domains annotation template from a locally saved PetEVAL split.

This script deliberately does not invent welfare labels. It exports note identifiers,
text, metadata, blank domain labels, evidence fields, certainty, and adjudication status.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_from_disk

DOMAINS = ["nutrition", "physical_environment", "health", "behavioural_interactions"]


def _choose_text_column(columns: list[str], requested: str | None) -> str:
    if requested:
        if requested not in columns:
            raise ValueError(f"Requested text column {requested!r} not found in {columns}")
        return requested
    candidates = ["text", "narrative", "clinical_text", "note", "consultation_text"]
    for column in candidates:
        if column in columns:
            return column
    raise ValueError(f"Could not infer the narrative column. Pass --text-column. Available: {columns}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/raw/peteval"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column")
    parser.add_argument("--id-column")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/annotation/peteval_five_domains.csv"))
    args = parser.parse_args()

    dataset = load_from_disk(str(args.dataset))
    if args.split not in dataset:
        raise ValueError(f"Split {args.split!r} not found. Available: {list(dataset.keys())}")
    split = dataset[args.split]
    text_column = _choose_text_column(split.column_names, args.text_column)

    frame = split.to_pandas()
    if args.sample_size and args.sample_size < len(frame):
        frame = frame.sample(args.sample_size, random_state=args.seed).reset_index(drop=True)

    output = pd.DataFrame({
        "record_id": frame[args.id_column].astype(str) if args.id_column else [f"{args.split}_{i:06d}" for i in range(len(frame))],
        "source_split": args.split,
        "text": frame[text_column].fillna("").astype(str),
    })
    for domain in DOMAINS:
        output[domain] = ""
        output[f"{domain}_evidence"] = ""
        output[f"{domain}_severity"] = ""
        output[f"{domain}_valence"] = ""
    output["mental_state"] = ""
    output["mental_state_evidence_basis"] = ""
    output["annotator_id"] = ""
    output["adjudication_status"] = "unreviewed"
    output["notes"] = ""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Exported {len(output):,} records to {args.output}")
    print("Labels are intentionally blank and require trained human annotation/adjudication.")


if __name__ == "__main__":
    main()
