"""Download and validate the gated SAVSNET/PetEVAL dataset from Hugging Face."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import DatasetDict, load_dataset

DATASET_ID = "SAVSNET/PetEVAL"


def _summarize(dataset: DatasetDict) -> dict[str, object]:
    return {
        "dataset_id": DATASET_ID,
        "splits": {
            name: {"rows": len(split), "columns": split.column_names}
            for name, split in dataset.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/peteval"))
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face token. Prefer setting HF_TOKEN instead of passing it on the command line.",
    )
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    if not args.token:
        raise SystemExit(
            "HF_TOKEN is required because PetEVAL is gated. Accept the dataset terms at "
            "https://huggingface.co/datasets/SAVSNET/PetEVAL and run `huggingface-cli login` "
            "or export HF_TOKEN=..."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET_ID, token=args.token, revision=args.revision)
    dataset.save_to_disk(str(args.output))

    summary = _summarize(dataset)
    summary_path = args.output.parent / "peteval_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved dataset to {args.output}")
    print(f"Saved manifest to {summary_path}")


if __name__ == "__main__":
    main()
