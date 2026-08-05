"""Download pinned foundation-model snapshots for reproducible experiments."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_MODELS = {
    "modernbert": "answerdotai/ModernBERT-base",
    "deberta": "microsoft/deberta-v3-base",
    "bioclinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=sorted(DEFAULT_MODELS), default=["modernbert"])
    parser.add_argument("--output", type=Path, default=Path("models/foundation"))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, str]] = {}

    for alias in args.models:
        repo_id = DEFAULT_MODELS[alias]
        local_dir = args.output / alias
        resolved = snapshot_download(
            repo_id=repo_id,
            revision=args.revision,
            token=args.token,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "*.py", "README.md"],
        )
        manifest[alias] = {"repo_id": repo_id, "revision": args.revision, "path": str(resolved)}
        print(f"Downloaded {repo_id} -> {resolved}")

    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved model manifest to {manifest_path}")


if __name__ == "__main__":
    main()
