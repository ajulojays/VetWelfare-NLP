"""Command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd

from .baseline import profile_text


@click.group()
def main() -> None:
    """VetWelfare-NLP utilities."""


@main.command()
@click.argument("input_csv", type=click.Path(exists=True, path_type=Path))
@click.option("--text-column", default="text", show_default=True)
@click.option("--output", type=click.Path(path_type=Path), default=Path("predictions.csv"))
def profile(input_csv: Path, text_column: str, output: Path) -> None:
    """Profile text records using the transparent lexicon baseline."""
    frame = pd.read_csv(input_csv)
    if text_column not in frame.columns:
        raise click.ClickException(f"Missing text column: {text_column}")
    frame["welfare_profile"] = frame[text_column].fillna("").map(
        lambda text: json.dumps(profile_text(str(text)), ensure_ascii=False)
    )
    frame.to_csv(output, index=False)
    click.echo(f"Saved {len(frame):,} profiles to {output}")


if __name__ == "__main__":
    main()
