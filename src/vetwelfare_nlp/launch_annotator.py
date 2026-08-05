from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click


@click.command()
@click.option(
    "--app",
    "app_path",
    type=click.Path(path_type=Path),
    default=Path("apps/annotator.py"),
    show_default=True,
)
@click.option("--port", default=8501, show_default=True, type=int)
def main(app_path: Path, port: int) -> None:
    """Launch the local VetWelfare Streamlit annotation interface."""
    if not app_path.exists():
        raise click.ClickException(
            f"Annotator app not found: {app_path}. Run this command from the repository root."
        )
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
    ]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
