#!/usr/bin/env bash
set -euo pipefail

ORG="${1:-VetAI-Lab}"
MANIFEST="ecosystem/repos.yaml"

if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: GitHub CLI (gh) is required. Install it and run 'gh auth login'." >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "ERROR: GitHub CLI is not authenticated. Run 'gh auth login'." >&2
  exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: Missing $MANIFEST" >&2
  exit 1
fi

python - "$ORG" "$MANIFEST" <<'PY'
import subprocess
import sys
from pathlib import Path

import yaml

org = sys.argv[1]
manifest_path = Path(sys.argv[2])
manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

for repo in manifest["repositories"]:
    name = repo["name"]
    description = repo["description"]
    visibility = repo.get("visibility", "public")
    full = f"{org}/{name}"

    exists = subprocess.run(
        ["gh", "repo", "view", full],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0

    if exists:
        print(f"[SKIP] {full} already exists")
        continue

    cmd = [
        "gh", "repo", "create", full,
        f"--{visibility}",
        "--description", description,
        "--add-readme",
    ]
    subprocess.run(cmd, check=True)
    print(f"[CREATED] {full}")
PY

echo
echo "VetAI-Lab repository bootstrap complete."
echo "Next: transfer or mirror the current VetWelfare-NLP repository and extract VetJudge."
