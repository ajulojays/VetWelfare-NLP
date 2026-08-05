#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
MODEL_SET="${MODEL_SET:-modernbert}"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -e ".[all]"

if [[ -z "${HF_TOKEN:-}" ]]; then
  cat <<'EOF'
HF_TOKEN is not set.
1. Accept the PetEVAL access conditions on Hugging Face.
2. Run: huggingface-cli login
3. Or export HF_TOKEN=your_token
EOF
  exit 2
fi

vetwelfare-fetch --output data/raw/peteval
# MODEL_SET is space-separated, for example: MODEL_SET="modernbert deberta"
# shellcheck disable=SC2086
vetwelfare-download-models --models $MODEL_SET

cat <<'EOF'
Bootstrap complete.

Next:
1. Inspect data/raw/peteval_manifest.json to identify the narrative column.
2. Create the human-annotation template, for example:

   vetwelfare-prepare-annotation \
     --dataset data/raw/peteval \
     --split train \
     --text-column YOUR_TEXT_COLUMN \
     --sample-size 2000

The pipeline stops here intentionally because Five-Domains labels must not be fabricated.
See docs/PIPELINE.md for supervised training commands after annotation.
EOF
