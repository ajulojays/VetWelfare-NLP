# End-to-end pipeline

This workflow separates **licensed source data**, **human welfare annotation**, and **model training**. PetEVAL does not contain Five-Domains welfare labels, so the repository never fabricates them.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[all]"
```

## 2. Authenticate with Hugging Face

PetEVAL is gated. First accept its access conditions on Hugging Face, then authenticate locally:

```bash
huggingface-cli login
export HF_TOKEN="your_token"   # optional after CLI login
```

For Weights & Biases tracking:

```bash
wandb login
export WANDB_PROJECT="VetWelfare-NLP"
```

## 3. Fetch PetEVAL

```bash
vetwelfare-fetch --output data/raw/peteval
```

The command saves the Hugging Face `DatasetDict` locally and writes `data/raw/peteval_manifest.json` containing split sizes and columns.

## 4. Download foundation models

Start with ModernBERT. Add the other encoders for controlled comparisons.

```bash
vetwelfare-download-models --models modernbert
vetwelfare-download-models --models modernbert deberta bioclinicalbert
```

Models are stored under `models/foundation/`. That directory should remain Git-ignored.

## 5. Create the annotation template

Inspect the PetEVAL manifest to identify the clinical narrative column, then run:

```bash
vetwelfare-prepare-annotation \
  --dataset data/raw/peteval \
  --split train \
  --text-column REPLACE_WITH_ACTUAL_TEXT_COLUMN \
  --sample-size 2000 \
  --output data/annotation/peteval_five_domains.csv
```

The output contains blank labels for Domains 1–4, evidence spans, severity, valence, inferred Mental State, annotator ID, and adjudication status. Labels require trained human annotation and adjudication.

## 6. Produce leakage-safe splits

After annotation, create train, validation, and test files. Prefer clinic-held-out splitting when clinic identifiers are available. At minimum, preserve record IDs and prevent duplicate notes from crossing splits.

Expected supervised columns:

```text
record_id,text,nutrition,physical_environment,health,behavioural_interactions
```

Labels must be binary integers (`0` or `1`). Missing or uncertain annotations should be resolved or excluded explicitly, not silently converted to zero.

## 7. Train the baseline

```bash
vetwelfare-train-baseline \
  --train data/processed/train.csv \
  --validation data/processed/validation.csv \
  --output artifacts/baseline
```

Outputs:

- `artifacts/baseline/model.joblib`
- `artifacts/baseline/metrics.json`

## 8. Validate the transformer configuration

```bash
vetwelfare-train-transformer \
  --train data/processed/train.csv \
  --validation data/processed/validation.csv \
  --model models/foundation/modernbert \
  --dry-run
```

## 9. Fine-tune ModernBERT

```bash
vetwelfare-train-transformer \
  --train data/processed/train.csv \
  --validation data/processed/validation.csv \
  --model models/foundation/modernbert \
  --epochs 3 \
  --batch-size 8 \
  --max-length 1024 \
  --output artifacts/modernbert
```

When `WANDB_API_KEY` is available, training metrics are logged automatically to Weights & Biases. Otherwise training runs fully offline.

## 10. Reproducibility rules

- Never commit PetEVAL records, tokens, model weights, or W&B credentials.
- Record the exact dataset revision and model revision in manifests.
- Use fixed seeds and clinic-held-out splits.
- Treat absent documentation as missing evidence, not proof of good welfare.
- Infer Domain 5 only from supported Domains 1–4 evidence and report abstention.
- Keep the untouched test set sealed until the analysis plan is frozen.
