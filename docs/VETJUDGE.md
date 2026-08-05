# VetJudge

**VetJudge: An Ensemble Agreement Framework for AI-Assisted Veterinary Clinical Annotation**

VetJudge converts predictions from multiple language models into a blinded, auditable consensus and human-review prioritization layer. It is designed for veterinary clinical annotation studies in which model agreement is analyzed before independent human annotation and adjudication.

## Scientific principle

Model consensus is not ground truth. VetJudge measures reproducibility, uncertainty, and evidence overlap across models. Final validity must be established using independently generated human annotations and senior adjudication.

## Workflow

```text
Veterinary clinical narrative
        ↓
Five independent model annotations
        ↓
Model identity blinding
        ↓
Categorical agreement + evidence overlap
        ↓
Consensus, entropy and disagreement
        ↓
Priority-ranked human review queue
        ↓
Independent veterinary annotation
        ↓
Adjudicated benchmark labels
```

## Core outputs

Running `vetjudge` creates:

- `blinded_predictions.csv` — model outputs with stable blinded identifiers.
- `pairwise_categorical_agreement.csv` — pairwise Cohen's kappa by annotation field.
- `pairwise_evidence_overlap.csv` — token-level evidence-span Jaccard overlap.
- `case_consensus_and_priority.csv` — consensus labels, vote distributions, entropy, disagreement and priority.
- `human_review_queue.csv` — compact ranked queue for annotation assignment.
- `vetjudge_summary.json` — run-level study summary.
- `model_identity_map.private.json` — private mapping from blinded IDs to actual model names.

Do not expose `model_identity_map.private.json` to annotators during blinded review.

## Priority score

The initial human-review priority score combines:

- 45% categorical disagreement;
- 30% normalized vote entropy;
- 20% evidence-span disagreement;
- 5% missing-model penalty.

These weights are explicit defaults, not validated clinical thresholds. They should be preregistered before the main study and tested in sensitivity analyses.

## Run

```bash
vetjudge \
  --predictions outputs/model_prefill/calibration_25_predictions.jsonl \
  --output-dir outputs/vetjudge/calibration_25
```

Use a private blinding salt in a real study:

```bash
export VETJUDGE_BLIND_SALT="replace-with-a-private-random-value"
```

## Recommended two-stage publication design

### Study 1 — Model agreement

Evaluate all five models under identical conditions:

- locked prompt and JSON schema;
- temperature 0;
- no cross-model communication;
- exact model/version logging;
- identical input text;
- no access to human adjudicated labels.

Primary outcomes:

- pairwise Cohen's kappa;
- multi-model agreement and entropy;
- evidence-span overlap;
- agreement by domain, note length and ICD stratum;
- frequency of unanimous but low-evidence cases;
- high-disagreement case characteristics.

### Study 2 — Human overlap and adjudication

Assign each case to at least three independent human annotators. Preserve a blinded human-only arm to avoid anchoring. Compare:

- human-human agreement;
- model-model agreement;
- human-model agreement;
- human adjudication versus model consensus;
- annotation time and correction rates;
- student versus veterinarian agreement;
- whether VetJudge priority predicts human disagreement.

## Safeguards

- Never label model consensus as gold truth.
- Preserve individual model and human annotations.
- Keep annotators blinded to model brand and identity.
- Validate that evidence spans occur in the source note.
- Report missing model outputs and parsing failures.
- Verify dataset licenses before sending records to external APIs.
- Prefer institution-controlled inference for gated clinical text.
