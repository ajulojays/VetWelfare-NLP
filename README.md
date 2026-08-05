# VetWelfare-NLP

**Explainable NLP for Five-Domains animal-welfare signal extraction from veterinary narratives.**

VetWelfare-NLP is an open research scaffold for converting unstructured veterinary text into an evidence-linked welfare profile based on the **2020 Five Domains Model**:

1. Nutrition
2. Physical Environment
3. Health
4. Behavioural Interactions
5. Mental State

The project does **not** claim to diagnose animal welfare automatically. It detects text-supported welfare indicators, highlights evidence, quantifies uncertainty, and abstains when the record is insufficient.

## Scientific premise

The first four domains describe welfare-relevant conditions and interactions. Domain 5 represents the animal's inferred mental or affective state arising from those conditions. Accordingly, VetWelfare-NLP uses a two-stage design:

1. Extract evidence for Domains 1–4 from text.
2. Infer Domain 5 only from supported evidence, with uncertainty and an abstention option.

This avoids treating Mental State as just another flat label.

## Initial research question

> Can NLP models produce calibrated, interpretable Five-Domains welfare-signal profiles from veterinary clinical narratives that generalize across clinics and species?

## MVP tasks

- Multi-label classification of Domains 1–4
- Evidence-span extraction
- Valence classification: negative, neutral/unclear, positive
- Severity estimation: none, mild, moderate, severe
- Domain 5 inference with abstention
- Clinic-held-out and species-stratified evaluation
- Bias, calibration, and error analysis

## Recommended data strategy

Start with **PetEVAL**, a benchmark of 17,600 professionally annotated veterinary EHRs from UK first-opinion practices. PetEVAL provides clinical narratives and clinic-separated train/evaluation/test partitions, but it does not directly provide Five-Domains labels. VetWelfare-NLP therefore adds an explicit welfare ontology and annotation layer.

Do not redistribute protected or license-restricted clinical text in this repository. Store only scripts, schemas, mappings, and permitted synthetic examples.

## Repository structure

```text
configs/                  Model and training configuration
data/sample/              Synthetic examples only
docs/                     Research protocol and literature deep dive
src/vetwelfare_nlp/       Ontology, rules, modeling, and evaluation code
tests/                    Unit tests
.github/workflows/        Continuous integration
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

vetwelfare profile data/sample/synthetic_notes.csv \
  --text-column text \
  --output predictions.csv
```

The initial CLI runs a transparent lexicon baseline. It is intended as a reproducible benchmark and annotation aid—not a clinical tool.

## Output schema

Each record receives:

- domain-level evidence spans
- domain presence probabilities or baseline scores
- positive/negative valence
- severity
- inferred Mental State
- confidence and abstention status

Example:

```json
{
  "nutrition": {"score": 0.75, "evidence": ["reduced appetite"]},
  "physical_environment": {"score": 0.0, "evidence": []},
  "health": {"score": 1.0, "evidence": ["painful on hip extension"]},
  "behavioural_interactions": {"score": 0.5, "evidence": ["reluctant to rise"]},
  "mental_state": {
    "label": "likely_negative",
    "confidence": 0.67,
    "abstain": false
  }
}
```

## Benchmark plan

1. Lexicon/rule baseline
2. TF-IDF + one-vs-rest logistic regression
3. Transformer multi-label classifier
4. Joint classifier + evidence-span extractor
5. Hierarchical model for Domains 1–4 → Domain 5
6. Optional instruction-tuned LLM baseline with fixed prompts

Primary metrics:

- Macro and micro F1
- Per-domain precision, recall, and PR-AUC
- Evidence-span F1
- Expected calibration error and Brier score
- Selective risk/coverage under abstention
- Worst-group performance by species, clinic, and note type

## Key safeguards

- Human review is required for consequential decisions.
- Absence of a domain mention is not evidence of good welfare.
- Clinical records systematically under-document environment, agency, positive experiences, and human–animal interactions.
- Domain 5 is an inference, not directly observable ground truth.
- The model must report missingness and uncertainty.

## Foundational references

- Mellor DJ et al. The 2020 Five Domains Model: Including Human–Animal Interactions in Assessments of Animal Welfare. *Animals*. 2020;10:1870. DOI: 10.3390/ani10101870.
- Farrell S et al. PetEVAL: A veterinary free text electronic health records benchmark. *BioNLP 2025*. DOI: 10.18653/v1/2025.bionlp-1.29.
- Nie A et al. DeepTag: inferring diagnoses from veterinary clinical notes in an under-resourced medical domain. *npj Digital Medicine*. 2019.

See [`docs/AI_NLP_ANIMAL_WELFARE_DEEP_DIVE.md`](docs/AI_NLP_ANIMAL_WELFARE_DEEP_DIVE.md) for the research landscape and study design.

## Status

Research scaffold / pre-registration stage. Not validated for clinical, regulatory, inspection, or enforcement use.

## License

MIT for code. Dataset licenses and institutional data-use agreements remain separate.
