# AI and NLP for Animal Welfare: Deep Dive and Research Blueprint

## Executive conclusion

Natural-language processing is established in veterinary medicine for diagnosis coding, syndrome surveillance, named-entity recognition, de-identification, and clinical information extraction. However, NLP explicitly organized around a validated animal-welfare framework remains comparatively underdeveloped. This creates a credible research gap—but only if the task is framed as **welfare-signal extraction and structured assessment support**, not automated determination of an animal's welfare state.

The highest-value initial project is a Five-Domains evidence extractor trained on veterinary narratives. It should identify documented positive and negative welfare indicators, preserve the distinction between observable evidence and inferred affect, expose missing information, and abstain when conclusions are unsupported.

## 1. Why NLP is relevant to animal welfare

Animal-welfare information is dispersed across clinical notes, herd-health records, shelter narratives, inspection reports, incident reports, owner communications, behavior consultations, treatment histories, and research observations. Much of this information is unstructured and therefore difficult to aggregate systematically.

NLP can support:

- extraction of welfare indicators and evidence spans;
- longitudinal tracking of welfare-relevant changes;
- screening records for cases needing expert review;
- audit of documentation completeness;
- population-level surveillance;
- harmonization of heterogeneous terminology;
- structured research datasets from narrative records.

The primary opportunity is not text generation. It is reliable extraction, normalization, uncertainty estimation, and human-centered decision support.

## 2. The Five Domains as the ontology

The 2020 Five Domains Model consists of:

1. **Nutrition** — food and water availability, intake, quality, feeding method, hunger, thirst, malnutrition, and feeding-related positive experiences.
2. **Physical Environment** — temperature, substrate, shelter, space, air quality, noise, weather exposure, physical comfort, and environmental opportunity.
3. **Health** — disease, injury, pain, functional impairment, physical fitness, and treatment effects.
4. **Behavioural Interactions** — interactions with the environment, other animals, and humans; agency, choice, restriction, social contact, play, exploration, fear-provoking handling, and positive human contact.
5. **Mental State** — negative and positive subjective experiences inferred from evidence collected in Domains 1–4.

### Critical modeling implication

The Five Domains are not five interchangeable labels. Domains 1–4 organize evidence about the animal's circumstances and functioning. Domain 5 integrates the likely affective consequences. A scientifically aligned architecture should therefore be hierarchical:

```text
Narrative text
   ↓
Evidence spans + valence + severity for Domains 1–4
   ↓
Missingness and confidence assessment
   ↓
Constrained inference for Domain 5
   ↓
Welfare profile for expert review
```

## 3. State of veterinary NLP

### Diagnosis coding

DeepTag demonstrated automated inference of diagnostic codes from more than 100,000 expert-annotated veterinary notes and introduced selective prediction, allowing the model to abstain on uncertain cases. Later work expanded veterinary diagnosis coding using transformer and foundation models over much larger coding spaces.

### Syndrome surveillance and NER

PetEVAL introduced 17,600 real-world veterinary EHRs with clinic-separated partitions, ICD-11 syndromic labels, disease named entities, and anonymization entities. This provides a strong base for evaluating cross-clinic generalization and for transferring representations to welfare tasks.

### LLM extraction

Recent veterinary studies indicate that general-purpose language models can extract specified clinical findings from EHRs, but performance depends on task definition, ambiguity, reproducibility, privacy, and evaluation against expert annotations. High apparent accuracy on narrow extraction tasks does not establish validity for welfare assessment.

### Clinical deployment considerations

Language-model applications in veterinary practice include documentation, client communication, information extraction, clinical decision support, and practice assessment. Their risks include hallucination, automation bias, privacy leakage, uneven performance, lack of calibration, and unclear accountability.

## 4. The specific research gap

Existing veterinary NLP usually predicts diagnoses, codes, signs, syndromes, or entities. Animal-welfare assessment requires additional reasoning dimensions:

- positive as well as negative experiences;
- environmental and behavioral context;
- opportunities, choice, agency, and restriction;
- interactions with humans and conspecifics;
- severity, duration, and reversibility;
- evidence provenance;
- explicit missingness;
- integration into an inferred mental state.

Clinical notes are especially rich for Domain 3 but can be sparse for Domains 2 and 4. A model trained only on clinical records may therefore learn a distorted definition of welfare centered on illness. The project must measure documentation bias rather than hide it.

## 5. Proposed annotation ontology

Each annotation should include:

- `domain`: nutrition, physical_environment, health, behavioural_interactions;
- `subdomain`: controlled vocabulary;
- `evidence_span`: exact text;
- `valence`: positive, negative, mixed, unclear;
- `severity`: none, mild, moderate, severe, unknown;
- `temporality`: current, historical, hypothetical, resolved;
- `experiencer`: index animal, another animal, human;
- `certainty`: asserted, probable, possible, negated;
- `source`: clinician observation, owner report, test result, inferred;
- `duration`: acute, chronic, recurrent, unknown;
- `actionability`: routine, review, urgent, unknown.

Domain 5 should be annotated separately by trained welfare experts using the collected evidence and should include confidence and rationale. Annotator disagreement is expected and scientifically meaningful.

## 6. Data plan

### Phase 1: Public benchmark prototyping

Use PetEVAL narratives subject to its license. Create a small, expert-annotated welfare extension:

- 1,000–2,000 records for initial development;
- dual independent annotation;
- adjudication by a veterinarian with animal-welfare expertise;
- clinic-held-out evaluation;
- species and note-type stratification.

### Phase 2: Broader document types

Add records that better represent environment and behavior:

- shelter intake and progress notes;
- behavior consultation notes;
- livestock welfare assessment narratives;
- farm audit and inspection reports;
- animal-cruelty forensic reports where ethically and legally appropriate;
- owner questionnaires and quality-of-life instruments.

### Phase 3: Multimodal linkage

Link text with structured clinical measurements, body-condition scores, pain scales, activity data, images, audio, or video. Text-only inference should remain clearly distinguished from multimodal assessment.

## 7. Modeling strategy

### Baseline A: Lexicon and rules

A transparent baseline establishes task feasibility and surfaces ontology errors. It should include negation, temporality, and experiencer handling.

### Baseline B: Sparse supervised model

TF-IDF with one-vs-rest logistic regression is strong for small datasets, fast to train, and easy to interpret.

### Baseline C: Transformer encoder

Fine-tune a suitable biomedical, clinical, veterinary, or modern general encoder for multi-label classification. Compare domain-pretrained and non-domain-pretrained models under identical splits.

### Joint evidence model

Use token classification or span extraction alongside document-level prediction. A correct label without defensible evidence is insufficient for this application.

### Hierarchical Domain 5 model

Predict Domain 5 from extracted domain evidence, valence, severity, duration, and missingness. Consider monotonic or rule-constrained components so severe negative evidence cannot be silently transformed into a positive inference.

### LLM baseline

Evaluate zero-shot and few-shot prompting using a locked schema. Measure run-to-run consistency, unsupported claims, evidence faithfulness, cost, latency, and privacy implications. Do not treat an LLM-generated explanation as evidence of correct reasoning.

## 8. Evaluation

### Predictive performance

- Macro/micro F1
- PR-AUC by domain and subdomain
- Sensitivity for severe negative signals
- Specificity for absent/negated indicators
- Evidence-span precision, recall, and F1

### Calibration and abstention

- Brier score
- Expected calibration error
- Reliability plots
- Risk–coverage curves
- Error rate among accepted versus abstained predictions

### Generalization

- clinic-held-out testing;
- species-held-out or species-stratified evaluation;
- temporal shift;
- document-type shift;
- external institutional validation.

### Fairness and robustness

Assess performance by species, breed where justified, age group, sex, clinical service, record length, documentation style, and source. Avoid presenting biologically confounded performance differences as social fairness conclusions without careful interpretation.

### Human factors

Measure:

- expert time saved;
- false reassurance;
- automation bias;
- interpretability usefulness;
- effect on inter-rater agreement;
- whether highlighted evidence improves or degrades decisions.

## 9. Risks and safeguards

### Construct validity

Welfare is not fully observable in clinical text. The model detects documented indicators, not the complete lived experience of the animal.

### Missing-not-at-random documentation

No mention of thirst, housing, play, or social interaction does not imply these domains are satisfactory. Outputs must distinguish `not documented` from `no concern`.

### Domain 5 overreach

Mental states are inferred, not directly read from notes. The system should provide bounded language such as `evidence suggests`, confidence intervals or categories, and abstention.

### Privacy

Veterinary EHRs may contain owner identifiers, addresses, financial information, staff names, and sensitive allegations. De-identification and data governance are mandatory.

### Enforcement misuse

A research model should not autonomously trigger punitive, regulatory, insurance, employment, or legal action. High-stakes use requires validation, governance, due process, and human review.

### Positive-welfare neglect

Clinical data overrepresent disease and negative states. Annotation guidelines must deliberately capture positive appetite, play, comfort, affiliative behavior, recovery, agency, and beneficial human interaction when present.

## 10. Publishable first study

### Working title

**VetWelfare-NLP: Evidence-Grounded Extraction of Five-Domains Animal-Welfare Signals from Veterinary Clinical Narratives**

### Primary aims

1. Develop and validate a Five-Domains text annotation ontology.
2. Benchmark rule-based, sparse, transformer, and prompted-LLM approaches.
3. Quantify cross-clinic generalization and domain-specific documentation gaps.
4. Evaluate evidence faithfulness, calibration, and selective prediction.

### Hypotheses

- Health indicators will be substantially easier to identify than environment or behavioral interactions in routine clinical notes.
- Joint evidence extraction will improve expert trust and error detection relative to label-only classifiers.
- Clinic-held-out performance will be lower than random-split performance.
- Explicit abstention will reduce harmful high-confidence errors.
- Domain 5 inference will be unreliable when Domains 2 and 4 are undocumented.

### Minimum viable paper

- ontology and annotation manual;
- at least 1,000 dual-annotated notes;
- inter-annotator agreement;
- four model families;
- clinic-held-out test;
- calibration and abstention;
- detailed error analysis;
- public code and synthetic examples.

## 11. Longer-term program

This project can expand into a broader veterinary AI program:

- longitudinal welfare trajectory modeling;
- livestock and shelter welfare surveillance;
- multilingual veterinary welfare NLP;
- retrieval-augmented welfare guidelines;
- multimodal fusion with activity, imaging, audio, and video;
- domain-adaptive foundation models for veterinary records;
- causal evaluation of interventions and welfare outcomes.

## 12. Strategic value for an AI-for-veterinary-medicine faculty application

VetWelfare-NLP complements projects in pathogen evolution, veterinary diagnostics, and therapeutic discovery by adding a direct animal-centered outcome. It demonstrates that the research program is not merely applying AI to biological data; it is developing accountable AI around a central mission of veterinary medicine: improving animal health and welfare.

## References

1. Mellor DJ, Beausoleil NJ, Littlewood KE, et al. The 2020 Five Domains Model: Including Human–Animal Interactions in Assessments of Animal Welfare. Animals. 2020;10(10):1870. doi:10.3390/ani10101870.
2. Farrell S, Radford A, Al Moubayed N, Noble P-J. PetEVAL: A veterinary free text electronic health records benchmark. Proceedings of BioNLP 2025. doi:10.18653/v1/2025.bionlp-1.29.
3. Nie A, Zehnder A, Page RL, et al. DeepTag: inferring diagnoses from veterinary clinical notes in an under-resourced medical domain. npj Digital Medicine. 2019.
4. Bollig N, Lustgarten JL, Venit E. Language Models in Veterinary Clinical Practice: Applications, Risks, and Practical Guidance. Vet Clin North Am Small Anim Pract. 2026. doi:10.1016/j.cvsm.2026.03.014.
