# VetWelfare-NLP Pilot Annotation Guidelines

## Purpose

These guidelines define how to annotate veterinary clinical narratives using the Five Domains framework while separating directly documented evidence from inferred affective state.

## Core principle

Annotate only what is supported by the note. Absence of a mention is **unknown**, not normal or positive.

## Allowed status values for Domains 1–4

- `negative`: documented adverse welfare-relevant evidence
- `positive`: documented favorable evidence or recovery
- `mixed`: both favorable and adverse evidence
- `neutral`: explicitly assessed with no meaningful welfare concern
- `unknown`: insufficient information

## Domain 1 — Nutrition

Include appetite, thirst, hydration, body condition, weight gain/loss, feeding ability, malnutrition, obesity, vomiting or diarrhea only when they materially affect intake or nutritional state.

Examples:

- `reduced appetite` → negative
- `eating and drinking normally` → neutral or positive depending on context
- `obese body condition` → negative
- isolated single vomiting episode with normal intake → usually unknown or mild negative, depending on context

## Domain 2 — Physical Environment

Include housing, temperature, shelter, bedding, confinement, transport, environmental hazards, exposure, cleanliness, and resource access.

Most routine clinical notes will be `unknown` for this domain.

Do not infer poor environment from disease alone.

## Domain 3 — Health

Include pain, injury, disease, inflammation, infection, impaired function, mobility limitation, respiratory difficulty, pruritus, wounds, masses, and treatment burden.

Normal examination findings may be annotated as neutral only when explicitly documented.

## Domain 4 — Behavioural Interactions

Include normal or abnormal behavior, mobility-related behavior, aggression, withdrawal, play, social interaction, stereotypy, fear responses, handling tolerance, and human–animal interaction.

Do not treat every activity-related phrase as behavioral evidence. For example, `reluctant to rise` may support both Health and Behaviour if it describes both physical impairment and expressed behavior.

## Domain 5 — Mental State

Domain 5 is inferred from supported evidence in Domains 1–4.

Allowed labels:

- `likely_negative`
- `likely_positive`
- `mixed`
- `uncertain`
- `unknown`

Use `unknown` when no welfare-relevant evidence is present. Use `uncertain` when evidence exists but affective consequences are ambiguous.

Examples:

- pain, pruritus, dyspnea, fear, or severe mobility restriction → likely_negative
- comfortable, active, eating normally, willingly interacting → likely_positive only when clearly documented
- normal exam plus one transient mild symptom → uncertain or mixed

## Evidence spans

Copy the shortest exact phrase that supports the annotation.

Good: `hips are a little stiff on extension`

Too broad: the entire note

Do not paraphrase evidence spans.

## Severity

- `none`: no adverse evidence
- `mild`: limited, transient, or low-intensity concern
- `moderate`: persistent or functionally meaningful concern
- `severe`: marked suffering, major functional compromise, or urgent threat
- `unknown`: insufficient information

## Overall valence

- `positive`
- `negative`
- `mixed`
- `neutral`
- `unknown`

## Pilot workflow

1. Annotate 25 records independently.
2. Record uncertainty in `notes`.
3. Review disagreements.
4. Revise these guidelines before annotating the remaining pilot set.
5. Do not train a model until the first adjudicated batch is complete.

## Example interpretations

### Example: stiff hips and unchanged fatty lumps

- Nutrition: unknown
- Environment: unknown
- Health: negative; evidence `Hips are a little stiff on extension`
- Behaviour: unknown unless mobility behavior is explicitly described
- Mental State: uncertain or likely_negative with low confidence
- Severity: mild

### Example: recurrent pruritus and inflamed paws

- Nutrition: unknown
- Environment: unknown
- Health: negative; evidence `mild inflammation over front paws`, `pruritic over caudal dorsum`
- Behaviour: negative; evidence `Biting at feet`
- Mental State: likely_negative
- Severity: moderate

### Example: one-off vomiting with otherwise normal exam

- Nutrition: mild negative or unknown depending on intake evidence
- Health: mild negative; evidence `Vomited last night`
- Behaviour: unknown
- Mental State: uncertain
- Severity: mild
