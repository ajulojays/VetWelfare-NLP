# VetAI-Lab Ecosystem

## Mission

VetAI-Lab develops open, governed, and interoperable infrastructure for curating veterinary clinical and diagnostic data, building trustworthy benchmarks, training foundation models, and translating artificial intelligence into clinical, diagnostic, animal-welfare, and One Health innovation.

## Repositories

- **VetAnnotate** — collaborative annotation, assignment, review, and adjudication.
- **VetJudge** — model-agnostic ensemble agreement, evidence comparison, uncertainty, and human-review prioritization.
- **VetBench** — standardized datasets, tasks, metrics, baselines, and leaderboards.
- **VetHub** — federated data catalog, metadata standards, governance, and institutional access pathways.
- **VetFM** — shared veterinary foundation-model infrastructure and pretrained models.
- **VetWelfare-NLP** — Five-Domains welfare reasoning from clinical narratives.
- **ParasiteFM** — parasitology microscopy and diagnostic imaging.
- **VetPathFM** — pathology, cytology, and histopathology.
- **VetGenomeFM** — genomics, pathogens, antimicrobial resistance, and biological sequences.
- **docs** — ecosystem-wide governance, architecture, roadmaps, and partnership documentation.

## Data-to-innovation pathway

```text
Veterinary schools, hospitals, diagnostic laboratories, industry, and government
                                  |
                                  v
                                VetHub
             discovery, governance, metadata, access, interoperability
                                  |
                                  v
                             VetAnnotate
                  human annotation and expert adjudication
                         |                    |
                         v                    v
                     VetJudge              VetBench
          AI-assisted curation,        datasets, tasks,
          agreement, uncertainty       metrics, leaderboards
                         |                    |
                         +---------+----------+
                                   |
                                   v
                                 VetFM
                                   |
              +--------------------+--------------------+
              |                    |                    |
              v                    v                    v
       VetWelfare-NLP         ParasiteFM            VetPathFM
                                   |
                                   v
                              VetGenomeFM
```

## Initial implementation sequence

1. Complete VetJudge v0.1 inside VetWelfare-NLP and validate the five-model pipeline.
2. Create the VetAI-Lab GitHub organization and repositories from `ecosystem/repos.yaml`.
3. Extract VetJudge into its own package while keeping a compatibility dependency in VetWelfare-NLP.
4. Move the collaborative platform into VetAnnotate.
5. Establish shared schemas between VetHub, VetAnnotate, VetJudge, and VetBench.
6. Use VetWelfare-NLP as the first end-to-end flagship benchmark.

## Repository boundaries

### VetJudge owns

- provider-neutral model prediction schemas;
- blinded model identities;
- categorical and evidence agreement;
- consensus and uncertainty;
- logical consistency checks;
- human-review prioritization;
- model calibration against adjudicated labels.

### VetAnnotate owns

- annotator authentication and roles;
- balanced case assignment;
- annotation user interfaces;
- audit trails and autosave;
- adjudication dashboards;
- export to VetBench-compatible formats.

### VetHub owns

- dataset metadata and discoverability;
- institutional ownership and access conditions;
- data-use agreements and governance status;
- modality and species standards;
- links to approved VetBench tasks without requiring centralized raw-data storage.

### VetBench owns

- immutable benchmark versions;
- train, validation, and test definitions;
- task-specific metrics;
- reference baselines and leaderboards;
- data cards and model cards.

## Current flagship workflow

```text
PetEVAL clinical narratives
        -> five-model structured predictions
        -> VetJudge consensus and review queue
        -> overlapping veterinary human annotation
        -> senior adjudication
        -> VetWelfare-NLP gold benchmark
        -> VetBench release, subject to dataset license
```
