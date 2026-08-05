# Collaborative annotation platform

## Architecture

- Streamlit annotator: `http://localhost:8501`
- FastAPI backend/docs: `http://localhost:8000/docs`
- PostgreSQL: `localhost:5432`
- Streamlit adjudication dashboard: `http://localhost:8502`

## Start the stack

```bash
docker compose up --build
```

## Load cases and annotators

Run this in a second terminal after the database is healthy:

```bash
docker compose exec api vetwelfare-seed-platform \
  data/annotation/pilot_500.csv \
  --annotations-per-case 3 \
  --annotators \
  student01:"Student One":veterinary_student \
  vet01:"Veterinarian One":veterinarian \
  expert01:"Welfare Expert":welfare_specialist
```

Use stable pseudonymous `external_id` values. Do not use email addresses as public identifiers.

## Assignment design

The initial engine uses deterministic rotating assignment. Each case receives the requested number of distinct annotators, and annotator pairs rotate across cases. This is appropriate for pilot deployment. Production extensions should add calibration eligibility, workload caps, hidden duplicates, and experience-group constraints.

## Workflow

1. Register each collaborator in the annotator interface.
2. Annotators receive one pending assignment at a time.
3. Submission stores the full structured payload, confidence, duration, and guideline version.
4. Cases with at least two completed annotations enter the adjudication queue.
5. An adjudicator reviews all individual labels and saves a consensus payload.

## Security status

This is a research pilot scaffold, not an internet-ready deployment. Before external hosting, add institutional authentication, TLS, role-based authorization, audit logs, database backups, rate limiting, secret management, and formal confirmation that the dataset license permits the planned access model. Never commit PetEVAL clinical narratives or database dumps to GitHub.

## Reset local database

```bash
docker compose down -v
```

This permanently removes the local PostgreSQL volume.
