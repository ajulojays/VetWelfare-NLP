from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import click
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

CATEGORICAL_FIELDS = [
    "nutrition_status",
    "environment_status",
    "health_status",
    "behaviour_status",
    "mental_state",
    "overall_valence",
    "severity",
]

EVIDENCE_FIELDS = [
    "nutrition_evidence",
    "environment_evidence",
    "health_evidence",
    "behaviour_evidence",
    "mental_state_evidence",
]


def _stable_blind_id(model_id: str, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}:{model_id}".encode("utf-8")).hexdigest()[:8]
    return f"model_{digest}"


def _shannon_entropy(values: Iterable[str]) -> float:
    values = [str(v) for v in values]
    if not values:
        return float("nan")
    counts = Counter(values)
    total = len(values)
    entropy = -sum((n / total) * math.log2(n / total) for n in counts.values())
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def _tokenize_evidence(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    if isinstance(value, list):
        text = " ".join(str(x) for x in value)
    else:
        text = str(value)
    return {token.strip(".,;:!?()[]{}\"'").lower() for token in text.split() if token.strip()}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _pairwise_kappa(frame: pd.DataFrame, field: str) -> list[dict[str, Any]]:
    wide = frame.pivot(index="case_id", columns="blind_model_id", values=field)
    models = list(wide.columns)
    rows: list[dict[str, Any]] = []
    for i, model_a in enumerate(models):
        for model_b in models[i + 1 :]:
            pair = wide[[model_a, model_b]].dropna()
            score = cohen_kappa_score(pair[model_a], pair[model_b]) if len(pair) else np.nan
            rows.append(
                {
                    "field": field,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n": len(pair),
                    "cohen_kappa": score,
                }
            )
    return rows


def _pairwise_evidence(frame: pd.DataFrame, field: str) -> list[dict[str, Any]]:
    models = sorted(frame["blind_model_id"].unique())
    rows: list[dict[str, Any]] = []
    for i, model_a in enumerate(models):
        for model_b in models[i + 1 :]:
            left = frame[frame["blind_model_id"] == model_a][["case_id", field]].rename(
                columns={field: "a"}
            )
            right = frame[frame["blind_model_id"] == model_b][["case_id", field]].rename(
                columns={field: "b"}
            )
            merged = left.merge(right, on="case_id", how="inner")
            scores = [
                _jaccard(_tokenize_evidence(row.a), _tokenize_evidence(row.b))
                for row in merged.itertuples(index=False)
            ]
            rows.append(
                {
                    "field": field,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n": len(scores),
                    "mean_token_jaccard": float(np.mean(scores)) if scores else np.nan,
                }
            )
    return rows


def _load_predictions(path: Path, blind_salt: str) -> tuple[pd.DataFrame, dict[str, str]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[dict[str, Any]] = []
    model_map: dict[str, str] = {}

    for record in records:
        if record.get("status") != "success":
            continue
        model_id = str(record["model_id"])
        blind_id = _stable_blind_id(model_id, blind_salt)
        model_map[blind_id] = model_id
        annotation = record.get("annotation", {})
        row: dict[str, Any] = {
            "case_id": str(record["case_id"]),
            "model_id": model_id,
            "blind_model_id": blind_id,
            "latency_seconds": record.get("latency_seconds"),
        }
        for field in CATEGORICAL_FIELDS:
            row[field] = annotation.get(field, "unknown")
        for field in EVIDENCE_FIELDS:
            row[field] = annotation.get(field, "")
        row["confidence"] = annotation.get("confidence")
        rows.append(row)

    return pd.DataFrame(rows), model_map


@click.command()
@click.option("--predictions", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("outputs/vetjudge"),
    show_default=True,
)
@click.option("--blind-salt", envvar="VETJUDGE_BLIND_SALT", default="vetjudge-v1", show_default=True)
@click.option("--min-models", default=3, type=int, show_default=True)
def main(predictions: Path, output_dir: Path, blind_salt: str, min_models: int) -> None:
    """Build blinded ensemble consensus, uncertainty, evidence-overlap, and review priorities."""
    frame, model_map = _load_predictions(predictions, blind_salt)
    if frame.empty:
        raise click.ClickException("No successful model predictions were found.")

    output_dir.mkdir(parents=True, exist_ok=True)
    frame.drop(columns=["model_id"]).to_csv(output_dir / "blinded_predictions.csv", index=False)

    kappa_rows: list[dict[str, Any]] = []
    for field in CATEGORICAL_FIELDS:
        kappa_rows.extend(_pairwise_kappa(frame, field))
    kappa = pd.DataFrame(kappa_rows)
    kappa.to_csv(output_dir / "pairwise_categorical_agreement.csv", index=False)

    evidence_rows: list[dict[str, Any]] = []
    for field in EVIDENCE_FIELDS:
        evidence_rows.extend(_pairwise_evidence(frame, field))
    evidence = pd.DataFrame(evidence_rows)
    evidence.to_csv(output_dir / "pairwise_evidence_overlap.csv", index=False)

    case_rows: list[dict[str, Any]] = []
    for case_id, group in frame.groupby("case_id", sort=False):
        row: dict[str, Any] = {
            "case_id": case_id,
            "n_models": int(group["blind_model_id"].nunique()),
        }
        field_disagreements: list[float] = []
        field_entropies: list[float] = []

        for field in CATEGORICAL_FIELDS:
            values = group[field].fillna("unknown").astype(str).tolist()
            counts = Counter(values)
            consensus, votes = counts.most_common(1)[0]
            agreement = votes / len(values)
            entropy = _shannon_entropy(values)
            row[f"{field}_consensus"] = consensus
            row[f"{field}_agreement"] = agreement
            row[f"{field}_entropy"] = entropy
            row[f"{field}_votes"] = json.dumps(dict(sorted(counts.items())), sort_keys=True)
            field_disagreements.append(1.0 - agreement)
            field_entropies.append(entropy)

        evidence_scores: list[float] = []
        for field in EVIDENCE_FIELDS:
            token_sets = [_tokenize_evidence(v) for v in group[field].tolist()]
            for i, left in enumerate(token_sets):
                for right in token_sets[i + 1 :]:
                    evidence_scores.append(_jaccard(left, right))

        row["mean_categorical_disagreement"] = float(np.mean(field_disagreements))
        row["mean_normalized_entropy"] = float(np.mean(field_entropies))
        row["mean_evidence_overlap"] = float(np.mean(evidence_scores)) if evidence_scores else np.nan
        row["mean_model_confidence"] = pd.to_numeric(group["confidence"], errors="coerce").mean()

        missing_penalty = max(0, min_models - row["n_models"]) / max(min_models, 1)
        evidence_disagreement = 1.0 - row["mean_evidence_overlap"] if not np.isnan(row["mean_evidence_overlap"]) else 0.5
        priority_score = (
            0.45 * row["mean_categorical_disagreement"]
            + 0.30 * row["mean_normalized_entropy"]
            + 0.20 * evidence_disagreement
            + 0.05 * missing_penalty
        )
        row["human_review_priority_score"] = float(priority_score)
        if priority_score >= 0.55:
            row["human_review_priority"] = "high"
        elif priority_score >= 0.30:
            row["human_review_priority"] = "medium"
        else:
            row["human_review_priority"] = "low"
        case_rows.append(row)

    cases = pd.DataFrame(case_rows).sort_values(
        ["human_review_priority_score", "case_id"], ascending=[False, True]
    )
    cases.to_csv(output_dir / "case_consensus_and_priority.csv", index=False)

    queue_columns = [
        "case_id",
        "n_models",
        "human_review_priority",
        "human_review_priority_score",
        "mean_categorical_disagreement",
        "mean_normalized_entropy",
        "mean_evidence_overlap",
    ]
    cases[queue_columns].to_csv(output_dir / "human_review_queue.csv", index=False)

    summary = {
        "framework": "VetJudge",
        "version": "0.1.0",
        "cases": int(cases["case_id"].nunique()),
        "successful_predictions": int(len(frame)),
        "blinded_models": sorted(frame["blind_model_id"].unique().tolist()),
        "high_priority_cases": int((cases["human_review_priority"] == "high").sum()),
        "medium_priority_cases": int((cases["human_review_priority"] == "medium").sum()),
        "low_priority_cases": int((cases["human_review_priority"] == "low").sum()),
        "mean_pairwise_kappa_by_field": (
            kappa.groupby("field")["cohen_kappa"].mean().dropna().to_dict()
            if not kappa.empty
            else {}
        ),
        "mean_evidence_overlap_by_field": (
            evidence.groupby("field")["mean_token_jaccard"].mean().dropna().to_dict()
            if not evidence.empty
            else {}
        ),
        "blinding": {
            "enabled": True,
            "mapping_file": "model_identity_map.private.json",
            "note": "Keep the identity map private during human review and adjudication.",
        },
    }
    (output_dir / "vetjudge_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "model_identity_map.private.json").write_text(
        json.dumps(model_map, indent=2, sort_keys=True), encoding="utf-8"
    )

    click.echo(f"VetJudge outputs written to {output_dir}")
    click.echo(f"Cases: {summary['cases']}; high-priority: {summary['high_priority_cases']}")


if __name__ == "__main__":
    main()
