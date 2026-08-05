from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

FIELDS = [
    "nutrition_status",
    "environment_status",
    "health_status",
    "behaviour_status",
    "mental_state",
    "overall_valence",
    "severity",
]


def _pairwise_kappa(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    wide = frame.pivot(index="case_id", columns="model_id", values=field)
    models = list(wide.columns)
    rows = []
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            pair = wide[[a, b]].dropna()
            score = cohen_kappa_score(pair[a], pair[b]) if len(pair) else np.nan
            rows.append({"field": field, "model_a": a, "model_b": b, "n": len(pair), "cohen_kappa": score})
    return pd.DataFrame(rows)


@click.command()
@click.option("--predictions", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("outputs/model_agreement"), show_default=True)
def main(predictions: Path, output_dir: Path) -> None:
    """Analyze five-model categorical agreement and consensus uncertainty."""
    records = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
    rows = []
    for record in records:
        if record.get("status") != "success":
            continue
        ann = record["annotation"]
        row = {"case_id": record["case_id"], "model_id": record["model_id"]}
        row.update({field: ann.get(field, "unknown") for field in FIELDS})
        rows.append(row)
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "model_predictions_flat.csv", index=False)

    kappa = pd.concat([_pairwise_kappa(frame, field) for field in FIELDS], ignore_index=True)
    kappa.to_csv(output_dir / "pairwise_kappa.csv", index=False)

    consensus_rows = []
    for case_id, group in frame.groupby("case_id"):
        result = {"case_id": case_id, "n_models": group["model_id"].nunique()}
        disagreement = []
        for field in FIELDS:
            values = group[field].fillna("unknown").tolist()
            counts = Counter(values)
            label, votes = counts.most_common(1)[0]
            agreement = votes / len(values)
            result[f"{field}_consensus"] = label
            result[f"{field}_agreement"] = agreement
            result[f"{field}_vote_distribution"] = json.dumps(dict(counts), sort_keys=True)
            disagreement.append(1 - agreement)
        result["mean_disagreement"] = float(np.mean(disagreement))
        result["priority_for_human_review"] = "high" if result["mean_disagreement"] >= 0.35 else ("medium" if result["mean_disagreement"] >= 0.15 else "low")
        consensus_rows.append(result)
    consensus = pd.DataFrame(consensus_rows).sort_values("mean_disagreement", ascending=False)
    consensus.to_csv(output_dir / "case_consensus.csv", index=False)

    summary = {
        "successful_predictions": int(len(frame)),
        "cases": int(frame["case_id"].nunique()) if not frame.empty else 0,
        "models": sorted(frame["model_id"].unique().tolist()) if not frame.empty else [],
        "mean_pairwise_kappa_by_field": kappa.groupby("field")["cohen_kappa"].mean().dropna().to_dict(),
        "high_disagreement_cases": int((consensus["priority_for_human_review"] == "high").sum()) if not consensus.empty else 0,
    }
    (output_dir / "agreement_summary.json").write_text(json.dumps(summary, indent=2))
    click.echo(f"Agreement outputs written to {output_dir}")


if __name__ == "__main__":
    main()
