from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import click
import pandas as pd
import requests
import yaml

SYSTEM_PROMPT = """You are annotating de-identified veterinary clinical text using the 2020 Five Domains Model.
Return JSON only. Do not diagnose beyond the note. Absence of mention is unknown, not normal.
Domains 1-4 status: negative, positive, mixed, neutral, unknown.
Mental state: likely_negative, likely_positive, mixed, uncertain, unknown.
Severity: none, mild, moderate, severe, unknown.
Overall valence: positive, negative, mixed, neutral, unknown.
Evidence must be the shortest exact phrase copied from the note. Use an empty list when unsupported.
"""

SCHEMA = {
    "nutrition_status": "unknown",
    "nutrition_evidence": [],
    "environment_status": "unknown",
    "environment_evidence": [],
    "health_status": "unknown",
    "health_evidence": [],
    "behaviour_status": "unknown",
    "behaviour_evidence": [],
    "mental_state": "unknown",
    "mental_state_evidence": [],
    "overall_valence": "unknown",
    "severity": "unknown",
    "confidence": 0.0,
    "rationale_short": "",
}


def _call_openai_compatible(model_cfg: dict[str, Any], note: str, temperature: float, max_tokens: int) -> dict[str, Any]:
    base_url = os.getenv(model_cfg["base_url_env"], "").rstrip("/")
    api_key = os.getenv(model_cfg["api_key_env"], "")
    if not base_url:
        raise RuntimeError(f"Missing {model_cfg['base_url_env']} for {model_cfg['id']}")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model_cfg["model"],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Clinical note:\n{note}\n\nReturn all fields in this template:\n{json.dumps(SCHEMA)}"},
        ],
    }
    response = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=180)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    return {**SCHEMA, **parsed}


@click.command()
@click.option("--input-csv", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), default=Path("configs/model_ensemble.yaml"), show_default=True)
@click.option("--output", type=click.Path(path_type=Path), default=Path("outputs/model_prefill/predictions.jsonl"), show_default=True)
@click.option("--text-column", default="sentence", show_default=True)
@click.option("--id-column", default="annotation_id", show_default=True)
@click.option("--limit", type=int, default=None)
@click.option("--resume/--no-resume", default=True, show_default=True)
def main(input_csv: Path, config_path: Path, output: Path, text_column: str, id_column: str, limit: int | None, resume: bool) -> None:
    """Run a fixed prompt across five OpenAI-compatible model endpoints."""
    cfg = yaml.safe_load(config_path.read_text())
    df = pd.read_csv(input_csv)
    if limit:
        df = df.head(limit)
    output.parent.mkdir(parents=True, exist_ok=True)

    completed: set[tuple[str, str]] = set()
    if resume and output.exists():
        for line in output.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                completed.add((str(row["case_id"]), row["model_id"]))

    with output.open("a", encoding="utf-8") as handle:
        for _, row in df.iterrows():
            case_id = str(row[id_column])
            note = str(row[text_column])
            for model in cfg["models"]:
                key = (case_id, model["id"])
                if key in completed:
                    continue
                started = time.time()
                record: dict[str, Any] = {
                    "study_name": cfg["study_name"],
                    "prompt_version": cfg["prompt_version"],
                    "case_id": case_id,
                    "model_id": model["id"],
                    "model_family": model["family"],
                }
                try:
                    annotation = _call_openai_compatible(
                        model,
                        note,
                        float(cfg.get("temperature", 0.0)),
                        int(cfg.get("max_output_tokens", 1200)),
                    )
                    record.update({"status": "success", "annotation": annotation})
                except Exception as exc:  # continue ensemble despite one endpoint failure
                    record.update({"status": "error", "error": str(exc)})
                record["latency_seconds"] = round(time.time() - started, 3)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                click.echo(f"{case_id} | {model['id']} | {record['status']}")


if __name__ == "__main__":
    main()
