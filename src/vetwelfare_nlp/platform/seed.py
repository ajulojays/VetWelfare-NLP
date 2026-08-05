from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from .db import Annotator, Assignment, Case, SessionLocal, init_db


def parse_value(value):
    if pd.isna(value): return None
    if isinstance(value, str):
        try: return ast.literal_eval(value)
        except Exception: return value
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--annotations-per-case", type=int, default=3)
    parser.add_argument("--annotators", nargs="+", required=True, help="external_id:name:experience_group")
    args = parser.parse_args()

    init_db(); db = SessionLocal()
    try:
        annotators = []
        for spec in args.annotators:
            external_id, name, *rest = spec.split(":")
            experience = rest[0] if rest else "unspecified"
            obj = db.scalar(select(Annotator).where(Annotator.external_id == external_id))
            if not obj:
                obj = Annotator(external_id=external_id, name=name, experience_group=experience)
                db.add(obj); db.flush()
            annotators.append(obj)

        df = pd.read_csv(args.csv)
        cases = []
        for _, row in df.iterrows():
            source_id = str(row.get("annotation_id") or row.get("id"))
            obj = db.scalar(select(Case).where(Case.source_id == source_id))
            if not obj:
                meta = {k: parse_value(row[k]) for k in row.index if k not in {"sentence"}}
                obj = Case(source_id=source_id, sentence=str(row["sentence"]), metadata_json=meta, target_annotations=args.annotations_per_case)
                db.add(obj); db.flush()
            cases.append(obj)

        for i, case in enumerate(cases):
            for offset in range(args.annotations_per_case):
                annotator = annotators[(i + offset) % len(annotators)]
                exists = db.scalar(select(Assignment).where(Assignment.case_id == case.id, Assignment.annotator_id == annotator.id))
                if not exists: db.add(Assignment(case_id=case.id, annotator_id=annotator.id))
        db.commit()
        print(f"Loaded {len(cases)} cases and assigned {args.annotations_per_case} annotators per case.")
    finally:
        db.close()


if __name__ == "__main__": main()
