from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .db import Adjudication, Annotation, Annotator, Assignment, Case, SessionLocal, init_db

app = FastAPI(title="VetWelfare Collaborative Annotation API", version="0.1.0")


class AnnotatorIn(BaseModel):
    external_id: str
    name: str
    role: str = "annotator"
    experience_group: str = "unspecified"


class AnnotationIn(BaseModel):
    payload: dict[str, Any]
    confidence: float | None = None
    guideline_version: str = "1.0"
    duration_seconds: float | None = None


class AdjudicationIn(BaseModel):
    adjudicator_external_id: str
    consensus_payload: dict[str, Any]
    notes: str = ""


def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/annotators")
def create_annotator(item: AnnotatorIn, db: Session = Depends(db_session)):
    existing = db.scalar(select(Annotator).where(Annotator.external_id == item.external_id))
    if existing:
        return existing
    obj = Annotator(**item.model_dump())
    db.add(obj); db.commit(); db.refresh(obj)
    return obj


@app.get("/assignments/next/{external_id}")
def next_assignment(external_id: str, db: Session = Depends(db_session)):
    annotator = db.scalar(select(Annotator).where(Annotator.external_id == external_id))
    if not annotator:
        raise HTTPException(404, "Annotator not registered")
    assignment = db.scalar(select(Assignment).where(Assignment.annotator_id == annotator.id, Assignment.status == "assigned").order_by(Assignment.id))
    if not assignment:
        return {"assignment": None}
    case = db.get(Case, assignment.case_id)
    return {"assignment": {"id": assignment.id, "case_id": case.id, "source_id": case.source_id, "sentence": case.sentence, "metadata": case.metadata_json}}


@app.post("/assignments/{assignment_id}/annotation")
def submit_annotation(assignment_id: int, item: AnnotationIn, db: Session = Depends(db_session)):
    assignment = db.get(Assignment, assignment_id)
    if not assignment:
        raise HTTPException(404, "Assignment not found")
    annotation = db.scalar(select(Annotation).where(Annotation.assignment_id == assignment_id))
    if annotation:
        for key, value in item.model_dump().items(): setattr(annotation, key, value)
    else:
        annotation = Annotation(assignment_id=assignment_id, **item.model_dump()); db.add(annotation)
    assignment.status = "completed"; assignment.completed_at = datetime.utcnow()
    db.commit(); db.refresh(annotation)
    return {"annotation_id": annotation.id, "status": assignment.status}


@app.get("/adjudication/queue")
def adjudication_queue(db: Session = Depends(db_session)):
    rows = db.execute(select(Case.id, Case.source_id, Case.sentence, func.count(Annotation.id).label("n")).join(Assignment, Assignment.case_id == Case.id).join(Annotation, Annotation.assignment_id == Assignment.id).outerjoin(Adjudication, Adjudication.case_id == Case.id).where(Adjudication.id.is_(None)).group_by(Case.id).having(func.count(Annotation.id) >= 2)).all()
    return [{"case_id": r.id, "source_id": r.source_id, "sentence": r.sentence, "annotation_count": r.n} for r in rows]


@app.get("/cases/{case_id}/annotations")
def case_annotations(case_id: int, db: Session = Depends(db_session)):
    rows = db.scalars(select(Annotation).join(Assignment).where(Assignment.case_id == case_id)).all()
    return [{"id": x.id, "payload": x.payload, "confidence": x.confidence} for x in rows]


@app.post("/cases/{case_id}/adjudicate")
def adjudicate(case_id: int, item: AdjudicationIn, db: Session = Depends(db_session)):
    adjudicator = db.scalar(select(Annotator).where(Annotator.external_id == item.adjudicator_external_id))
    if not adjudicator: raise HTTPException(404, "Adjudicator not registered")
    obj = Adjudication(case_id=case_id, adjudicator_id=adjudicator.id, consensus_payload=item.consensus_payload, notes=item.notes)
    db.add(obj); db.commit(); db.refresh(obj)
    return {"adjudication_id": obj.id}
