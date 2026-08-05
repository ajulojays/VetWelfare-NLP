from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

DOMAIN_STATUS = ["unknown", "neutral", "positive", "negative", "mixed"]
MENTAL_STATE = ["unknown", "uncertain", "likely_positive", "likely_negative", "mixed"]
VALENCE = ["unknown", "neutral", "positive", "negative", "mixed"]
SEVERITY = ["unknown", "none", "mild", "moderate", "severe"]
REVIEW_STATUS = [
    "ai_prefilled_needs_human_review",
    "human_verified",
    "human_modified",
    "needs_discussion",
    "adjudicated",
]
ANNOTATION_COLUMNS = [
    "nutrition_status",
    "nutrition_evidence",
    "environment_status",
    "environment_evidence",
    "health_status",
    "health_evidence",
    "behaviour_status",
    "behaviour_evidence",
    "mental_state",
    "mental_state_evidence",
    "overall_valence",
    "severity",
    "annotator_confidence",
    "annotator_id",
    "review_status",
    "notes",
]


def _safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _select_index(options: list[str], value: Any) -> int:
    text = _safe_text(value).strip()
    return options.index(text) if text in options else 0


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "annotation_id" not in frame.columns or "sentence" not in frame.columns:
        raise ValueError("CSV must contain annotation_id and sentence columns")
    for column in ANNOTATION_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame


def save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def completed_mask(frame: pd.DataFrame) -> pd.Series:
    required = [
        "nutrition_status",
        "environment_status",
        "health_status",
        "behaviour_status",
        "mental_state",
        "overall_valence",
        "severity",
        "review_status",
    ]
    return frame[required].fillna("").apply(
        lambda row: all(str(value).strip() not in {"", "unknown", "ai_prefilled_needs_human_review"} for value in row),
        axis=1,
    )


def main() -> None:
    st.set_page_config(page_title="VetWelfare Annotator", page_icon="🐾", layout="wide")
    st.title("VetWelfare Annotator")
    st.caption("Evidence-grounded Five-Domains annotation for veterinary clinical narratives")

    with st.sidebar:
        st.header("Dataset")
        default_path = "data/annotation/calibration_25_annotated.csv"
        csv_path_text = st.text_input("Annotation CSV", value=default_path)
        csv_path = Path(csv_path_text).expanduser()
        annotator = st.text_input("Annotator ID", value="Samuel Ajulo")
        autosave = st.checkbox("Autosave on Save & Next", value=True)
        st.markdown("[Annotation guidelines](../docs/ANNOTATION_GUIDELINES.md)")

    try:
        frame = load_table(csv_path)
    except Exception as exc:
        st.error(f"Could not load {csv_path}: {exc}")
        st.stop()

    signature = f"{csv_path.resolve()}::{csv_path.stat().st_mtime_ns}"
    if st.session_state.get("dataset_signature") != signature:
        st.session_state.dataset_signature = signature
        st.session_state.frame = frame
        st.session_state.row_index = 0
    frame = st.session_state.frame

    done = completed_mask(frame)
    total = len(frame)
    completed = int(done.sum())

    with st.sidebar:
        st.metric("Completed", f"{completed}/{total}")
        st.progress(completed / max(total, 1))
        show_filter = st.selectbox("Show", ["all", "needs review", "completed"])

        if show_filter == "needs review":
            visible_indices = frame.index[~done].tolist()
        elif show_filter == "completed":
            visible_indices = frame.index[done].tolist()
        else:
            visible_indices = frame.index.tolist()
        if not visible_indices:
            st.success("No records match this filter.")
            st.stop()

        current = st.session_state.row_index
        if current not in visible_indices:
            current = visible_indices[0]
            st.session_state.row_index = current

        labels = [f"{frame.at[i, 'annotation_id']} — row {i + 1}" for i in visible_indices]
        current_position = visible_indices.index(current)
        selected_label = st.selectbox("Record", labels, index=current_position)
        selected = visible_indices[labels.index(selected_label)]
        if selected != current:
            st.session_state.row_index = selected
            st.rerun()

    idx = st.session_state.row_index
    row = frame.loc[idx]

    header_left, header_right = st.columns([3, 1])
    with header_left:
        st.subheader(f"{row['annotation_id']}")
    with header_right:
        st.write(f"Record {idx + 1} of {total}")

    metadata = {
        "id": _safe_text(row.get("id", "")),
        "primary_icd": _safe_text(row.get("primary_icd", "")),
        "length_bin": _safe_text(row.get("length_bin", "")),
        "word_count": _safe_text(row.get("word_count", "")),
        "has_disease_entity": _safe_text(row.get("has_disease_entity", "")),
    }
    with st.expander("Record metadata", expanded=False):
        st.json(metadata)
        if _safe_text(row.get("icd_label", "")):
            st.write("**ICD labels:**", _safe_text(row.get("icd_label", "")))
        if _safe_text(row.get("disease", "")):
            st.write("**Disease spans:**", _safe_text(row.get("disease", "")))

    st.markdown("### Clinical narrative")
    st.info(_safe_text(row["sentence"]))
    st.caption("Copy the shortest exact supporting phrase into each evidence field. Missing information is unknown, not normal.")

    with st.form(key=f"annotation_form_{idx}"):
        domain_columns = st.columns(2)
        values: dict[str, Any] = {}

        domain_specs = [
            ("Nutrition", "nutrition_status", "nutrition_evidence"),
            ("Physical Environment", "environment_status", "environment_evidence"),
            ("Health", "health_status", "health_evidence"),
            ("Behavioural Interactions", "behaviour_status", "behaviour_evidence"),
        ]
        for position, (title, status_column, evidence_column) in enumerate(domain_specs):
            with domain_columns[position % 2]:
                st.markdown(f"#### {title}")
                values[status_column] = st.selectbox(
                    "Status",
                    DOMAIN_STATUS,
                    index=_select_index(DOMAIN_STATUS, row[status_column]),
                    key=f"{status_column}_{idx}",
                )
                values[evidence_column] = st.text_area(
                    "Exact evidence spans (separate multiple spans with ` | `)",
                    value=_safe_text(row[evidence_column]),
                    key=f"{evidence_column}_{idx}",
                    height=90,
                )

        st.divider()
        mental_left, mental_right = st.columns(2)
        with mental_left:
            st.markdown("#### Mental State")
            values["mental_state"] = st.selectbox(
                "Inferred state",
                MENTAL_STATE,
                index=_select_index(MENTAL_STATE, row["mental_state"]),
            )
            values["mental_state_evidence"] = st.text_area(
                "Supporting evidence from Domains 1–4",
                value=_safe_text(row["mental_state_evidence"]),
                height=90,
            )
        with mental_right:
            st.markdown("#### Overall assessment")
            values["overall_valence"] = st.selectbox(
                "Overall valence",
                VALENCE,
                index=_select_index(VALENCE, row["overall_valence"]),
            )
            values["severity"] = st.selectbox(
                "Severity",
                SEVERITY,
                index=_select_index(SEVERITY, row["severity"]),
            )
            confidence_default = pd.to_numeric(pd.Series([row["annotator_confidence"]]), errors="coerce").iloc[0]
            if pd.isna(confidence_default):
                confidence_default = 0.80
            values["annotator_confidence"] = st.slider(
                "Annotator confidence",
                min_value=0.0,
                max_value=1.0,
                value=float(max(0.0, min(1.0, confidence_default))),
                step=0.01,
            )

        values["review_status"] = st.selectbox(
            "Review status",
            REVIEW_STATUS,
            index=_select_index(REVIEW_STATUS, row["review_status"]),
        )
        values["notes"] = st.text_area("Notes / ambiguity", value=_safe_text(row["notes"]), height=80)

        previous_col, save_col, next_col = st.columns([1, 2, 1])
        previous_clicked = previous_col.form_submit_button("← Previous", use_container_width=True)
        save_clicked = save_col.form_submit_button("Save", type="primary", use_container_width=True)
        next_clicked = next_col.form_submit_button("Save & Next →", use_container_width=True)

    def persist() -> None:
        for column, value in values.items():
            frame.at[idx, column] = value
        frame.at[idx, "annotator_id"] = annotator.strip() or "unknown"
        st.session_state.frame = frame
        save_table(frame, csv_path)

    if previous_clicked:
        persist()
        position = visible_indices.index(idx)
        st.session_state.row_index = visible_indices[max(0, position - 1)]
        st.rerun()
    if save_clicked:
        persist()
        st.success(f"Saved {row['annotation_id']} to {csv_path}")
    if next_clicked:
        persist()
        position = visible_indices.index(idx)
        st.session_state.row_index = visible_indices[min(len(visible_indices) - 1, position + 1)]
        st.rerun()

    st.divider()
    export_col, manifest_col = st.columns(2)
    with export_col:
        csv_bytes = frame.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download current CSV",
            data=csv_bytes,
            file_name=csv_path.name,
            mime="text/csv",
            use_container_width=True,
        )
    with manifest_col:
        manifest = {
            "source": str(csv_path),
            "rows": total,
            "completed": completed,
            "annotator": annotator,
            "review_status_counts": frame["review_status"].fillna("").value_counts().to_dict(),
        }
        st.download_button(
            "Download annotation manifest",
            data=json.dumps(manifest, indent=2),
            file_name=f"{csv_path.stem}.manifest.json",
            mime="application/json",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
