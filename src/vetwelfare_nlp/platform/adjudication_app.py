from __future__ import annotations

import os
import requests
import streamlit as st

API = os.getenv("VETWELFARE_API_URL", "http://localhost:8000")

st.set_page_config(page_title="VetWelfare Adjudication", layout="wide")
st.title("VetWelfare Adjudication Dashboard")
adjudicator_id = st.sidebar.text_input("Adjudicator ID")
queue = requests.get(f"{API}/adjudication/queue", timeout=20).json()
if not queue:
    st.success("No cases currently ready for adjudication.")
    st.stop()

labels = [f"{x['source_id']} ({x['annotation_count']} annotations)" for x in queue]
selected = queue[st.selectbox("Case", range(len(queue)), format_func=lambda i: labels[i])]
st.markdown(f"### Clinical narrative\n{selected['sentence']}")
annotations = requests.get(f"{API}/cases/{selected['case_id']}/annotations", timeout=20).json()
for i, ann in enumerate(annotations, start=1):
    with st.expander(f"Annotation {i} · confidence={ann.get('confidence')}", expanded=True):
        st.json(ann["payload"])

st.subheader("Consensus annotation")
consensus = st.text_area("Consensus JSON", value="{}", height=320)
notes = st.text_area("Adjudication notes")
if st.button("Save adjudication"):
    import json
    try:
        payload = json.loads(consensus)
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON: {exc}")
    else:
        response = requests.post(
            f"{API}/cases/{selected['case_id']}/adjudicate",
            json={"adjudicator_external_id": adjudicator_id, "consensus_payload": payload, "notes": notes},
            timeout=20,
        )
        if response.ok:
            st.success("Adjudication saved.")
            st.rerun()
        else:
            st.error(response.text)
