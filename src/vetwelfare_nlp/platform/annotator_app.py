from __future__ import annotations

import os, time
import requests
import streamlit as st

API = os.getenv("VETWELFARE_API_URL", "http://localhost:8000")
STATUS = ["unknown", "neutral", "positive", "negative", "mixed"]
MENTAL = ["unknown", "uncertain", "likely_positive", "likely_negative", "mixed"]

st.set_page_config(page_title="VetWelfare Annotator", layout="wide")
st.title("VetWelfare Annotator")
external_id = st.sidebar.text_input("Annotator ID")
name = st.sidebar.text_input("Name")
experience = st.sidebar.selectbox("Experience", ["veterinary_student", "veterinarian", "welfare_specialist", "other"])

if st.sidebar.button("Register") and external_id and name:
    requests.post(f"{API}/annotators", json={"external_id": external_id, "name": name, "experience_group": experience}, timeout=20).raise_for_status()
    st.sidebar.success("Registered")

if "started" not in st.session_state: st.session_state.started = time.time()
if external_id:
    data = requests.get(f"{API}/assignments/next/{external_id}", timeout=20).json().get("assignment")
    if not data:
        st.success("No pending assignments.")
        st.stop()
    st.caption(f"Case {data['source_id']} · Assignment {data['id']}")
    st.markdown(f"### Clinical narrative\n{data['sentence']}")
    with st.form("annotation"):
        payload = {}
        cols = st.columns(2)
        domains = [("nutrition", "Nutrition"), ("environment", "Physical environment"), ("health", "Health"), ("behaviour", "Behavioural interactions")]
        for i, (key, label) in enumerate(domains):
            with cols[i % 2]:
                payload[f"{key}_status"] = st.selectbox(label, STATUS, key=f"{key}_status")
                payload[f"{key}_evidence"] = st.text_area(f"{label} evidence", key=f"{key}_evidence")
        payload["mental_state"] = st.selectbox("Mental state", MENTAL)
        payload["mental_state_evidence"] = st.text_area("Mental-state evidence")
        payload["overall_valence"] = st.selectbox("Overall valence", ["unknown", "neutral", "positive", "negative", "mixed"])
        payload["severity"] = st.selectbox("Severity", ["unknown", "none", "mild", "moderate", "severe"])
        confidence = st.slider("Confidence", 0.0, 1.0, 0.8, 0.01)
        payload["notes"] = st.text_area("Notes")
        if st.form_submit_button("Submit & next"):
            body = {"payload": payload, "confidence": confidence, "duration_seconds": time.time() - st.session_state.started}
            response = requests.post(f"{API}/assignments/{data['id']}/annotation", json=body, timeout=20)
            response.raise_for_status(); st.session_state.started = time.time(); st.rerun()
else:
    st.info("Enter your annotator ID in the sidebar.")
