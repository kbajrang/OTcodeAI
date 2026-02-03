import streamlit as st
import requests

API_URL = "http://localhost:8000/query"

st.title("GraphRAG Code Intelligence")
question = st.text_area("Ask a question about your codebase")

if st.button("Ask") and question.strip():
    try:
        response = requests.post(API_URL, json={"question": question}, timeout=60)
        response.raise_for_status()
        st.write(response.json().get("answer"))
    except requests.RequestException as exc:
        st.error(f"Backend not reachable at {API_URL}. Is the API running? ({exc})")
