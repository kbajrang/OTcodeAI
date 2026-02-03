from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

DEFAULT_SAMPLE_PATH = (
    r"C:\Users\bkailasa\Desktop\zenworks\zenworks-rest-api\zenworks-rest-core"
)

st.set_page_config(page_title="GraphRAG Code Intelligence", layout="wide")

st.title("GraphRAG Code Intelligence")

api_base_url = st.sidebar.text_input(
    "API base URL",
    value=os.getenv("OTCODEAI_API_URL", "http://localhost:8000"),
)


def _api_url(path: str) -> str:
    return f"{api_base_url}{path}"


def _safe_json(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"raw": response.text}


with st.sidebar:
    st.subheader("Backend")
    try:
        health = requests.get(_api_url("/health"), timeout=3)
        if health.ok:
            st.success("API reachable")
        else:
            st.error(f"API error: {health.status_code}")
    except requests.RequestException as exc:
        st.error(f"API unreachable: {exc}")
        st.caption("Start it with `.\\scripts\\run_all.ps1` (starts API + UI) or run `python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000`.")

tabs = st.tabs(["Workspace", "Index", "Ask"])

with tabs[0]:
    st.subheader("Modules")

    col_left, col_right = st.columns(2)
    with col_left:
        source_path = st.text_input("Source path", value=DEFAULT_SAMPLE_PATH)
    with col_right:
        module_name = st.text_input("Module name", value="zenworks-rest-core")

    overwrite = st.checkbox("Overwrite if exists", value=False)
    if st.button("Import module"):
        try:
            resp = requests.post(
                _api_url("/modules/import"),
                json={
                    "source_path": source_path,
                    "module_name": module_name,
                    "overwrite": overwrite,
                },
                timeout=300,
            )
            if resp.ok:
                st.success("Imported module")
                st.json(_safe_json(resp))
            else:
                st.error(_safe_json(resp))
        except requests.RequestException as exc:
            st.error(str(exc))

    if st.button("Refresh modules"):
        pass

    try:
        resp = requests.get(_api_url("/modules"), timeout=10)
        if resp.ok:
            modules = resp.json()
            st.dataframe(modules, use_container_width=True)
        else:
            st.error(_safe_json(resp))
    except requests.RequestException as exc:
        st.error(str(exc))

with tabs[1]:
    st.subheader("Indexing")

    repo_path = st.text_input(
        "Repo/workspace path to index",
        value="modules",
        help="Use 'modules' with Workspace=true to index all imported modules together.",
    )
    workspace = st.checkbox("Workspace (treat each top-level folder as a module)", value=True)
    attach_module = st.text_input(
        "Module name (optional for single repo)",
        value="",
        help="If Workspace is false, you can attach a module name for disambiguation.",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Start indexing"):
            payload = {
                "repo_path": repo_path,
                "workspace": workspace,
                "module_name": attach_module.strip() or None,
                "reset_logs": True,
            }
            try:
                resp = requests.post(_api_url("/index"), json=payload, timeout=10)
                st.json(_safe_json(resp))
            except requests.RequestException as exc:
                st.error(str(exc))
    with col_b:
        if st.button("Refresh status"):
            pass

    try:
        status_resp = requests.get(_api_url("/index/status"), timeout=10)
        status_json = _safe_json(status_resp)
        if status_resp.ok:
            st.json({k: v for k, v in status_json.items() if k != "logs"})
            logs = status_json.get("logs", [])
            if logs:
                st.text_area("Indexer logs", value="\n".join(logs[-400:]), height=300)
        else:
            st.error(status_json)
    except requests.RequestException as exc:
        st.error(str(exc))

with tabs[2]:
    st.subheader("Ask")

    question = st.text_area("Question", height=120)
    col_k, col_debug, col_btn = st.columns([1, 1, 2])
    with col_k:
        k = st.number_input("Top-K", min_value=1, max_value=25, value=5, step=1)
    with col_debug:
        debug = st.checkbox("Show retrieved context", value=True)
    with col_btn:
        ask = st.button("Ask")

    if ask and question.strip():
        try:
            response = requests.post(
                _api_url("/query"),
                json={"question": question, "k": int(k), "debug": debug},
                timeout=None,
            )
            if not response.ok:
                st.error(_safe_json(response))
            data = _safe_json(response)
            st.markdown("### Answer")
            st.write(data.get("answer"))

            if debug:
                retrieved = data.get("retrieved", [])
                meta = data.get("meta", {})
                with st.expander("Retrieval metadata", expanded=False):
                    st.json(meta)
                with st.expander("Retrieved items", expanded=True):
                    st.dataframe(retrieved, use_container_width=True)
                with st.expander("Context sent to Ollama", expanded=False):
                    st.text_area("Context", value=data.get("context", ""), height=300)
                with st.expander("Prompt (preview)", expanded=False):
                    prompt = data.get("prompt", "")
                    st.text_area("Prompt", value=prompt[:20000], height=300)
        except requests.RequestException as exc:
            st.error(f"Backend not reachable at {api_base_url}: {exc}")
