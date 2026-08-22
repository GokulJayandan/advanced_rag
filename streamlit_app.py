"""
Streamlit UI for Advanced Agentic RAG Platform.

Provides an interactive dashboard for document ingestion, query execution,
and inspectable execution traces (citations, evidence quality, agent decisions).
"""

from __future__ import annotations

import os
import time
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Advanced Agentic RAG Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .citation-badge {
        display: inline-block;
        background-color: #EEF2FF;
        color: #4F46E5;
        border: 1px solid #C7D2FE;
        border-radius: 4px;
        padding: 2px 6px;
        font-size: 0.85rem;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .badge-valid {
        border-color: #86EFAC;
        background-color: #F0FDF4;
        color: #166534;
    }
    .badge-invalid {
        border-color: #FCA5A5;
        background-color: #FEF2F2;
        color: #991B1B;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_url" not in st.session_state:
    st.session_state.api_url = os.getenv("API_URL", "http://127.0.0.1:8000")

API_TIMEOUT_HEALTH = 10    # seconds – quick ping
API_TIMEOUT_INGEST = 30    # seconds – just to queue the job (fast)
API_TIMEOUT_POLL   = 10    # seconds – polling status
API_TIMEOUT_QUERY  = 180   # seconds – LLM + reranker can be slow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_api_health(api_url: str):
    try:
        res = requests.get(f"{api_url}/health", timeout=API_TIMEOUT_HEALTH)
        return res.status_code == 200, res.json() if res.status_code == 200 else {}
    except Exception as e:
        return False, {"error": str(e)}


def get_stats(api_url: str):
    try:
        res = requests.get(f"{api_url}/stats", timeout=API_TIMEOUT_HEALTH)
        return res.json() if res.status_code == 200 else {}
    except Exception:
        return {}


def _poll_ingest_job(api_url: str, job_id: str, poll_interval: float = 2.0, max_wait: float = 600.0):
    """
    Poll /ingest/status/{job_id} until the job is done or failed.
    Returns (success: bool, result_or_error: dict).
    """
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            res = requests.get(f"{api_url}/ingest/status/{job_id}", timeout=API_TIMEOUT_POLL)
            if res.status_code != 200:
                return False, {"error": f"Status check failed: {res.status_code}"}
            data = res.json()
            status = data.get("status", "")
            if status == "done":
                return True, data.get("result", {})
            if status == "failed":
                return False, {"error": data.get("error", "Unknown error")}
            # Still queued / running – wait and retry
            time.sleep(poll_interval)
        except Exception as exc:
            return False, {"error": str(exc)}
    return False, {"error": f"Timed out waiting for ingestion after {max_wait}s"}


def _render_diagnostics_tabs(meta: dict):
    """Render the 4-tab diagnostics panel."""
    tabs = st.tabs(["Citations & Sources", "Evidence Quality", "Agent Decisions", "Metrics"])

    with tabs[0]:
        citations = meta.get("citations", [])
        if citations:
            st.markdown("**Inline Citations Extracted & Validated:**")
            for cit in citations:
                badge_class = "badge-valid" if cit.get("is_valid") else "badge-invalid"
                valid_label = "VALID" if cit.get("is_valid") else "INVALID"
                st.markdown(
                    f'<span class="citation-badge {badge_class}">[{valid_label}] {cit.get("citation_text")}</span>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No inline citations detected in response.")
        sources = meta.get("sources", [])
        if sources:
            st.markdown("**Referenced Sources:**")
            st.json(sources)

    with tabs[1]:
        eq = meta.get("evidence_quality", {})
        st.metric("Evidence Quality Score", f"{eq.get('score', 0.0):.2f}")
        st.write(f"**Sufficiency:** {'✅ Yes' if eq.get('is_sufficient') else '❌ No'}")
        st.write(f"**Explanation:** {eq.get('explanation', '')}")

    with tabs[2]:
        decisions = meta.get("agent_decisions", [])
        if decisions:
            for d in decisions:
                st.markdown(f"- **{d.get('step')}**: `{d.get('decision')}`\n  _{d.get('reasoning')}_")
        else:
            st.info("Direct execution path without sub-queries.")

    with tabs[3]:
        st.json({
            "Query ID": meta.get("query_id"),
            "Query Type": meta.get("query_type"),
            "Latency": f"{meta.get('total_latency_ms', 0):.2f} ms",
            "Retrieval Iterations": meta.get("retrieval_iterations", 1),
        })


# ---------------------------------------------------------------------------
# Sidebar: System Status & Document Management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ RAG Controller")
    api_url = st.text_input("API URL", value=st.session_state.api_url)
    st.session_state.api_url = api_url

    is_healthy, health_data = check_api_health(api_url)
    if is_healthy:
        st.success(f"Backend Connected (v{health_data.get('version', '1.0')})")
        providers = health_data.get("providers", [])
        if providers:
            st.caption(f"Active Providers: {', '.join(providers)}")
    else:
        st.error(f"Backend Disconnected ({health_data.get('error', 'unreachable')})")

    st.divider()

    st.subheader("📊 System Stats")
    stats = get_stats(api_url)
    col1, col2 = st.columns(2)
    col1.metric("Documents", stats.get("total_documents", 0))
    col2.metric("Total Chunks", stats.get("total_chunks", 0))
    st.caption(f"BM25 Indexed: {stats.get('bm25_indexed', 0)}")
    cache_info = stats.get("embedding_cache", {})
    if cache_info:
        st.caption(f"Embedding Cache Size: {cache_info.get('size', 0)} (Hits: {cache_info.get('hits', 0)})")

    st.divider()

    st.subheader("📄 Ingest Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT, or MD",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Ingest Files", use_container_width=True):
        progress_bar = st.progress(0)
        status_box = st.empty()
        total = len(uploaded_files)

        for idx, file in enumerate(uploaded_files):
            status_box.info(f"⏫ Uploading **{file.name}**…")
            files = {"file": (file.name, file.getvalue(), file.type)}

            try:
                # Step 1: Queue the job (returns almost instantly)
                res = requests.post(f"{api_url}/ingest", files=files, timeout=API_TIMEOUT_INGEST)
                if res.status_code != 200:
                    st.toast(f"❌ Upload failed for {file.name}: {res.text}", icon="⚠️")
                    progress_bar.progress((idx + 1) / total)
                    continue

                job_id = res.json()["job_id"]
                status_box.info(f"⚙️ Processing **{file.name}** (job `{job_id[:8]}…`)…")

                # Step 2: Poll until done
                ok, result = _poll_ingest_job(api_url, job_id)

                if ok:
                    chunks = result.get("total_chunks", "?")
                    already = result.get("already_exists", False)
                    label = "already indexed" if already else f"{chunks} chunks"
                    st.toast(f"✅ {file.name} → {label}", icon="📄")
                else:
                    st.toast(f"❌ {file.name} ingestion failed: {result.get('error')}", icon="⚠️")

            except Exception as e:
                st.error(f"Error uploading {file.name}: {e}")

            progress_bar.progress((idx + 1) / total)

        status_box.success("✅ All files processed!")
        time.sleep(1)
        st.rerun()


# ---------------------------------------------------------------------------
# Main Chat & Query Interface
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">🧠 Advanced Agentic RAG Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Hybrid Retrieval (Dense + BM25 + RRF) • Neural Reranking (BGE-v2-m3) • Adaptive Agent • Grounded Citations</div>',
    unsafe_allow_html=True,
)

# Display historical messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg:
            with st.expander("🔍 Diagnostics & Citations"):
                _render_diagnostics_tabs(msg["metadata"])

# User Input
if prompt := st.chat_input("Ask a question based on your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent analyzing, retrieving & verifying…"):
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            payload = {
                "query": prompt,
                "conversation_history": history,
                "include_diagnostics": True,
            }

            try:
                res = requests.post(
                    f"{st.session_state.api_url}/query",
                    json=payload,
                    timeout=API_TIMEOUT_QUERY,
                )

                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "No answer provided.")
                    st.markdown(answer)

                    meta_data = {
                        "query_id": data.get("query_id"),
                        "citations": data.get("citations", []),
                        "evidence_quality": data.get("evidence_quality", {}),
                        "sources": data.get("sources", []),
                        "query_type": data.get("query_type"),
                        "total_latency_ms": data.get("total_latency_ms"),
                        "retrieval_iterations": data.get("retrieval_iterations"),
                        "agent_decisions": data.get("agent_decisions", []),
                    }

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": meta_data,
                    })

                    with st.expander("🔍 Diagnostics & Citations"):
                        _render_diagnostics_tabs(meta_data)

                else:
                    error_msg = f"Error: Received status {res.status_code} — {res.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

            except requests.exceptions.Timeout:
                msg = (
                    "⏳ The query timed out after 3 minutes. "
                    "This usually means the BGE reranker model is still loading in the background. "
                    "Please wait 30–60 seconds and try again."
                )
                st.warning(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})

            except Exception as e:
                error_msg = f"Failed to connect to RAG API: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
