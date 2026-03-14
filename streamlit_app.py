"""
streamlit_app.py
----------------
Streamlit frontend for the AI Legal Assistant.

Run after starting the FastAPI backend:
  Terminal 1: uvicorn main:app --reload --port 8000
  Terminal 2: streamlit run streamlit_app.py
"""

import os
from uuid import uuid4

import requests
import streamlit as st

# Page config
st.set_page_config(
    page_title="AI Legal Assistant",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8000").rstrip("/")
API_BASE = f"{BACKEND_BASE_URL}/api/v1"
CHAT_URL = f"{API_BASE}/chat"
STREAM_URL = f"{API_BASE}/chat/stream"
ADVICE_URL = f"{API_BASE}/advice"
UPLOAD_URL = f"{API_BASE}/documents/upload"
IMPACT_URL = f"{API_BASE}/documents/impact"
HEALTH_URL = f"{BACKEND_BASE_URL}/health"

# CSS
st.markdown(
    """
<style>
    .stApp { background-color: #0d1b2a !important; }
    .stApp p, .stApp div, .stApp span, .stApp label,
    .stMarkdown, .stMarkdown p { color: #e8e8e8 !important; }
    section[data-testid="stSidebar"] { background-color: #112236 !important; }
    section[data-testid="stSidebar"] * { color: #e8e8e8 !important; }
    .stChatMessage { border-radius: 10px; margin-bottom: 8px; }
    .stChatMessage[data-testid="stChatMessageUser"] { background-color: #1a4a7a !important; }
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: #162535 !important;
        border: 1px solid #2a4a6a !important;
    }
    .stButton > button {
        background-color: #1a3a5c !important;
        color: #e8e8e8 !important;
        border: 1px solid #2a5a8c !important;
    }
    .stButton > button:hover {
        background-color: #c8922a !important;
        color: #ffffff !important;
        border-color: #c8922a !important;
    }
    .source-badge {
        display: inline-block;
        background: #1a3a5c;
        color: #90c8f0;
        border: 1px solid #2a5a8c;
        border-radius: 4px;
        padding: 2px 10px;
        font-size: 12px;
        margin: 3px 4px 3px 0;
        font-family: monospace;
    }
    .status-online  { color: #4ecb82 !important; font-weight: 600; }
    .status-offline { color: #ff6b6b !important; font-weight: 600; }
    .sidebar-header {
        font-weight: 700;
        color: #c8922a;
        border-bottom: 2px solid #c8922a;
        padding-bottom: 4px;
        margin-bottom: 8px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources_history" not in st.session_state:
    st.session_state.sources_history = {}
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []


# Helpers
def check_health() -> bool:
    try:
        response = requests.get(HEALTH_URL, timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def stream_answer(question: str):
    try:
        with requests.post(STREAM_URL, json={"question": question}, stream=True, timeout=120) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    yield chunk
    except Exception as exc:
        yield f"Streaming error: {exc}"


def get_answer_with_sources(question: str) -> tuple[str, list]:
    try:
        response = requests.post(CHAT_URL, json={"question": question}, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("answer", ""), data.get("sources", [])
    except requests.exceptions.ConnectionError:
        return "Cannot connect to backend. Is FastAPI running on port 8000?", []
    except Exception as exc:
        return f"Error: {str(exc)}", []


def get_advice(problem: str) -> tuple[str, list]:
    try:
        response = requests.post(ADVICE_URL, json={"problem": problem}, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("advice", ""), data.get("sources", [])
    except requests.exceptions.ConnectionError:
        return "Cannot connect to backend. Is FastAPI running on port 8000?", []
    except Exception as exc:
        return f"Error: {str(exc)}", []


def upload_documents(files) -> tuple[bool, str]:
    try:
        payload = []
        for f in files:
            payload.append(("files", (f.name, f.getvalue(), f.type or "application/octet-stream")))

        response = requests.post(
            UPLOAD_URL,
            data={"session_id": st.session_state.session_id},
            files=payload,
            timeout=180,
        )
        response.raise_for_status()

        data = response.json()
        uploaded = data.get("uploaded", [])
        st.session_state.uploaded_docs.extend(uploaded)
        return True, f"Uploaded {len(uploaded)} file(s) successfully."
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to backend. Is FastAPI running on port 8000?"
    except Exception as exc:
        return False, f"Upload failed: {exc}"


def get_impact(problem: str, scope: str, selected_document_ids: list[str]) -> tuple[str, list]:
    payload = {
        "problem": problem,
        "session_id": st.session_state.session_id,
        "scope": scope,
        "document_ids": selected_document_ids or None,
    }

    try:
        response = requests.post(IMPACT_URL, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        return data.get("analysis", ""), data.get("sources", [])
    except requests.exceptions.ConnectionError:
        return "Cannot connect to backend. Is FastAPI running on port 8000?", []
    except Exception as exc:
        return f"Impact analysis failed: {exc}", []


def render_sources(sources: list):
    if sources:
        with st.expander(f"Sources ({len(sources)} documents referenced)", expanded=False):
            seen = set()
            for src in sources:
                key = f"{src['source_file']}|{src['page']}"
                if key in seen:
                    continue
                seen.add(key)
                st.markdown(
                    f'<span class="source-badge">File: {src["source_file"]} | Page {src["page"]}</span>',
                    unsafe_allow_html=True,
                )


# Sidebar
with st.sidebar:
    st.title("Legal Assistant")
    st.caption("Groq | Llama-3.1 | ChromaDB | LangChain")
    st.divider()

    st.markdown('<div class="sidebar-header">API Status</div>', unsafe_allow_html=True)
    api_ok = check_health()
    if api_ok:
        st.markdown('<span class="status-online">Online</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-offline">Offline - start FastAPI first</span>', unsafe_allow_html=True)
        st.code("uvicorn main:app --reload --port 8000", language="bash")

    st.divider()
    st.markdown('<div class="sidebar-header">Session</div>', unsafe_allow_html=True)
    st.code(st.session_state.session_id)

    if st.button("Reset Upload Session", use_container_width=True):
        st.session_state.session_id = str(uuid4())
        st.session_state.uploaded_docs = []
        st.rerun()

    st.divider()
    st.markdown('<div class="sidebar-header">Settings</div>', unsafe_allow_html=True)
    use_streaming = st.checkbox("Stream tokens", value=True)

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources_history = {}
        st.rerun()

    st.caption("Educational use only. Not legal advice.")


# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("AI Legal Assistant")
    st.caption("Ask about Indian law and analyze uploaded documents")
with col2:
    if api_ok:
        st.success("API Online")
    else:
        st.error("API Offline")

st.divider()

# Tabs
tab1, tab2, tab3 = st.tabs(["Ask a Legal Question", "Describe My Problem", "Upload & Impact"])


# Tab 1: Q&A
with tab1:
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and i in st.session_state.sources_history:
                render_sources(st.session_state.sources_history[i])

    question = st.chat_input("Ask about Indian law (for example, 'What is Article 21?')")

    if question:
        if not api_ok:
            st.error("Start FastAPI first: uvicorn main:app --reload --port 8000")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if use_streaming:
                full_answer = st.write_stream(stream_answer(question))
                with st.spinner("Fetching citations..."):
                    _, sources = get_answer_with_sources(question)
            else:
                with st.spinner("Consulting legal documents..."):
                    full_answer, sources = get_answer_with_sources(question)
                st.markdown(full_answer)
            render_sources(sources)

        idx = len(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": full_answer})
        st.session_state.sources_history[idx] = sources


# Tab 2: Advice
with tab2:
    st.markdown("### Describe Your Legal Problem")
    problem = st.text_area(
        "Your situation:",
        height=120,
        placeholder="Describe your legal situation here...",
    )

    if st.button("Get Legal Advice", use_container_width=True, type="primary"):
        if not problem.strip():
            st.warning("Please describe your problem first.")
        elif not api_ok:
            st.error("Start FastAPI first: uvicorn main:app --reload --port 8000")
        else:
            with st.spinner("Analyzing your situation..."):
                advice, sources = get_advice(problem)
            st.markdown("---")
            st.markdown(advice)
            render_sources(sources)


# Tab 3: Upload and Impact
with tab3:
    st.markdown("### Upload Documents")
    st.caption("Supported files: PDF, TXT")

    uploads = st.file_uploader(
        "Select legal documents",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Upload Selected Documents", use_container_width=True):
        if not uploads:
            st.warning("Please choose at least one file.")
        elif not api_ok:
            st.error("Start FastAPI first: uvicorn main:app --reload --port 8000")
        else:
            with st.spinner("Uploading and indexing documents..."):
                ok, message = upload_documents(uploads)
            if ok:
                st.success(message)
            else:
                st.error(message)

    st.markdown("### Uploaded Documents")
    if st.session_state.uploaded_docs:
        for item in st.session_state.uploaded_docs:
            st.write(f"- {item['file_name']} ({item['chunks_indexed']} chunks)")
    else:
        st.info("No uploaded documents in this session yet.")

    st.markdown("### Analyze Impact")
    impact_problem = st.text_area(
        "Problem description for impact analysis:",
        height=120,
        placeholder="Explain your issue and how these documents may relate...",
        key="impact_problem",
    )

    scope_map = {
        "Hybrid (Legal corpus + uploaded docs)": "hybrid",
        "Uploaded docs only": "uploaded_docs_only",
        "Legal corpus only": "legal_corpus_only",
    }
    scope_label = st.selectbox("Analysis scope", list(scope_map.keys()))
    scope_value = scope_map[scope_label]

    selected_doc_ids = []
    if st.session_state.uploaded_docs:
        id_to_name = {item["document_id"]: item["file_name"] for item in st.session_state.uploaded_docs}
        selected_doc_ids = st.multiselect(
            "Limit analysis to selected uploaded documents (optional)",
            options=list(id_to_name.keys()),
            format_func=lambda x: id_to_name.get(x, x),
        )

    if st.button("Run Impact Analysis", use_container_width=True, type="primary"):
        if not impact_problem.strip():
            st.warning("Please describe the problem first.")
        elif not api_ok:
            st.error("Start FastAPI first: uvicorn main:app --reload --port 8000")
        else:
            with st.spinner("Running impact analysis..."):
                analysis, sources = get_impact(impact_problem, scope_value, selected_doc_ids)
            st.markdown("---")
            st.markdown(analysis)
            render_sources(sources)
