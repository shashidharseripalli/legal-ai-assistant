"""
streamlit_app.py
----------------
Streamlit frontend for the AI Legal Assistant.

Connects to the FastAPI backend at http://localhost:8000.
Run AFTER starting the FastAPI server:

  Terminal 1:  ollama serve
  Terminal 2:  uvicorn main:app --reload --port 8000
  Terminal 3:  streamlit run streamlit_app.py

Features:
  - Chat interface with message history
  - Real-time streaming responses (token by token)
  - Source citations panel for every answer
  - Sidebar with quick sample questions
  - Session-based conversation memory
"""

import streamlit as st
import requests
import json

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
API_BASE   = "http://localhost:8000/api/v1"
CHAT_URL   = f"{API_BASE}/chat"
STREAM_URL = f"{API_BASE}/chat/stream"
HEALTH_URL = "http://localhost:8000/health"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background — dark navy */
    .stApp { background-color: #0d1b2a !important; }

    /* Global text colour */
    .stApp p, .stApp div, .stApp span, .stApp label,
    .stMarkdown, .stMarkdown p { color: #e8e8e8 !important; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #112236 !important; }
    section[data-testid="stSidebar"] * { color: #e8e8e8 !important; }

    /* Chat messages container */
    .stChatMessage { border-radius: 10px; margin-bottom: 8px; }

    /* User bubble */
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #1a4a7a !important;
    }
    .stChatMessage[data-testid="stChatMessageUser"] p { color: #ffffff !important; }

    /* Assistant bubble */
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: #162535 !important;
        border: 1px solid #2a4a6a !important;
    }
    .stChatMessage[data-testid="stChatMessageAssistant"] p,
    .stChatMessage[data-testid="stChatMessageAssistant"] li,
    .stChatMessage[data-testid="stChatMessageAssistant"] span { color: #e8e8e8 !important; }

    /* Chat input */
    .stChatInput textarea {
        background-color: #162535 !important;
        color: #e8e8e8 !important;
        border: 1px solid #2a4a6a !important;
    }

    /* Buttons */
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

    /* Expander */
    details { background-color: #162535 !important; border: 1px solid #2a4a6a !important; }
    details summary { color: #e8e8e8 !important; }
    details > div { background-color: #0d1b2a !important; color: #e8e8e8 !important; }

    /* Source badge */
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

    /* Status */
    .status-online  { color: #4ecb82 !important; font-weight: 600; }
    .status-offline { color: #ff6b6b !important; font-weight: 600; }

    /* Sidebar section headers */
    .sidebar-header {
        font-weight: 700;
        color: #c8922a;
        border-bottom: 2px solid #c8922a;
        padding-bottom: 4px;
        margin-bottom: 8px;
    }

    /* Divider */
    hr { border-color: #2a4a6a !important; }

    /* Title and captions */
    h1, h2, h3 { color: #e8e8e8 !important; }
    .stCaption, small { color: #a0b4c8 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources_history" not in st.session_state:
    st.session_state.sources_history = {}  # maps message index → sources list


# ── Helper: check API health ──────────────────────────────────────────────────
def check_api_health() -> bool:
    try:
        r = requests.get(HEALTH_URL, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ── Helper: stream response from FastAPI ─────────────────────────────────────
def stream_answer(question: str):
    """
    Calls /chat/stream and yields tokens one by one.
    Falls back to /chat if streaming fails.
    """
    try:
        with requests.post(
            STREAM_URL,
            json={"question": question},
            stream=True,
            timeout=120,
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    yield chunk
    except Exception as e:
        yield f"⚠️ Streaming error: {e}. Retrying without streaming...\n\n"
        # Fallback to non-streaming
        try:
            r = requests.post(CHAT_URL, json={"question": question}, timeout=120)
            data = r.json()
            yield data.get("answer", "No answer returned.")
        except Exception as e2:
            yield f"❌ API error: {e2}"


# ── Helper: get answer + sources (non-streaming) ──────────────────────────────
def get_answer_with_sources(question: str) -> tuple[str, list]:
    """Returns (answer_text, sources_list)"""
    try:
        r = requests.post(CHAT_URL, json={"question": question}, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("answer", ""), data.get("sources", [])
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to FastAPI backend. Is it running on port 8000?", []
    except Exception as e:
        return f"❌ Error: {str(e)}", []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/55/Emblem_of_India.svg", width=60)
    st.title("⚖️ Legal Assistant")
    st.caption("Powered by Mistral · ChromaDB · LangChain")

    st.divider()

    # API Status
    st.markdown('<div class="sidebar-header">🔌 API Status</div>', unsafe_allow_html=True)
    api_ok = check_api_health()
    if api_ok:
        st.markdown('<span class="status-online">● Online — FastAPI running</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-offline">● Offline — Start FastAPI first</span>', unsafe_allow_html=True)
        st.code("uvicorn main:app --reload --port 8000", language="bash")

    st.divider()

    # Streaming toggle
    st.markdown('<div class="sidebar-header">⚙️ Settings</div>', unsafe_allow_html=True)
    use_streaming = st.toggle("Stream response tokens", value=True,
                              help="See the answer appear word by word. Disable for faster citation display.")

    st.divider()

    # Sample questions
    st.markdown('<div class="sidebar-header">💡 Sample Questions</div>', unsafe_allow_html=True)

    sample_questions = [
        "What is Article 21 of the Constitution?",
        "What is the punishment for murder under IPC?",
        "What are Fundamental Rights in India?",
        "Explain Section 420 of IPC.",
        "What does Article 14 say about equality?",
        "What is the punishment for theft under IPC?",
        "What are Directive Principles of State Policy?",
        "Explain Section 302 of IPC.",
    ]

    for i, q in enumerate(sample_questions):
        if st.button(q, use_container_width=True, key=f"sample_{i}"):
            st.session_state["prefill_question"] = q

    st.divider()

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources_history = {}
        st.rerun()

    st.caption("⚠️ For educational use only. Not legal advice.")


# ── Main header ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    st.title("⚖️ AI Legal Assistant")
    st.caption("Ask questions about the **Constitution of India** and **Indian Penal Code (IPC)**")

with col2:
    if api_ok:
        st.success("API Online", icon="✅")
    else:
        st.error("API Offline", icon="🔴")

st.divider()


# ── Chat history display ──────────────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "⚖️"):
        st.markdown(msg["content"])

        # Show sources under assistant messages
        if msg["role"] == "assistant" and i in st.session_state.sources_history:
            sources = st.session_state.sources_history[i]
            if sources:
                with st.expander(f"📚 Sources ({len(sources)} documents referenced)", expanded=False):
                    seen = set()
                    for src in sources:
                        key = f"{src['source_file']}|{src['page']}"
                        if key not in seen:
                            seen.add(key)
                            st.markdown(
                                f'<span class="source-badge">📄 {src["source_file"]} · Page {src["page"]}</span>',
                                unsafe_allow_html=True,
                            )


# ── Chat input ────────────────────────────────────────────────────────────────
# Pre-fill from sidebar sample question click
prefill = st.session_state.pop("prefill_question", "")

question = st.chat_input(
    "Ask about Indian law (e.g. 'What is Article 21?')",
    key="chat_input",
)

# Use prefill if sidebar button was clicked
if prefill and not question:
    question = prefill

if question:
    if not api_ok:
        st.error("⚠️ FastAPI backend is not running. Start it first:\n```\nuvicorn main:app --reload --port 8000\n```")
        st.stop()

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    # Generate assistant response
    with st.chat_message("assistant", avatar="⚖️"):
        if use_streaming:
            # ── Streaming mode ────────────────────────────────────────────
            full_answer = st.write_stream(stream_answer(question))

            # After streaming finishes, fetch sources separately
            with st.spinner("Fetching citations..."):
                _, sources = get_answer_with_sources(question)

        else:
            # ── Non-streaming mode ────────────────────────────────────────
            with st.spinner("Consulting legal documents..."):
                full_answer, sources = get_answer_with_sources(question)
            st.markdown(full_answer)

        # Display sources inline
        if sources:
            with st.expander(f"📚 Sources ({len(sources)} documents referenced)", expanded=True):
                seen = set()
                for src in sources:
                    key = f"{src['source_file']}|{src['page']}"
                    if key not in seen:
                        seen.add(key)
                        st.markdown(
                            f'<span class="source-badge">📄 {src["source_file"]} · Page {src["page"]}</span>',
                            unsafe_allow_html=True,
                        )

    # Save to history
    msg_index = len(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": full_answer})
    st.session_state.sources_history[msg_index] = sources