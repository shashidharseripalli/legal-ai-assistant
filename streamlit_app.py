"""
streamlit_app.py
----------------
Streamlit frontend for the AI Legal Assistant.

Run AFTER starting the FastAPI backend:
  Terminal 1: uvicorn main:app --reload --port 8000
  Terminal 2: streamlit run streamlit_app.py

Two tabs:
  1. Legal Q&A   — ask questions, get cited answers, real-time streaming
  2. Legal Advice — describe a problem, get structured step-by-step advice
"""

import streamlit as st
import requests

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
API_BASE    = "http://localhost:8000/api/v1"
CHAT_URL    = f"{API_BASE}/chat"
STREAM_URL  = f"{API_BASE}/chat/stream"
ADVICE_URL  = f"{API_BASE}/advice"
HEALTH_URL  = "http://localhost:8000/health"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0d1b2a !important; }
    .stApp p, .stApp div, .stApp span, .stApp label,
    .stMarkdown, .stMarkdown p { color: #e8e8e8 !important; }
    section[data-testid="stSidebar"] { background-color: #112236 !important; }
    section[data-testid="stSidebar"] * { color: #e8e8e8 !important; }
    .stChatMessage { border-radius: 10px; margin-bottom: 8px; }
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #1a4a7a !important;
    }
    .stChatMessage[data-testid="stChatMessageUser"] p { color: #ffffff !important; }
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: #162535 !important;
        border: 1px solid #2a4a6a !important;
    }
    .stChatMessage[data-testid="stChatMessageAssistant"] p,
    .stChatMessage[data-testid="stChatMessageAssistant"] li,
    .stChatMessage[data-testid="stChatMessageAssistant"] span { color: #e8e8e8 !important; }
    .stChatInput textarea {
        background-color: #162535 !important;
        color: #e8e8e8 !important;
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
    details { background-color: #162535 !important; border: 1px solid #2a4a6a !important; }
    details summary { color: #e8e8e8 !important; }
    details > div { background-color: #0d1b2a !important; color: #e8e8e8 !important; }
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
    hr { border-color: #2a4a6a !important; }
    h1, h2, h3 { color: #e8e8e8 !important; }
    .stCaption, small { color: #a0b4c8 !important; }
    .stTextArea textarea {
        background-color: #162535 !important;
        color: #e8e8e8 !important;
        border: 1px solid #2a4a6a !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources_history" not in st.session_state:
    st.session_state.sources_history = {}


# ── Helpers ───────────────────────────────────────────────────────────────────
def check_health() -> bool:
    try:
        r = requests.get(HEALTH_URL, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def stream_answer(question: str):
    try:
        with requests.post(STREAM_URL, json={"question": question}, stream=True, timeout=120) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    yield chunk
    except Exception as e:
        yield f"⚠️ Streaming error: {e}"


def get_answer_with_sources(question: str) -> tuple[str, list]:
    try:
        r = requests.post(CHAT_URL, json={"question": question}, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("answer", ""), data.get("sources", [])
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to backend. Is FastAPI running on port 8000?", []
    except Exception as e:
        return f"❌ Error: {str(e)}", []


def get_advice(problem: str) -> tuple[str, list]:
    try:
        r = requests.post(ADVICE_URL, json={"problem": problem}, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("advice", ""), data.get("sources", [])
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to backend. Is FastAPI running on port 8000?", []
    except Exception as e:
        return f"❌ Error: {str(e)}", []


def render_sources(sources: list):
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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/55/Emblem_of_India.svg", width=60)
    st.title("⚖️ Legal Assistant")
    st.caption("Groq · Llama-3.1 · ChromaDB · LangChain")
    st.divider()

    st.markdown('<div class="sidebar-header">🔌 API Status</div>', unsafe_allow_html=True)
    api_ok = check_health()
    if api_ok:
        st.markdown('<span class="status-online">● Online</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-offline">● Offline — start FastAPI first</span>', unsafe_allow_html=True)
        st.code("uvicorn main:app --reload --port 8000", language="bash")

    st.divider()

    st.markdown('<div class="sidebar-header">⚙️ Settings</div>', unsafe_allow_html=True)
    use_streaming = st.toggle("Stream tokens", value=True)

    st.divider()

    st.markdown('<div class="sidebar-header">💡 Sample Questions</div>', unsafe_allow_html=True)
    samples = [
        "What is Article 21 of the Constitution?",
        "What is the punishment for murder under IPC?",
        "What are Fundamental Rights in India?",
        "Explain Section 420 of IPC.",
        "What does Article 14 say about equality?",
        "What is the punishment for theft under IPC?",
        "What are Directive Principles of State Policy?",
        "Explain Section 302 of IPC.",
    ]
    for i, q in enumerate(samples):
        if st.button(q, use_container_width=True, key=f"sample_{i}"):
            st.session_state["prefill_question"] = q

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources_history = {}
        st.rerun()

    st.caption("⚠️ Educational use only. Not legal advice.")


# ── Header ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    st.title("⚖️ AI Legal Assistant")
    st.caption("Ask about the **Constitution of India** and **Indian Penal Code (IPC)**")
with col2:
    st.success("API Online", icon="✅") if api_ok else st.error("API Offline", icon="🔴")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💬 Ask a Legal Question", "🆘 Describe My Problem"])


# ── TAB 1: Q&A ────────────────────────────────────────────────────────────────
with tab1:
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "⚖️"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and i in st.session_state.sources_history:
                render_sources(st.session_state.sources_history[i])

    prefill = st.session_state.pop("prefill_question", "")
    question = st.chat_input("Ask about Indian law (e.g. 'What is Article 21?')")
    if prefill and not question:
        question = prefill

    if question:
        if not api_ok:
            st.error("⚠️ Start FastAPI first: `uvicorn main:app --reload --port 8000`")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="👤"):
            st.markdown(question)

        with st.chat_message("assistant", avatar="⚖️"):
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


# ── TAB 2: Legal Advice ───────────────────────────────────────────────────────
with tab2:
    st.markdown("### 🆘 Describe Your Legal Problem")
    st.caption("Describe your situation in plain language. The AI will classify it and give structured advice.")

    example_problems = [
        "My landlord locked me out without notice and won't return my deposit.",
        "I was arrested without being told the reason. What are my rights?",
        "Someone is threatening me online and demanding money.",
        "My employer hasn't paid my salary for 3 months.",
    ]

    st.markdown("**Example problems:**")
    cols = st.columns(2)
    for i, prob in enumerate(example_problems):
        with cols[i % 2]:
            if st.button(prob, use_container_width=True, key=f"prob_{i}"):
                st.session_state["prefill_problem"] = prob

    prefill_prob = st.session_state.pop("prefill_problem", "")
    problem = st.text_area(
        "Your situation:",
        value=prefill_prob,
        height=120,
        placeholder="Describe your legal situation here...",
    )

    if st.button("⚖️ Get Legal Advice", use_container_width=True, type="primary"):
        if not problem.strip():
            st.warning("Please describe your problem first.")
        elif not api_ok:
            st.error("⚠️ Start FastAPI first: `uvicorn main:app --reload --port 8000`")
        else:
            with st.spinner("Analyzing your situation..."):
                advice, sources = get_advice(problem)
            st.markdown("---")
            st.markdown(advice)
            render_sources(sources)
