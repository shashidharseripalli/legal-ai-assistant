# ⚖️ AI Legal Assistant

A production-quality RAG (Retrieval-Augmented Generation) application for Indian law.
Ask legal questions or describe your problem — get grounded answers with citations.

**Sources:** Constitution of India · Indian Penal Code (IPC)  
**LLM:** Groq · Llama-3.1-8b-instant (free API, no RAM crash)  
**Embeddings:** Ollama · nomic-embed-text (local, lightweight)  
**Stack:** FastAPI · LangChain · ChromaDB · Streamlit · Python 3.10+

---

## ✨ Features

- 💬 **Legal Q&A Chat** — Ask anything about the Constitution or IPC, get cited answers
- 🆘 **Legal Problem Advisor** — Describe your situation, get structured step-by-step advice
- ⚡ **Streaming responses** — See answers appear token by token in real time
- 📚 **Source citations** — Every answer shows which PDF and page it came from
- ☁️ **Groq API** — Fast cloud LLM, free tier, no local GPU or RAM needed
- 🖥️ **Streamlit UI** — Clean dark-themed chat interface with sidebar sample questions

---

## 📁 Project Structure

```
legal-ai-assistant/
│
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── chat.py              ← FastAPI routes (/chat, /chat/stream, /advice)
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py            ← All settings loaded from .env
│   └── __init__.py
│
├── rag/
│   ├── __init__.py
│   ├── vector.py                ← ChromaDB singleton + MMR retriever
│   └── rag_pipeline.py          ← RAG chain + legal Q&A prompt + advice prompt
│
├── data/
│   └── raw/
│       ├── constitution_of_india.pdf   ← ADD YOUR PDF HERE
│       └── ipc.pdf                     ← ADD YOUR PDF HERE
│
├── scripts/
│   └── ingest_documents.py      ← Run once to chunk, embed, and store PDFs
│
├── vectorstore/                 ← Auto-created by ChromaDB after ingestion
│
├── streamlit_app.py             ← Streamlit frontend
├── main.py                      ← FastAPI app entry point
├── .env                         ← Your local environment variables (not committed)
├── .env.example                 ← Template — copy this to .env
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Run

### 1. Get a free Groq API key
Go to **https://console.groq.com** → Sign up → API Keys → Create key.
Takes 2 minutes. Free tier is generous.

### 2. Install Ollama (for embeddings only)
Download from **https://ollama.com** and install.
On Windows it runs automatically in the background after installation.

### 3. Pull the embedding model
```bash
ollama pull nomic-embed-text
```
This is only ~270MB and uses very little RAM. It's only for embeddings, not the LLM.

### 4. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/legal-ai-assistant.git
cd legal-ai-assistant
```

### 5. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 6. Install dependencies
```bash
pip install -r requirements.txt
```

### 7. Set up environment variables
```bash
# Windows
copy .env.example .env

# Mac / Linux
cp .env.example .env
```

Open `.env` and add your Groq API key:
```
GROQ_API_KEY=gsk_your_actual_key_here
```

### 8. Add your PDF documents
Place the following files inside `data/raw/`:
- `constitution_of_india.pdf`
- `ipc.pdf`

### 9. Run document ingestion (ONE TIME only)
```bash
python scripts/ingest_documents.py
```
This chunks the PDFs, embeds them with `nomic-embed-text`, and stores them in ChromaDB.
Re-run only if you add new documents.

### 10. Start the FastAPI backend
```bash
# Terminal 1
uvicorn main:app --reload --port 8000
```
Wait until you see: `✅ Legal AI Assistant is ready!`

### 11. Start the Streamlit frontend
```bash
# Terminal 2
streamlit run streamlit_app.py
```
Opens at **http://localhost:8501**

---

## 🖥️ Streamlit UI

Two tabs at **http://localhost:8501**:

### Tab 1 — 💬 Ask a Legal Question
- Type any question about Indian law
- Get a cited answer grounded in the Constitution / IPC
- Real-time streaming (token by token)
- Source panel shows which PDF + page was used
- 8 sample questions in the sidebar to get started

### Tab 2 — 🆘 Describe My Problem
- Describe your legal situation in plain language
- Returns structured advice in this format:
  - 🔍 Case Type
  - 📋 Relevant Laws (specific Sections / Articles)
  - ⚖️ Your Legal Situation (explained simply)
  - ✅ Steps To Take Right Now
  - ⚠️ Important Warnings
  - 👨‍⚖️ Type of Lawyer to Consult

---

## 🔌 API Endpoints

All endpoints at `http://localhost:8000`  
Interactive docs: **http://localhost:8000/docs**

### POST `/api/v1/chat`
Ask a legal question — returns full answer + source citations.

**Request:**
```json
{ "question": "What does Article 21 say about the right to life?" }
```

**Response:**
```json
{
  "answer": "Article 21 of the Constitution states...",
  "sources": [
    { "source_file": "constitution_of_india.pdf", "page": 14 }
  ]
}
```

### POST `/api/v1/chat/stream`
Same as `/chat` but streams tokens in real time as plain text.

### POST `/api/v1/advice`
Describe a legal problem — returns structured advice.

**Request:**
```json
{ "problem": "My landlord locked me out without notice and won't return my deposit." }
```

### GET `/health`
```json
{
  "status": "ok",
  "llm": "llama-3.1-8b-instant (groq)",
  "embeddings": "nomic-embed-text (ollama)",
  "vectorstore": "chromadb"
}
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| LLM | Groq · Llama-3.1-8b-instant | Free, fast, no RAM crash |
| Embeddings | Ollama · nomic-embed-text | Lightweight, runs locally |
| RAG Framework | LangChain (LCEL) | Clean chain composition |
| Vector Database | ChromaDB (persistent) | Simple, local, no setup |
| Backend API | FastAPI + Uvicorn | Async, auto-docs |
| Frontend UI | Streamlit | Fast to build, looks good |
| PDF Loading | PyPDFLoader | Reliable PDF parsing |
| Chunking | RecursiveCharacterTextSplitter | Legal-aware chunking |

---

## 🔄 Why Groq Instead of Ollama?

The original version used Ollama to run Mistral locally, which requires 4.5GB RAM.
This caused a crash on machines with less available memory:

```
ValueError: model requires more system memory (4.5 GiB) than is available (3.0 GiB)
```

**Solution:** Groq runs Llama-3.1 on their custom hardware in the cloud.
- ✅ Free tier — no cost for normal usage
- ✅ Fast — 2-5s responses vs 15-40s local
- ✅ No RAM usage — LLM runs on Groq's servers
- ✅ Ollama still used for embeddings (tiny, ~200MB RAM, no problem)

---

## ⚠️ Disclaimer

This tool provides **general legal information only** and is intended for educational purposes.
It is **not a substitute for professional legal advice**.
Always consult a qualified lawyer for your specific legal situation.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## 📄 License

MIT License
