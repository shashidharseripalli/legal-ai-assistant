# ⚖️ AI Legal Assistant

A production-quality RAG (Retrieval-Augmented Generation) application for Indian law.
Ask legal questions or describe your problem — get grounded answers with citations, powered entirely by a local AI model.

**Sources:** Constitution of India · Indian Penal Code (IPC)  
**Model:** Mistral (via Ollama — runs 100% locally, no API key needed)  
**Stack:** FastAPI · LangChain · ChromaDB · Streamlit · Python 3.10+

---

## ✨ Features

- 💬 **Legal Q&A Chat** — Ask anything about the Constitution or IPC, get cited answers
- 🆘 **Legal Problem Advisor** — Describe your criminal or civil problem, get structured step-by-step advice
- ⚡ **Streaming responses** — See answers appear token by token in real time
- 📚 **Source citations** — Every answer shows which PDF and page it came from
- 🔒 **100% local** — Nothing leaves your machine; no OpenAI, no cloud API
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
├── streamlit_app.py             ← Streamlit frontend (chat UI + advisor UI)
├── main.py                      ← FastAPI app entry point
├── .env                         ← Your local environment variables (not committed)
├── .env.example                 ← Template — copy this to .env
├── requirements.txt             ← All dependencies
└── README.md
```

---

## 🚀 Setup & Run

### 1. Install Ollama
Download and install from https://ollama.com  
On Windows, Ollama runs automatically in the background after installation — you do **not** need to run `ollama serve` manually.

### 2. Pull required models
```bash
ollama pull mistral
ollama pull nomic-embed-text
```

> `mistral` — the LLM that generates answers  
> `nomic-embed-text` — the embedding model for semantic search (do NOT skip this)

### 3. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/legal-ai-assistant.git
cd legal-ai-assistant
```

### 4. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 5. Install dependencies
```bash
pip install -r requirements.txt
pip install pydantic-settings
```

### 6. Set up environment variables
```bash
# Windows
copy .env.example .env

# Mac / Linux
cp .env.example .env
```
Open `.env` and confirm the values match your setup (defaults work out of the box).

### 7. Add your PDF documents
Place the following files inside `data/raw/`:
- `constitution_of_india.pdf`
- `ipc.pdf`

### 8. Run document ingestion (ONE TIME only)
```bash
python scripts/ingest_documents.py
```
This chunks the PDFs, embeds them with `nomic-embed-text`, and stores them in ChromaDB.  
Re-run this only if you add new documents.

### 9. Start the FastAPI backend
```bash
# Terminal 1
uvicorn main:app --reload --port 8000
```
Wait until you see: `✅ Legal AI Assistant is ready!`

### 10. Start the Streamlit frontend
```bash
# Terminal 2
streamlit run streamlit_app.py
```
Opens at **http://localhost:8501**

---

## 🖥️ Streamlit UI

Two tabs available at **http://localhost:8501**:

### Tab 1 — 💬 Ask a Legal Question
- Type any question about Indian law
- Get a cited answer grounded in the Constitution / IPC
- Real-time streaming (token by token)
- Source panel shows which PDF + page was used
- 8 sample questions in the sidebar to get started

### Tab 2 — 🆘 Describe My Problem
- Describe your criminal or civil legal situation in plain language
- AI classifies your case (Criminal / Civil / Mixed)
- Returns structured advice in this format:
  - 🔍 Case Type
  - 📋 Relevant Laws (specific Sections / Articles)
  - ⚖️ Your Legal Situation (explained simply)
  - ✅ Steps To Take Right Now
  - ⚠️ Important Warnings (deadlines, risks)
  - 👨‍⚖️ Type of Lawyer to Consult
- 4 example problems to click and test

---

## 🔌 API Endpoints

All endpoints available at `http://localhost:8000`  
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
  "answer": "Article 21 of the Constitution states that no person shall be deprived of their life or personal liberty except according to procedure established by law...",
  "sources": [
    { "source_file": "constitution_of_india.pdf", "page": 14 }
  ]
}
```

### POST `/api/v1/chat/stream`
Same as `/chat` but streams tokens in real time as plain text.

### POST `/api/v1/advice`
Describe a legal problem — returns structured advice with case classification.

**Request:**
```json
{ "problem": "My landlord locked me out without notice and won't return my deposit." }
```

**Response:**
```json
{
  "advice": "🔍 CASE TYPE: Civil\n\n📋 RELEVANT LAWS:\n...",
  "sources": [
    { "source_file": "ipc.pdf", "page": 87 }
  ]
}
```

### GET `/health`
```json
{ "status": "ok", "model": "mistral", "vectorstore": "chromadb" }
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Ollama · Mistral 7B (local) |
| Embeddings | Ollama · nomic-embed-text (local) |
| RAG Framework | LangChain (LCEL) |
| Vector Database | ChromaDB (persistent, local) |
| Backend API | FastAPI + Uvicorn |
| Frontend UI | Streamlit |
| PDF Loading | PyPDFLoader (LangChain) |
| Chunking | RecursiveCharacterTextSplitter (legal-aware) |

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
