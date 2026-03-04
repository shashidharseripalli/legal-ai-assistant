# ⚖️ AI Legal Assistant

RAG-powered assistant for Indian law using FastAPI + LangChain + Ollama + ChromaDB.

**Sources:** Constitution of India · Indian Penal Code (IPC)  
**Model:** Mistral (via Ollama, runs 100% locally)

---

## 📁 Project Structure

```
legal-ai-assistant/
│
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── chat.py              ← FastAPI routes (/chat, /chat/stream)
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py            ← All settings from .env
│   └── __init__.py
│
├── rag/
│   ├── __init__.py
│   ├── vector.py                ← ChromaDB singleton + MMR retriever
│   └── rag_pipeline.py          ← LangChain RAG chain + legal prompt
│
├── data/
│   └── raw/
│       ├── constitution_of_india.pdf   ← ADD YOUR PDF HERE
│       └── ipc.pdf                     ← ADD YOUR PDF HERE
│
├── scripts/
│   └── ingest_documents.py      ← Run once to embed + store PDFs
│
├── vectorstore/                 ← Auto-created by ChromaDB
│
├── main.py                      ← FastAPI app entry point
├── .env                         ← Environment variables
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Run

### 1. Install Ollama
Download from https://ollama.com and install for Windows.

### 2. Pull required models
```bash
ollama pull mistral
ollama pull nomic-embed-text
```

### 3. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
pip install pydantic-settings  # required for config.py
```

### 5. Add your PDFs
Place these files in `data/raw/`:
- `constitution_of_india.pdf`
- `ipc.pdf`

### 6. Run ingestion (ONE TIME only)
```bash
python scripts/ingest_documents.py
```

### 7. Start Ollama server (Terminal 1)
```bash
ollama serve
```

### 8. Start the FastAPI backend (Terminal 2)
```bash
uvicorn main:app --reload --port 8000
```

### 9. Start the Streamlit frontend (Terminal 3)
```bash
streamlit run streamlit_app.py
```
Opens at http://localhost:8501

---

## 🖥️ Streamlit Frontend

Run the chat UI after the FastAPI server is up:

```bash
streamlit run streamlit_app.py
```

Opens at **http://localhost:8501**

**Features:**
- 💬 Full chat interface with message history
- ⚡ Real-time streaming responses (token by token)
- 📚 Source citations panel (shows which PDF + page answered the question)
- 💡 Sidebar with 8 sample questions to click
- 🔌 Live API status indicator

---

## 🔌 API Endpoints

### POST `/api/v1/chat`
Ask a question, get a full answer with source citations.

**Request:**
```json
{ "question": "What does Article 21 say about the right to life?" }
```

**Response:**
```json
{
  "answer": "Article 21 of the Constitution of India states that no person shall be deprived of their life or personal liberty except according to procedure established by law...",
  "sources": [
    { "source_file": "constitution_of_india.pdf", "page": 14 }
  ]
}
```

### POST `/api/v1/chat/stream`
Same question, but streams tokens as they are generated (better UX).

### GET `/health`
Returns `{ "status": "ok" }`

---

## 📖 Interactive Docs
Visit http://localhost:8000/docs after starting the server.
