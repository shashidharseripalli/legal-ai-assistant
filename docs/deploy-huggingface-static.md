# Hugging Face Spaces (Static SDK) deployment

This repository's `index.html` is a static public demo. The project backend (`main.py`, FastAPI, ChromaDB and RAG chain) does not run in a `sdk: static` Space. This is an intentional boundary: Hugging Face says Static Spaces are served directly without compute, while Python/Docker Spaces require compute and may require a paid subscription. Read the current [Static Spaces docs](https://huggingface.co/docs/hub/main/spaces-sdks-static).

## Deploy the public static page for free

1. Sign in/create a free account at https://huggingface.co.
2. Open https://huggingface.co/new-space.
3. Set Space name to `legal-ai-assistant`.
4. Select **Static** as SDK and **Public** as visibility, then create it.
5. In that Space's **Files** page, use the repository's static files below (or clone/push the Space Git repo):
   - `README.md` from this deployment folder (contains `sdk: static` YAML metadata)
   - `index.html`
6. If using git, follow the clone command shown by Hugging Face. Copy these two files into that Space repository, then:
   ```bash
   git add README.md index.html
   git commit -m "Deploy Legal AI Assistant static frontend"
   git push
   ```
7. Wait for the Space's build/status to show running. Public URL will be:
   `https://huggingface.co/spaces/YOUR_HF_USERNAME/legal-ai-assistant`
   Replace `YOUR_HF_USERNAME` with your Hugging Face username.

## To enable real AI answers

The static page is only the UI. The Python API must live somewhere else and be reachable over HTTPS. For a backend host:

1. Deploy this GitHub repo using a Python service/container runtime (FastAPI startup command: `uvicorn main:app --host 0.0.0.0 --port $PORT`, adapting to the platform's required port).
2. Add `GROQ_API_KEY` to that host's secret environment variables; never commit it or place it in `index.html`.
3. Download the official PDFs and build the Chroma index on the backend's available storage:
   ```bash
   python scripts/download_legal_corpus.py
   python scripts/ingest_documents.py
   ```
   Storage is rebuildable/non-persistent in this plan. A host restart may require rebuilding the index.
4. Configure FastAPI CORS to permit the exact Space origin: `https://huggingface.co` is not the right origin for the iframe in every context; use the actual public Space URL origin shown in the browser address bar and test it. Do not use `allow_origins=["*"]` together with credentials.
5. Enter the backend HTTPS base URL in the static page and select **Check connection**.

## Important limits

- No secret belongs in a public Space or GitHub repository.
- Free/static web hosting does not provide Python runtime for this backend.
- Public users can access and send input to any connected API; set rate limits and upload size/request controls before sharing widely.
- Because uploaded documents/vector indexes live on local disk in this current app, treat them as ephemeral on a non-persistent backend host.
- Legal answers are educational and must be verified; they do not replace qualified legal advice.
