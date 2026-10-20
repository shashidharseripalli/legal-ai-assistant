---
title: AI Legal Assistant
emoji: ⚖️
colorFrom: blue
colorTo: purple
sdk: static
app_file: index.html
fullWidth: true
---

# AI Legal Assistant — Static Demo

This free Static Space publishes a browser-based interface. Static Spaces do not run the repository's Python FastAPI, ChromaDB, Streamlit, or document ingestion code.

## Connect a working backend

Deploy the Python backend separately, then paste its HTTPS base URL into the web page. The API should support:

- `GET /health`
- `POST /api/v1/chat` with `{"question":"..."}`
- `POST /api/v1/advice` with `{"problem":"..."}`

The API must configure CORS to allow this Space's origin. Keep `GROQ_API_KEY` only in the backend host's secrets/environment; never put it in this static repository or browser JavaScript.

## Free static hosting

Hugging Face Static Spaces are served as static assets without compute, so this page itself is free to host. That does **not** make the Python AI backend run for free here. If you require an all-in-one live Python backend with no paid subscription, use a host that currently offers an eligible free application runtime, or run the backend locally. Review provider limits before deploying.

Source code: https://github.com/shashidharseripalli/legal-ai-assistant

**Disclaimer:** educational legal information only, not legal advice. Verify current law with official publications and qualified counsel.
