"""SoldrinAI RAG Chatbot API (FastAPI). Stateless, so it runs on Vercel serverless.

Flow: /api/parse (file -> units) -> /api/embed (batched) ... everything else
(semantic chunking, BM25 + cosine retrieval and the LLM) runs in the visitor's browser.
Nothing is stored: files and text only live in memory for the length of a request.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))  # sibling imports on Vercel

import engine  # noqa: E402
import files  # noqa: E402

MAX_EMBED_TEXTS = 128
MAX_EMBED_CHARS = 1000

app = FastAPI(title="SoldrinAI RAG Chatbot API", docs_url="/api/docs", openapi_url="/api/openapi.json")


class EmbedIn(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=MAX_EMBED_TEXTS)


@app.get("/api/health")
def health():
    return {"status": "ok", "model": engine.MODEL_NAME}


@app.get("/api/limits")
def limits():
    return {
        "maxUploadBytes": files.MAX_UPLOAD, "extensions": files.SUPPORTED,
        "maxRows": files.MAX_ROWS, "maxColumns": files.MAX_COLUMNS,
        "maxChars": files.MAX_CHARS, "maxPdfPages": files.MAX_PDF_PAGES,
    }


@app.post("/api/parse")
async def parse(file: UploadFile = File(...)):
    content = await file.read(files.MAX_UPLOAD + 1)
    if len(content) > files.MAX_UPLOAD:
        raise HTTPException(413, "File is too large. Keep uploads under 4 MB.")
    return files.parse(file.filename or "document", content)


@app.post("/api/embed")
def embed(body: EmbedIn):
    texts = [t[:MAX_EMBED_CHARS] for t in body.texts]
    return {"vectors": engine.encode_vectors(engine.embed(texts)), "count": len(texts)}
