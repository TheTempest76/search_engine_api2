# server.py
import os
import json
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Your existing modules
from faiss_embedding import load_faiss_index, search_faiss, model as embed_model
from llmthrow import answer_query_with_context  # uses Gemini

# ---------- Config ----------
INDEX_DIR = os.getenv("INDEX_DIR", "index")
INDEX_PATH = os.path.join(INDEX_DIR, "index.bin")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.json")

# CORS for browser usage
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# ---------- App ----------
app = FastAPI(title="RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- In-memory state ----------
faiss_index = None
all_chunks: List[str] = []

# ---------- Models ----------
class RagRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: int = Field(8, ge=1, le=64)
    format_json: bool = True  # pass through to your Gemini prompt (JSON vs natural reply)

class RagSource(BaseModel):
    rank: int
    score: float
    preview: str

class RagResponse(BaseModel):
    answer: str
    sources: List[RagSource]

# ---------- Helpers ----------
def _load_chunks_or_fail():
    global all_chunks
    if not Path(CHUNKS_PATH).exists():
        raise RuntimeError(
            f"Chunk file not found at {CHUNKS_PATH}. "
            "Run `python load.py` so it writes index/index.bin and index/chunks.json."
        )
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
    if not isinstance(all_chunks, list) or not all_chunks:
        raise RuntimeError("chunks.json is empty or invalid.")

def _load_index_or_fail():
    global faiss_index
    if not Path(INDEX_PATH).exists():
        raise RuntimeError(
            f"FAISS index not found at {INDEX_PATH}. "
            "Run `python load.py` to build it."
        )
    faiss_index = load_faiss_index(INDEX_PATH)

def _ensure_ready():
    if not all_chunks:
        _load_chunks_or_fail()
    if faiss_index is None:
        _load_index_or_fail()

# ---------- Startup ----------
@app.on_event("startup")
def startup_event():
    try:
        _ensure_ready()
        # Touch the embed model once so first query is fast
        _ = embed_model.encode(["warmup"], convert_to_numpy=True)
        print(f"Loaded FAISS index and {len(all_chunks)} chunks.")
    except Exception as e:
        # Don't crash startup—surface via /health
        print(f"[Startup warning] {e}")

# ---------- Routes ----------
@app.get("/health")
def health():
    status = {
        "index_loaded": faiss_index is not None,
        "chunks_loaded": bool(all_chunks),
        "chunks_count": len(all_chunks) if all_chunks else 0,
        "index_path": INDEX_PATH,
        "chunks_path": CHUNKS_PATH,
    }
    return {"ok": status["index_loaded"] and status["chunks_loaded"], **status}

@app.post("/rag", response_model=RagResponse)
def rag(req: RagRequest):
    try:
        _ensure_ready()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 1) retrieve top-k chunks by vector search
    try:
        hits = search_faiss(req.query, all_chunks, faiss_index, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    if not hits:
        return RagResponse(answer="No relevant context found.", sources=[])

    # 2) generate grounded answer with Gemini (your llmthrow.py)
    try:
        answer = answer_query_with_context(req.query, hits, format_json=req.format_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    # Prepare source previews (these are the raw chunks returned by search_faiss)
    sources = []
    for i, ch in enumerate(hits, start=1):
        preview = (ch[:240] + "…") if len(ch) > 240 else ch
        # Note: `search_faiss` in your code currently doesn't expose numeric scores;
        # if you want scores, you can adapt it to return (chunk, score) pairs.
        sources.append(RagSource(rank=i, score=0.0, preview=preview))

    return RagResponse(answer=answer, sources=sources)

# (Optional) Re-index endpoint (calls your load.py to rebuild)
@app.post("/reindex")
def reindex():
    """
    Run `load.py` via subprocess to rebuild the FAISS index + chunks.json.
    Useful if you updated data/wiki_1k.json and want a live refresh.
    """
    import subprocess, sys
    try:
        result = subprocess.run([sys.executable, "load.py"], capture_output=True, text=True, check=True)
        # Reload into memory
        _load_chunks_or_fail()
        _load_index_or_fail()
        return {"ok": True, "stdout": result.stdout}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Reindex failed: {e.stdout}\n{e.stderr}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
