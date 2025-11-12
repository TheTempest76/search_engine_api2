# ---- Base image ----
FROM python:3.11-slim

# Prevents Python from buffering stdout/stderr and writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps (build tools for some pip wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Workdir ----
WORKDIR /

# ---- Python deps ----
# Copy only requirements first for better layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ---- App code ----
# Make sure your repo includes:
# - server.py (FastAPI)
# - load.py (creates index/index.bin + index/chunks.json)
# - faiss_embedding.py, llm_router.py (Gemini only; no Ollama calls)
# - data/wiki_1k.json (or set DATA_JSON_PATH env to your path)
COPY . .

# ---- Build-time indexing (optional but recommended) ----
# This will run your load.py so the image ships with a ready FAISS index.
# If your data changes often, you can skip this and call /reindex on boot instead.
RUN python load.py

# ---- Runtime env ----
# Render sets $PORT automatically; default to 8000 if missing (local runs)
ENV INDEX_DIR="index" \
    DATA_JSON_PATH="data/wiki_1k.json" \
    ALLOWED_ORIGINS="*" \
    PORT=8000

# ---- Expose & start ----
EXPOSE 8000

# Use uvicorn; bind to 0.0.0.0 and $PORT (Render requirement)
CMD ["bash", "-lc", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]