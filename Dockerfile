# Multimodal RAG - FastAPI backend image.
#
# Railway auto-detects a Dockerfile named `Dockerfile` at the repo root and
# uses it instead of the built-in builders.  The `PORT` env var set by Railway
# (default 8000) is bound at startup.
#
# Build:   docker build -t multimodal-rag-api .
# Run:     docker run --rm -p 8000:8000 --env-file .env multimodal-rag-api

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf

WORKDIR /app

# System libraries required by the ML stack (OpenCV/libGL, glib) at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first: the API never runs on a GPU, and the default
# torch wheels pull ~2GB of unused CUDA libraries.  The pip install below will
# then see torch already satisfied and skip those downloads.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Backend-only dependencies (ingestion/dashboard libs stay out of the image).
# Copy requirements first so layer caching kicks in on code-only changes.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Application code plus the pre-built vector store (dbv2/) and PDF assets.
COPY . .

# Writable runtime artifacts (logs are also mirrored to stdout).
RUN mkdir -p /app/logs && chmod -R u+w /app

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.getenv('PORT', '8000'), timeout=5)"

# Single worker: the vector store + embedding client are heavy in-memory objects.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
