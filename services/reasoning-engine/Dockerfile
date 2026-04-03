###############################################################################
# Reasoning-Engine Service
# — LLM inference, LangGraph, Entropy Pruning, Self-Correction Loop
###############################################################################
FROM python:3.11-slim AS base

LABEL maintainer="cortex-ai-act"
LABEL service="reasoning-engine"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY services/reasoning-engine/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Warm-up: pre-download models so first request is fast.
# Models are cached in /app/.cache which is bind-mounted at runtime,
# so this only downloads if the host cache is empty.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')"

COPY services/reasoning-engine/ .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
