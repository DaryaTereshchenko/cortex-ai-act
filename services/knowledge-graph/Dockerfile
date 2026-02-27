###############################################################################
# Knowledge-Graph Service
# — PDF parsing, Neo4j ingestion, Cypher query API
###############################################################################
FROM python:3.11-slim AS base

LABEL maintainer="cortex-ai-act"
LABEL service="knowledge-graph"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for PDF processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        poppler-utils \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY services/knowledge-graph/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/knowledge-graph/ .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
