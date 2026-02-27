###############################################################################
# Benchmarking Service
# — Performance metrics, sustainability tracking, experiment logging
###############################################################################
FROM python:3.11-slim AS base

LABEL maintainer="cortex-ai-act"
LABEL service="benchmarking"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY services/benchmarking/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/benchmarking/ .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
