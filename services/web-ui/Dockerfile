###############################################################################
# Web-UI Service
# — Streamlit dashboard with graph visualization
###############################################################################
FROM python:3.11-slim AS base

LABEL maintainer="cortex-ai-act"
LABEL service="web-ui"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY services/web-ui/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/web-ui/ .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
