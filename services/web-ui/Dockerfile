###############################################################################
# Web-UI Service
# — Streamlit dashboard with graph visualization
###############################################################################
FROM python:3.11-slim

LABEL maintainer="cortex-ai-act"
LABEL service="web-ui"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY services/web-ui/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/web-ui/ .

# Expose ports for both Streamlit and FastAPI
EXPOSE 8501 8000

# Health check on FastAPI endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start both FastAPI and Streamlit
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless=true"]
