# CORTEX-AI-ACT — Developer Setup Guide

> Everything you need to know about Docker, local development, and how each piece fits together.

---

## Table of Contents

1. [Do I need Docker?](#do-i-need-docker)
2. [Architecture overview](#architecture-overview)
3. [Option A: Work on your service only (no Docker)](#option-a-work-on-your-service-only-no-docker)
4. [Option B: Run the full stack locally (Docker)](#option-b-run-the-full-stack-locally-docker)
5. [Option C: Use GitHub Codespaces (zero install)](#option-c-use-github-codespaces-zero-install)
6. [Docker explained](#docker-explained)
7. [Environment variables](#environment-variables)
8. [Common tasks](#common-tasks)
9. [Troubleshooting](#troubleshooting)

---

## Do I need Docker?

**Short answer: No — not for day-to-day coding.**

| What you're doing | Docker needed? |
|---|---|
| Writing code for your service | No |
| Running your service's tests | No |
| Running the linter | No |
| Testing how your service talks to Neo4j or other services | Yes (or use Codespaces) |
| Running the full CORTEX stack end-to-end | Yes (or use Codespaces) |

Most of the time, you only need **Python 3.11** and your service's `requirements.txt`.

---

## Architecture overview

CORTEX is made up of 4 independent services + a database:

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│   Web UI    │────▶│ Reasoning Engine │────▶│    Neo4j     │
│  (Streamlit)│     │ (LLM + LangGraph)│     │ (Graph DB)   │
│  port 8501  │     │   port 8002      │     │ port 7474    │
└─────────────┘     └──────────────────┘     └──────┬───────┘
                                                     │
┌─────────────┐     ┌──────────────────┐             │
│Benchmarking │────▶│ Knowledge Graph  │─────────────┘
│  port 8003  │     │   port 8001      │
└─────────────┘     └──────────────────┘
```

Each service is a Python app (FastAPI or Streamlit) that runs on its own port. Docker packages each one into a container so they can all run together.

**Key insight:** You don't need to run all 5 things to work on your part. You only need them when testing how services talk to each other.

---

## Option A: Work on your service only (no Docker)

This is the **recommended daily workflow**. You work in your service folder, run the app directly with Python, and only use Docker when you need integration testing.

### 1. Clone and set up

```bash
git clone <repo-url>
cd cortex-ai-act
```

### 2. Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Linux/Mac
# or: .venv\Scripts\activate  # Windows
```

### 3. Install YOUR service's dependencies

You only install what you need:

```bash
# Data Architect:
pip install -r services/knowledge-graph/requirements.txt

# LLM Engineer (Malai):
pip install -r services/reasoning-engine/requirements.txt

# Full-Stack Developer:
pip install -r services/web-ui/requirements.txt

# Systems Lead:
pip install -r services/benchmarking/requirements.txt
```

### 4. Run your service

```bash
# Knowledge-Graph, Reasoning-Engine, or Benchmarking (FastAPI):
cd services/knowledge-graph    # (or reasoning-engine, benchmarking)
uvicorn main:app --reload --port 8001

# Web UI (Streamlit):
cd services/web-ui
streamlit run app.py
```

### 5. Run tests

```bash
cd services/knowledge-graph   # (your service folder)
python -m pytest tests/ -v
```

### 6. Run the linter

```bash
# From the project root:
ruff check services/knowledge-graph/
ruff format services/knowledge-graph/
```

### What if my service needs Neo4j?

The **knowledge-graph** and **reasoning-engine** services talk to Neo4j. If you need it while coding, you can run just Neo4j in Docker without starting anything else:

```bash
docker compose up -d neo4j
```

This starts Neo4j on `localhost:7474` (browser) and `localhost:7687` (bolt). Your Python code running outside Docker can connect to it normally. Stop it with:

```bash
docker compose down
```

---

## Option B: Run the full stack locally (Docker)

Use this when you want to test everything together — all services + Neo4j.

### Prerequisites

- Docker Desktop (or Docker Engine on Linux)
- Docker Compose v2+

### Steps

```bash
# 1. Copy environment config
cp .env.example .env

# 2. Build and start everything
make up-build

# 3. Check status
make ps

# 4. View logs
make logs               # all services
make logs-neo4j         # specific service
make logs-web-ui        # specific service
```

### What's running

| Service | URL | What it does |
|---|---|---|
| Neo4j Browser | http://localhost:7474 | Graph database admin UI |
| Knowledge-Graph API | http://localhost:8001 | PDF parsing & Cypher queries |
| Reasoning-Engine API | http://localhost:8002 | LLM + entropy pruning |
| Web UI | http://localhost:8501 | The user-facing dashboard |
| Benchmarking API | http://localhost:8003 | Performance metrics |

### Common Docker commands

```bash
make up          # Start all services (uses existing builds)
make up-build    # Rebuild then start
make down        # Stop everything
make down-v      # Stop and delete all data (volumes)
make restart     # Restart all services
make clean       # Remove stopped containers and caches
```

### Running only specific services

You don't have to run everything:

```bash
# Just Neo4j + knowledge-graph:
docker compose up -d neo4j knowledge-graph

# Just Neo4j + web-ui + reasoning-engine:
docker compose up -d neo4j reasoning-engine web-ui
```

---

## Option C: Use GitHub Codespaces (zero install)

If you don't want to install Docker or fight with local setup:

1. Go to the GitHub repo
2. Click **Code** → **Codespaces** → **Create codespace on main**
3. Wait ~2 minutes — the full stack starts automatically
4. Click the **Ports** tab to access each service

Free tier gives you **60 core-hours/month** (~30 hours on a 2-core machine).

---

## Docker explained

If you've never used Docker, here's what each file does:

### `docker-compose.yml` (the orchestra conductor)

Defines all 5 services (Neo4j + 4 Python apps), how they connect to each other, which ports they expose, and where they store data. When you run `make up-build`, Docker reads this file and starts everything.

### `services/<name>/Dockerfile` (the recipe for one service)

Each service has its own Dockerfile — a step-by-step recipe that says:
1. Start with Python 3.11
2. Install system tools (e.g., PDF reader for knowledge-graph)
3. Install Python packages from `requirements.txt`
4. Copy the service code
5. Define how to start the service

### `services/<name>/requirements.txt` (Python packages for one service)

Each service lists only the packages it needs. They're separate because:
- **reasoning-engine** needs PyTorch (~2GB) — web-ui doesn't
- **web-ui** needs Streamlit — no other service does
- Separate files = smaller Docker images + faster builds + no conflicts

### `.env` / `.env.example` (passwords and settings)

Configuration that changes between environments (passwords, ports, model IDs). You copy `.env.example` to `.env` and fill in real values. **Never commit `.env` to Git** — it's in `.gitignore`.

### `docker/nginx/nginx.conf` (reverse proxy)

Routes web traffic to the right service. Only used in production (the `--profile production` flag).

### `.devcontainer/` (Codespaces config)

Tells GitHub Codespaces how to set up the environment when someone opens the repo there.

---

## Environment variables

Copy the template and edit as needed:

```bash
cp .env.example .env
```

| Variable | Default | Who needs it |
|---|---|---|
| `NEO4J_USER` | `neo4j` | knowledge-graph, reasoning-engine |
| `NEO4J_PASSWORD` | `changeme` | knowledge-graph, reasoning-engine |
| `MODEL_ID` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | reasoning-engine |
| `WANDB_API_KEY` | (empty) | benchmarking |
| `MLFLOW_TRACKING_URI` | (empty) | benchmarking |

When running without Docker, you can set these in your shell:

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=changeme
```

---

## Common tasks

### Add a Python package to your service

```bash
# 1. Add it to your service's requirements.txt
echo "some-package>=1.0,<2" >> services/knowledge-graph/requirements.txt

# 2. Install it locally
pip install -r services/knowledge-graph/requirements.txt

# 3. If using Docker, rebuild your service
docker compose build knowledge-graph
docker compose up -d knowledge-graph
```

### Create a new API endpoint

Edit `main.py` in your service folder. Example for knowledge-graph:

```python
@app.post("/parse")
async def parse_document(file: UploadFile):
    # your code here
    return {"status": "parsed"}
```

If running with `uvicorn --reload`, it auto-reloads on save.

### Run the full test suite (like CI does)

```bash
make lint    # ruff check across all services
make test    # pytest across all services
```

### Check what CI will do before pushing

```bash
# Lint your service:
ruff check services/knowledge-graph/ --output-format=github
ruff format --check services/knowledge-graph/

# Test your service:
cd services/knowledge-graph && python -m pytest tests/ -v
```

---

## Troubleshooting

### "Docker permission denied"

```bash
sudo usermod -aG docker $USER
newgrp docker
```

### "docker-credential-desktop.exe: exec format error" (WSL)

```bash
# Edit ~/.docker/config.json and set:
# { "credStore": "" }
sed -i 's/"credsStore".*/"credStore": ""/' ~/.docker/config.json
```

### Neo4j won't start / "port already in use"

```bash
# Check what's using the port:
lsof -i :7474

# Kill it or change the port in .env:
NEO4J_HTTP_PORT=7475
```

### "ModuleNotFoundError" when running a service

Make sure you installed that service's requirements:

```bash
pip install -r services/<your-service>/requirements.txt
```

### Docker images are too big / slow to build

The `.dockerignore` file keeps unnecessary files out of images. If builds are slow, check that model weights (`.safetensors`, `.bin`) and `node_modules` aren't being copied in.

### CI fails but it works locally

CI runs `ruff check` and `ruff format --check` — if you haven't formatted your code, it will fail. Run this before pushing:

```bash
ruff format services/<your-service>/
ruff check services/<your-service>/ --fix
```

---

## Summary: What each team member needs

| Role | Daily tools | When to use Docker |
|---|---|---|
| **Data Architect** (knowledge-graph) | Python, Neo4j | `docker compose up -d neo4j` for local Neo4j |
| **LLM Engineer** (reasoning-engine) | Python, PyTorch | `docker compose up -d neo4j` when testing graph queries |
| **Full-Stack** (web-ui) | Python, Streamlit | Full stack (`make up-build`) to test UI ↔ API integration |
| **Systems Lead** (benchmarking) | Python, Docker | Full stack for benchmarking runs |
