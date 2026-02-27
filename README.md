# CORTEX-AI-ACT

> **EU AI Act Compliance Assistant** — Knowledge-graph-augmented reasoning with entropy pruning and self-correction.

---

## Architecture

| Service | Port | Description |
|---------|------|-------------|
| **Neo4j** | 7474 / 7687 | Knowledge Graph database (EU AI Act articles as nodes/edges) |
| **Knowledge-Graph** | 8001 | PDF→Markdown parsing, Neo4j schema management, Cypher query API |
| **Reasoning-Engine** | 8002 | Llama 3.1 8B + LangGraph — Entropy Pruning & Self-Correction Loop |
| **Web-UI** | 8501 | Streamlit dashboard with graph visualization |
| **Benchmarking** | 8003 | Token-reduction, latency, sustainability metrics |
| **NGINX** | 80 | Reverse proxy (production profile only) |

## Quick Start

### Prerequisites

- Docker & Docker Compose v2+
- (Optional) NVIDIA Container Toolkit for GPU inference

### 1. Clone & configure

```bash
git clone <repo-url> && cd cortex-ai-act
cp .env.example .env          # edit .env with your credentials
```

### 2. Start the stack

```bash
make up-build                  # or: docker compose up -d --build
```

### 3. Verify

```bash
make ps                        # check container status
curl http://localhost:8001/health
curl http://localhost:8002/health
open http://localhost:8501      # Streamlit UI
open http://localhost:7474      # Neo4j Browser
```

### Alternative: GitHub Codespaces (zero install)

Click **Code → Codespaces → Create codespace on main** in the GitHub UI. The full stack starts automatically — no Docker or local setup needed. Look in the **Ports** tab to access each service.

Free tier: **60 core-hours/month** (~30 hours on a 2-core machine).

## Development

```bash
make lint          # ruff lint all services
make format        # ruff auto-format
make test          # pytest all services
make logs          # tail logs
make logs-neo4j    # tail a specific service
```

## CI/CD Pipeline

### Continuous Integration (`.github/workflows/ci.yml`)

Triggers on every push/PR to `main` or `develop`:

1. **Lint** — `ruff check` + `ruff format --check` per service (parallel matrix)
2. **Test** — `pytest` per service
3. **Build** — Docker images built & pushed to GHCR
4. **Security** — Trivy vulnerability scan uploaded to GitHub Security tab

### Continuous Deployment (`.github/workflows/cd.yml`)

| Trigger | Target |
|---------|--------|
| Push to `main` | **staging** via Codespaces (auto, free) |
| GitHub Release published | **production** (requires approval) |
| Manual dispatch | choice of staging/production |

Three deploy methods (set `DEPLOY_METHOD` variable in GitHub environment settings):

| Method | Value | Requirements | Best for |
|--------|-------|-------------|----------|
| **Codespaces** | `codespaces` (default) | None — works immediately | Staging / demos |
| **SSH** | `ssh` | `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY` secrets | Nuvelos / GPU server |
| **Docker context** | `docker-context` | Same SSH secrets | Advanced remote Docker |

### Preview Environments (`.github/workflows/preview.yml`)

Every PR automatically gets a Codespaces preview environment with the full running stack. Auto-deleted when the PR closes.

### Required GitHub Secrets & Variables

**For Codespaces deploy (staging):** No secrets needed — works out of the box.

**For SSH deploy (production / Nuvelos):**

| Name | Type | Description |
|------|------|-------------|
| `DEPLOY_HOST` | Secret | SSH hostname of deploy server |
| `DEPLOY_USER` | Secret | SSH username |
| `DEPLOY_SSH_KEY` | Secret | Private SSH key |
| `DEPLOY_URL` | Variable | Base URL for smoke tests |
| `DEPLOY_PATH` | Variable | Path on server (default: `~/cortex-ai-act`) |
| `DEPLOY_METHOD` | Variable | `ssh` or `docker-context` |

### GPU Deployment (Nuvelos / Tesla)

Uncomment the GPU `deploy` block in `docker-compose.yml` under `reasoning-engine`, then:

```bash
docker compose up -d reasoning-engine
```

## Project Structure

```
cortex-ai-act/
├── .github/workflows/
│   ├── ci.yml                 # Lint → Test → Build → Security
│   └── cd.yml                 # Deploy to staging / production
├── docker/
│   ├── neo4j/init-cypher/     # Neo4j seed scripts
│   └── nginx/nginx.conf       # Reverse proxy config
├── services/
│   ├── knowledge-graph/       # Data Architect's domain
│   ├── reasoning-engine/      # LLM Engineer's domain
│   ├── web-ui/                # Full-Stack Developer's domain
│   └── benchmarking/          # Systems & Efficiency Lead's domain
├── scripts/                   # Utility scripts
├── tests/                     # Integration / end-to-end tests
├── docker-compose.yml
├── .env.example
├── Makefile
└── pyproject.toml
```

## Team Ownership

| Directory | Owner |
|-----------|-------|
| `services/knowledge-graph/` | Data Architect (Knowledge Graph Lead) |
| `services/reasoning-engine/` | LLM Engineer (Malai) |
| `services/web-ui/` | Full-Stack Developer |
| `services/benchmarking/` | Systems & Efficiency Lead |

## License

TBD
