# Models & Caching

> How CORTEX handles LLM and embedding models — locally and in Docker.

---

## Models used

| Model | Type | Default for | Framework |
|-------|------|-------------|-----------|
| `all-MiniLM-L6-v2` | Embedding (384-dim) | Dense baseline, Judge RAG re-ranking, Cortex pruning engine | sentence-transformers |
| `llama3.1:8b` | LLM | Judge RAG query rewriting & generation, Cortex reasoning engine | Ollama |
| `phi4:latest` | LLM | Judge RAG judge evaluation | Ollama |

All defaults are overridable via CLI arguments or environment variables.

---

## Running locally (no Docker)

### 1. Pre-download embedding model

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

Cached in `~/.cache/torch/sentence_transformers/` by default (or wherever `SENTENCE_TRANSFORMERS_HOME` points).

### 2. Pull Ollama models

Make sure the Ollama server is running, then:

```bash
ollama pull llama3.1:8b
ollama pull phi4:latest
```

Models are stored by the Ollama server (typically `~/.ollama/models/`).

### 3. Run evaluations

```bash
# All baselines + Cortex
python -m baselines.evaluation.run_eval --models all

# Judge RAG only (uses llama3.1:8b for generation, phi4 for judge)
python -m baselines.evaluation.run_eval --models judge

# Override models via CLI
python -m baselines.judge_rag.run_judge_rag \
    --question "What does Article 5 prohibit?" \
    --generation-model llama3.1:8b \
    --judge-model phi4:latest
```

---

## Running in Docker

### Architecture

```
┌─────────────────┐     ┌──────────────────┐
│ Reasoning Engine│────▶│     Ollama       │
│  (pruning uses  │     │  llama3.1:8b     │
│  all-MiniLM)    │     │  phi4:latest     │
│  port 8002      │     │  port 11434      │
└─────────────────┘     └──────────────────┘
```

### Start the stack

```bash
cp .env.example .env    # first time only
docker compose up -d
```

This starts Neo4j, all services, **and the Ollama server**.

### Pull models into the Ollama container

After `ollama` is healthy, pull the required models:

```bash
docker exec cortex-ollama ollama pull llama3.1:8b
docker exec cortex-ollama ollama pull phi4:latest
```

These are stored in the `ollama_models` Docker volume and persist across container restarts.

### Verify models are available

```bash
docker exec cortex-ollama ollama list
```

---

## How caching works

### Embedding models (sentence-transformers, HuggingFace)

Both `knowledge-graph` and `reasoning-engine` bind-mount your **host cache** into the container:

```
Host                                          Container
~/.cache/huggingface                    →     /app/.cache/huggingface
~/.cache/torch/sentence_transformers    →     /app/.cache/sentence_transformers
```

This means:

- **Download once, use everywhere** — local runs and Docker share the same files.
- **Survives `docker compose down -v`** — the files live on your host, not in a Docker volume.
- **The reasoning-engine Dockerfile** includes a warm-up `RUN` step that pre-downloads `all-MiniLM-L6-v2` at build time as a fallback.

You can override the host paths via environment variables:

```bash
# In .env or shell
HF_HOME=/path/to/huggingface/cache
SENTENCE_TRANSFORMERS_HOME=/path/to/sentence_transformers/cache
```

### Ollama models

Ollama models are managed by the Ollama server and stored in the `ollama_models` named Docker volume (mounted at `/root/.ollama` inside the container).

- Persists across `docker compose restart` and `docker compose down`.
- **Lost on `docker compose down -v`** (which removes all volumes). Re-pull after if needed.

### Environment variables reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_ID` | `llama3.1:8b` | Primary LLM for reasoning engine |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace model/tokenizer cache |
| `SENTENCE_TRANSFORMERS_HOME` | `~/.cache/torch/sentence_transformers` | Sentence-transformers model cache |
| `OLLAMA_PORT` | `11434` | Ollama server port |

---

## Changing models

### Use a different Ollama model for generation

```bash
# CLI
python -m baselines.judge_rag.run_judge_rag \
    --question "..." \
    --generation-model mistral:7b

# Eval
python -m baselines.evaluation.run_eval \
    --models judge \
    --judge-rag-generation-model mistral:7b
```

### Use a different model for the Cortex reasoning engine

```bash
# .env
MODEL_ID=llama3.1:8b

# Or inline
MODEL_ID=mistral:7b docker compose up -d reasoning-engine
```

Phi-4 is a 14B parameter, state-of-the-art open model built upon a blend of synthetic datasets, data from filtered public domain websites, and acquired academic books and Q&A datasets.

### Use a different embedding model

```bash
python -m baselines.judge_rag.run_judge_rag \
    --question "..." \
    --embedding-model all-mpnet-base-v2
```

---

## GPU support

Both the reasoning-engine and Ollama services have GPU support commented out in `docker-compose.yml`. To enable:

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
2. Uncomment the `deploy.resources.reservations` block for the desired service
3. Restart: `docker compose up -d`

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

---

## Troubleshooting

### "Connection refused" when Judge RAG calls Ollama

Make sure Ollama is running:
- **Local**: `ollama serve` (or check `systemctl status ollama`)
- **Docker**: `docker compose up -d ollama` then verify with `docker exec cortex-ollama ollama list`

### Models re-download every time in Docker

Check that the bind mounts exist on your host:

```bash
ls ~/.cache/huggingface/
ls ~/.cache/torch/sentence_transformers/
```

If the directories don't exist, create them before starting Docker:

```bash
mkdir -p ~/.cache/huggingface ~/.cache/torch/sentence_transformers
```

### Ollama models lost after `docker compose down -v`

The `-v` flag removes all volumes including `ollama_models`. Re-pull:

```bash
docker compose up -d ollama
docker exec cortex-ollama ollama pull llama3.1:8b
docker exec cortex-ollama ollama pull phi4:latest
```
