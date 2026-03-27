# Judge RAG Pipeline

Hybrid retrieval pipeline with LLM-as-judge evaluation loop for EU AI Act and DSA legal Q&A.

## Architecture Overview

```text
User Query
    │
    ▼
┌──────────────────┐
│ 1. Chunking      │  Extract KG nodes → legal-aware chunks
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 2. Query Rewrite │  ← Ollama LLM (--retrieval-model)
│    (sub-queries) │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 3. Hybrid        │  BM25 → BERT re-rank → Adaptive-K
│    Retrieval     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 4. Retrieval     │  cosine-sim eval metric
│    Scoring       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 5. Answer        │  ← Ollama LLM (--generation-model)
│    Generation    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌──────────────────┐
│ 6. Judge LLM     │────▶│ 7. Decision      │
│  (--judge-model) │     │ accept or retry  │
└──────────────────┘     │ (≤ 3 attempts)   │
                         └──────┬───────────┘
                            retry│     │accept
                                 ▼     ▼
                          back to 5   END
```

The pipeline is orchestrated as a **LangGraph state-machine**. Each box above is a node in the graph. The conditional edge from the decision node either terminates the run or loops back to re-generate with the judge's feedback injected into the prompt.

---

## Step-by-Step Pipeline Description

### Step 1 — Chunking (`chunker.py`)

**Goal:** Extract every text-bearing node from the Neo4j Knowledge Graph and split the content into retrieval-ready chunks.

**What happens:**

1. Connects to Neo4j using the configured credentials.
2. Runs six Cypher queries to pull all **Articles** (`full_text`), **Paragraphs** (`text`), **SubParagraphs** (`text`), **Recitals** (`text`), **Definitions** (`definition_text`), and **AnnexSections** (`content`) for the selected regulation (`eu_ai_act`, `dsa`, or `both`).
3. For each node, the raw text is extracted **word-by-word without any rendering or transformation** — the exact text stored in Neo4j is preserved.
4. If a node's text fits within the configured `chunk_size` (default 750 tokens), it becomes a single chunk.
5. Longer texts are split using a **sentence-boundary-aware algorithm**:
   - Text is split on sentence-ending punctuation followed by an uppercase letter or parenthesis.
   - Sentences are accumulated until the token budget is reached.
   - An **overlap window** (default 125 tokens) carries trailing sentences from one chunk into the next, so no context is lost at boundaries.
   - Text is **never split mid-sentence**.
6. Each chunk carries full provenance metadata: `source_id` (e.g. `eu_ai_act_art_6`), `source_type` (e.g. `Article`), `regulation`, `title`, `number`, `summary`, and `key_topics`.

**Configurable parameters:** `--chunk-size` (default 750, recommended 500–1000), `--chunk-overlap` (default 125, recommended 100–150), `--regulation`.

---

### Step 2 — LLM Query Rewriting (`query_rewriter.py`)

**Goal:** Transform an ambiguous or broad user question into 1–3 focused legal sub-queries that maximise retrieval precision.

**What happens:**

1. The user's original query is sent to the **retrieval LLM** (any Ollama model specified via `--retrieval-model`).
2. The LLM receives a system prompt instructing it to:
   - Use specific legal terminology (e.g. "high-risk AI system", "conformity assessment").
   - Target distinct aspects of the original question.
   - Include synonyms for key legal terms.
3. The LLM returns a JSON array of 1–3 sub-query strings.
4. These sub-queries are used in the retrieval step — results from all sub-queries are merged.
5. If the LLM call fails or returns unparseable output, the pipeline **falls back to the original query** (no crash).

**Why this matters:** Legal documents use precise terminology. A user asking "What's banned?" benefits from rewriting to `["What AI practices are prohibited under Article 5?", "Which AI systems are classified as unacceptable risk?"]`.

**Can be disabled:** `--no-query-rewrite`.

---

### Step 3 — Hybrid Retrieval (`retriever.py`)

**Goal:** Find the most relevant chunks using a multi-stage pipeline that combines keyword matching with semantic understanding.

The retrieval pipeline has three sub-stages:

#### Stage 3a — BM25 Keyword Recall

1. A BM25 (Okapi) index is built over all chunk texts at startup.
2. Each sub-query (from Step 2) is tokenised and scored against the index.
3. Scores across sub-queries are merged (element-wise max).
4. The top `bm25_top_k` (default 25) chunks are selected as the **candidate pool**.

**Why BM25 first:** BM25 is fast and excels at exact keyword matching — it catches chunks that mention specific article numbers, defined terms, or legal phrases even when their overall semantics differ from the query.

#### Stage 3b — Semantic Re-ranking with Sentence Transformers

1. The query is encoded into a dense vector using a sentence-transformer model (default `all-MiniLM-L6-v2`, configurable via `--embedding-model`).
2. All candidate chunks from 3a are also encoded (embeddings are pre-computed at startup for speed).
3. Cosine similarity between the query embedding and each chunk embedding is computed.
4. A **combined score** is calculated: `0.4 × normalised_BM25 + 0.6 × cosine_similarity`.
5. Candidates are sorted by combined score.

**Why this blend:** BM25 catches exact term matches; the dense encoder captures paraphrases and semantic similarity. The 40/60 weighting favours semantic understanding while still rewarding keyword hits.

#### Stage 3c — Adaptive-K Selection

Instead of always returning a fixed number of chunks, the number of results adapts based on:

1. **Query complexity** — measured as `min(1.0, total_query_tokens / 30)`. More complex queries get more chunks.
2. **Score density** — how quickly combined scores drop off relative to the top score. If many chunks score similarly (dense), more are kept; if scores drop sharply (sparse), fewer are needed.
3. These two factors are blended equally: `K = K_min + (K_max - K_min) × (0.5 × complexity + 0.5 × density)`.
4. K is clamped to `[adaptive_k_min, adaptive_k_max]` (default 3–10).

**Configurable parameters:** `--bm25-top-k`, `--embedding-model`.

---

### Step 4 — Retrieval Scoring (`pipeline.py → _score_retrieval_node`)

**Goal:** Produce a single numeric **retrieval quality metric** that measures how relevant the retrieved chunks are to the original query.

**What happens:**

1. The original user query (not the rewritten sub-queries) is encoded into a dense vector.
2. Each retrieved chunk is independently encoded.
3. Cosine similarity between the query vector and each chunk vector is computed.
4. The **mean cosine similarity** across all retrieved chunks is stored as `retrieval_similarity`.

**Why this metric:** It provides a fast, model-independent evaluation signal. A `retrieval_similarity` of 0.75 means the retrieved context is, on average, highly aligned with the question. This metric is included in the final output for benchmarking and comparison against other pipelines.

---

### Step 5 — Answer Generation (`generator.py`)

**Goal:** Produce a grounded legal answer using the retrieved context and an Ollama LLM.

**What happens:**

1. Retrieved chunks are formatted into a numbered context block, each prefixed with its source type and ID (e.g. `[1] Article eu_ai_act_art_5 — Prohibited AI practices`).
2. A **system prompt** instructs the LLM to:
   - Answer using ONLY the provided context.
   - Cite specific Article, Recital, or Annex numbers.
   - Never fabricate information — if the context is insufficient, say so.
   - Use precise legal language with structured bullet points.
3. The user message contains the context block followed by the question.
4. On **retry attempts** (when the judge rejected a prior answer), the judge's written feedback is appended to the user message under a "Previous Attempt Feedback" section, so the LLM can address the specific issues raised.
5. The answer is generated by the `--generation-model` (any Ollama model) with low temperature (0.2) for factual consistency.

**Configurable:** `--generation-model` — use any model available on your Ollama server (e.g. `llama3.1:8b`, `mistral:7b`, `gemma2:27b`, `qwen2:72b`).

---

### Step 6 — Judge Evaluation (`judge.py`)

**Goal:** Automatically evaluate the generated answer for quality and decide whether to accept it or request a revision.

**What happens:**

1. A **judge LLM** (configurable via `--judge-model`, can be different from the generation model) receives:
   - The original user question.
   - The retrieved context (first 600 chars per chunk for efficiency).
   - The candidate answer from Step 5.
2. The judge evaluates four dimensions, each scored 1–10:

   | Dimension | What it measures |
   | --- | --- |
   | **Relevance** | Does the answer actually address the question? |
   | **Faithfulness** | Is every claim supported by the retrieved context? Penalises hallucinations. |
   | **Completeness** | Are all relevant legal provisions from the context mentioned? |
   | **Precision** | Are Article/Recital citations accurate (correct numbers, correct regulation)? |

3. The judge returns a JSON object with per-dimension scores, an overall score, and written feedback describing specific issues.
4. The overall score is compared against the **acceptance threshold** (default 7.0/10, configurable via `--judge-threshold`).
5. An **uncertainty label** is derived from the overall score:
   - ≥ 8.0 → `low` (high confidence)
   - ≥ 6.0 → `medium`
   - < 6.0 → `high` (low confidence)
6. Temperature is set to 0.0 for deterministic evaluation.

**Configurable:** `--judge-model`, `--judge-threshold`.

---

### Step 7 — Decision & Retry Loop (`pipeline.py → _decide_edge`)

**Goal:** Decide whether to accept the current answer or loop back for regeneration.

**What happens:**

1. If the judge score ≥ threshold → **accept**. The pipeline terminates and returns the answer.
2. If the judge score < threshold AND fewer than `max_attempts` have been used → **retry**. The pipeline loops back to Step 5 (generation) with the judge's feedback injected.
3. If the judge score < threshold AND `max_attempts` is reached → **accept with high uncertainty**. The best answer so far is returned with the `uncertainty` field reflecting the quality concern.

**Retry mechanics:**
- Maximum 3 attempts by default (`--max-attempts`).
- On each retry, the judge's written feedback from the previous attempt is passed to the generator so it can address specific issues (e.g. "The answer incorrectly cited Article 6 instead of Article 5").
- The attempt counter increments, and per-attempt judge scores are stored in `judge_details` for post-hoc analysis.
- The retrieval results are **not** re-fetched on retries — only the generation and judging steps repeat. This is intentional: the context is fixed, and the LLM is asked to produce a better answer from the same evidence.

---

## Module-to-File Map

| Module | File | Responsibility |
| --- | --- | --- |
| Configuration | `config.py` | `JudgeRAGConfig` dataclass with all tuneable parameters; `Chunk`, `RetrievedChunk`, and `PipelineState` type definitions |
| Chunking | `chunker.py` | Neo4j text extraction + sentence-boundary-aware splitting with overlap |
| Hybrid Retrieval | `retriever.py` | BM25 index, dense encoder, combined scoring, adaptive-K |
| Query Rewriting | `query_rewriter.py` | Ollama-based query expansion into legal sub-queries |
| Answer Generation | `generator.py` | Context-grounded answer generation via Ollama |
| Judge | `judge.py` | LLM-as-judge scoring (4 dimensions), retry logic, uncertainty labelling |
| LangGraph Orchestrator | `pipeline.py` | State-machine graph wiring, `run_judge_rag()` public entry point |
| CLI | `run_judge_rag.py` | Argument parsing, human-readable terminal output |

---

## Usage

```bash
# Basic — default llama3.1:8b for all stages
python -m baselines.judge_rag.run_judge_rag \
    --question "What does Article 5 of the AI Act prohibit?"

# Different models for each stage
python -m baselines.judge_rag.run_judge_rag \
    --question "What are the transparency obligations for high-risk AI?" \
    --retrieval-model mistral:7b \
    --generation-model llama3.1:8b \
    --judge-model llama3.1:70b

# Disable query rewriting, search DSA only
python -m baselines.judge_rag.run_judge_rag \
    --question "What are the obligations for online platforms?" \
    --no-query-rewrite \
    --regulation dsa

# Custom chunking, stricter judge, output to file
python -m baselines.judge_rag.run_judge_rag \
    --question "Explain conformity assessment" \
    --chunk-size 500 --chunk-overlap 100 \
    --judge-threshold 8.0 \
    --output result.json

# Verbose logging to see every pipeline step
python -m baselines.judge_rag.run_judge_rag \
    --question "What are risk management obligations?" \
    --verbose
```

## CLI Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--question` / `-q` | *(required)* | The legal question to answer |
| `--retrieval-model` | `llama3.1:8b` | Ollama model used for query rewriting (Step 2) |
| `--generation-model` | `llama3.1:8b` | Ollama model used for answer generation (Step 5) |
| `--judge-model` | `llama3.1:8b` | Ollama model used for judge evaluation (Step 6) |
| `--ollama-base-url` | `http://localhost:11434` | Ollama server URL |
| `--chunk-size` | `750` | Target chunk size in tokens (recommended 500–1000) |
| `--chunk-overlap` | `125` | Chunk overlap in tokens (recommended 100–150) |
| `--bm25-top-k` | `25` | Number of chunks in BM25 initial recall pool |
| `--rerank-top-k` | `10` | Semantic re-ranking pool size |
| `--embedding-model` | `all-MiniLM-L6-v2` | Sentence-transformers model for dense re-ranking |
| `--no-query-rewrite` | `false` | Disable LLM query rewriting (use original query as-is) |
| `--max-attempts` | `3` | Maximum judge retry attempts before forced acceptance |
| `--judge-threshold` | `7.0` | Minimum overall judge score (1–10) to accept an answer |
| `--regulation` | `eu_ai_act` | Regulation scope: `eu_ai_act`, `dsa`, or `both` |
| `--neo4j-uri` | `bolt://localhost:7687` | Neo4j connection URI |
| `--neo4j-user` | `neo4j` | Neo4j username |
| `--neo4j-password` | `changeme` | Neo4j password |
| `--output` / `-o` | stdout | Write full JSON result to a file |
| `--verbose` / `-v` | `false` | Enable DEBUG-level logging |

## Prerequisites

1. **Neo4j** running with the knowledge graph ingested (default port 7687). Run the KG service's `/graph/ingest` endpoint first.
2. **Ollama** running with the desired models pulled (default port 11434). Example: `ollama pull llama3.1:8b`.
3. Install dependencies:
   ```bash
   pip install -r baselines/judge_rag/requirements.txt
   ```

## Output Schema

The pipeline returns a JSON result with the following fields:

```json
{
  "query": "What does Article 5 prohibit?",
  "answer": "Article 5 of the EU AI Act prohibits the following AI practices: …",
  "uncertainty": "low",
  "judge_score": 8.5,
  "judge_feedback": "No issues found.",
  "attempts_used": 1,
  "retrieval_similarity": 0.72,
  "retrieved_ids": ["eu_ai_act_art_5", "eu_ai_act_rec_29"],
  "retrieved_context": "Full text of retrieved chunks concatenated …",
  "rewritten_queries": ["What AI practices does Article 5 prohibit?", "What is classified as unacceptable risk?"],
  "latency_seconds": 12.3,
  "judge_details": {
    "judge_attempt_1": {
      "relevance": 9.0,
      "faithfulness": 8.0,
      "completeness": 8.5,
      "precision": 9.0,
      "overall": 8.5,
      "feedback": "No issues found."
    }
  },
  "config": {
    "retrieval_model": "llama3.1:8b",
    "generation_model": "llama3.1:8b",
    "judge_model": "llama3.1:8b",
    "chunk_size": 750,
    "chunk_overlap": 125,
    "regulation": "eu_ai_act"
  }
}
```

| Field | Type | Description |
| --- | --- | --- |
| `query` | string | Original user question |
| `answer` | string | Final generated answer (best attempt) |
| `uncertainty` | string | Confidence level: `low` (≥8), `medium` (≥6), `high` (<6) |
| `judge_score` | float | Overall judge score of the final answer (1–10) |
| `judge_feedback` | string | Judge's written feedback on the final answer |
| `attempts_used` | int | Number of generation attempts used (1–3) |
| `retrieval_similarity` | float | Mean cosine similarity of retrieved chunks vs. original query |
| `retrieved_ids` | list[str] | De-duplicated source node IDs of retrieved chunks |
| `retrieved_context` | string | Concatenated text of all retrieved chunks |
| `rewritten_queries` | list[str] | Sub-queries produced by the query rewriter |
| `latency_seconds` | float | Total pipeline wall-clock time |
| `judge_details` | dict | Per-attempt breakdown of judge scores |
| `config` | dict | Pipeline configuration used for this run |
