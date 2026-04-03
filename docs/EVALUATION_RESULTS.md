# Evaluation Results — RAG Pipeline Comparison

> **Status**: Tables below are templates. Run the full evaluation on a GPU machine to populate with actual numbers. See [How to Run](#how-to-run-the-full-evaluation) at the bottom.

## Table 1: Retrieval Performance

Measures how accurately each pipeline retrieves the correct legal provisions from the knowledge graph.

| Model | Precision ↑ | Recall ↑ | Avg Context Tokens | Token Efficiency Ratio ↓ |
|-------|:-----------:|:--------:|:------------------:|:------------------------:|
| Naive (Neo4j full-text) | — | — | — | 1.00 |
| BM25 (rank_bm25) | — | — | — | — |
| Dense (all-MiniLM-L6-v2) | — | — | — | — |
| Advanced (CORTEX, no pruning/critic) | — | — | — | — |
| Cortex Pruner Only | — | — | — | — |
| Cortex Critic Only | — | — | — | — |
| **Cortex (full pipeline)** | — | — | — | — |
| **Judge RAG (phi-4)** | — | — | — | — |

## Table 2: Generation Quality

Measures the quality of generated answers against ground-truth expected answers.

| Model | Exact Match ↑ | F1 Score ↑ | Faithfulness (1–5) ↑ |
|-------|:-------------:|:----------:|:--------------------:|
| Naive (Neo4j full-text) | — | — | — |
| BM25 (rank_bm25) | — | — | — |
| Dense (all-MiniLM-L6-v2) | — | — | — |
| Advanced (CORTEX, no pruning/critic) | — | — | — |
| Cortex Pruner Only | — | — | — |
| Cortex Critic Only | — | — | — |
| **Cortex (full pipeline)** | — | — | — |
| **Judge RAG (phi-4)** | — | — | — |

> **Note:** Naive, BM25, and Dense are retrieval-only baselines; their EM and F1 scores compare empty responses to the expected answer and are expected to be 0. Faithfulness is also expected to be `null` for retrieval-only models.

## Table 3: Judge RAG — Internal LLM-as-Judge Evaluation

The Judge RAG pipeline includes an internal LLM-as-judge that scores its own generated answers on four dimensions and re-generates up to 3 times if the score is below threshold (7.0/10).

| Metric | Mean Score |
|--------|:----------:|
| Overall Judge Score (1–10) | — |
| Relevance (1–10) | — |
| Faithfulness (1–10) | — |
| Completeness (1–10) | — |
| Precision (1–10) | — |
| Avg Attempts Used | — |
| Retrieval Similarity | — |
| Uncertainty: Low (%) | — |
| Uncertainty: Medium (%) | — |
| Uncertainty: High (%) | — |

## Table 4: Judge RAG — Model Ablation

Compare different LLMs used as judge/generator in the Judge RAG pipeline. Each row is a separate evaluation run with `--judge-rag-judge-model` and `--judge-rag-generation-model` set accordingly.

| Judge/Gen Model | F1 ↑ | EM ↑ | Faithfulness ↑ | Judge Score ↑ | Avg Attempts |
|-----------------|:----:|:----:|:--------------:|:-------------:|:------------:|
| phi4:latest | — | — | — | — | — |
| llama3.1:8b | — | — | — | — | — |
| llama3.1:70b | — | — | — | — | — |
| mistral:7b | — | — | — | — | — |

> To fill this table, run separate evaluations per model:
> ```bash
> python -m baselines.evaluation.run_eval \
>   --models judge \
>   --judge-rag-judge-model phi4:latest \
>   --judge-rag-generation-model phi4:latest \
>   --artifact-dir artifacts/eval_judge_phi4
> ```

## Table 5: Combined Summary

End-to-end comparison of all pipelines.

| Model | Retrieval P ↑ | Retrieval R ↑ | EM ↑ | F1 ↑ | Faithfulness ↑ | Context Tokens |
|-------|:-------------:|:-------------:|:----:|:----:|:--------------:|:--------------:|
| Naive | — | — | — | — | — | — |
| BM25 | — | — | — | — | — | — |
| Dense | — | — | — | — | — | — |
| Advanced | — | — | — | — | — | — |
| Cortex Pruner Only | — | — | — | — | — | — |
| Cortex Critic Only | — | — | — | — | — | — |
| **Cortex** | — | — | — | — | — | — |
| **Judge RAG** | — | — | — | — | — | — |

---

## How to Run the Full Evaluation

### Prerequisites

1. **GPU machine** with NVIDIA GPU (≥ 16 GB VRAM recommended for phi-4)
2. **Ollama** installed and running with the desired model(s) pulled:
   ```bash
   # Install Ollama (if not already)
   curl -fsSL https://ollama.com/install.sh | sh

   # Pull the default model
   ollama pull phi4:latest

   # Optional: pull other models for ablation
   ollama pull llama3.1:8b
   ollama pull llama3.1:70b
   ollama pull mistral:7b
   ```
3. **Neo4j** running with the knowledge graph loaded (see docker-compose.yml)
4. **Python dependencies** installed:
   ```bash
   pip install -r baselines/requirements.txt
   pip install -r baselines/judge_rag/requirements.txt
   ```
5. **(Optional)** `OPENAI_API_KEY` env var set for external faithfulness scoring

### Running the Evaluation

#### Full evaluation (all pipelines, all 200 questions)

```bash
# Start infrastructure
docker compose up -d

# Run the full evaluation
python -m baselines.evaluation.run_eval \
  --models naive,bm25,dense,advanced,cortex-pruner-only,cortex-critic-only,cortex,judge \
  --artifact-dir artifacts/eval_full

# Output files:
#   artifacts/eval_full/results_per_question.csv
#   artifacts/eval_full/summary_metrics.csv
#   artifacts/eval_full/top5_case_studies.csv
```

#### Judge RAG only (fastest for testing)

```bash
python -m baselines.evaluation.run_eval \
  --models judge \
  --artifact-dir artifacts/eval_judge
```

#### Limit to N questions (for quick smoke tests)

```bash
python -m baselines.evaluation.run_eval \
  --models naive,judge \
  --max-rows 20 \
  --artifact-dir artifacts/eval_smoke
```

#### Judge RAG with a different model

```bash
python -m baselines.evaluation.run_eval \
  --models judge \
  --judge-rag-retrieval-model llama3.1:8b \
  --judge-rag-generation-model llama3.1:8b \
  --judge-rag-judge-model llama3.1:70b \
  --artifact-dir artifacts/eval_judge_llama70b
```

#### CORTEX pipeline only

```bash
python -m baselines.evaluation.run_eval \
  --models cortex \
  --api-base-url http://localhost:8000/api \
  --artifact-dir artifacts/eval_cortex
```

### Generated Files

After each evaluation run, the following files are generated in `--artifact-dir`:

| File | Description |
|------|-------------|
| `results_per_question.csv` | Per-question raw scores for each model. One row per question, columns for every metric per model. |
| `summary_metrics.csv` | Mean of each metric aggregated per model. One row per model. |
| `top5_case_studies.csv` | Top 5 questions where Cortex showed the largest F1 improvement over Naive. |

### Populating the Tables Above

After running the full evaluation, use the summary CSV to fill in the tables:

```python
import pandas as pd
df = pd.read_csv("artifacts/eval_full/summary_metrics.csv")
print(df.to_markdown(index=False))
```
