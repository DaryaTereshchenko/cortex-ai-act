# Update for Malai: Evaluation Framework Ready

## Summary

I've completed the evaluation framework on branch `eval-advanced-rag`:

✅ **Advanced RAG baseline runner** (`baselines/advanced_rag/`)
✅ **Unified evaluation script** (`baselines/evaluation/run_eval.py`)
✅ **Updated SOP** (`docs/EVALUATION_PROCEDURE_UPDATED.md`)
✅ **Requirements** (`baselines/requirements.txt`)
✅ **Full 25-row test run** completed with outputs

Model naming used in the evaluator/results:
- `naive` = Naive baseline (Neo4j full-text lexical)
- `bm25` = BM25 baseline (rank_bm25 lexical)
- `advanced` = Advanced RAG baseline (my implementation in `baselines/advanced_rag/`, pruning/self-correction OFF)
- `cortex` = Main Cortex mode (advanced pipeline with pruning/self-correction ON)

Backward compatibility: `challenger` is accepted as an alias for `advanced`.

## Branch
`eval-advanced-rag` → Ready for PR

## How to Use

### 1. Install dependencies
```powershell
pip install -r baselines/requirements.txt
```

### 2. Start services (with Neo4j Desktop running)
```powershell
# Terminal A: Knowledge Graph
cd services/knowledge-graph
uvicorn main:app --host 0.0.0.0 --port 8001

# Terminal B: Reasoning Engine  
cd services/reasoning-engine
uvicorn reasoning_api:app --host 0.0.0.0 --port 8002

# Terminal C: Web API (gateway)
cd services/web-ui
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 3. Run evaluation (any terminal)
```powershell
cd c:/path/to/cortex-ai-act
python -m baselines.evaluation.run_eval `
  --input "eval Qs.xlsx" `
  --api-base-url http://localhost:8000/api `
  --models naive,bm25,advanced,cortex `
  --max-rows 25
```

Or full 200 rows (when dataset is ready):
```powershell
python -m baselines.evaluation.run_eval `
  --input "eval Qs.xlsx" `
  --api-base-url http://localhost:8000/api `
  --models naive,bm25,advanced,cortex
```

If running in Docker, only change the API host (for example `web-ui:8000` inside Compose network).

## Outputs
Three CSVs written to `artifacts/eval/`:
- `results_per_question.csv` — per-row metrics for all 4 metrics
- `summary_metrics.csv` — averaged metrics by model
- `top5_case_studies.csv` — strongest Cortex-over-Naive performance deltas

## What It Does

### Phase 1: Dataset Setup
- Maps your `eval Qs.xlsx` columns (`Question` → `question`, `Correct Answer` → `expected_answer`)
- Synthesizes `golden_ids` from `Doc`, `Article`, `Paragraph` if not provided
- Ready to accept curated `golden_ids` for final publication

### Phase 2: Model Runs  
Executes four pipelines:
- **Naive**: Full-text search on Neo4j only
- **BM25**: rank_bm25 lexical retrieval baseline
- **Advanced**: Advanced RAG (pruning + self-correction disabled)
- **Cortex**: Advanced RAG (pruning + self-correction enabled)

Per-row output: response, retrieved IDs, retrieved context, status, error

### Phase 3: 4 Metrics
1. **Retrieval Precision & Recall** — Did we find the right articles?
2. **Faithfulness** — Optional LLM-as-judge scoring (requires `OPENAI_API_KEY`)
3. **Token Efficiency Ratio** — Context reduction vs Naive
4. **EM & F1** — Text similarity vs expected answer

### Phase 4: Analysis
- Summary table (for your paper comparison)
- Top 5 case studies (for detailed analysis)

## Technical Notes

- Script is model/endpoint-agnostic — works with future versions (Nuvelos, etc.)
- 25 rows validates pipeline; scales to 200+ without code changes
- All metrics normalized (stopword removal, punctuation stripping)
- Graceful handling of missing LLM judge (faithfulness stays blank)

## Next Steps

1. For now: use 25 rows to validate pipeline stability
2. Once Chris completes 200-question dataset: update `golden_ids` + run full eval
3. Generated artifacts are publication-ready (just format for paper)

---

**Branch:** `eval-advanced-rag`
**PR:** Ready to open at https://github.com/DaryaTereshchenko/cortex-ai-act/pull/new/eval-advanced-rag
**Status:** All infrastructure in place; ready for full evaluation runs.
