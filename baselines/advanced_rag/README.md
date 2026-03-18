# Advanced RAG Baseline Runner

This folder contains a thin runner for the integrated CORTEX advanced RAG pipeline.

## What it does

- Sends a question to `POST /api/query`
- Polls `GET /api/query/{query_id}` until complete/failed
- Extracts eval-ready fields:
  - `response`
  - `retrieved_ids`
  - `retrieved_context`
  - `latency_seconds`

## Example

```bash
python baselines/advanced_rag/run_advanced_baseline.py \
  --question "What does Article 5 prohibit?" \
  --api-base-url http://localhost:8000/api \
  --regulation both
```

## Notes

- `retrieved_ids` are composed from reasoning steps, graph nodes, and citations.
- `retrieved_context` prefers explicit context fields, then falls back to graph previews.
