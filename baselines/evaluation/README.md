# Evaluation Runner

This folder contains the unified evaluation script for:

- `naive` baseline
- `bm25` lexical baseline (rank_bm25)
- `dense` semantic baseline (sentence-transformers)
- `advanced` (advanced pipeline with pruning/self-correction disabled)
- `cortex` (advanced pipeline enabled)

Backward compatibility: `challenger` is accepted as an alias for `advanced`.

## Dataset input

The script accepts `.xlsx` or `.csv` and expects at least:

- `Question` or `question`
- `Correct Answer` or `expected_answer`

Optional:

- `golden_ids` (comma/semicolon/JSON list)

If `golden_ids` is missing, the script synthesizes article-level IDs from `Doc`, `Article`, and `Paragraph` as a bootstrap.

## Run

```powershell
python -m baselines.evaluation.run_eval --input "eval Qs.xlsx" --api-base-url http://localhost:8000/api --models naive,bm25,dense,advanced,cortex --max-rows 25
```

PowerShell multiline version:

```powershell
python -m baselines.evaluation.run_eval `
  --input "eval Qs.xlsx" `
  --api-base-url http://localhost:8000/api `
  --models naive,bm25,dense,advanced,cortex `
  --max-rows 25
```

## Artifacts

Written to `artifacts/eval/` by default:

- `results_per_question.csv`
- `summary_metrics.csv`
- `top5_case_studies.csv` (when Naive and Cortex columns exist)

## Faithfulness judge

Set `OPENAI_API_KEY` to enable LLM-as-a-judge scoring.
If missing, faithfulness columns remain empty.
