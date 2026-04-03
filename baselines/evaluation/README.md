# Evaluation Runner

This folder contains the unified evaluation script for:

- `naive` baseline
- `bm25` lexical baseline (rank_bm25)
- `dense` semantic baseline (sentence-transformers)
- `advanced` (advanced pipeline with pruning/self-correction disabled)
- `cortex-pruner-only` (pruning enabled, critic/self-correction disabled)
- `cortex-critic-only` (pruning disabled, critic/self-correction enabled)
- `cortex` (advanced pipeline enabled)

Backward compatibility: `challenger` is accepted as an alias for `advanced`.

## Dataset input

The script accepts `.xlsx` or `.csv` and expects at least:

- `Question` or `question`
- `Correct Answer` or `expected_answer`

Optional:

- `golden_ids` (comma/semicolon/JSON list)
- `Cortex Expected Answer` or `cortex_expected_answer` (used only for Cortex models when `--cortex-only-evals` flag is set)

If `golden_ids` is missing, the script synthesizes article-level IDs from `Doc`, `Article`, and `Paragraph` as a bootstrap.

If `cortex_expected_answer` is missing but `--cortex-only-evals` is set, the script falls back to `expected_answer`.

Indexed-mode behavior (used when input has at least 12 columns):

- Question source: column E (index 4)
- Golden IDs source: column K (index 10)
- F1/EM target source: column L (index 11)

In indexed mode, the script also strips pandas metadata fragments like `Name: ... dtype: ...` from expected-answer text before EM/F1 scoring.

Row limit policy:

- Evaluation always runs on top 100 rows maximum.
- `--max-rows` is supported, but still capped at 100.

## Run

```powershell
python -m baselines.evaluation.run_eval --input "eval Qs.xlsx" --api-base-url http://localhost:8000/api --models naive,bm25,dense,advanced,cortex-pruner-only,cortex-critic-only,cortex --pruning-threshold 0.45 --max-rows 25
```

### Using cortex-specific expected answers

If the dataset includes `cortex_expected_answer` or `Cortex Expected Answer` column with ground truth answers for the Cortex models only:

```powershell
python -m baselines.evaluation.run_eval --input "eval_cortex_answers.xlsx" --api-base-url http://localhost:8000/api --models cortex --cortex-only-evals
```

This will:
- Use `cortex_expected_answer` for EM and F1 scoring of Cortex models
- Use `expected_answer` for all other baselines
- Not affect precision/recall metrics (always based on `golden_ids`)

Sensitivity sweep example:

```powershell
python -m baselines.evaluation.run_eval --input "eval Qs.xlsx" --api-base-url http://localhost:8000/api --models advanced,cortex-pruner-only,cortex --pruning-threshold 0.7 --max-rows 25
```

## Artifacts

Written to `artifacts/eval/` by default:

- `results_per_question.csv`
- `summary_metrics.csv`
- `top5_case_studies.csv` (when Naive and Cortex columns exist)

## Faithfulness judge

Set `OPENAI_API_KEY` to enable LLM-as-a-judge scoring.
If missing, faithfulness columns remain empty.
