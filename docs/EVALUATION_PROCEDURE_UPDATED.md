# Ground Truth Evaluation SOP (Updated for CORTEX Baselines)

## Scope

This SOP operationalizes evaluation for six model variants in this repository:

- **Naive**: `baselines/naive_baseline.py`
- **BM25**: `baselines/bm25_baseline.py` lexical baseline
- **Advanced**: advanced pipeline with pruning and self-correction disabled
- **Cortex-Pruner-Only**: pruning enabled and self-correction disabled
- **Cortex-Critic-Only**: pruning disabled and self-correction enabled
- **Cortex**: advanced pipeline with pruning and self-correction enabled

Advanced query orchestration is implemented in:

- `baselines/advanced_rag/run_advanced_baseline.py`
- `baselines/evaluation/run_eval.py`

## Phase 1: Dataset Setup

Before running evaluation, finalize dataset schema.

Each row should include:

- `question`: input query text
- `golden_ids`: official legal IDs expected to be retrieved
- `expected_answer`: one-sentence legal fact summary

Supported source columns in the provided spreadsheet:

- `Question` -> mapped to `question`
- `Correct Answer` -> mapped to `expected_answer`

If `golden_ids` is absent, the eval script bootstraps article-level IDs from `Doc`, `Article`, and `Paragraph`. This is useful for initial runs but should be replaced with curated IDs for final paper reporting.

## Phase 2: Model Runs

Run all rows through all target models and produce per-row outputs:

- `[Model]_Response`
- `[Model]_Retrieved_IDs`
- `[Model]_Retrieved_Context`

Run command:

```powershell
python -m baselines.evaluation.run_eval --input "eval Qs.xlsx" --api-base-url http://localhost:8000/api --models naive,bm25,dense,advanced,cortex-pruner-only,cortex-critic-only,cortex --pruning-threshold 0.45 --max-rows 25
```

Optional flags:

- `--max-rows 25` for quick validation
- `--artifact-dir artifacts/eval`
- `--judge-model gpt-4o-mini`
- `--pruning-threshold 0.45`

## Phase 3: 4 Metrics Execution

### Metric 1: Retrieval Precision and Recall

Goal: did model retrieve the correct legal units?

Per row:

- `Precision = |Retrieved ∩ Golden| / |Retrieved|`
- `Recall = |Retrieved ∩ Golden| / |Golden|`

### Metric 2: Faithfulness (Groundedness)

Goal: detect hallucination against retrieved context.

Method:

- LLM-as-a-judge prompt scores 1-5
- Inputs: `[Model]_Retrieved_Context` and `[Model]_Response`

Enable by setting `OPENAI_API_KEY`. Without key, faithfulness stays blank.

### Metric 3: Token Efficiency Ratio

Goal: quantify context reduction vs Naive baseline.

Per row:

- `Model_Token_Efficiency_Ratio = Model_Context_Tokens / Naive_Context_Tokens`

Interpretation:

- `0.30` means 70% context reduction.

### Metric 4: EM and F1

Goal: compare answer text with expected answer.

Normalization in script:

- lowercase
- remove punctuation
- remove stopwords (`a`, `the`, `is`, etc.)

Per row:

- `EM`: exact normalized match (0/1)
- `F1`: token overlap harmonic mean

## Phase 4: Final Analysis and Visualization Inputs

Script outputs:

- `artifacts/eval/results_per_question.csv`
- `artifacts/eval/summary_metrics.csv`
- `artifacts/eval/top5_case_studies.csv`

`summary_metrics.csv` is the source for your comparison table.

`top5_case_studies.csv` surfaces strongest Cortex-over-Naive deltas for paper case studies.

## Phase 5: Ablation and Sensitivity Studies (Part 2)

Goal: isolate Pruner and Critic contributions and assess pruning aggressiveness.

Ablation matrix in the same evaluation loop/output files:

- `advanced`: `enable_pruning=False`, `enable_self_correction=False`
- `cortex-pruner-only`: `enable_pruning=True`, `enable_self_correction=False`
- `cortex-critic-only`: `enable_pruning=False`, `enable_self_correction=True`
- `cortex`: `enable_pruning=True`, `enable_self_correction=True`

All profiles write to the same:

- `results_per_question.csv`
- `summary_metrics.csv`

Sensitivity study:

Sweep `--pruning-threshold` values (for example 0.5, 0.7, 0.9) and compare `avg_F1` versus `avg_Token_Efficiency_Ratio`.

Recommended runs:

```powershell
python -m baselines.evaluation.run_eval --input "eval Qs.xlsx" --api-base-url http://localhost:8000/api --models advanced,cortex-pruner-only,cortex-critic-only,cortex --pruning-threshold 0.5 --artifact-dir artifacts/eval_t05
python -m baselines.evaluation.run_eval --input "eval Qs.xlsx" --api-base-url http://localhost:8000/api --models advanced,cortex-pruner-only,cortex-critic-only,cortex --pruning-threshold 0.7 --artifact-dir artifacts/eval_t07
python -m baselines.evaluation.run_eval --input "eval Qs.xlsx" --api-base-url http://localhost:8000/api --models advanced,cortex-pruner-only,cortex-critic-only,cortex --pruning-threshold 0.9 --artifact-dir artifacts/eval_t09
```

Plotting guidance:

- X-axis: pruning threshold
- Y-axis 1: `avg_F1` (Cortex or Pruner-Only profile)
- Y-axis 2: `avg_Token_Efficiency_Ratio`
- Look for the threshold point where F1 drops sharply while token savings continue rising.

## Reproducibility Checklist

1. Start required services (Neo4j, KG, web-ui API, reasoning-engine).
2. Confirm API health at `http://localhost:8000/api/health`.
3. Keep dataset version fixed during a run.
4. Run evaluation with the same command and model list.
5. Archive generated CSV artifacts with commit hash.

## Notes for Team Handoff

- This setup is validated for 25 rows and scales to 200+ rows without code changes.
- Final publication-quality retrieval metrics require curated `golden_ids` in the dataset.
- The script is intentionally model-agnostic at the interface layer and can evaluate future Nuvolos-connected versions with the same output schema.
