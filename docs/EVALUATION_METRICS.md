# Evaluation Metrics Reference

This document describes every metric computed by the evaluation pipeline (`baselines/evaluation/run_eval.py`) across all RAG systems.

## 1. Retrieval Metrics

These metrics evaluate how well the retrieval component identifies the correct legal provisions from the knowledge graph.

| Metric | Description | Range |
|--------|-------------|-------|
| **Precision** | Fraction of retrieved document IDs that appear in the golden (ground-truth) ID set. High precision means few irrelevant documents retrieved. | 0.0 – 1.0 |
| **Recall** | Fraction of golden IDs that were actually retrieved. High recall means the system finds most of the relevant provisions. | 0.0 – 1.0 |
| **Retrieval Similarity** | Average cosine similarity between the query embedding and the retrieved chunk embeddings. Measures semantic relevance of retrieved context. *(Judge RAG only)* | 0.0 – 1.0 |
| **Context Tokens** | Total number of whitespace-delimited tokens in the retrieved context. Indicates retrieval verbosity. | ≥ 0 |
| **Token Efficiency Ratio** | Ratio of a model's context tokens to the Naive baseline's context tokens. Values < 1.0 mean more concise retrieval. | ≥ 0.0 |

## 2. Generation Metrics

These metrics evaluate the quality of the generated answer against the ground-truth expected answer.

| Metric | Description | Range |
|--------|-------------|-------|
| **Exact Match (EM)** | Binary score — 1.0 if the normalised prediction exactly matches the normalised expected answer, 0.0 otherwise. Normalisation: lowercase, strip punctuation, remove stopwords. | 0.0 or 1.0 |
| **F1 Score** | Token-level F1 between the normalised prediction and the normalised expected answer. Balances precision (fraction of predicted tokens that are correct) and recall (fraction of gold tokens that are predicted). | 0.0 – 1.0 |
| **Faithfulness** | LLM-judged faithfulness score (1–5) indicating whether the generated answer is grounded in the retrieved context. 5 = fully grounded, 1 = severe hallucination. Requires an OpenAI API key (`OPENAI_API_KEY` env var); `null` if unavailable. | 1.0 – 5.0 |

### Normalisation Details

Both EM and F1 apply the following normalisation before comparison:
1. Lowercase all text
2. Remove all punctuation
3. Remove common English stopwords (a, an, the, is, are, was, were, be, to, of, in, for, and, or, on, it, this, that)
4. Split on whitespace

## 3. Judge-Specific Metrics (Judge RAG Only)

The Judge RAG pipeline includes an internal LLM-as-judge loop that evaluates its own generated answers. These metrics capture the judge's self-assessment.

| Metric | Description | Range |
|--------|-------------|-------|
| **Judge LLM Score** | Overall quality score assigned by the judge LLM. The pipeline re-generates if below the acceptance threshold (default: 7.0). | 1.0 – 10.0 |
| **Judge LLM Relevance** | Does the answer address the user's question? | 1 – 10 |
| **Judge LLM Faithfulness** | Is every claim in the answer supported by the retrieved context? Penalises hallucinations. | 1 – 10 |
| **Judge LLM Completeness** | Are all relevant legal provisions from the context mentioned? | 1 – 10 |
| **Judge LLM Precision** | Are Article / Recital citations accurate? | 1 – 10 |
| **Judge Attempts** | Number of generate-judge iterations used before the answer was accepted (or max attempts reached). | 1 – max_attempts |
| **Judge Uncertainty** | Qualitative uncertainty label derived from the judge score: `low` (≥ 8), `medium` (5–8), `high` (< 5). | low / medium / high |

## 4. Pipeline-Level Summary Metrics

The summary CSV (`summary_metrics.csv`) reports the **mean** of each metric across all evaluated questions, per model. For the Judge model, the summary additionally includes:

- Mean LLM Score, Relevance, Faithfulness, Completeness, Precision
- Mean number of attempts
- Mean retrieval similarity
- Uncertainty distribution (JSON counts of low/medium/high)

## 5. Evaluated RAG Pipelines

| Model Key | Description |
|-----------|-------------|
| **Naive** | Neo4j full-text lexical search baseline (retrieval only, no generation) |
| **BM25** | rank_bm25 lexical baseline over article corpus (retrieval only) |
| **Dense** | all-MiniLM-L6-v2 dense embedding semantic retrieval baseline (retrieval only) |
| **Advanced** | CORTEX pipeline with pruning=off, critic=off |
| **Cortex_Pruner_Only** | CORTEX pipeline with pruning=on, critic=off |
| **Cortex_Critic_Only** | CORTEX pipeline with pruning=off, critic=on |
| **Cortex** | Full CORTEX pipeline with pruning=on, critic=on |
| **Judge** | Judge RAG — hybrid BM25+semantic retrieval with LLM-as-judge re-generation loop (default model: phi-4) |

## 6. Dataset

The evaluation uses `EU AI and DSA Compliance Dataset.xlsx` (200 questions across the EU AI Act and Digital Services Act). Columns:

| Column | Description |
|--------|-------------|
| Doc | Source regulation (EU AI Act / DSA) |
| Chapter / Article / Paragraph | Legal provision reference |
| Question | Evaluation question |
| Correct Answer | Ground-truth correct answer |
| Alt_1 – Alt_4 | Distractor alternatives (multiple-choice) |
| golden_ids | Comma-separated ground-truth document IDs |
| expected_answer | Extended expected answer for free-form evaluation |

## 7. Output Artifacts

| File | Contents |
|------|----------|
| `results_per_question.csv` | Per-question scores for every model and metric |
| `summary_metrics.csv` | Aggregated mean metrics per model |
| `top5_case_studies.csv` | Top 5 questions with largest Cortex-vs-Naive F1 improvement |
