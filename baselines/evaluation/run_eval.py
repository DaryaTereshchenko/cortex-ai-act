"""Unified evaluation runner for Naive, BM25, Advanced, and CORTEX models."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# Ensure repository root is importable when executed as a script path.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.advanced_rag.run_advanced_baseline import run_advanced_query
from baselines.bm25_baseline import run_bm25_rag_benchmark
from baselines.dense_embedding_baseline import run_dense_embedding_rag_benchmark
from baselines.naive_baseline import run_naive_rag_benchmark


MODEL_DISPLAY_NAMES = {
    "Naive": "Naive (Neo4j full-text lexical baseline)",
    "BM25": "BM25 (rank_bm25 lexical baseline)",
    "Dense": "Dense Embedding (all-MiniLM-L6-v2 semantic retrieval baseline)",
    "Advanced": "Advanced RAG (CORTEX pipeline; pruning=off, critic=off)",
    "Cortex": "Cortex Main (CORTEX pipeline; pruning=on, critic=on)",
}

STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "to",
    "of",
    "in",
    "for",
    "and",
    "or",
    "on",
    "it",
    "this",
    "that",
}


@dataclass
class ModelOutput:
    response: str
    retrieved_ids: list[str]
    retrieved_context: str
    status: str = "completed"
    error: str = ""


def normalize_text(text: str) -> str:
    lowered = text.lower()
    no_punct = lowered.translate(str.maketrans("", "", string.punctuation))
    tokens = [tok for tok in no_punct.split() if tok and tok not in STOPWORDS]
    return " ".join(tokens)


def f1_score(prediction: str, target: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(target).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1

    overlap = 0
    for tok in gold_tokens:
        if pred_counts.get(tok, 0) > 0:
            overlap += 1
            pred_counts[tok] -= 1

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, target: str) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(target) else 0.0


def parse_id_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except Exception:
            pass

    chunks = re.split(r"[;,|]", text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def regulation_from_doc(doc_value: Any) -> str:
    doc = str(doc_value).strip().lower()
    if "dsa" in doc:
        return "dsa"
    if "ai" in doc:
        return "eu_ai_act"
    return "both"


def synthesize_golden_ids(row: pd.Series) -> list[str]:
    doc = str(row.get("Doc", "")).strip().lower()
    article = str(row.get("Article", "")).strip()
    paragraph = str(row.get("Paragraph", "")).strip()

    if not article or article.lower() == "nan":
        return []

    prefix = "dsa" if "dsa" in doc else "eu_ai_act"
    article_num = re.sub(r"[^0-9]", "", article)
    if not article_num:
        return []

    ids = [f"{prefix}_art_{article_num}"]

    para_digits = re.sub(r"[^0-9]", "", paragraph)
    if para_digits:
        ids.append(f"{prefix}_art_{article_num}_par_{para_digits}")

    return ids


def load_eval_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    rename_map = {
        "Question": "question",
        "question": "question",
        "Correct Answer": "expected_answer",
        "expected_answer": "expected_answer",
        "golden_ids": "golden_ids",
    }
    for src, dst in rename_map.items():
        if src in df.columns and src != dst:
            df = df.rename(columns={src: dst})

    if "question" not in df.columns:
        raise ValueError("Input file must include 'Question' or 'question' column")
    if "expected_answer" not in df.columns:
        raise ValueError("Input file must include 'Correct Answer' or 'expected_answer' column")

    if "golden_ids" not in df.columns:
        df["golden_ids"] = df.apply(synthesize_golden_ids, axis=1)
    else:
        df["golden_ids"] = df["golden_ids"].apply(parse_id_list)

    return df


def precision_recall(retrieved: list[str], gold: list[str]) -> tuple[float, float]:
    rset = set(retrieved)
    gset = set(gold)

    if not rset and not gset:
        return 1.0, 1.0
    if not rset and gset:
        return 0.0, 0.0

    true_pos = len(rset.intersection(gset))
    precision = true_pos / len(rset) if rset else 0.0
    recall = true_pos / len(gset) if gset else 1.0
    return precision, recall


def context_token_count(text: str) -> int:
    return len(str(text).split())


def score_faithfulness_with_judge(
    *,
    retrieved_context: str,
    response: str,
    judge_model: str,
) -> float | None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    prompt = (
        "You are evaluating legal QA faithfulness. Score from 1 to 5.\n"
        "5 = answer strictly grounded in provided context.\n"
        "1 = severe hallucination.\n"
        "Return JSON only: {\"score\": number, \"reason\": string}."
    )

    user = (
        f"Retrieved context:\n{retrieved_context}\n\n"
        f"AI answer:\n{response}\n\n"
        "Does the answer contain only information grounded in the context?"
    )

    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=90,
    )
    resp.raise_for_status()
    data = resp.json()
    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "{\"score\": null}")
    )

    try:
        parsed = json.loads(content)
        score = parsed.get("score")
        return float(score) if score is not None else None
    except Exception:
        match = re.search(r"([1-5](?:\.\d+)?)", str(content))
        return float(match.group(1)) if match else None


def run_naive(question: str) -> ModelOutput:
    result = run_naive_rag_benchmark(question)
    context = str(result.get("retrieved_context") or "")
    node_count = int(result.get("nodes_found", 0))

    return ModelOutput(
        response="",  # Naive baseline currently retrieval-only
        retrieved_ids=[],
        retrieved_context=context,
        status="completed" if node_count >= 0 else "failed",
        error="",
    )


def run_bm25(question: str) -> ModelOutput:
    result = run_bm25_rag_benchmark(question, top_k=5)
    context = str(result.get("retrieved_context") or "")
    retrieved_ids = [str(v).strip() for v in result.get("retrieved_ids", []) if str(v).strip()]

    return ModelOutput(
        response="",  # BM25 baseline currently retrieval-only
        retrieved_ids=retrieved_ids,
        retrieved_context=context,
        status="completed",
        error="",
    )


def run_dense(question: str) -> ModelOutput:
    result = run_dense_embedding_rag_benchmark(question, top_k=5)
    context = str(result.get("retrieved_context") or "")
    retrieved_ids = [str(v).strip() for v in result.get("retrieved_ids", []) if str(v).strip()]

    return ModelOutput(
        response="",  # Dense baseline currently retrieval-only
        retrieved_ids=retrieved_ids,
        retrieved_context=context,
        status="completed",
        error="",
    )


def run_advanced(
    *,
    question: str,
    api_base_url: str,
    regulation: str,
    max_hops: int,
    enable_pruning: bool,
    enable_self_correction: bool,
) -> ModelOutput:
    result = run_advanced_query(
        question,
        api_base_url=api_base_url,
        regulation=regulation,
        max_hops=max_hops,
        enable_pruning=enable_pruning,
        enable_self_correction=enable_self_correction,
        poll_interval_seconds=1.0,
        timeout_seconds=240.0,
    )
    return ModelOutput(
        response=result["response"],
        retrieved_ids=result["retrieved_ids"],
        retrieved_context=result["retrieved_context"],
        status=result["status"],
        error=result["error"],
    )


def ensure_artifact_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ground-truth evaluation")
    parser.add_argument("--input", default="eval Qs.xlsx", help="Input dataset path (.xlsx or .csv)")
    parser.add_argument("--api-base-url", default="http://localhost:8000/api")
    parser.add_argument("--max-rows", type=int, default=0, help="0 = all rows")
    parser.add_argument("--artifact-dir", default="artifacts/eval")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument(
        "--models",
        default="naive,advanced,cortex",
        help=(
            "Comma list from: naive,bm25,dense,advanced,cortex "
            "(challenger alias -> advanced)"
        ),
    )
    args = parser.parse_args()

    selected_models = {m.strip().lower() for m in args.models.split(",") if m.strip()}
    if "challenger" in selected_models:
        selected_models.remove("challenger")
        selected_models.add("advanced")

    valid = {"naive", "bm25", "dense", "advanced", "cortex"}
    unknown = selected_models.difference(valid)
    if unknown:
        raise ValueError(f"Unknown model(s): {sorted(unknown)}")

    df = load_eval_dataframe(Path(args.input))
    if args.max_rows > 0:
        df = df.head(args.max_rows)

    artifact_dir = Path(args.artifact_dir)
    ensure_artifact_dir(artifact_dir)

    rows_out: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        question = str(row["question"])
        expected = str(row["expected_answer"])
        gold_ids = parse_id_list(row["golden_ids"])
        regulation = regulation_from_doc(row.get("Doc", ""))

        out_row: dict[str, Any] = {
            "row_index": int(idx),
            "question": question,
            "expected_answer": expected,
            "golden_ids": json.dumps(gold_ids),
            "regulation": regulation,
        }

        if "naive" in selected_models:
            naive = run_naive(question)
            out_row["Naive_Response"] = naive.response
            out_row["Naive_Retrieved_IDs"] = json.dumps(naive.retrieved_ids)
            out_row["Naive_Retrieved_Context"] = naive.retrieved_context
            out_row["Naive_Status"] = naive.status
            out_row["Naive_Error"] = naive.error
            out_row["Naive_Context_Tokens"] = context_token_count(naive.retrieved_context)

            p, r = precision_recall(naive.retrieved_ids, gold_ids)
            out_row["Naive_Precision"] = p
            out_row["Naive_Recall"] = r
            out_row["Naive_EM"] = exact_match(naive.response, expected)
            out_row["Naive_F1"] = f1_score(naive.response, expected)
            out_row["Naive_Faithfulness"] = score_faithfulness_with_judge(
                retrieved_context=naive.retrieved_context,
                response=naive.response,
                judge_model=args.judge_model,
            )

        if "bm25" in selected_models:
            bm25 = run_bm25(question)
            out_row["BM25_Response"] = bm25.response
            out_row["BM25_Retrieved_IDs"] = json.dumps(bm25.retrieved_ids)
            out_row["BM25_Retrieved_Context"] = bm25.retrieved_context
            out_row["BM25_Status"] = bm25.status
            out_row["BM25_Error"] = bm25.error
            out_row["BM25_Context_Tokens"] = context_token_count(bm25.retrieved_context)

            p, r = precision_recall(bm25.retrieved_ids, gold_ids)
            out_row["BM25_Precision"] = p
            out_row["BM25_Recall"] = r
            out_row["BM25_EM"] = exact_match(bm25.response, expected)
            out_row["BM25_F1"] = f1_score(bm25.response, expected)
            out_row["BM25_Faithfulness"] = score_faithfulness_with_judge(
                retrieved_context=bm25.retrieved_context,
                response=bm25.response,
                judge_model=args.judge_model,
            )

        if "dense" in selected_models:
            dense = run_dense(question)
            out_row["Dense_Response"] = dense.response
            out_row["Dense_Retrieved_IDs"] = json.dumps(dense.retrieved_ids)
            out_row["Dense_Retrieved_Context"] = dense.retrieved_context
            out_row["Dense_Status"] = dense.status
            out_row["Dense_Error"] = dense.error
            out_row["Dense_Context_Tokens"] = context_token_count(dense.retrieved_context)

            p, r = precision_recall(dense.retrieved_ids, gold_ids)
            out_row["Dense_Precision"] = p
            out_row["Dense_Recall"] = r
            out_row["Dense_EM"] = exact_match(dense.response, expected)
            out_row["Dense_F1"] = f1_score(dense.response, expected)
            out_row["Dense_Faithfulness"] = score_faithfulness_with_judge(
                retrieved_context=dense.retrieved_context,
                response=dense.response,
                judge_model=args.judge_model,
            )

        if "advanced" in selected_models:
            advanced = run_advanced(
                question=question,
                api_base_url=args.api_base_url.rstrip("/"),
                regulation=regulation,
                max_hops=3,
                enable_pruning=False,
                enable_self_correction=False,
            )
            out_row["Advanced_Response"] = advanced.response
            out_row["Advanced_Retrieved_IDs"] = json.dumps(advanced.retrieved_ids)
            out_row["Advanced_Retrieved_Context"] = advanced.retrieved_context
            out_row["Advanced_Status"] = advanced.status
            out_row["Advanced_Error"] = advanced.error
            out_row["Advanced_Context_Tokens"] = context_token_count(advanced.retrieved_context)

            p, r = precision_recall(advanced.retrieved_ids, gold_ids)
            out_row["Advanced_Precision"] = p
            out_row["Advanced_Recall"] = r
            out_row["Advanced_EM"] = exact_match(advanced.response, expected)
            out_row["Advanced_F1"] = f1_score(advanced.response, expected)
            out_row["Advanced_Faithfulness"] = score_faithfulness_with_judge(
                retrieved_context=advanced.retrieved_context,
                response=advanced.response,
                judge_model=args.judge_model,
            )

        if "cortex" in selected_models:
            cortex = run_advanced(
                question=question,
                api_base_url=args.api_base_url.rstrip("/"),
                regulation=regulation,
                max_hops=3,
                enable_pruning=True,
                enable_self_correction=True,
            )
            out_row["Cortex_Response"] = cortex.response
            out_row["Cortex_Retrieved_IDs"] = json.dumps(cortex.retrieved_ids)
            out_row["Cortex_Retrieved_Context"] = cortex.retrieved_context
            out_row["Cortex_Status"] = cortex.status
            out_row["Cortex_Error"] = cortex.error
            out_row["Cortex_Context_Tokens"] = context_token_count(cortex.retrieved_context)

            p, r = precision_recall(cortex.retrieved_ids, gold_ids)
            out_row["Cortex_Precision"] = p
            out_row["Cortex_Recall"] = r
            out_row["Cortex_EM"] = exact_match(cortex.response, expected)
            out_row["Cortex_F1"] = f1_score(cortex.response, expected)
            out_row["Cortex_Faithfulness"] = score_faithfulness_with_judge(
                retrieved_context=cortex.retrieved_context,
                response=cortex.response,
                judge_model=args.judge_model,
            )

        naive_tokens = float(out_row.get("Naive_Context_Tokens", 0) or 0)
        if "bm25" in selected_models:
            bm25_tokens = float(out_row.get("BM25_Context_Tokens", 0) or 0)
            out_row["BM25_Token_Efficiency_Ratio"] = (
                bm25_tokens / naive_tokens if naive_tokens > 0 else None
            )
        if "dense" in selected_models:
            dense_tokens = float(out_row.get("Dense_Context_Tokens", 0) or 0)
            out_row["Dense_Token_Efficiency_Ratio"] = (
                dense_tokens / naive_tokens if naive_tokens > 0 else None
            )
        if "advanced" in selected_models:
            advanced_tokens = float(out_row.get("Advanced_Context_Tokens", 0) or 0)
            out_row["Advanced_Token_Efficiency_Ratio"] = (
                advanced_tokens / naive_tokens if naive_tokens > 0 else None
            )
        if "cortex" in selected_models:
            cortex_tokens = float(out_row.get("Cortex_Context_Tokens", 0) or 0)
            out_row["Cortex_Token_Efficiency_Ratio"] = (
                cortex_tokens / naive_tokens if naive_tokens > 0 else None
            )

        rows_out.append(out_row)
        print(f"[{len(rows_out):03d}] Completed: {question[:70]}")

    results_df = pd.DataFrame(rows_out)
    results_path = artifact_dir / "results_per_question.csv"
    results_df.to_csv(results_path, index=False)

    summary_rows: list[dict[str, Any]] = []
    for model_name in ["Naive", "BM25", "Dense", "Advanced", "Cortex"]:
        model_cols = [c for c in results_df.columns if c.startswith(f"{model_name}_")]
        if not model_cols:
            continue

        row_summary: dict[str, Any] = {
            "model": model_name,
            "model_display": MODEL_DISPLAY_NAMES.get(model_name, model_name),
        }
        for metric in [
            "Precision",
            "Recall",
            "Faithfulness",
            "EM",
            "F1",
            "Token_Efficiency_Ratio",
            "Context_Tokens",
        ]:
            col = f"{model_name}_{metric}"
            row_summary[f"avg_{metric}"] = (
                float(results_df[col].dropna().mean()) if col in results_df.columns else None
            )
        summary_rows.append(row_summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = artifact_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False)

    case_cols = [
        "question",
        "expected_answer",
        "Naive_F1",
        "Cortex_F1",
        "Naive_Precision",
        "Cortex_Precision",
        "Naive_Recall",
        "Cortex_Recall",
    ]
    available_case_cols = [c for c in case_cols if c in results_df.columns]
    if {"Naive_F1", "Cortex_F1"}.issubset(results_df.columns):
        cases = results_df.copy()
        cases["delta_f1"] = cases["Cortex_F1"] - cases["Naive_F1"]
        cases = cases.sort_values(by="delta_f1", ascending=False).head(5)
        cases_path = artifact_dir / "top5_case_studies.csv"
        cases[available_case_cols + ["delta_f1"]].to_csv(cases_path, index=False)

    print("\\nEvaluation complete.")
    print(f"Per-question results: {results_path}")
    print(f"Summary metrics:      {summary_path}")


if __name__ == "__main__":
    main()
