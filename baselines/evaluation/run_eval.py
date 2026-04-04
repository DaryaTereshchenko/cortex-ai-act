"""Unified evaluation runner for Naive, BM25, Advanced, and CORTEX models."""

from __future__ import annotations

import argparse
import json
import re
import sys
import string
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Ensure repository root is importable when executed as a script path.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.advanced_rag.run_advanced_baseline import run_advanced_query
from baselines.bm25_baseline import run_bm25_rag_benchmark
from baselines.dense_embedding_baseline import run_dense_embedding_rag_benchmark
from baselines.evaluation.metrics import (
    golden_id_metrics,
    generator_semantic_metrics,
    generator_statistical_metrics,
    mcq_retrieval_metrics,
    retrieval_text_metrics,
    source_identification_metrics,
)
from baselines.judge_rag.config import JudgeRAGConfig
from baselines.judge_rag.pipeline import run_judge_rag
from baselines.naive_baseline import run_naive_rag_benchmark


MODEL_DISPLAY_NAMES = {
    "Naive": "Naive (Neo4j full-text lexical baseline)",
    "BM25": "BM25 (rank_bm25 lexical baseline)",
    "Dense": "Dense Embedding (all-MiniLM-L6-v2 semantic retrieval baseline)",
    "Advanced": "Advanced RAG (CORTEX pipeline; pruning=off, critic=off)",
    "Cortex_Pruner_Only": "Cortex Pruner Only (pruning=on, critic=off)",
    "Cortex_Critic_Only": "Cortex Critic Only (pruning=off, critic=on)",
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
    lowered = str(text).lower()
    no_punct = lowered.translate(str.maketrans("", "", string.punctuation))
    tokens = [tok for tok in no_punct.split() if tok and tok not in STOPWORDS]
    return " ".join(tokens)


def sanitize_expected_text(text: Any) -> str:
    """Strip noisy pandas Series metadata before text-based scoring."""
    return re.sub(r"Name:.*dtype:.*", "", str(text), flags=re.DOTALL).strip()


def get_expected_answer(row: pd.Series, model_name: str, use_cortex_evals: bool) -> str:
    """
    Get the expected answer for a model.
    If use_cortex_evals is True and cortex_expected_answer exists, use it for Cortex models.
    Otherwise use expected_answer.
    """
    is_cortex_model = model_name in ("Cortex_Pruner_Only", "Cortex_Critic_Only", "Cortex")
    if use_cortex_evals and is_cortex_model and "cortex_expected_answer" in row.index:
        return str(row.get("cortex_expected_answer", row["expected_answer"]))
    return str(row["expected_answer"])


def f1_score(prediction: str, target: str) -> float:
    target = sanitize_expected_text(target)
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
    target = sanitize_expected_text(target)
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

    chunks = re.split(r"[;,|\s]", text)
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

    # Unified indexing required by eval handoff:
    # Col E (index 4)  -> question
    # Col K (index 10) -> golden_ids
    # Col L (index 11) -> expected_answer
    if len(df.columns) >= 12:
        df = df.copy()
        df["question"] = df.iloc[:, 4].astype(str)
        df["golden_ids"] = df.iloc[:, 10].apply(parse_id_list)
        df["expected_answer"] = df.iloc[:, 11].astype(str)

        # Keep cortex_expected_answer aligned to col L in indexed mode unless explicitly provided.
        if "cortex_expected_answer" in df.columns:
            df["cortex_expected_answer"] = df["cortex_expected_answer"].fillna(df["expected_answer"])
        else:
            df["cortex_expected_answer"] = df["expected_answer"]
        return df

    rename_map = {
        "Question": "question",
        "question": "question",
        "Correct Answer": "expected_answer",
        "expected_answer": "expected_answer",
        "golden_ids": "golden_ids",
        "Cortex Expected Answer": "cortex_expected_answer",
        "cortex_expected_answer": "cortex_expected_answer",
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

    # If cortex_expected_answer column exists but is incomplete, fill with expected_answer
    if "cortex_expected_answer" in df.columns:
        df["cortex_expected_answer"] = df["cortex_expected_answer"].fillna(df["expected_answer"])
    
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


def compute_comprehensive_metrics(
    prefix: str,
    model_out: "ModelOutput",
    row: pd.Series,
    gold_ids: list[str],
    expected_answer: str,
    has_generator: bool,
    encoder: Any = None,
) -> dict[str, Any]:
    """Compute all metric groups for a single model on a single question.

    Returns a flat dict with keys prefixed by *prefix* (e.g. 'Naive_').
    """
    out: dict[str, Any] = {}

    # -- raw outputs --
    out[f"{prefix}_Response"] = model_out.response
    out[f"{prefix}_Retrieved_IDs"] = json.dumps(model_out.retrieved_ids)
    out[f"{prefix}_Retrieved_Context"] = model_out.retrieved_context
    out[f"{prefix}_Status"] = model_out.status
    out[f"{prefix}_Error"] = model_out.error
    out[f"{prefix}_Context_Tokens"] = context_token_count(model_out.retrieved_context)

    # --- 1. Retrieval Source Identification ---
    gold_doc = str(row.get("Doc", ""))
    gold_chapter = str(row.get("Chapter", ""))
    gold_article = row.get("Article", "")
    gold_paragraph = row.get("Paragraph", "")

    src_metrics = source_identification_metrics(
        model_out.retrieved_ids, gold_doc, gold_chapter, gold_article, gold_paragraph,
    )
    for k, v in src_metrics.items():
        out[f"{prefix}_{k}"] = v

    # --- 2. Retrieval Text Quality (Correct Answer vs context) ---
    correct_answer = str(row.get("Correct Answer", row.get("correct_answer", "")))
    if correct_answer and correct_answer.lower() not in ("nan", "none", ""):
        txt_metrics = retrieval_text_metrics(model_out.retrieved_context, correct_answer)
        for k, v in txt_metrics.items():
            out[f"{prefix}_{k}"] = v

    # --- 3. MCQ Retrieval ---
    alternatives = []
    for alt_col in ["Alt_1", "Alt_2", "Alt_3", "Alt_4"]:
        alt_val = str(row.get(alt_col, ""))
        if alt_val and alt_val.lower() not in ("nan", "none", ""):
            alternatives.append(alt_val)

    if correct_answer and correct_answer.lower() not in ("nan", "none", "") and alternatives:
        mcq_metrics = mcq_retrieval_metrics(model_out.retrieved_context, correct_answer, alternatives)
        for k, v in mcq_metrics.items():
            out[f"{prefix}_{k}"] = v

    # --- 4. Golden ID Retrieval ---
    gid_metrics = golden_id_metrics(model_out.retrieved_ids, gold_ids)
    for k, v in gid_metrics.items():
        out[f"{prefix}_{k}"] = v

    # --- 5. Generator Quality ---
    if has_generator and model_out.response.strip():
        gen_stat = generator_statistical_metrics(model_out.response, expected_answer)
        for k, v in gen_stat.items():
            out[f"{prefix}_{k}"] = v

        gen_sem = generator_semantic_metrics(model_out.response, expected_answer, encoder=encoder)
        for k, v in gen_sem.items():
            out[f"{prefix}_{k}"] = v

    return out


def run_naive(question: str) -> ModelOutput:
    result = run_naive_rag_benchmark(question)
    context = str(result.get("retrieved_context") or "")
    node_count = int(result.get("nodes_found", 0))
    retrieved_ids = [str(v).strip() for v in result.get("retrieved_ids", []) if str(v).strip()]

    return ModelOutput(
        response="",  # Naive baseline currently retrieval-only
        retrieved_ids=retrieved_ids,
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
    regulation: str,
    max_hops: int,
    enable_pruning: bool,
    enable_self_correction: bool,
    pruning_threshold: float,
) -> ModelOutput:
    result = run_advanced_query(
        question,
        regulation=regulation,
        max_hops=max_hops,
        enable_pruning=enable_pruning,
        enable_self_correction=enable_self_correction,
        pruning_threshold=pruning_threshold,
    )
    return ModelOutput(
        response=result["response"],
        retrieved_ids=result["retrieved_ids"],
        retrieved_context=result["retrieved_context"],
        status=result["status"],
        error=result["error"],
    )


@dataclass
class JudgeModelOutput(ModelOutput):
    """Extended output for Judge RAG with judge-specific metrics."""
    judge_score: float = 0.0
    judge_attempts: int = 0
    retrieval_similarity: float = 0.0
    uncertainty: str = "high"
    judge_relevance: float = 0.0
    judge_faithfulness: float = 0.0
    judge_completeness: float = 0.0
    judge_precision_score: float = 0.0


def run_judge(
    *,
    question: str,
    regulation: str,
    judge_rag_retrieval_model: str,
    judge_rag_generation_model: str,
    judge_rag_judge_model: str,
    ollama_base_url: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
) -> JudgeModelOutput:
    cfg = JudgeRAGConfig(
        regulation=regulation,
        retrieval_model=judge_rag_retrieval_model,
        generation_model=judge_rag_generation_model,
        judge_model=judge_rag_judge_model,
        ollama_base_url=ollama_base_url,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
    )
    try:
        result = run_judge_rag(question, cfg)
        # Extract per-dimension judge scores from the last attempt
        judge_details = result.get("judge_details", {})
        last_attempt_key = f"judge_attempt_{result.get('attempts_used', 1)}"
        last_judge = judge_details.get(last_attempt_key, {})

        return JudgeModelOutput(
            response=result["answer"],
            retrieved_ids=result["retrieved_ids"],
            retrieved_context=result["retrieved_context"],
            status="completed",
            error="",
            judge_score=result.get("judge_score", 0.0),
            judge_attempts=result.get("attempts_used", 0),
            retrieval_similarity=result.get("retrieval_similarity", 0.0),
            uncertainty=result.get("uncertainty", "high"),
            judge_relevance=last_judge.get("relevance", 0.0),
            judge_faithfulness=last_judge.get("faithfulness", 0.0),
            judge_completeness=last_judge.get("completeness", 0.0),
            judge_precision_score=last_judge.get("precision", 0.0),
        )
    except Exception as exc:
        return JudgeModelOutput(
            response="",
            retrieved_ids=[],
            retrieved_context="",
            status="failed",
            error=str(exc),
        )


def ensure_artifact_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ground-truth evaluation")
    parser.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parent / "data" / "EU AI and DSA Compliance Dataset.xlsx"),
        help="Input dataset path (.xlsx or .csv)",
    )
    parser.add_argument("--reasoning-engine-url", default="http://localhost:8002",
                        help="(unused, kept for CLI compat)")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Maximum rows to evaluate (0 = all rows, default: 0)",
    )
    parser.add_argument("--artifact-dir", default="artifacts/eval")
    parser.add_argument(
        "--pruning-threshold",
        type=float,
        default=0.45,
        help="Pruning threshold for profiles that enable pruning (0.0-1.0)",
    )
    parser.add_argument(
        "--models",
        default="naive,bm25,dense,advanced,cortex,judge",
        help=(
            "Comma list of models to evaluate. "
            "Options: naive, bm25, dense, advanced, cortex-pruner-only, "
            "cortex-critic-only, cortex, judge"
        ),
    )
    parser.add_argument(
        "--cortex-only-evals",
        action="store_true",
        default=False,
        help="Use cortex_expected_answer column for Cortex models when available",
    )
    # Judge RAG pipeline options
    parser.add_argument("--judge-rag-retrieval-model", default="llama3.1:8b",
                        help="Ollama model for Judge RAG query rewriting (default: llama3.1:8b)")
    parser.add_argument("--judge-rag-generation-model", default="llama3.1:8b",
                        help="Ollama model for Judge RAG answer generation (default: llama3.1:8b)")
    parser.add_argument("--judge-rag-judge-model", default="phi4:latest",
                        help="Ollama model for Judge RAG judge evaluation (default: phi4:latest)")
    parser.add_argument("--ollama-base-url", default="http://localhost:11434")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="changeme")
    args = parser.parse_args()

    selected_models = {m.strip().lower() for m in args.models.split(",") if m.strip()}
    valid = {
        "naive", "bm25", "dense", "advanced",
        "cortex-pruner-only", "cortex-critic-only", "cortex", "judge",
    }
    unknown = selected_models.difference(valid)
    if unknown:
        raise ValueError(f"Unknown model(s): {sorted(unknown)}. Valid: {sorted(valid)}")

    df = load_eval_dataframe(Path(args.input))
    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows)

    artifact_dir = Path(args.artifact_dir)
    ensure_artifact_dir(artifact_dir)

    # Load semantic encoder once for all generator semantic metrics
    _semantic_encoder = None
    has_generation_model = any(
        m in selected_models
        for m in ("advanced", "cortex-pruner-only", "cortex-critic-only", "cortex", "judge")
    )
    if has_generation_model:
        try:
            from sentence_transformers import SentenceTransformer
            _semantic_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            print("WARNING: sentence-transformers not available; semantic metrics will be NaN")

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
            "doc": str(row.get("Doc", "")),
            "chapter": str(row.get("Chapter", "")),
            "article": str(row.get("Article", "")),
            "paragraph": str(row.get("Paragraph", "")),
        }

        if "naive" in selected_models:
            naive = run_naive(question)
            out_row.update(compute_comprehensive_metrics(
                "Naive", naive, row, gold_ids, expected,
                has_generator=False, encoder=_semantic_encoder,
            ))

        if "bm25" in selected_models:
            bm25 = run_bm25(question)
            out_row.update(compute_comprehensive_metrics(
                "BM25", bm25, row, gold_ids, expected,
                has_generator=False, encoder=_semantic_encoder,
            ))

        if "dense" in selected_models:
            dense = run_dense(question)
            out_row.update(compute_comprehensive_metrics(
                "Dense", dense, row, gold_ids, expected,
                has_generator=False, encoder=_semantic_encoder,
            ))

        if "advanced" in selected_models:
            advanced = run_advanced(
                question=question,
                regulation=regulation,
                max_hops=3,
                enable_pruning=False,
                enable_self_correction=False,
                pruning_threshold=args.pruning_threshold,
            )
            out_row.update(compute_comprehensive_metrics(
                "Advanced", advanced, row, gold_ids, expected,
                has_generator=True, encoder=_semantic_encoder,
            ))

        if "cortex-pruner-only" in selected_models:
            cortex_pruner_only = run_advanced(
                question=question,
                regulation=regulation,
                max_hops=3,
                enable_pruning=True,
                enable_self_correction=False,
                pruning_threshold=args.pruning_threshold,
            )
            cortex_pruner_expected = get_expected_answer(row, "Cortex_Pruner_Only", args.cortex_only_evals)
            out_row.update(compute_comprehensive_metrics(
                "Cortex_Pruner_Only", cortex_pruner_only, row, gold_ids, cortex_pruner_expected,
                has_generator=True, encoder=_semantic_encoder,
            ))

        if "cortex-critic-only" in selected_models:
            cortex_critic_only = run_advanced(
                question=question,
                regulation=regulation,
                max_hops=3,
                enable_pruning=False,
                enable_self_correction=True,
                pruning_threshold=args.pruning_threshold,
            )
            cortex_critic_expected = get_expected_answer(row, "Cortex_Critic_Only", args.cortex_only_evals)
            out_row.update(compute_comprehensive_metrics(
                "Cortex_Critic_Only", cortex_critic_only, row, gold_ids, cortex_critic_expected,
                has_generator=True, encoder=_semantic_encoder,
            ))

        if "cortex" in selected_models:
            cortex = run_advanced(
                question=question,
                regulation=regulation,
                max_hops=3,
                enable_pruning=True,
                enable_self_correction=True,
                pruning_threshold=args.pruning_threshold,
            )
            cortex_main_expected = get_expected_answer(row, "Cortex", args.cortex_only_evals)
            out_row.update(compute_comprehensive_metrics(
                "Cortex", cortex, row, gold_ids, cortex_main_expected,
                has_generator=True, encoder=_semantic_encoder,
            ))

        if "judge" in selected_models:
            judge_out = run_judge(
                question=question,
                regulation=regulation,
                judge_rag_retrieval_model=args.judge_rag_retrieval_model,
                judge_rag_generation_model=args.judge_rag_generation_model,
                judge_rag_judge_model=args.judge_rag_judge_model,
                ollama_base_url=args.ollama_base_url,
                neo4j_uri=args.neo4j_uri,
                neo4j_user=args.neo4j_user,
                neo4j_password=args.neo4j_password,
            )
            # Wrap in ModelOutput for comprehensive metrics
            judge_model_out = ModelOutput(
                response=judge_out.response,
                retrieved_ids=judge_out.retrieved_ids,
                retrieved_context=judge_out.retrieved_context,
                status=judge_out.status,
                error=judge_out.error,
            )
            out_row.update(compute_comprehensive_metrics(
                "Judge", judge_model_out, row, gold_ids, expected,
                has_generator=True, encoder=_semantic_encoder,
            ))

            # Judge-specific LLM-as-judge metrics (unique to this pipeline)
            out_row["Judge_Retrieval_Similarity"] = judge_out.retrieval_similarity
            out_row["Judge_LLM_Score"] = judge_out.judge_score
            out_row["Judge_LLM_Relevance"] = judge_out.judge_relevance
            out_row["Judge_LLM_Faithfulness"] = judge_out.judge_faithfulness
            out_row["Judge_LLM_Completeness"] = judge_out.judge_completeness
            out_row["Judge_LLM_Precision"] = judge_out.judge_precision_score
            out_row["Judge_Attempts"] = judge_out.judge_attempts
            out_row["Judge_Uncertainty"] = judge_out.uncertainty

        # Token efficiency ratios (relative to Naive baseline)
        naive_tokens = float(out_row.get("Naive_Context_Tokens", 0) or 0)
        for model_key, sel_key in [
            ("BM25", "bm25"), ("Dense", "dense"), ("Advanced", "advanced"),
            ("Cortex_Pruner_Only", "cortex-pruner-only"),
            ("Cortex_Critic_Only", "cortex-critic-only"),
            ("Cortex", "cortex"), ("Judge", "judge"),
        ]:
            if sel_key in selected_models:
                tok = float(out_row.get(f"{model_key}_Context_Tokens", 0) or 0)
                out_row[f"{model_key}_Token_Efficiency_Ratio"] = (
                    tok / naive_tokens if naive_tokens > 0 else None
                )

        rows_out.append(out_row)
        print(f"[{len(rows_out):03d}] Completed: {question[:70]}")

    results_df = pd.DataFrame(rows_out)
    results_path = artifact_dir / "results_per_question.csv"
    results_df.to_csv(results_path, index=False)

    # ── Build separate metric CSVs per group ───────────────────────────────
    all_model_prefixes = [
        "Naive", "BM25", "Dense", "Advanced",
        "Cortex_Pruner_Only", "Cortex_Critic_Only", "Cortex", "Judge",
    ]
    active_prefixes = [
        p for p in all_model_prefixes
        if any(c.startswith(f"{p}_") for c in results_df.columns)
    ]

    # Helper: extract per-model average for given metric suffixes
    def _model_avgs(metric_suffixes: list[str]) -> list[dict[str, Any]]:
        rows = []
        for model in active_prefixes:
            r: dict[str, Any] = {
                "model": model,
                "model_display": MODEL_DISPLAY_NAMES.get(model, model),
            }
            for sfx in metric_suffixes:
                col = f"{model}_{sfx}"
                if col in results_df.columns:
                    vals = pd.to_numeric(results_df[col], errors="coerce").dropna()
                    r[f"avg_{sfx}"] = float(vals.mean()) if len(vals) else None
                    r[f"std_{sfx}"] = float(vals.std()) if len(vals) > 1 else None
                else:
                    r[f"avg_{sfx}"] = None
                    r[f"std_{sfx}"] = None
            rows.append(r)
        return rows

    # 1. Retrieval Source Identification
    source_metrics = _model_avgs([
        "doc_accuracy", "article_accuracy", "paragraph_accuracy",
        "doc_hit", "article_hit", "paragraph_hit",
    ])
    pd.DataFrame(source_metrics).to_csv(artifact_dir / "retrieval_source_metrics.csv", index=False)

    # 2. Retrieval Text Quality (Correct Answer overlap)
    text_metrics = _model_avgs([
        "retrieval_token_precision", "retrieval_token_recall", "retrieval_token_f1",
        "retrieval_rouge1_f1", "retrieval_rouge1_recall",
        "retrieval_rouge2_f1", "retrieval_rouge2_recall",
        "retrieval_rouge_l_f1", "retrieval_rouge_l_recall",
        "retrieval_exact_containment", "retrieval_substring_containment",
    ])
    pd.DataFrame(text_metrics).to_csv(artifact_dir / "retrieval_text_metrics.csv", index=False)

    # 3. Retrieval MCQ
    mcq_metrics = _model_avgs([
        "mcq_accuracy", "mcq_correct_rank", "mcq_mrr",
        "mcq_margin", "mcq_normalized_margin", "mcq_discrimination_score",
    ])
    pd.DataFrame(mcq_metrics).to_csv(artifact_dir / "retrieval_mcq_metrics.csv", index=False)

    # 4. Golden ID Retrieval
    id_metrics = _model_avgs([
        "id_precision", "id_recall", "id_f1", "id_jaccard",
        "id_hit_at_1", "id_hit_at_3", "id_hit_at_5",
        "id_mrr", "id_map",
        "id_ndcg_at_3", "id_ndcg_at_5", "id_ndcg_at_10",
    ])
    pd.DataFrame(id_metrics).to_csv(artifact_dir / "retrieval_id_metrics.csv", index=False)

    # 5. Generator Quality
    gen_metrics = _model_avgs([
        "gen_exact_match", "gen_token_precision", "gen_token_recall", "gen_token_f1",
        "gen_rouge1_precision", "gen_rouge1_recall", "gen_rouge1_f1",
        "gen_rouge2_precision", "gen_rouge2_recall", "gen_rouge2_f1",
        "gen_rouge_l_precision", "gen_rouge_l_recall", "gen_rouge_l_f1",
        "gen_bleu", "gen_meteor", "gen_length_ratio",
        "gen_semantic_similarity", "gen_bertscore_precision", "gen_bertscore_recall", "gen_bertscore_f1",
    ])
    pd.DataFrame(gen_metrics).to_csv(artifact_dir / "generator_metrics.csv", index=False)

    # 6. Efficiency + Context metrics
    eff_metrics = _model_avgs(["Context_Tokens", "Token_Efficiency_Ratio"])
    pd.DataFrame(eff_metrics).to_csv(artifact_dir / "efficiency_metrics.csv", index=False)

    # ── Combined summary (backward-compatible + new metrics) ───────────────
    summary_rows: list[dict[str, Any]] = []
    for model_name in active_prefixes:
        row_summary: dict[str, Any] = {
            "model": model_name,
            "model_display": MODEL_DISPLAY_NAMES.get(model_name, model_name),
        }
        # Key retrieval metrics
        for metric in [
            "id_precision", "id_recall", "id_f1", "id_hit_at_5", "id_mrr", "id_map",
            "doc_accuracy", "article_accuracy",
            "retrieval_token_recall", "retrieval_rouge_l_f1",
            "mcq_accuracy", "mcq_mrr",
        ]:
            col = f"{model_name}_{metric}"
            if col in results_df.columns:
                vals = pd.to_numeric(results_df[col], errors="coerce").dropna()
                row_summary[f"avg_{metric}"] = float(vals.mean()) if len(vals) else None

        # Key generation metrics
        for metric in [
            "gen_exact_match", "gen_token_f1", "gen_rouge_l_f1", "gen_bleu", "gen_meteor",
            "gen_semantic_similarity", "gen_bertscore_f1",
        ]:
            col = f"{model_name}_{metric}"
            if col in results_df.columns:
                vals = pd.to_numeric(results_df[col], errors="coerce").dropna()
                row_summary[f"avg_{metric}"] = float(vals.mean()) if len(vals) else None

        # Efficiency
        for metric in ["Context_Tokens", "Token_Efficiency_Ratio"]:
            col = f"{model_name}_{metric}"
            if col in results_df.columns:
                vals = pd.to_numeric(results_df[col], errors="coerce").dropna()
                row_summary[f"avg_{metric}"] = float(vals.mean()) if len(vals) else None

        # Judge-specific metrics
        if model_name == "Judge":
            for metric in [
                "LLM_Score", "LLM_Relevance", "LLM_Faithfulness",
                "LLM_Completeness", "LLM_Precision", "Attempts",
                "Retrieval_Similarity",
            ]:
                col = f"Judge_{metric}"
                if col in results_df.columns:
                    vals = pd.to_numeric(results_df[col], errors="coerce").dropna()
                    row_summary[f"avg_{metric}"] = float(vals.mean()) if len(vals) else None
            unc_col = "Judge_Uncertainty"
            if unc_col in results_df.columns:
                unc_counts = results_df[unc_col].value_counts().to_dict()
                row_summary["uncertainty_distribution"] = json.dumps(unc_counts)

        summary_rows.append(row_summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = artifact_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False)

    # ── Write eval metadata (models, params, timestamp) ────────────────────
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_file": str(Path(args.input).name),
        "total_questions": len(results_df),
        "models_evaluated": sorted(selected_models),
        "embedding_model": "all-MiniLM-L6-v2",
        "judge_rag": {
            "retrieval_model": args.judge_rag_retrieval_model,
            "generation_model": args.judge_rag_generation_model,
            "judge_model": args.judge_rag_judge_model,
            "ollama_base_url": args.ollama_base_url,
        },
        "cortex": {
            "pruning_threshold": args.pruning_threshold,
        },
        "neo4j_uri": args.neo4j_uri,
        "metric_groups": [
            "retrieval_source_metrics.csv",
            "retrieval_text_metrics.csv",
            "retrieval_mcq_metrics.csv",
            "retrieval_id_metrics.csv",
            "generator_metrics.csv",
            "efficiency_metrics.csv",
            "summary_metrics.csv",
        ],
    }
    metadata_path = artifact_dir / "eval_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    # ── Top-5 case studies (backward-compat) ───────────────────────────────
    case_cols = ["question", "expected_answer"]
    for prefix in all_model_prefixes:
        for metric in ["gen_token_f1", "id_precision", "id_recall"]:
            col = f"{prefix}_{metric}"
            if col in results_df.columns:
                case_cols.append(col)
    available_case_cols = [c for c in case_cols if c in results_df.columns]
    if {"Naive_gen_token_f1", "Cortex_gen_token_f1"}.issubset(results_df.columns):
        cases = results_df.copy()
        cases["delta_f1"] = cases["Cortex_gen_token_f1"] - cases["Naive_gen_token_f1"]
        cases = cases.sort_values(by="delta_f1", ascending=False).head(5)
        cases_path = artifact_dir / "top5_case_studies.csv"
        cases[available_case_cols + ["delta_f1"]].to_csv(cases_path, index=False)

    print("\nEvaluation complete.")
    print(f"Per-question results:          {results_path}")
    print(f"Summary metrics:               {summary_path}")
    print(f"Retrieval source metrics:      {artifact_dir / 'retrieval_source_metrics.csv'}")
    print(f"Retrieval text metrics:        {artifact_dir / 'retrieval_text_metrics.csv'}")
    print(f"Retrieval MCQ metrics:         {artifact_dir / 'retrieval_mcq_metrics.csv'}")
    print(f"Retrieval ID metrics:          {artifact_dir / 'retrieval_id_metrics.csv'}")
    print(f"Generator metrics:             {artifact_dir / 'generator_metrics.csv'}")
    print(f"Efficiency metrics:            {artifact_dir / 'efficiency_metrics.csv'}")
    print(f"Eval metadata:                 {metadata_path}")


if __name__ == "__main__":
    main()
