from __future__ import annotations
import sys, os
sys.path.insert(0, "/files/cortex-ai-act")

import argparse
import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# Exact imports from your original plumbing
from baselines.naive_baseline import run_naive_rag_benchmark

STOPWORDS = {"a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "for", "and", "or", "on", "it", "this", "that"}

@dataclass
class ModelOutput:
    response: str
    retrieved_ids: list[str]
    retrieved_context: str
    status: str = "completed"
    error: str = ""

def normalize_text(text: str) -> str:
    # Handle non-string inputs gracefully
    lowered = str(text).lower()
    no_punct = lowered.translate(str.maketrans("", "", string.punctuation))
    tokens = [tok for tok in no_punct.split() if tok and tok not in STOPWORDS]
    return " ".join(tokens)

def f1_score(prediction: str, target: str) -> float:
    # CLEANING: Remove the Pandas metadata noise (Name: ... dtype: ...)
    target = re.sub(r"Name:.*dtype:.*", "", str(target), flags=re.DOTALL).strip()
    
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(target).split()
    if not pred_tokens and not gold_tokens: return 1.0
    if not pred_tokens or not gold_tokens: return 0.0
    
    # Using set intersection for efficiency
    pred_counts = {}
    for tok in pred_tokens: pred_counts[tok] = pred_counts.get(tok, 0) + 1
    overlap = 0
    for tok in gold_tokens:
        if pred_counts.get(tok, 0) > 0:
            overlap += 1
            pred_counts[tok] -= 1
            
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    if precision + recall == 0: return 0.0
    return 2 * precision * recall / (precision + recall)

def parse_id_list(value: Any) -> list[str]:
    if value is None or pd.isna(value): return []
    if isinstance(value, list): return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text: return []
    # Handles list-like strings [id1, id2] or plain CSV strings
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text.replace("'", '"'))
            if isinstance(parsed, list): return [str(v).strip() for v in parsed if str(v).strip()]
        except Exception: pass
    chunks = re.split(r"[;,|\s]", text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def precision_recall(retrieved: list[str], gold: list[str]) -> tuple[float, float]:
    rset = set([str(i).strip() for i in retrieved if i])
    gset = set([str(i).strip() for i in gold if i])
    if not gset:
        return (1.0, 1.0) if not rset else (0.0, 0.0)
    if not rset:
        return 0.0, 0.0
    true_pos = len(rset.intersection(gset))
    return true_pos / len(rset), true_pos / len(gset)

def run_naive(question: str) -> ModelOutput:
    result = run_naive_rag_benchmark(question)
    return ModelOutput(
        response=str(result.get("answer") or ""),
        retrieved_ids=result.get("retrieved_ids", []),
        retrieved_context=str(result.get("retrieved_context") or ""),
        status="completed"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="eval Qs.xlsx")
    parser.add_argument("--max-rows", type=int, default=1) 
    args = parser.parse_args()

    path = Path(args.input)
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if args.max_rows > 0: df = df.head(args.max_rows)

    rows_out = []
    for idx, row in df.iterrows():
        # UNIFIED INDEXING: 
        # Col E (4) = Question
        # Col L (11) = Expected Answer (Golden)
        # Col K (10) = Golden IDs
        question = str(row.iloc[4])
        expected = str(row.iloc[11])
        gold_ids = parse_id_list(row.iloc[10])
        
        # Execute Naive RAG
        naive = run_naive(question)
        p, r = precision_recall(naive.retrieved_ids, gold_ids)
        
        rows_out.append({
            "question": question,
            "expected_answer": expected,
            "Naive_Response": naive.response,
            "Naive_F1": f1_score(naive.response, expected),
            "Naive_Precision": p,
            "Naive_Recall": r,
            "Naive_Status": naive.status
        })
        print(f"[{idx+1}] Completed Naive Eval: {question[:50]}...")

    pd.DataFrame(rows_out).to_csv("naive_results_only.csv", index=False)
    print("\n✅ Done. Results saved to naive_results_only.csv")

if __name__ == "__main__":
    main()