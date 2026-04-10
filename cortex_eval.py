import sys, os
sys.path.insert(0, "/files/cortex-ai-act")

import argparse
import pandas as pd
import requests
import re
from pathlib import Path
from typing import Any
import string

STOPWORDS = {"a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "for", "and", "or", "on", "it", "this", "that"}

def normalize_text(text: str) -> str:
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
    overlap = len(set(pred_tokens) & set(gold_tokens))
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

def run_cortex_request(question: str):
    """Hits the main /api/reason endpoint with Turbo settings."""
    payload = {
        "question": question,
        "max_hops": 1, # Sustainable setting for 1-core CPU
        "enable_pruning": True,
        "enable_self_correction": True
    }
    try:
        resp = requests.post("http://127.0.0.1:8002/api/reason", json=payload, timeout=180)
        return resp.json()
    except Exception as e:
        return {"final_answer": f"Error: {e}", "metrics": {}, "citations": []}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="eval Qs.xlsx")
    parser.add_argument("--max-rows", type=int, default=10)
    args = parser.parse_args()

    path = Path(args.input)
    print(f"--- Loading evaluation data from: {path.name} ---")
    
    try:
        if path.suffix.lower() == ".xlsx":
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
    except Exception as e:
        print(f"❌ Failed to load file: {e}")
        return

    if args.max_rows > 0: 
        df = df.head(args.max_rows)

    results = []
    for idx, row in df.iterrows():
        # UNIFIED INDEXING:
        # Col E (4) = Question
        # Col L (11) = Expected Answer (Golden)
        # Col K (10) = Golden IDs
        question = str(row.iloc[4]) 
        expected = str(row.iloc[11]) 
        
        raw_gold = str(row.iloc[10])
        gold_ids = set([i.strip() for i in re.split(r'[;,|\s]', raw_gold) if i.strip()])

        print(f"[{idx+1}] Processing Cortex: {question[:50]}...")
        
        output = run_cortex_request(question)
        answer = output.get("final_answer", "")
        
        # Extract base IDs from citations (stripping parenthetical info)
        citations_list = output.get("citations", [])
        citations = set([c.split(" ")[0] for c in citations_list if c])
        
        metrics = output.get("metrics", {})

        # Calculate Retrieval Accuracy
        intersect = citations.intersection(gold_ids)
        p = len(intersect) / len(citations) if citations else 0.0
        r = len(intersect) / len(gold_ids) if gold_ids else 0.0

        results.append({
            "question": question,
            "Cortex_Answer": answer,
            "Cortex_F1": f1_score(answer, expected),
            "Cortex_Precision": p,
            "Cortex_Recall": r,
            "Hops": metrics.get("reasoning_steps", 0),
            "Nodes_Pruned": metrics.get("nodes_pruned", 0),
            "Optimization_Ratio": metrics.get("entropy_reduction", 0.0),
            "Latency": metrics.get("latency_seconds", 0.0)
        })

    pd.DataFrame(results).to_csv("cortex_results_final.csv", index=False)
    print("\n✅ Cortex Evaluation Complete. Results saved to cortex_results_final.csv")

if __name__ == "__main__":
    main()