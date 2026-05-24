import sys, os
import argparse
import pandas as pd
import requests
import re
import string
import json
from pathlib import Path

# Ensure API path is correct
sys.path.insert(0, "/files/cortex-ai-act")

STOPWORDS = {"a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "for", "and", "or", "on", "it", "this", "that"}

def normalize_text(text: str) -> str:
    lowered = str(text).lower()
    no_punct = lowered.translate(str.maketrans("", "", string.punctuation))
    tokens = [tok for tok in no_punct.split() if tok and tok not in STOPWORDS]
    return " ".join(tokens)

def f1_score(prediction: str, target: str) -> float:
    target = re.sub(r"Name:.*dtype:.*", "", str(target), flags=re.DOTALL).strip()
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(target).split()
    if not pred_tokens and not gold_tokens: return 1.0
    if not pred_tokens or not gold_tokens: return 0.0
    overlap = len(set(pred_tokens) & set(gold_tokens))
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

def run_ablation_request(question: str, pruning: bool, critic: bool):
    """Hits the API with specific components toggled for ablation study."""
    payload = {
        "question": question,
        "max_hops": 3 if critic else 1,
        "enable_pruning": pruning,
        "enable_self_correction": critic
    }
    try:
        resp = requests.post("http://127.0.0.1:8002/api/reason", json=payload, timeout=180)
        return resp.json()
    except Exception as e:
        return {"final_answer": f"Error: {e}", "metrics": {}, "citations": []}

def main():
    parser = argparse.ArgumentParser(description="Cortex Ablation Study Runner")
    parser.add_argument("--input", default="eval Qs.xlsx", help="Path to evaluation Excel file")
    parser.add_argument("--max-rows", type=int, default=25, help="Number of rows to sample")
    args = parser.parse_args()

    print(f"--- Loading: {args.input} ---")
    
    try:
        full_df = pd.read_excel(args.input)
        # SCIENTIFIC SAMPLING: We take 25 random rows using a fixed seed for reproducibility
        if args.max_rows < len(full_df):
            print(f"🎲 Randomly sampling {args.max_rows} queries for ablation study...")
            df = full_df.sample(n=args.max_rows, random_state=42)
        else:
            df = full_df
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    versions = [
        {"name": "Pruner_Only", "pruning": True, "critic": False},
        {"name": "Critic_Only", "pruning": False, "critic": True}
    ]

    for v in versions:
        print(f"\n🚀 STARTING ABLATION: {v['name']}")
        results = []
        
        counter = 0
        for idx, row in df.iterrows():
            counter += 1
            question = str(row.iloc[4]) 
            expected = str(row.iloc[11]) 
            raw_gold = str(row.iloc[10])
            gold_ids = set([i.strip() for i in re.split(r'[;,|\s]', raw_gold) if i.strip()])

            output = run_ablation_request(question, v['pruning'], v['critic'])
            answer = output.get("final_answer", "")
            
            citations_list = output.get("citations", [])
            citations = set([c.split(" ")[0] for c in citations_list if c])
            intersect = citations.intersection(gold_ids)
            
            p = len(intersect) / len(citations) if citations else 0.0
            r = len(intersect) / len(gold_ids) if gold_ids else 0.0

            results.append({
                "question": question,
                f"{v['name']}_F1": f1_score(answer, expected),
                f"{v['name']}_Precision": p,
                f"{v['name']}_Recall": r,
                f"{v['name']}_Latency": output.get("metrics", {}).get("latency_seconds", 0.0)
            })
            print(f"   [{counter}/{len(df)}] {v['name']} query completed...")

        output_file = f"cortex_ablation_{v['name']}.csv"
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"✅ Saved results to {output_file}")

    print("\n🎉 ABLATION SAMPLING RUN COMPLETE.")

if __name__ == "__main__":
    main()