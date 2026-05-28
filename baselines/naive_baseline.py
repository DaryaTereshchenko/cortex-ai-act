import requests
import re
from neo4j import GraphDatabase

# Configuration
URI = "bolt://127.0.0.1:7687"
USER = "neo4j"
PASSWORD = "changeme"

def clean_query_for_lexical(text: str):
    """Simple regex to remove question filler for better keyword matching."""
    text = re.sub(r'[?]', '', text)
    fillers = r'\b(which|of|the|is|a|an|what|according|to|in|regard|with|how|ensure|shall|for|are|following)\b'
    text = re.sub(fillers, '', text, flags=re.IGNORECASE)
    return " ".join(text.split())

def run_naive_rag_benchmark(query_text: str):
    # --- 1. RETRIEVAL (Neo4j) ---
    search_term = clean_query_for_lexical(query_text)
    
    cypher_query = """
    CALL db.index.fulltext.queryNodes("article_fulltext", $searchTerm)
    YIELD node, score
    RETURN node.id AS id, node.full_text AS text, score
    LIMIT 5
    """
    
    try:
        with GraphDatabase.driver(URI, auth=(USER, PASSWORD)) as driver, driver.session() as session:
            result = session.run(cypher_query, searchTerm=search_term)
            records = [{"id": r["id"], "text": r["text"]} for r in result if r["text"]]
    except Exception as e:
        print(f"❌ Neo4j Error: {e}")
        records = []

    # --- INTRINSIC SPEED HACK (EMNLP COMPLIANT) ---
    # We prune the top-K to 2 and truncate the text. 
    # This ensures the 1-core CPU can process the prompt within the 120s timeout.
    naive_records = records[:2]
    for r in naive_records:
        if r["text"]:
            r["text"] = r["text"][:600] # Take only the core snippet

    context_text = "\n\n".join([r["text"] for r in naive_records])
    retrieved_ids = [r["id"] for r in records] # Keep all 5 IDs for retrieval metrics

    print(f"DEBUG: Found {len(records)} records. Sending TOP 2 to Qwen.")

    # --- 2. GENERATION (The Turbo Payload) ---
    # We add a specific instruction for brevity to speed up generation
    turbo_question = f"{query_text} (Answer briefly in 1-2 sentences)"

    payload = {
        "question": turbo_question,
        "kg_hits": [{"content": r["text"], "id": r["id"]} for r in naive_records]
    }

    try:
        # Hitting the /api/naive side-door we created
        resp = requests.post("http://127.0.0.1:8002/api/naive", json=payload, timeout=120)
        
        if resp.status_code == 200:
            data = resp.json()
            answer = data.get("final_answer") or "No answer generated."
        else:
            answer = f"Error: API returned status {resp.status_code}"

    except Exception as e:
        answer = f"Generation failed: {e}"

    return {
        "answer": answer,
        "nodes_found": len(records),
        "retrieved_ids": retrieved_ids,
        "retrieved_context": context_text,
    }