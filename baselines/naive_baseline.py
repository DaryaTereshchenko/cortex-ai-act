from neo4j import GraphDatabase

# 1. CONNECTION
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "changeme"


def run_naive_rag_benchmark(query_text: str):
    print("\n--- RUNNING DIRECT FULLTEXT BENCHMARK ---")

    # We use $searchTerm to avoid the Python naming conflict
    cypher_query = """
    CALL db.index.fulltext.queryNodes("chunk_text_fulltext", $searchTerm)
    YIELD node, score
    RETURN node.text AS text
    LIMIT 5
    """

    with GraphDatabase.driver(URI, auth=(USER, PASSWORD)) as driver, driver.session() as session:
        # Pass the query as 'searchTerm'
        result = session.run(cypher_query, searchTerm=query_text)
        texts = [record["text"] for record in result if record["text"]]

    context_text = "\n\n".join(texts)
    word_count = len(context_text.split())

    return {
        "nodes_found": len(texts),
        "total_word_count": word_count,
        "retrieved_context": context_text,
    }


if __name__ == "__main__":
    # Simplified search terms for the keyword index
    test_query = "transparency obligations AI systems"
    res = run_naive_rag_benchmark(test_query)

    print("\n[BENCHMARK DATA]")
    print(f"Nodes Retrieved: {res['nodes_found']}")
    print(f"Total Context Size: {res['total_word_count']} words")

    if res["nodes_found"] > 0:
        print("\n[SAMPLE RETRIEVED TEXT]")
        print(f"{res['retrieved_context'][:500]}...")

    print("\nSUCCESS: Data captured for CORTEX comparison.")
