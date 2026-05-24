import traceback

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# IMPORTANT: Do NOT import run_cortex_engine here at the top.
# This prevents the model from trying to load before the API is fully ready.

app = FastAPI(title="Cortex Reasoning Engine API")


class ReasonRequest(BaseModel):
    question: str
    regulation: str = "both"
    max_hops: int = 3
    enable_pruning: bool = True
    enable_self_correction: bool = True
    pruning_threshold: float = 0.45
    kg_hits: list = []

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/api/reason")
async def reason(request: ReasonRequest):
    try:
        # We import here so the GPU/CPU handshake happens ONLY when a request arrives
        from main_orchestrator import run_cortex_engine

        print(f"--- Processing Query: {request.question} ---")

        # Run the full agentic loop (Retriever -> Pruner -> Critic -> Synthesizer)
        result = run_cortex_engine(
            request.question,
            max_hops=request.max_hops,
            enable_pruning=request.enable_pruning,
            enable_self_correction=request.enable_self_correction,
            pruning_threshold=request.pruning_threshold,
        )

        # Ensure metrics are always present for the frontend
        if "metrics" not in result:
            result["metrics"] = {}

        return result

    except Exception as e:
        print(f"❌ API Error: {traceback.format_exc()}")
        # This sends the error back to the Gateway so it knows exactly what failed
        raise HTTPException(status_code=500, detail=str(e)) from e


# --- FOR NAIVE RAG ---

@app.post("/api/naive")
async def naive_synthesis(request: ReasonRequest):
    try:
        # Import the actual function used in your orchestrator
        from main_orchestrator import synthesis_node

        print(f"--- Naive Synthesis for: {request.question[:50]}... ---")

        # Prepare a manual "Fake" State that the synthesis_node expects
        # We put your retrieved hits directly into 'pruned_context'
        # so the model thinks the Pruner already approved them.
        manual_state = {
            "query": request.question,
            "pruned_context": [
                {
                    "id": hit.get("id", "unknown"),
                    "content": hit.get("content") or hit.get("result", {}).get("text", ""),
                }
                for hit in request.kg_hits
            ],
            "final_answer": "",
        }

        # Call the existing synthesis logic directly
        final_state = synthesis_node(manual_state)

        return {
            "final_answer": final_state["final_answer"],
            "status": "completed",
            "metrics": {"mode": "naive_bypass_v2"},
        }
    except Exception as e:
        print(f"❌ Naive API Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e)) from e
