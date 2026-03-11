import traceback

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from main_orchestrator import run_cortex_engine

app = FastAPI(title="Cortex Reasoning Engine API")


class ReasonRequest(BaseModel):
    question: str
    regulation: str = "both"
    max_hops: int = 3


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/reason")
async def reason(request: ReasonRequest):
    try:
        # Run the full engine
        result = run_cortex_engine(request.question)

        # Ensure result contains the keys expected by QueryResponse schema
        if "metrics" not in result:
            result["metrics"] = {}

        return result

    except Exception as e:
        print(f"❌ API Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e)) from e
