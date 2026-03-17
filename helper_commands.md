Open a NEW Terminal in VS Code.
Navigate to the KG folder:

- python -m uvicorn main:app --port 8001 for c:\Users\Samsa\.vscode\coding\cortex-ai-act\services\knowledge-graph
- python -m uvicorn reasoning_api:app --port 8002 for C:\Users\Samsa\.vscode\coding\cortex-ai-act\services\reasoning-engine
- python -m streamlit run app.py --server.port 8502 for C:\Users\Samsa\.vscode\coding\cortex-ai-act\services\web-ui
- $env:KNOWLEDGE_GRAPH_URL="http://localhost:8001"; python -m uvicorn api.main:app --port 8000 for C:\Users\Samsa\.vscode\coding\cortex-ai-act\services\web-ui

desktop orchestrator:
C:\Users\Samsa\.vscode\coding\cortex-ai-act\services\reasoning-engine, type: python -m main_orchestrator

Optional step:
To Ingest (open another 2nd terminal)
Invoke-RestMethod -Uri "http://127.0.0.1:8001/graph/ingest" -Method Post -InFile "data/ai_act_extracted_enriched.json" -ContentType "application/json"

To view knowledge graph in explorer
Open another 3rd terminal
cd C:\Users\Samsa\.vscode\coding\cortex-ai-act\services\knowledge-graph\ui
python -m streamlit run app.py

To open user dashboard
cd C:\Users\Samsa\.vscode\coding\cortex-ai-act\services\web-ui
python -m streamlit run app.py --server.port 8502

RUFF
python -m ruff check services/reasoning-engine --fix

BLACK
python -m black services/reasoning-engine

Naive Rag:
python "C:\Users\Samsa\.vscode\coding\cortex-ai-act\baselines\naive_baseline.py"