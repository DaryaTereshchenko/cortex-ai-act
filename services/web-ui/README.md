# CORTEX-RAG Web UI

Interactive dashboard for regulatory compliance queries on the **EU AI Act** and **Digital Services Act (DSA)**.

## Features

- 🔍 **Natural Language Queries** - Ask compliance questions in plain English
- 🕸️ **Knowledge Graph Visualization** - See reasoning paths through regulations
- 📊 **Entropy Metrics** - Track context optimization in real-time
- 🔄 **Reasoning Transparency** - View each step of the agentic loop
- 📚 **Automated Citations** - Get precise article references

## Architecture

```
User Browser
     ↓
Streamlit Frontend (Port 8501)
     ↓
FastAPI Gateway (Port 8000)
     │
     ├─→ Reasoning Engine API
     └─→ Knowledge Graph API
```

## Local Development

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Git

### Setup

1. **Create virtual environment:**

   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/Mac:
   # source venv/bin/activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**

   ```bash
   cp .env.example .env
   # Edit .env with your service URLs if needed
   ```

4. **Run locally:**

   ```bash
   # Terminal 1: FastAPI
   uvicorn api.main:app --reload --port 8000

   # Terminal 2: Streamlit
   streamlit run app.py --server.port 8501
   ```

5. **Access:**
   - Streamlit UI: http://localhost:8501
   - API Docs: http://localhost:8000/docs
   - OpenAPI Schema: http://localhost:8000/openapi.json

### Docker

```bash
# Build and run
docker-compose up web-ui

# With rebuild
docker-compose up --build web-ui

# Specific services
docker-compose up -d neo4j knowledge-graph reasoning-engine web-ui
```

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=api --cov-report=html

# Specific test file
pytest tests/test_api.py -v

# Watch mode (requires pytest-watch)
ptw
```

## API Endpoints

### Health Check

```
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "web-ui",
  "reasoning_engine_available": true,
  "knowledge_graph_available": true,
  "timestamp": "2026-03-05T10:00:00"
}
```

### Submit Query

```
POST /api/query
Content-Type: application/json

{
  "question": "What are the obligations for high-risk AI providers?",
  "regulation": "eu_ai_act",
  "max_hops": 3,
  "enable_pruning": true,
  "enable_self_correction": true
}
```

**Response:**
```json
{
  "query_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing"
}
```

### Get Query Status

```
GET /api/query/{query_id}
```

**Response:**
```json
{
  "query_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "question": "...",
  "final_answer": "...",
  "reasoning_steps": [...],
  "graph_data": {...},
  "citations": [...],
  "metrics": {...}
}
```

### Get Graph Visualization

```
GET /api/graph/{query_id}
```

### Delete Query

```
DELETE /api/query/{query_id}
```

### WebSocket Real-time Updates

```
WS /ws/{query_id}
```

## Configuration

Key environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `REASONING_ENGINE_URL` | `http://reasoning-engine:8001` | Reasoning engine endpoint |
| `KNOWLEDGE_GRAPH_URL` | `http://knowledge-graph:8002` | Knowledge graph endpoint |
| `API_BASE_URL` | `http://localhost:8000/api` | API base for frontend |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `ENVIRONMENT` | `development` | Environment type |

## File Structure

```
services/web-ui/
├── api/                      # FastAPI application
│   ├── __init__.py
│   ├── main.py              # Main FastAPI app
│   └── schemas.py           # Pydantic models
├── tests/                    # Test suite
│   ├── __init__.py
│   └── test_api.py          # API tests
├── app.py                    # Streamlit entry point
├── Dockerfile                # Container definition
├── requirements.txt          # Python dependencies
├── .env.example             # Environment template
└── README.md                # This file
```

## Troubleshooting

### Services Unreachable

```bash
# Check if services are running
docker-compose ps

# Check network connectivity
docker network inspect cortex-net

# View service logs
docker-compose logs reasoning-engine
docker-compose logs knowledge-graph
```

### Slow Query Processing

- Reduce `max_hops` parameter
- Check GPU availability (if using)
- Verify network connectivity between services
- Check service logs for errors

### Graph Visualization Not Loading

- Check browser console for JavaScript errors
- Verify graph data in API response
- Clear browser cache
- Try a different browser

### Port Already in Use

```bash
# Find process using port 8501
lsof -i :8501

# Kill process
kill -9 <PID>
```

## Development Workflow

1. Create feature branch from `fullstack-development`
2. Implement changes with tests
3. Run tests locally: `pytest -v`
4. Commit with descriptive message
5. Push to remote
6. Submit PR to `fullstack-development`

## Code Quality

- **Format:** `black .`
- **Lint:** `ruff check .`
- **Type Check:** `mypy api/`
- **Tests:** `pytest --cov`

## Integration with Other Services

### Reasoning Engine

- **Type:** LLM Engineer's service
- **URL:** `REASONING_ENGINE_URL`
- **Expected Endpoint:** `/health` and `/api/reason`

### Knowledge Graph

- **Type:** Data Engineer's Neo4j service
- **URL:** `KNOWLEDGE_GRAPH_URL`
- **Expected Endpoint:** `/health` and graph query endpoints

## Performance Targets

| Metric | Target |
| --- | --- |
| Initial Page Load | < 2 seconds |
| Query Submission | < 500ms |
| Graph Rendering (50 nodes) | < 1 second |
| WebSocket Delay | < 100ms |
| Memory Usage | < 512MB |

## Contributing

1. Follow PEP 8 style guide
2. Write tests for new features
3. Keep commits atomic and descriptive
4. Update documentation as needed
5. Ensure CI passes before merging

## License

MIT License - See LICENSE file for details

## Support

For issues or questions:
1. Check Troubleshooting section
2. Review service logs
3. File GitHub issue with details
4. Contact development team

---

*Last Updated: March 5, 2026*
