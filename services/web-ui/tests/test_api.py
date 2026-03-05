"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["service"] == "web-ui"


def test_submit_query():
    """Test query submission."""
    payload = {
        "question": "What are the prohibited AI practices under Article 5?",
        "regulation": "eu_ai_act",
        "max_hops": 3,
        "enable_pruning": True,
        "enable_self_correction": True,
    }

    response = client.post("/api/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "query_id" in data
    assert data["status"] == "processing"


def test_get_query_status():
    """Test retrieving query status."""
    # Submit a query
    payload = {
        "question": "Test question for compliance",
        "regulation": "eu_ai_act",
        "max_hops": 2,
    }

    response = client.post("/api/query", json=payload)
    query_id = response.json()["query_id"]

    # Get status
    status_response = client.get(f"/api/query/{query_id}")
    assert status_response.status_code == 200
    data = status_response.json()
    assert data["query_id"] == query_id


def test_invalid_query_too_short():
    """Test validation of query that is too short."""
    payload = {
        "question": "Too short",
        "regulation": "eu_ai_act",
    }

    response = client.post("/api/query", json=payload)
    assert response.status_code == 422  # Validation error


def test_get_nonexistent_query():
    """Test retrieving nonexistent query."""
    response = client.get("/api/query/nonexistent-id")
    assert response.status_code == 404


def test_delete_query():
    """Test deleting a query."""
    # Submit a query
    payload = {
        "question": "What is high-risk AI classification",
        "regulation": "eu_ai_act",
    }

    response = client.post("/api/query", json=payload)
    query_id = response.json()["query_id"]

    # Delete it
    delete_response = client.delete(f"/api/query/{query_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["status"] == "deleted"

    # Verify it's gone
    get_response = client.get(f"/api/query/{query_id}")
    assert get_response.status_code == 404
