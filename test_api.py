import pytest
from fastapi.testclient import TestClient
from capstone.api import app

client = TestClient(app)

def test_health_endpoint_returns_success():
    """Verify that public system tracking routes respond cleanly"""
    response = client.get("/health")
    assert response.status_code == 200

def test_unauthenticated_research_route_blocked():
    """Verify that security rules intercept unauthenticated requests with a 401"""
    # Assuming your POST endpoint expects data; this checks that the verify_api_key security blocks it first
    response = client.post("/research", json={"query": "test pipeline mapping"})
    assert response.status_code == 401
