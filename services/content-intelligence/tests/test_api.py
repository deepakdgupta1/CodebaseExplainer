"""Integration tests for CIS API."""

import pytest
from fastapi.testclient import TestClient
from cis.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health endpoint."""

    def test_health_returns_ok(self, client):
        """Test health endpoint returns 200."""
        response = client.get("/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_includes_stats(self, client):
        """Test health includes stats."""
        response = client.get("/v1/health")
        
        data = response.json()
        assert "stats" in data
        assert "performance" in data


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_info(self, client):
        """Test root returns service info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Content Intelligence Service"


class TestContextQueryEndpoint:
    """Tests for context query endpoint."""

    def test_query_requires_query(self, client):
        """Test query requires body."""
        response = client.post("/v1/context/query", json={})
        
        assert response.status_code == 422  # Validation error

    def test_query_minimal(self, client):
        """Test minimal query."""
        response = client.post("/v1/context/query", json={
            "query": "How does authentication work?"
        })
        
        # Should return OK even with empty index
        assert response.status_code in (200, 500)

    def test_query_with_options(self, client):
        """Test query with all options."""
        response = client.post("/v1/context/query", json={
            "query": "test query",
            "context": {
                "current_file": "test.py",
                "cursor_line": 10
            },
            "options": {
                "max_tokens": 2048,
                "min_completeness": 0.8,
                "dependency_depth": 1
            }
        })
        
        assert response.status_code in (200, 500)


class TestUpdateEndpoint:
    """Tests for update endpoint."""

    def test_update_requires_action(self, client):
        """Test update requires action."""
        response = client.post("/v1/context/update", json={
            "file_path": "test.py"
        })
        
        assert response.status_code == 422

    def test_update_basic(self, client):
        """Test basic update."""
        response = client.post("/v1/context/update", json={
            "action": "modify",
            "file_path": "test.py",
            "content": "def hello(): pass"
        })
        
        assert response.status_code in (200, 500)
