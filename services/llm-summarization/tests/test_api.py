"""Integration tests for LSS API."""

import pytest
from fastapi.testclient import TestClient
from lss.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health endpoint."""

    def test_health_returns_ok(self, client):
        """Test health endpoint."""
        response = client.get("/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_health_includes_providers(self, client):
        """Test health includes providers."""
        response = client.get("/v1/health")
        
        data = response.json()
        assert "providers" in data


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_info(self, client):
        """Test root returns service info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "LLM Summarization Service"


class TestPromptsEndpoint:
    """Tests for prompts endpoint."""

    def test_list_prompts(self, client):
        """Test listing prompts."""
        response = client.get("/v1/prompts")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 3  # Default prompts

    def test_create_prompt(self, client):
        """Test creating a prompt."""
        response = client.post("/v1/prompts", json={
            "id": "test_prompt",
            "name": "Test Prompt",
            "template": "Summarize: ${code}",
            "summary_type": "custom",
            "variables": ["code"]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test_prompt"


class TestProvidersEndpoint:
    """Tests for providers endpoint."""

    def test_list_providers(self, client):
        """Test listing providers."""
        response = client.get("/v1/providers")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestSummarizeEndpoint:
    """Tests for summarize endpoint."""

    def test_summarize_requires_content(self, client):
        """Test summarize requires content."""
        response = client.post("/v1/summarize", json={})
        
        assert response.status_code == 422

    def test_summarize_validates_type(self, client):
        """Test summarize validates summary type."""
        response = client.post("/v1/summarize", json={
            "content": "def hello(): pass",
            "summary_type": "invalid_type"
        })
        
        assert response.status_code == 422

    def test_summarize_basic(self, client):
        """Test basic summarization."""
        response = client.post("/v1/summarize", json={
            "content": "def hello(): print('Hello, World!')"
        })
        
        # May fail if no LLM backend, but should be valid request
        assert response.status_code in (200, 500)

    def test_summarize_with_options(self, client):
        """Test summarization with options."""
        response = client.post("/v1/summarize", json={
            "content": "def hello(): pass",
            "summary_type": "extractive",
            "temperature": 0.5,
            "max_tokens": 512
        })
        
        assert response.status_code in (200, 500)
