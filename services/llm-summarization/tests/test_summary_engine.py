"""Unit tests for SummaryEngine."""

import pytest
from unittest.mock import Mock, AsyncMock
from lss.core.summary_engine import SummaryEngine, SummaryResult
from lss.backends.base import BaseLLMBackend


class MockBackend(BaseLLMBackend):
    """Mock LLM backend for testing."""
    
    def __init__(self, response: str = "Test summary"):
        self.response = response
        self.generate_called = False
    
    async def generate(self, prompt, max_tokens=1024, temperature=0.2, **kwargs):
        self.generate_called = True
        return self.response
    
    async def generate_stream(self, prompt, max_tokens=1024, temperature=0.2, **kwargs):
        self.generate_called = True
        for word in self.response.split():
            yield word + " "
    
    async def is_healthy(self):
        return True
    
    @property
    def provider_name(self):
        return "mock"
    
    @property
    def model_id(self):
        return "mock-model"


class TestSummaryEngine:
    """Tests for SummaryEngine class."""

    @pytest.fixture
    def mock_backend(self):
        """Create mock backend."""
        return MockBackend()

    @pytest.fixture
    def engine(self, mock_backend):
        """Create engine with mock backend."""
        return SummaryEngine(mock_backend)

    @pytest.mark.asyncio
    async def test_summarize_basic(self, engine, mock_backend):
        """Test basic summarization."""
        result = await engine.summarize(
            content="def hello(): pass",
            summary_type="abstractive"
        )
        
        assert isinstance(result, SummaryResult)
        assert result.summary == "Test summary"
        assert mock_backend.generate_called

    @pytest.mark.asyncio
    async def test_summarize_with_context(self, engine):
        """Test summarization with context."""
        result = await engine.summarize(
            content="def hello(): pass",
            context="This is a greeting function"
        )
        
        assert result.provider == "mock"

    @pytest.mark.asyncio
    async def test_summarize_caching(self, engine):
        """Test that results are cached."""
        await engine.summarize("content1")
        await engine.summarize("content1")
        
        assert len(engine._cache) == 1

    @pytest.mark.asyncio
    async def test_summarize_no_cache(self, engine):
        """Test summarization with caching disabled."""
        await engine.summarize("content1", use_cache=False)
        
        assert len(engine._cache) == 0

    @pytest.mark.asyncio
    async def test_summarize_stream(self, engine):
        """Test streaming summarization."""
        chunks = []
        async for chunk in engine.summarize_stream("def hello(): pass"):
            chunks.append(chunk)
        
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_summarize_batch(self, engine):
        """Test batch summarization."""
        items = [
            {"content": "def a(): pass"},
            {"content": "def b(): pass"},
        ]
        
        results = await engine.summarize_batch(items)
        
        assert len(results) == 2

    def test_clear_cache(self, engine):
        """Test clearing cache."""
        engine._cache["key"] = "value"
        engine.clear_cache()
        
        assert len(engine._cache) == 0

    def test_estimate_quality(self, engine):
        """Test quality estimation."""
        # Good summary
        quality1 = engine._estimate_quality(
            "This function greets the world",
            "def hello(): print('hello')"
        )
        
        # Too short
        quality2 = engine._estimate_quality(
            "Hi",
            "def hello(): print('hello')"
        )
        
        assert quality1 > quality2

    def test_default_prompt_for_type(self, engine):
        """Test default prompt selection."""
        assert engine._default_prompt_for_type("abstractive") == "code_abstractive"
        assert engine._default_prompt_for_type("extractive") == "code_extractive"
        assert engine._default_prompt_for_type("unknown") == "code_abstractive"


class TestSummaryResult:
    """Tests for SummaryResult dataclass."""

    def test_create_result(self):
        """Test creating a result."""
        result = SummaryResult(
            summary="Test summary",
            quality_score=0.8,
            tokens_used=100,
            provider="openai",
            model="gpt-4",
            latency_ms=500
        )
        
        assert result.summary == "Test summary"
        assert result.cached is False
