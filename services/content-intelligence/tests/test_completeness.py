"""Unit tests for CompletenessScorer."""

import pytest
from cis.core.completeness import CompletenessScorer, CompletenessResult


class TestCompletenessScorer:
    """Tests for CompletenessScorer class."""

    @pytest.fixture
    def scorer(self):
        """Create scorer instance."""
        return CompletenessScorer(min_threshold=0.9)

    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for testing."""
        return [
            {
                "chunk_id": "file.py:func1:10",
                "content": "def func1(): return func2()",
                "symbol_name": "func1"
            },
            {
                "chunk_id": "file.py:func2:20",
                "content": "def func2(): return 42",
                "symbol_name": "func2"
            },
        ]

    def test_init(self, scorer):
        """Test scorer initialization."""
        assert scorer.symbol_weight == 0.7
        assert scorer.dependency_weight == 0.3
        assert scorer.min_threshold == 0.9

    def test_score_complete_context(self, scorer, sample_chunks):
        """Test scoring complete context."""
        result = scorer.score(sample_chunks)
        
        assert isinstance(result, CompletenessResult)
        assert 0 <= result.overall_score <= 1.0
        assert 0 <= result.symbol_resolution <= 1.0

    def test_score_empty_chunks(self, scorer):
        """Test scoring empty chunks list."""
        result = scorer.score([])
        
        assert result.symbol_resolution == 1.0  # No refs to resolve

    def test_is_complete_above_threshold(self, scorer):
        """Test completeness check above threshold."""
        result = CompletenessResult(
            overall_score=0.95,
            symbol_resolution=0.95,
            dependency_coverage=1.0,
            unresolved_symbols=[],
            missing_dependencies=[]
        )
        
        assert scorer.is_complete(result) is True

    def test_is_complete_below_threshold(self, scorer):
        """Test completeness check below threshold."""
        result = CompletenessResult(
            overall_score=0.5,
            symbol_resolution=0.5,
            dependency_coverage=0.5,
            unresolved_symbols=["missing"],
            missing_dependencies=["dep"]
        )
        
        assert scorer.is_complete(result) is False

    def test_extract_references(self, scorer):
        """Test reference extraction."""
        code = '''
def foo():
    bar()
    baz.method()
    return MyClass()
'''
        refs = scorer._extract_references(code)
        
        assert "bar" in refs
        assert "MyClass" in refs

    def test_extract_references_filters_keywords(self, scorer):
        """Test that common keywords are filtered."""
        code = "def foo(): return str(len(x))"
        refs = scorer._extract_references(code)
        
        assert "str" not in refs
        assert "len" not in refs
        assert "return" not in refs


class TestCompletenessResult:
    """Tests for CompletenessResult dataclass."""

    def test_create_result(self):
        """Test creating a result."""
        result = CompletenessResult(
            overall_score=0.85,
            symbol_resolution=0.9,
            dependency_coverage=0.8,
            unresolved_symbols=["missing"],
            missing_dependencies=["dep1", "dep2"]
        )
        
        assert result.overall_score == 0.85
        assert len(result.unresolved_symbols) == 1
        assert len(result.missing_dependencies) == 2
