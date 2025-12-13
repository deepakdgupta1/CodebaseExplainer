"""Unit tests for HybridIndex."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from cis.core.hybrid_index import HybridIndex, SearchResult


class TestHybridIndex:
    """Tests for HybridIndex class."""

    @pytest.fixture
    def mock_model(self):
        """Mock SentenceTransformer model."""
        model = Mock()
        model.encode.return_value = np.random.rand(10, 768).astype('float32')
        return model

    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for testing."""
        return [
            {
                "chunk_id": "file.py:func1:10",
                "content": "def func1(): pass",
                "metadata": {"type": "function"}
            },
            {
                "chunk_id": "file.py:func2:20",
                "content": "def func2(): return 42",
                "metadata": {"type": "function"}
            },
            {
                "chunk_id": "file.py:Class1:30",
                "content": "class Class1: pass",
                "metadata": {"type": "class"}
            },
        ]

    @patch('cis.core.hybrid_index.SentenceTransformer')
    def test_init(self, mock_st):
        """Test HybridIndex initialization."""
        mock_st.return_value = Mock()
        
        index = HybridIndex(
            model_name="test-model",
            dimension=768
        )
        
        assert index.dimension == 768
        assert index.faiss_index is None
        assert index.bm25_index is None
        mock_st.assert_called_once()

    @patch('cis.core.hybrid_index.SentenceTransformer')
    @patch('cis.core.hybrid_index.faiss')
    def test_build_index(self, mock_faiss, mock_st, sample_chunks):
        """Test building hybrid index."""
        # Setup mocks
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(3, 768).astype('float32')
        mock_st.return_value = mock_model
        
        mock_index = Mock()
        mock_faiss.IndexHNSWFlat.return_value = mock_index
        mock_faiss.METRIC_INNER_PRODUCT = 0
        
        index = HybridIndex()
        index.build_index(sample_chunks)
        
        # Verify chunks stored
        assert len(index.chunk_store) == 3
        assert "file.py:func1:10" in index.chunk_store
        
        # Verify BM25 index built
        assert index.bm25_index is not None
        assert len(index.bm25_ids) == 3

    @patch('cis.core.hybrid_index.SentenceTransformer')
    def test_tokenize(self, mock_st):
        """Test text tokenization."""
        mock_st.return_value = Mock()
        index = HybridIndex()
        
        tokens = index._tokenize("Hello World! This is a test.")
        
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    @patch('cis.core.hybrid_index.SentenceTransformer')
    def test_rrf_fusion(self, mock_st):
        """Test Reciprocal Rank Fusion."""
        mock_st.return_value = Mock()
        index = HybridIndex()
        
        dense_results = [("chunk1", 0.9), ("chunk2", 0.8), ("chunk3", 0.7)]
        sparse_results = [("chunk2", 5.0), ("chunk4", 4.0), ("chunk1", 3.0)]
        
        fused = index._rrf_fusion(dense_results, sparse_results, alpha=0.5)
        
        # chunk2 should be highest (ranked 1 in sparse, 2 in dense)
        assert "chunk2" in fused
        assert "chunk1" in fused
        assert "chunk4" in fused
        # chunk2 should have higher score than chunk3
        assert fused["chunk2"] > fused.get("chunk3", 0)

    @patch('cis.core.hybrid_index.SentenceTransformer')
    def test_search_empty_index(self, mock_st):
        """Test search with empty index."""
        mock_st.return_value = Mock()
        index = HybridIndex()
        
        results = index.search("test query")
        
        assert results == []

    @patch('cis.core.hybrid_index.SentenceTransformer')
    def test_build_index_empty_chunks(self, mock_st):
        """Test building index with empty chunks list."""
        mock_st.return_value = Mock()
        index = HybridIndex()
        
        index.build_index([])
        
        assert index.faiss_index is None
        assert index.bm25_index is None
