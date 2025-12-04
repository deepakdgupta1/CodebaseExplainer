import pytest
import numpy as np
from codehierarchy.search.embedder import HighQualityEmbedder

@pytest.fixture
def embedder():
    # Use a small model or mock for testing to avoid heavy download?
    # For now, let's assume the environment has the model or we mock SentenceTransformer
    # But real integration test is better if model is cached.
    # To be safe and fast, let's mock SentenceTransformer
    with pytest.mock.patch('codehierarchy.search.embedder.SentenceTransformer') as MockST:
        instance = MockST.return_value
        instance.encode.return_value = np.random.rand(2, 768).astype('float32')
        yield HighQualityEmbedder()

# Actually, I can't use pytest.mock inside fixture easily without importing unittest.mock
from unittest.mock import patch, MagicMock

def test_encode_batch():
    with patch('codehierarchy.search.embedder.SentenceTransformer') as MockST:
        mock_model = MockST.return_value
        # Mock encode to return numpy array
        mock_model.encode.return_value = np.zeros((1, 768), dtype='float32')
        
        embedder = HighQualityEmbedder()
        embeddings = embedder.encode_batch(["test"])
        
        assert embeddings.shape == (1, 768)
        mock_model.encode.assert_called_once()

def test_build_index():
    with patch('codehierarchy.search.embedder.SentenceTransformer') as MockST:
        mock_model = MockST.return_value
        mock_model.encode.return_value = np.random.rand(5, 768).astype('float32')
        
        embedder = HighQualityEmbedder()
        summaries = {f"id{i}": f"text{i}" for i in range(5)}
        
        index, mapping = embedder.build_index(summaries)
        
        assert index.ntotal == 5
        assert len(mapping) == 5
