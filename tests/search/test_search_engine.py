import pytest
from pathlib import Path
from codehierarchy.search.search_engine import EnterpriseSearchEngine
from codehierarchy.search.result import Result

@pytest.fixture
def search_engine(tmp_path):
    # Mock components inside EnterpriseSearchEngine if needed, 
    # but for unit test of the engine logic (hybrid search), we can mock the sub-searchers.
    pass

# Since EnterpriseSearchEngine loads models in __init__, it's hard to unit test without mocking.
# Let's write a test that mocks the internal searchers.

from unittest.mock import MagicMock, patch

def test_hybrid_search_logic():
    with patch('codehierarchy.search.search_engine.HighQualityEmbedder') as MockEmbedder, \
         patch('codehierarchy.search.search_engine.KeywordSearch') as MockKeyword:
        
        # Setup mocks
        mock_embedder = MockEmbedder.return_value
        mock_embedder.load_index.return_value = (MagicMock(), {})
        
        mock_keyword = MockKeyword.return_value
        
        engine = EnterpriseSearchEngine(Path("dummy"))
        
        # Mock results
        res1 = Result(node_id="1", name="n1", file="f1", line=1, summary="s1", score=0.9)
        res2 = Result(node_id="2", name="n2", file="f2", line=2, summary="s2", score=0.8)
        
        mock_keyword.search.return_value = [res1]
        
        # Mock semantic search (private method, or mock vector index search)
        # Easier to mock _semantic_search if we could, but it's part of the class.
        # Let's mock the vector index search return
        engine.vector_index.search.return_value = ([[0.85]], [[0]]) # score, index
        engine.id_mapping = {0: "2"}
        
        # We need to ensure _semantic_search returns res2-like result
        # But _semantic_search creates new Result objects.
        # Let's just trust the logic or mock _semantic_search?
        # Mocking methods on the object under test is partial mocking.
        
        with patch.object(engine, '_semantic_search', return_value=[res2]):
            results = engine.search("query", mode='hybrid', top_k=2)
            
            # RRF Logic:
            # res1 rank 1 in keyword -> score += 1/61
            # res2 rank 1 in semantic -> score += 1/61
            # Both have equal score.
            
            assert len(results) == 2
            ids = {r.node_id for r in results}
            assert "1" in ids
            assert "2" in ids
