import pytest
from unittest.mock import Mock, patch
from codehierarchy.core.llm.deepseek_summarizer import DeepSeekSummarizer
from codehierarchy.config.schema import LLMConfig

@pytest.fixture
def mock_ollama():
    with patch('codehierarchy.llm.deepseek_summarizer.ollama') as mock:
        yield mock

def test_summarize_batch(mock_ollama):
    config = LLMConfig()
    summarizer = DeepSeekSummarizer(config, "System Prompt")
    
    # Mock builder and context
    builder = MagicMock()
    builder.get_node_with_context.return_value = {
        'node': {'type': 'function', 'name': 'test', 'file': 'test.py'},
        'source': {'source_code': 'def test(): pass', 'docstring': None},
        'metadata': {},
        'parents': [],
        'children': []
    }
    
    # Mock LLM response
    mock_ollama.chat.return_value = {
        'message': {
            'content': "[test.py:test:1] This is a summary."
        }
    }
    
    summaries = summarizer.summarize_batch(["test.py:test:1"], builder)
    
    assert len(summaries) == 1
    assert summaries["test.py:test:1"] == "This is a summary."
    
def test_parse_batch_response():
    config = LLMConfig()
    summarizer = DeepSeekSummarizer(config, "")
    
    response = """
    [id1] Summary 1
    continued.
    
    [id2] Summary 2.
    """
    
    summaries = summarizer._parse_batch_response(response, ["id1", "id2"])
    
    assert summaries["id1"] == "Summary 1 continued."
    assert summaries["id2"] == "Summary 2."
