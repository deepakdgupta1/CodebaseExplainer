import pytest
from unittest.mock import Mock, patch, MagicMock
from codehierarchy.core.llm.lmstudio_summarizer import LMStudioSummarizer
from codehierarchy.config.schema import LLMConfig

@pytest.fixture
def mock_openai():
    with patch('codehierarchy.core.llm.lmstudio_summarizer.OpenAI') as mock:
        yield mock

def test_summarize_batch(mock_openai):
    config = LLMConfig()
    with patch('codehierarchy.core.llm.lmstudio_summarizer.ModelManager') as MockManager:
        summarizer = LMStudioSummarizer(config, "System Prompt")
    
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
    mock_client = mock_openai.return_value
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "[test.py:test:1] This is a very long summary for the test function that should definitely pass the minimum length check of fifty characters."
    mock_client.chat.completions.create.return_value = mock_response
    
    summaries = summarizer.summarize_batch(["test.py:test:1"], builder)
    
    assert len(summaries) == 1
    assert summaries["test.py:test:1"] == "This is a very long summary for the test function that should definitely pass the minimum length check of fifty characters."
    
def test_parse_batch_response():
    config = LLMConfig()
    # Mock OpenAI client creation in init
    # Mock OpenAI client creation in init
    with patch('codehierarchy.core.llm.lmstudio_summarizer.OpenAI'), \
         patch('codehierarchy.core.llm.lmstudio_summarizer.ModelManager'):
        summarizer = LMStudioSummarizer(config, "")
    
    response = """
    [id1] Summary 1
    continued.
    
    [id2] Summary 2.
    """
    
    summaries = summarizer._parse_batch_response(response, ["id1", "id2"])
    
    assert summaries["id1"] == "Summary 1 continued."
    assert summaries["id2"] == "Summary 2."
