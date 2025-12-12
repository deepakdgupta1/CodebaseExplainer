
import pytest
from unittest.mock import Mock, patch, MagicMock
from openai import APIConnectionError, BadRequestError
from codehierarchy.core.llm.lmstudio_summarizer import LMStudioSummarizer
from codehierarchy.config.schema import LLMConfig
from codehierarchy.analysis.graph.graph_builder import InMemoryGraphBuilder

@pytest.fixture
def mock_config():
    config = MagicMock(spec=LLMConfig)
    config.base_url = "http://localhost:1234/v1"
    config.api_key = "lm-studio"
    config.model_name = "test-model"
    config.context_window = 2048
    config.temperature = 0.7
    config.top_p = 0.9
    config.top_k = 40
    config.repeat_penalty = 1.1
    config.min_p = 0.05
    config.context_overflow_policy = "stopAtLimit"
    return config

@pytest.fixture
def mock_builder():
    builder = MagicMock(spec=InMemoryGraphBuilder)
    # The summarizer calls get_node_with_context(nid)
    # We need to ensure it returns valid data for the requested nids
    def get_context(nid):
        return {
            'node': {'type': 'function', 'name': f'func_{nid}', 'file': 'test.py'},
            'source': {'source_code': 'def test(): pass', 'docstring': None},
            'parents': [],
            'children': []
        }
    builder.get_node_with_context.side_effect = get_context
    return builder

@pytest.fixture
def mock_model_manager():
    """Mock the create_backend factory to return a mock backend."""
    with patch(
        "codehierarchy.core.llm.lmstudio_summarizer.create_backend"
    ) as mock_factory:
        mock_backend = MagicMock()
        mock_backend.setup.return_value = None
        mock_backend.load_model.return_value = "test-model"
        mock_backend.base_url = "http://localhost:1234/v1"
        mock_backend.is_healthy.return_value = True
        mock_factory.return_value = mock_backend
        yield mock_factory

class TestLMStudioSummarizer:
    
    @patch("codehierarchy.core.llm.lmstudio_summarizer.OpenAI")
    def test_summarize_batch_success(
        self, mock_openai, mock_config, mock_builder, mock_model_manager
    ):
        """Test successful batch summarization."""
        # Setup mock response
        node_id_1 = "test.py:function:func_1"
        node_id_2 = "test.py:function:func_2"

        mock_response = MagicMock()
        summary_1 = (
            "This is a detailed summary for function func_1 that is definitely "
            "longer than fifty characters to pass validation."
        )
        summary_2 = (
            "This is a detailed summary for function func_2 that is definitely "
            "longer than fifty characters to pass validation."
        )
        mock_response.choices[0].message.content = (
            f"[{node_id_1}] {summary_1}\n[{node_id_2}] {summary_2}"
        )
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        summarizer = LMStudioSummarizer(mock_config, "Template")
        result = summarizer.summarize_batch(
            [node_id_1, node_id_2], mock_builder
        )

        assert len(result) == 2
        assert result[node_id_1] == summary_1
        assert result[node_id_2] == summary_2
        assert summarizer.enabled is True

    @patch("codehierarchy.core.llm.lmstudio_summarizer.OpenAI")
    def test_connection_error_disables_summarizer(
        self, mock_openai, mock_config, mock_builder, mock_model_manager
    ):
        """Test that APIConnectionError disables the summarizer."""
        # Setup mock to raise error
        # APIConnectionError signature: (message, *, request)
        mock_openai.return_value.chat.completions.create.side_effect = (
            APIConnectionError(message="Connection refused", request=MagicMock())
        )

        summarizer = LMStudioSummarizer(mock_config, "Template")
        result = summarizer.summarize_batch(["node1"], mock_builder)

        assert result == {}
        assert summarizer.enabled is False

        # Verify subsequent calls return empty immediately
        result2 = summarizer.summarize_batch(["node1"], mock_builder)
        assert result2 == {}
        # Ensure create was not called again
        assert mock_openai.return_value.chat.completions.create.call_count == 1

    @patch("codehierarchy.core.llm.lmstudio_summarizer.OpenAI")
    def test_bad_request_error_handled(
        self, mock_openai, mock_config, mock_builder, mock_model_manager
    ):
        """Test that BadRequestError is caught and returns empty dict."""
        mock_openai.return_value.chat.completions.create.side_effect = (
            BadRequestError(
                message="Bad Request", response=MagicMock(), body=None
            )
        )

        summarizer = LMStudioSummarizer(mock_config, "Template")
        result = summarizer.summarize_batch(["node1"], mock_builder)

        assert result == {}
        # Should NOT disable for bad request (e.g. context too long)
        assert summarizer.enabled is True

    def test_init_load_model_failure(self, mock_config, mock_model_manager):
        """Test that summarizer is disabled if model load fails."""
        # Set backend.load_model to return None (failure)
        mock_model_manager.return_value.load_model.return_value = None

        with patch("codehierarchy.core.llm.lmstudio_summarizer.OpenAI"):
            summarizer = LMStudioSummarizer(mock_config, "Template")

        assert summarizer.enabled is False
        
    def test_summarize_batch_skips_if_disabled(self, mock_config, mock_builder, mock_model_manager):
        """Test that summarize_batch returns early if enabled=False."""
        with patch("codehierarchy.core.llm.lmstudio_summarizer.OpenAI"):
            summarizer = LMStudioSummarizer(mock_config, "Template")
            summarizer.enabled = False
            
            result = summarizer.summarize_batch(["node1"], mock_builder)
            assert result == {}
