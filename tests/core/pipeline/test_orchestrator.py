"""
Tests for Pipeline Orchestrator module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from codehierarchy.core.pipeline.orchestrator import Orchestrator
from codehierarchy.config.schema import Config, SystemConfig, ParsingConfig, LLMConfig, EmbeddingsConfig

@pytest.fixture
def mock_config(tmp_path):
    return Config(
        system=SystemConfig(output_dir=tmp_path / "output"),
        parsing=ParsingConfig(),
        llm=LLMConfig(),
        embeddings=EmbeddingsConfig()
    )

@pytest.fixture
def orchestrator(mock_config):
    return Orchestrator(mock_config)

@patch('codehierarchy.core.pipeline.orchestrator.FileScanner')
@patch('codehierarchy.core.pipeline.orchestrator.ParallelParser')
@patch('codehierarchy.core.pipeline.orchestrator.InMemoryGraphBuilder')
@patch('codehierarchy.core.pipeline.orchestrator.DeepSeekSummarizer')
@patch('codehierarchy.core.pipeline.orchestrator.HighQualityEmbedder')
@patch('codehierarchy.core.pipeline.orchestrator.KeywordSearch')
def test_pipeline_execution_mocked(
    MockKeywordSearch, MockEmbedder, MockSummarizer, 
    MockGraphBuilder, MockParser, MockScanner, 
    orchestrator, tmp_path
):
    """Test full pipeline execution with mocked components."""
    
    # Setup mocks
    mock_scanner_instance = MockScanner.return_value
    mock_scanner_instance.scan_directory.return_value = [Path("test.py")]
    
    mock_parser_instance = MockParser.return_value
    mock_parser_instance.parse_repository.return_value = {"test.py": MagicMock()}
    
    mock_graph_builder_instance = MockGraphBuilder.return_value
    mock_graph = MagicMock()
    mock_graph.number_of_nodes.return_value = 10
    mock_graph.number_of_edges.return_value = 5
    mock_graph.nodes.return_value = ["node1"]
    mock_graph_builder_instance.build_from_results.return_value = mock_graph
    
    mock_summarizer_instance = MockSummarizer.return_value
    mock_summarizer_instance.summarize_batch.return_value = {"node1": "summary"}
    
    mock_embedder_instance = MockEmbedder.return_value
    mock_embedder_instance.build_index.return_value = (MagicMock(), {})
    
    # Run pipeline
    results = orchestrator.run_pipeline(tmp_path)
    
    # Verify execution flow
    mock_scanner_instance.scan_directory.assert_called_once()
    mock_parser_instance.parse_repository.assert_called_once()
    mock_graph_builder_instance.build_from_results.assert_called_once()
    mock_summarizer_instance.summarize_batch.assert_called()
    mock_embedder_instance.build_index.assert_called_once()
    
    assert results is not None
    assert 'graph' in results
    assert 'summaries' in results

def test_pipeline_no_files(orchestrator, tmp_path):
    """Test pipeline handles empty repository gracefully."""
    with patch('codehierarchy.core.pipeline.orchestrator.FileScanner') as MockScanner:
        mock_scanner = MockScanner.return_value
        mock_scanner.scan_directory.return_value = []
        
        results = orchestrator.run_pipeline(tmp_path)
        
        assert results == {}
