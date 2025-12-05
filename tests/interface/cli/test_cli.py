"""
Tests for CLI module.
"""

import pytest
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from codehierarchy.interface.cli.cli import cli

@pytest.fixture
def runner():
    return CliRunner()

def test_help(runner):
    """Test that help command works."""
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert "CodeHierarchy Explainer CLI" in result.output

@patch('codehierarchy.interface.cli.cli.load_config')
@patch('codehierarchy.interface.cli.cli.Orchestrator')
def test_analyze_command(MockOrchestrator, MockLoadConfig, runner, tmp_path):
    """Test analyze command execution."""
    mock_orchestrator = MockOrchestrator.return_value
    
    mock_graph = MagicMock()
    mock_graph.number_of_nodes.return_value = 10
    
    mock_orchestrator.run_pipeline.return_value = {"graph": mock_graph, "summaries": {}}
    
    # Mock config
    MockLoadConfig.return_value = MagicMock()
    MockLoadConfig.return_value.system.output_dir = tmp_path
    
    # Create a dummy directory to analyze
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    
    result = runner.invoke(cli, ['analyze', str(target_dir), '--output', str(tmp_path)], catch_exceptions=False)
    
    print(f"CLI Output: {result.output}")
    assert result.exit_code == 0
    assert "Analysis complete" in result.output
    mock_orchestrator.run_pipeline.assert_called_once()

@patch('codehierarchy.interface.cli.cli.Orchestrator')
def test_analyze_command_invalid_path(MockOrchestrator, runner):
    """Test analyze command with invalid path."""
    result = runner.invoke(cli, ['analyze', '/non/existent/path'])
    
    assert result.exit_code != 0
    assert "Error" in result.output

@patch('codehierarchy.interface.cli.cli.EnterpriseSearchEngine')
def test_search_command(MockSearchEngine, runner, tmp_path):
    """Test search command execution."""
    mock_result = MagicMock()
    mock_result.name = "test_node"
    mock_result.score = 0.9
    mock_result.file = "test.py"
    mock_result.summary = "test content summary"
    
    mock_engine = MockSearchEngine.return_value
    mock_engine.search.return_value = [mock_result]
    
    # Create dummy index files
    (tmp_path / "codebase_index.faiss").touch()
    (tmp_path / "metadata.pkl").touch()
    
    result = runner.invoke(cli, ['search', 'query', '--index-dir', str(tmp_path)])
    
    assert result.exit_code == 0
    assert "test_node" in result.output
