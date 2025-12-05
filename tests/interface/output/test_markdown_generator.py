"""
Tests for Markdown Generator module.
"""

import pytest
import networkx as nx
from unittest.mock import MagicMock
from codehierarchy.interface.output.markdown_generator import MarkdownGenerator

@pytest.fixture
def generator(tmp_path):
    return MarkdownGenerator(tmp_path)

def test_generate_documentation(generator, tmp_path):
    """Test documentation generation."""
    # Use real graph instead of mock to avoid iteration issues
    graph = nx.DiGraph()
    graph.add_node("node1", file="file1.py", line=10, type="function", name="func1")
    graph.add_node("node2", file="file2.py", line=20, type="class", name="Class1")
    
    # Mock summaries
    summaries = {
        "node1": "Summary for node 1",
        "node2": "Summary for node 2"
    }
    
    generator.generate_documentation(graph, summaries)
    
    output_dir = generator.output_dir
    assert output_dir.exists()
    
    output_file = output_dir / "CODEBASE_EXPLAINER.md"
    assert output_file.exists()
    content = output_file.read_text()
    assert "Summary for node 1" in content
    assert "func1" in content
    assert "Class1" in content

def test_generate_empty_documentation(generator, tmp_path):
    """Test generation with empty data."""
    graph = nx.DiGraph()
    summaries = {}
    
    output_dir = tmp_path / "empty_docs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update generator output_dir since fixture sets it to a different tmp_path
    generator.output_dir = output_dir
    generator.generate_documentation(graph, summaries)
    
    assert output_dir.exists()
    assert (output_dir / "CODEBASE_EXPLAINER.md").exists()
