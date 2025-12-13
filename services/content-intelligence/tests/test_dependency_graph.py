"""Unit tests for DependencyGraph."""

import pytest
from pathlib import Path
import tempfile

from cis.core.dependency_graph import DependencyGraph, DependencyNode


class TestDependencyGraph:
    """Tests for DependencyGraph class."""

    @pytest.fixture
    def graph(self):
        """Create a fresh graph."""
        return DependencyGraph(max_depth=2)

    def test_init(self, graph):
        """Test graph initialization."""
        assert graph.max_depth == 2
        assert len(graph.node_cache) == 0

    def test_add_node(self, graph):
        """Test adding a node."""
        fingerprint = graph.add_node(
            node_id="file.py:func:10",
            node_type="function",
            file_path="file.py",
            symbol_name="func",
            content="def func(): pass"
        )
        
        assert "file.py:func:10" in graph.node_cache
        assert len(fingerprint) == 64  # SHA-256 hex

    def test_add_edge(self, graph):
        """Test adding an edge."""
        graph.add_node("a", "function", "a.py", "a", "content a")
        graph.add_node("b", "function", "b.py", "b", "content b")
        
        graph.add_edge("a", "b", "calls")
        
        assert graph.graph.has_edge("a", "b")

    def test_get_dependencies(self, graph):
        """Test getting dependencies."""
        graph.add_node("a", "function", "a.py", "a", "a")
        graph.add_node("b", "function", "b.py", "b", "b")
        graph.add_node("c", "function", "c.py", "c", "c")
        
        graph.add_edge("a", "b", "calls")
        graph.add_edge("b", "c", "calls")
        
        deps = graph.get_dependencies("a", max_depth=2)
        
        assert "b" in deps
        assert "c" in deps

    def test_get_dependencies_depth_limit(self, graph):
        """Test depth limit on dependencies."""
        graph.add_node("a", "function", "a.py", "a", "a")
        graph.add_node("b", "function", "b.py", "b", "b")
        graph.add_node("c", "function", "c.py", "c", "c")
        graph.add_node("d", "function", "d.py", "d", "d")
        
        graph.add_edge("a", "b", "calls")
        graph.add_edge("b", "c", "calls")
        graph.add_edge("c", "d", "calls")
        
        deps = graph.get_dependencies("a", max_depth=1)
        
        assert "b" in deps
        assert "c" not in deps
        assert "d" not in deps

    def test_get_dependents(self, graph):
        """Test getting reverse dependencies."""
        graph.add_node("a", "function", "a.py", "a", "a")
        graph.add_node("b", "function", "b.py", "b", "b")
        
        graph.add_edge("a", "b", "calls")
        
        dependents = graph.get_dependents("b")
        
        assert "a" in dependents

    def test_mark_dirty(self, graph):
        """Test marking nodes as dirty."""
        graph.add_node("a", "function", "a.py", "a", "a")
        graph.add_node("b", "function", "b.py", "b", "b")
        graph.add_edge("a", "b", "calls")
        
        dirty = graph.mark_dirty("b")
        
        assert "b" in dirty
        assert "a" in dirty  # Dependent should also be dirty

    def test_mark_clean(self, graph):
        """Test marking node as clean."""
        graph.add_node("a", "function", "a.py", "a", "a")
        graph.mark_dirty("a")
        
        assert graph.node_cache["a"].is_dirty is True
        
        graph.mark_clean("a")
        
        assert graph.node_cache["a"].is_dirty is False

    def test_update_node_unchanged(self, graph):
        """Test updating node with same content."""
        content = "def func(): pass"
        graph.add_node("a", "function", "a.py", "a", content)
        
        dirty = graph.update_node("a", content)
        
        assert len(dirty) == 0  # No change

    def test_update_node_changed(self, graph):
        """Test updating node with changed content."""
        graph.add_node("a", "function", "a.py", "a", "original")
        
        dirty = graph.update_node("a", "modified")
        
        assert "a" in dirty

    def test_get_dirty_nodes(self, graph):
        """Test getting all dirty nodes."""
        graph.add_node("a", "function", "a.py", "a", "a")
        graph.add_node("b", "function", "b.py", "b", "b")
        
        graph.mark_dirty("a")
        
        dirty = graph.get_dirty_nodes()
        
        assert "a" in dirty
        assert "b" not in dirty

    def test_save_and_load(self, graph):
        """Test saving and loading graph."""
        graph.add_node("a", "function", "a.py", "a", "content")
        graph.add_edge("a", "a", "self")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph.pkl"
            graph.save(path)
            
            new_graph = DependencyGraph()
            new_graph.load(path)
            
            assert "a" in new_graph.node_cache

    def test_compute_fingerprint(self, graph):
        """Test fingerprint computation."""
        fp1 = graph._compute_fingerprint("content1")
        fp2 = graph._compute_fingerprint("content2")
        fp3 = graph._compute_fingerprint("content1")
        
        assert fp1 != fp2
        assert fp1 == fp3
        assert len(fp1) == 64
