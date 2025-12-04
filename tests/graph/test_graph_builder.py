import pytest
from pathlib import Path
from codehierarchy.graph.graph_builder import InMemoryGraphBuilder
from codehierarchy.parser.parallel_parser import ParseResult
from codehierarchy.parser.node_extractor import NodeInfo
from codehierarchy.parser.call_graph_analyzer import Edge

@pytest.fixture
def sample_results():
    file_path = Path("test.py")
    nodes = [
        NodeInfo(type="function", name="func1", line=1, end_line=2, source_code="def func1(): pass"),
        NodeInfo(type="function", name="func2", line=4, end_line=5, source_code="def func2(): func1()")
    ]
    edges = [
        Edge(source="func2", target="func1", type="call", confidence=1.0)
    ]
    return {
        file_path: ParseResult(nodes=nodes, edges=edges)
    }

def test_build_graph(sample_results):
    builder = InMemoryGraphBuilder()
    graph = builder.build_from_results(sample_results)
    
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    
    # Check node data
    node_id = "test.py:func1:1"
    assert graph.has_node(node_id)
    assert graph.nodes[node_id]['name'] == 'func1'
    
    # Check edge
    source_id = "test.py:func2:4"
    assert graph.has_edge(source_id, node_id)

def test_context_retrieval(sample_results):
    builder = InMemoryGraphBuilder()
    builder.build_from_results(sample_results)
    
    node_id = "test.py:func1:1"
    context = builder.get_node_with_context(node_id)
    
    assert context['node']['name'] == 'func1'
    assert context['source']['source_code'] == "def func1(): pass"
    assert len(context['parents']) == 1 # func2 calls func1
