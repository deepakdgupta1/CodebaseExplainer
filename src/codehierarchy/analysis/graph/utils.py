import networkx as nx
from pathlib import Path
from typing import List

def get_module_subgraph(graph: nx.DiGraph, module_path: str) -> nx.DiGraph:
    """
    Extract subgraph containing only nodes from a specific module/file.
    """
    nodes = [n for n, d in graph.nodes(data=True) if d.get('file') == module_path]
    # subgraph returns a view, copy returns a new Graph/DiGraph
    # We ensure it's a DiGraph by casting or relying on input being DiGraph
    sub = graph.subgraph(nodes).copy()
    return sub # type: ignore

def get_dependency_chain(graph: nx.DiGraph, node_id: str) -> List[str]:
    """
    Get all transitive dependencies of a node (BFS).
    """
    return list(nx.bfs_tree(graph, node_id))

def export_graph(graph: nx.DiGraph, output_path: Path, format: str = 'graphml') -> None:
    """
    Export graph to file for visualization.
    """
    try:
        if format == 'graphml':
            nx.write_graphml(graph, output_path)
        elif format == 'gexf':
            nx.write_gexf(graph, output_path)
        elif format == 'dot':
            # Requires pydot or networkx dot support
            nx.drawing.nx_pydot.write_dot(graph, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    except Exception as e:
        print(f"Failed to export graph: {e}")
