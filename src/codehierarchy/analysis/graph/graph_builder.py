from typing import Dict, List, Any, Optional
import networkx as nx
from pathlib import Path
import logging
from codehierarchy.analysis.parser.parallel_parser import ParseResult


class InMemoryGraphBuilder:
    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self.node_cache: Dict[str, Dict[str, Any]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

        # Index for resolution: name -> list of node_ids
        self.name_index: Dict[str, List[str]] = {}
        # Index: file_path -> name -> node_id
        self.file_index: Dict[str, Dict[str, str]] = {}

    def build_from_results(
            self, results: Dict[Path, ParseResult]) -> nx.DiGraph:
        """
        Build the dependency graph from parse results.

        Args:
            results: A dictionary mapping file paths to their parse results.

        Returns:
            A NetworkX DiGraph representing the codebase structure.
        """
        logging.info("Building graph from parse results...")

        # Phase 1: Add all nodes
        for file_path, result in results.items():
            if result.error or result.skipped:
                continue

            file_str = str(file_path)
            self.file_index[file_str] = {}

            for node in result.nodes:
                # Create unique ID
                node_id = f"{file_str}:{node.name}:{node.line}"

                # Add to graph
                self.graph.add_node(
                    node_id,
                    type=node.type,
                    name=node.name,
                    file=file_str,
                    line=node.line,
                    end_line=node.end_line,
                    complexity=node.complexity,
                    loc=node.loc
                )

                # Cache content
                self.node_cache[node_id] = {
                    'source_code': node.source_code,
                    'docstring': node.docstring,
                    'signature': node.signature
                }

                # Store metadata
                self.metadata[node_id] = {
                    'complexity': node.complexity,
                    'loc': node.loc
                }

                # Update indices
                if node.name not in self.name_index:
                    self.name_index[node.name] = []
                self.name_index[node.name].append(node_id)

                self.file_index[file_str][node.name] = node_id

        # Phase 2: Add edges
        for file_path, result in results.items():
            if result.error or result.skipped:
                continue

            file_str = str(file_path)

            for edge in result.edges:
                source_id = self._resolve_node(file_str, edge.source)
                target_id = self._resolve_node(file_str, edge.target)

                if source_id and target_id:
                    # Avoid self-loops if desired, or keep them
                    if source_id != target_id:
                        self.graph.add_edge(
                            source_id,
                            target_id,
                            type=edge.type,
                            weight=edge.confidence
                        )
                elif source_id and edge.type == 'import':
                    # For imports, target might be a module not a node
                    # We can create a module node if it doesn't exist
                    # But for now, let's skip or handle differently
                    pass

        # Phase 3: Compute metrics
        self._compute_metrics()

        return self.graph

    def _resolve_node(self, current_file: str, name: str) -> Optional[str]:
        """
        Resolve a name to a node ID.
        Strategy:
        1. Check current file.
        2. Check global index (if unique).
        3. (Future) Check imports.
        """
        # 1. Local resolution
        if current_file in self.file_index and name in self.file_index[current_file]:
            return self.file_index[current_file][name]

        # 2. Global resolution
        if name in self.name_index:
            candidates = self.name_index[name]
            if len(candidates) == 1:
                return candidates[0]
            # If multiple, we could try to use imports to disambiguate
            # For now, return None to avoid wrong edges
            return None

        return None

    def _compute_metrics(self) -> None:
        """
        Compute graph-theoretic metrics.
        """
        if not self.graph:
            return

        # PageRank for centrality
        try:
            pagerank = nx.pagerank(self.graph)
            for node_id, score in pagerank.items():
                if node_id in self.metadata:
                    self.metadata[node_id]['centrality'] = score
        except Exception as e:
            logging.warning(f"Failed to compute PageRank: {e}")

        # Identify critical paths
        # (simplified: high centrality + high complexity)
        # ...

    def get_node_with_context(
            self, node_id: str, depth: int = 2) -> Dict[str, Any]:
        """
        Retrieve node with full context for LLM.

        Args:
            node_id: The unique identifier of the node.
            depth: The depth of context to retrieve (currently unused but reserved for future).

        Returns:
            A dictionary containing node data, source code, metadata, parents, and children.
        """
        if node_id not in self.graph:
            return {}

        node_data = self.graph.nodes[node_id]
        cache_data = self.node_cache.get(node_id, {})
        meta_data = self.metadata.get(node_id, {})

        # Get neighbors
        parents = list(self.graph.predecessors(node_id))
        children = list(self.graph.successors(node_id))

        return {
            'node': node_data,
            'source': cache_data,
            'metadata': meta_data,
            'parents': parents,
            'children': children
        }
