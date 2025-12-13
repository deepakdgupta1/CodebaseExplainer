"""
Dependency Graph with red-green marking for incremental updates.

Tracks relationships between code entities and propagates
changes efficiently using fingerprint-based dirty detection.
"""

import hashlib
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Optional, Any

import networkx as nx


@dataclass
class DependencyNode:
    """A node in the dependency graph."""
    node_id: str
    node_type: str  # file, function, class, import
    file_path: str
    symbol_name: str
    fingerprint: str
    is_dirty: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class DependencyGraph:
    """
    Dependency graph with red-green marking for incremental updates.

    Tracks:
    - Import relationships
    - Function calls
    - Class inheritance
    - Module dependencies

    Uses fingerprinting to detect changes and mark affected
    nodes as "dirty" (red) for reprocessing.
    """

    def __init__(self, max_depth: int = 2) -> None:
        """
        Initialize the dependency graph.

        Args:
            max_depth: Maximum depth for dependency traversal.
        """
        self.graph = nx.DiGraph()
        self.max_depth = max_depth
        self.node_cache: Dict[str, DependencyNode] = {}

    def add_node(
        self,
        node_id: str,
        node_type: str,
        file_path: str,
        symbol_name: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a node to the dependency graph.

        Args:
            node_id: Unique identifier for the node.
            node_type: Type of node (function, class, etc).
            file_path: Path to source file.
            symbol_name: Name of the symbol.
            content: Source content for fingerprinting.
            metadata: Additional metadata.

        Returns:
            The node fingerprint.
        """
        fingerprint = self._compute_fingerprint(content)

        node = DependencyNode(
            node_id=node_id,
            node_type=node_type,
            file_path=file_path,
            symbol_name=symbol_name,
            fingerprint=fingerprint,
            metadata=metadata or {}
        )

        self.graph.add_node(node_id, **node.__dict__)
        self.node_cache[node_id] = node

        return fingerprint

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = 1.0
    ) -> None:
        """
        Add a dependency edge.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            edge_type: Type of relationship (calls, imports, inherits).
            weight: Edge weight for ranking.
        """
        self.graph.add_edge(
            source_id,
            target_id,
            type=edge_type,
            weight=weight
        )

    def get_dependencies(
        self,
        node_id: str,
        edge_types: Optional[List[str]] = None,
        max_depth: Optional[int] = None
    ) -> List[str]:
        """
        Get dependencies of a node.

        Args:
            node_id: Node to get dependencies for.
            edge_types: Filter by edge types.
            max_depth: Override default max depth.

        Returns:
            List of dependent node IDs.
        """
        if node_id not in self.graph:
            return []

        depth = max_depth or self.max_depth
        dependencies = set()

        def traverse(nid: str, current_depth: int):
            if current_depth > depth:
                return

            for successor in self.graph.successors(nid):
                edge_data = self.graph.get_edge_data(nid, successor)
                if edge_types is None or edge_data.get("type") in edge_types:
                    dependencies.add(successor)
                    traverse(successor, current_depth + 1)

        traverse(node_id, 1)
        return list(dependencies)

    def get_dependents(
        self,
        node_id: str,
        max_depth: Optional[int] = None
    ) -> List[str]:
        """
        Get nodes that depend on this node (reverse dependencies).

        Args:
            node_id: Node to find dependents for.
            max_depth: Override default max depth.

        Returns:
            List of dependent node IDs.
        """
        if node_id not in self.graph:
            return []

        depth = max_depth or self.max_depth
        dependents = set()

        def traverse(nid: str, current_depth: int):
            if current_depth > depth:
                return

            for predecessor in self.graph.predecessors(nid):
                dependents.add(predecessor)
                traverse(predecessor, current_depth + 1)

        traverse(node_id, 1)
        return list(dependents)

    def mark_dirty(self, node_id: str) -> Set[str]:
        """
        Mark a node as dirty and propagate to dependents.

        Args:
            node_id: Node that changed.

        Returns:
            Set of all dirty node IDs (including propagated).
        """
        dirty_nodes = set()

        if node_id in self.node_cache:
            self.node_cache[node_id].is_dirty = True
            dirty_nodes.add(node_id)

            # Propagate to dependents (red marking)
            dependents = self.get_dependents(node_id, max_depth=3)
            for dep_id in dependents:
                if dep_id in self.node_cache:
                    self.node_cache[dep_id].is_dirty = True
                    dirty_nodes.add(dep_id)

        return dirty_nodes

    def mark_clean(self, node_id: str) -> None:
        """
        Mark a node as clean after reprocessing.

        Args:
            node_id: Node that was reprocessed.
        """
        if node_id in self.node_cache:
            self.node_cache[node_id].is_dirty = False

    def update_node(
        self,
        node_id: str,
        new_content: str,
        new_edges: Optional[List[tuple]] = None
    ) -> Set[str]:
        """
        Update a node with new content and edges.

        Performs red-green marking:
        1. Compute new fingerprint
        2. If changed, mark node and dependents as dirty
        3. Update edges

        Args:
            node_id: Node to update.
            new_content: New source content.
            new_edges: New edges as (target_id, edge_type) tuples.

        Returns:
            Set of dirty node IDs.
        """
        dirty_nodes = set()

        if node_id not in self.node_cache:
            logging.warning(f"Node {node_id} not found for update")
            return dirty_nodes

        node = self.node_cache[node_id]
        new_fingerprint = self._compute_fingerprint(new_content)

        if new_fingerprint != node.fingerprint:
            # Content changed - mark dirty
            dirty_nodes = self.mark_dirty(node_id)
            node.fingerprint = new_fingerprint

            # Update graph node
            self.graph.nodes[node_id]['fingerprint'] = new_fingerprint

            logging.info(f"Node {node_id} changed, {len(dirty_nodes)} nodes marked dirty")

        # Update edges if provided
        if new_edges is not None:
            # Remove old edges from this node
            old_edges = list(self.graph.out_edges(node_id))
            self.graph.remove_edges_from(old_edges)

            # Add new edges
            for target_id, edge_type in new_edges:
                self.add_edge(node_id, target_id, edge_type)

        return dirty_nodes

    def get_dirty_nodes(self) -> List[str]:
        """Get all nodes marked as dirty."""
        return [nid for nid, node in self.node_cache.items() if node.is_dirty]

    def _compute_fingerprint(self, content: str) -> str:
        """Compute SHA-256 fingerprint of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def save(self, path: Path) -> None:
        """
        Save graph to disk.

        Args:
            path: Path to save file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'graph': nx.node_link_data(self.graph),
            'node_cache': {k: v.__dict__ for k, v in self.node_cache.items()},
            'max_depth': self.max_depth
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        logging.info(f"Saved dependency graph to {path}")

    def load(self, path: Path) -> None:
        """
        Load graph from disk.

        Args:
            path: Path to load file.
        """
        if not path.exists():
            logging.warning(f"Graph file {path} not found")
            return

        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.graph = nx.node_link_graph(data['graph'])
        self.max_depth = data.get('max_depth', 2)

        # Reconstruct node cache
        self.node_cache = {}
        for nid, node_data in data.get('node_cache', {}).items():
            self.node_cache[nid] = DependencyNode(**node_data)

        logging.info(f"Loaded dependency graph from {path}")
