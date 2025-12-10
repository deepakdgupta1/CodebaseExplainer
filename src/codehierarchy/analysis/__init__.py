"""
Analysis package for code parsing, scanning, and graph building.

This package contains modules for:
- parser: AST parsing using Tree-sitter
- scanner: File system scanning
- graph: Dependency graph construction
"""

from codehierarchy.analysis.parser.tree_sitter_parser import (
    TreeSitterParser
)
from codehierarchy.analysis.parser.parallel_parser import ParallelParser
from codehierarchy.analysis.parser.node_extractor import NodeExtractor
from codehierarchy.analysis.parser.call_graph_analyzer import (
    CallGraphAnalyzer
)
from codehierarchy.analysis.scanner.file_scanner import FileScanner
from codehierarchy.analysis.graph.graph_builder import InMemoryGraphBuilder

__all__ = [
    # Re-export key classes for convenience
    "TreeSitterParser",
    "ParallelParser",
    "NodeExtractor",
    "CallGraphAnalyzer",
    "FileScanner",
    "InMemoryGraphBuilder",
]
