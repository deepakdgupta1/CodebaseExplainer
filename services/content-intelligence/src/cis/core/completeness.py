"""
Completeness Scorer for context quality assessment.

Measures how completely the retrieved context covers the
required symbols and dependencies for a query.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Set, Dict, Any, Optional


@dataclass
class CompletenessResult:
    """Result of completeness scoring."""
    overall_score: float
    symbol_resolution: float
    dependency_coverage: float
    unresolved_symbols: List[str]
    missing_dependencies: List[str]


class CompletenessScorer:
    """
    Scores context completeness for retrieved chunks.

    Completeness Formula:
        score = 0.7 * symbol_resolution + 0.3 * dependency_coverage

    Where:
        symbol_resolution = defined_symbols / referenced_symbols
        dependency_coverage = included_deps / critical_deps

    Target: >0.90
    """

    def __init__(
        self,
        symbol_weight: float = 0.7,
        dependency_weight: float = 0.3,
        min_threshold: float = 0.9
    ) -> None:
        """
        Initialize the completeness scorer.

        Args:
            symbol_weight: Weight for symbol resolution (0-1).
            dependency_weight: Weight for dependency coverage.
            min_threshold: Minimum acceptable completeness.
        """
        self.symbol_weight = symbol_weight
        self.dependency_weight = dependency_weight
        self.min_threshold = min_threshold

    def score(
        self,
        chunks: List[Dict[str, Any]],
        dependencies: Optional[List[str]] = None
    ) -> CompletenessResult:
        """
        Calculate completeness score for retrieved chunks.

        Args:
            chunks: List of chunk dicts with 'content', 'symbol_name'.
            dependencies: Optional list of critical dependency IDs.

        Returns:
            CompletenessResult with scores and details.
        """
        # Extract symbols from chunks
        defined_symbols = set()
        referenced_symbols = set()

        for chunk in chunks:
            content = chunk.get("content", "")
            symbol = chunk.get("symbol_name", "")

            if symbol:
                defined_symbols.add(symbol)

            # Extract references from content
            refs = self._extract_references(content)
            referenced_symbols.update(refs)

        # Calculate symbol resolution
        unresolved = referenced_symbols - defined_symbols
        resolved_count = len(referenced_symbols) - len(unresolved)
        symbol_resolution = (
            resolved_count / len(referenced_symbols)
            if referenced_symbols else 1.0
        )

        # Calculate dependency coverage
        missing_deps = []
        if dependencies:
            included = set(c.get("chunk_id", "") for c in chunks)
            missing_deps = [d for d in dependencies if d not in included]
            included_count = len(dependencies) - len(missing_deps)
            dependency_coverage = (
                included_count / len(dependencies)
                if dependencies else 1.0
            )
        else:
            dependency_coverage = 1.0

        # Calculate overall score
        overall = (
            self.symbol_weight * symbol_resolution +
            self.dependency_weight * dependency_coverage
        )

        return CompletenessResult(
            overall_score=overall,
            symbol_resolution=symbol_resolution,
            dependency_coverage=dependency_coverage,
            unresolved_symbols=list(unresolved),
            missing_dependencies=missing_deps
        )

    def is_complete(self, result: CompletenessResult) -> bool:
        """Check if result meets minimum threshold."""
        return result.overall_score >= self.min_threshold

    def suggest_additions(
        self,
        result: CompletenessResult,
        graph: Any  # DependencyGraph
    ) -> List[str]:
        """
        Suggest additional chunks to improve completeness.

        Args:
            result: Current completeness result.
            graph: DependencyGraph for lookups.

        Returns:
            List of suggested chunk IDs to add.
        """
        suggestions = []

        # Add chunks for unresolved symbols
        for symbol in result.unresolved_symbols[:10]:  # Limit suggestions
            # Look up in graph by symbol name
            matches = self._find_by_symbol(symbol, graph)
            suggestions.extend(matches)

        # Add missing critical dependencies
        suggestions.extend(result.missing_dependencies[:5])

        return list(set(suggestions))  # Deduplicate

    def _extract_references(self, content: str) -> Set[str]:
        """
        Extract symbol references from code content.

        Looks for:
        - Function calls: name(...)
        - Class references: ClassName
        - Attribute access: self.method
        """
        references = set()

        # Function calls
        call_pattern = re.compile(r'\b([a-zA-Z_]\w*)\s*\(')
        for match in call_pattern.finditer(content):
            name = match.group(1)
            # Filter common keywords
            if name not in ('if', 'for', 'while', 'with', 'def', 'class', 'return', 'print', 'str', 'int', 'list', 'dict', 'set', 'len', 'range'):
                references.add(name)

        # Class inheritance
        inherit_pattern = re.compile(r'class\s+\w+\s*\(\s*([^)]+)\)')
        for match in inherit_pattern.finditer(content):
            bases = match.group(1).split(',')
            for base in bases:
                name = base.strip().split('.')[0]
                if name:
                    references.add(name)

        # Import statements (extract imported names)
        import_pattern = re.compile(r'from\s+\S+\s+import\s+([^#\n]+)')
        for match in import_pattern.finditer(content):
            imports = match.group(1).split(',')
            for imp in imports:
                name = imp.strip().split(' as ')[0].strip()
                if name:
                    references.add(name)

        return references

    def _find_by_symbol(self, symbol: str, graph: Any) -> List[str]:
        """Find chunk IDs matching a symbol name."""
        matches = []

        if hasattr(graph, 'node_cache'):
            for node_id, node in graph.node_cache.items():
                if node.symbol_name == symbol:
                    matches.append(node_id)

        return matches
