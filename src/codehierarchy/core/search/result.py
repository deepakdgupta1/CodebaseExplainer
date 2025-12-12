"""
Search result data structure.

This module defines the Result class which represents a single
search result from either semantic, keyword, or hybrid search.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Result:
    """
    Represents a single search result.

    Contains metadata about the matched code element along with
    relevance scoring and optional context.

    Attributes:
        node_id: Unique identifier for the code element.
        name: Name of the function, class, or module.
        file: Path to the source file containing this element.
        line: Line number where the element is defined.
        summary: LLM-generated summary of the code element.
        score: Relevance score (higher is better).
        snippet: Optional source code snippet.
        explanation: Optional explanation of why this result matched.
    """

    node_id: str
    name: str
    file: str
    line: int
    summary: str
    score: float
    snippet: Optional[str] = None
    explanation: Optional[str] = None
