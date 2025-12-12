"""
Core package for pipeline orchestration, LLM integration, and search.

This package contains modules for:
- pipeline: Main orchestration logic
- llm: LLM summarization and checkpointing
- search: Semantic and keyword search
"""

from codehierarchy.core.pipeline.orchestrator import Orchestrator
from codehierarchy.core.llm.lmstudio_summarizer import (
    LMStudioSummarizer
)
from codehierarchy.core.search.embedder import HighQualityEmbedder
from codehierarchy.core.search.search_engine import EnterpriseSearchEngine
from codehierarchy.core.search.keyword_search import KeywordSearch

__all__ = [
    # Re-export key classes for convenience
    "Orchestrator",
    "LMStudioSummarizer",
    "HighQualityEmbedder",
    "EnterpriseSearchEngine",
    "KeywordSearch",
]
