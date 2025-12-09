"""
CodeHierarchy Explainer - A high-performance codebase documentation and search system.

This package provides tools for:
- Analyzing codebases (parsing, scanning, dependency graphs)
- Generating AI-powered summaries using LLMs
- Searching code semantically and with keywords
- Creating comprehensive markdown documentation

The package is organized into functional groups:
- analysis: Code parsing, scanning, and graph building
- core: Pipeline orchestration, LLM integration, and search
- interface: CLI and output generation
- config: Configuration management
- utils: Shared utilities

For backward compatibility, key classes are re-exported at the top level.
"""

__version__ = "0.1.0"

# Re-export key modules for convenience and backward compatibility
from codehierarchy.analysis.parser import (
    TreeSitterParser,
    ParallelParser,
    NodeExtractor,
    CallGraphAnalyzer,
)

from codehierarchy.analysis.scanner import FileScanner
from codehierarchy.analysis.graph import InMemoryGraphBuilder

from codehierarchy.core import (
    Orchestrator,
    LMStudioSummarizer,
    HighQualityEmbedder,
    EnterpriseSearchEngine,
    KeywordSearch,
)

from codehierarchy.interface.cli import main
from codehierarchy.interface.output import MarkdownGenerator

from codehierarchy.config import Config, load_config
from codehierarchy.utils import setup_logging, Profiler, detect_language

__all__ = [
    # Version
    "__version__",

    # Analysis
    "TreeSitterParser",
    "ParallelParser",
    "NodeExtractor",
    "CallGraphAnalyzer",
    "FileScanner",
    "InMemoryGraphBuilder",

    # Core
    "Orchestrator",
    "LMStudioSummarizer",
    "HighQualityEmbedder",
    "EnterpriseSearchEngine",
    "KeywordSearch",

    # Interface
    "main",
    "MarkdownGenerator",

    # Config & Utils
    "Config",
    "load_config",
    "setup_logging",
    "Profiler",
    "detect_language",
]
