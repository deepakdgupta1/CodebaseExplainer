"""
Interface package for CLI and output generation.

This package contains modules for:
- cli: Command-line interface
- output: Markdown and report generation
"""

from codehierarchy.interface.cli.cli import main
from codehierarchy.interface.output.markdown_generator import (
    MarkdownGenerator
)

__all__ = [
    # Re-export key classes for convenience
    "main",
    "MarkdownGenerator",
]
