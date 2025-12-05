"""
Interface package for CLI and output generation.

This package contains modules for:
- cli: Command-line interface
- output: Markdown and report generation
"""

from codehierarchy.interface.cli import *
from codehierarchy.interface.output import *

__all__ = [
    # Re-export key classes for convenience
    "main",
    "MarkdownGenerator",
]
