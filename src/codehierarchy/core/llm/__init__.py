"""
LLM submodule for language model integration.

Supports multiple backends via configuration:
- lmstudio: LM Studio with optional xvfb for headless
- llamacpp: Direct llama-server process management
"""

from .lmstudio_summarizer import LMStudioSummarizer
from .checkpoint import save_checkpoint, load_checkpoint
from .validator import validate_summary
from .backends import create_backend, BaseLLMBackend

# Alias for backward compatibility
Summarizer = LMStudioSummarizer

__all__ = [
    "LMStudioSummarizer",
    "Summarizer",
    "save_checkpoint",
    "load_checkpoint",
    "validate_summary",
    "create_backend",
    "BaseLLMBackend",
]
