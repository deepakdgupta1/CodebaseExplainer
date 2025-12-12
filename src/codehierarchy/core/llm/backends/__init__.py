"""
LLM Backends package.

Provides pluggable backend implementations for LLM inference:
- LlamaCppBackend: Direct llama-server process management
- LMStudioBackend: LM Studio with optional xvfb automation

Use create_backend() factory to instantiate based on config.
"""

from .base import BaseLLMBackend
from .factory import create_backend

__all__ = [
    "BaseLLMBackend",
    "create_backend",
]
