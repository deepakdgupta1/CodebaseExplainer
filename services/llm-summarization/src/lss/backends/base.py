"""
LLM Backend Base Class.

Provides abstract interface for different LLM providers,
enabling seamless switching between local and API backends.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncGenerator


class BaseLLMBackend(ABC):
    """
    Abstract base class for LLM backends.

    Implementations handle:
    - Connection/authentication
    - Request formatting
    - Response parsing
    - Streaming support
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        **kwargs
    ) -> str:
        """
        Generate text completion.

        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Provider-specific parameters.

        Returns:
            Generated text.
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream text completion.

        Args:
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Provider-specific parameters.

        Yields:
            Text chunks as they are generated.
        """
        pass

    @abstractmethod
    async def is_healthy(self) -> bool:
        """Check if backend is available."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider name."""
        pass

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return current model identifier."""
        pass

    def get_extra_body(self) -> Dict[str, Any]:
        """Return provider-specific API parameters."""
        return {}
