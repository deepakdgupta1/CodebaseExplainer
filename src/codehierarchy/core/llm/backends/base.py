"""
Abstract base class for LLM server backends.

This module defines the interface that all LLM backends must implement,
enabling configuration-based switching between providers like llama.cpp
and LM Studio.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseLLMBackend(ABC):
    """
    Abstract base class for LLM server backends.

    All backend implementations must provide methods for:
    - Server lifecycle (setup, shutdown)
    - Model loading and health checking
    - Configuration access (base_url, model_id)
    """

    @abstractmethod
    def setup(self) -> None:
        """
        Initialize and start the backend server.

        This method should:
        - Start any required processes
        - Wait for server to be ready
        - Register cleanup handlers

        Raises:
            RuntimeError: If server fails to start.
        """
        pass

    @abstractmethod
    def load_model(self) -> Optional[str]:
        """
        Load the configured model.

        Returns:
            Model identifier string if successful, None if failed.
        """
        pass

    @abstractmethod
    def is_healthy(self) -> bool:
        """
        Check if the backend server is responding.

        Returns:
            True if server is healthy and ready for requests.
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """
        Gracefully shutdown the backend server.

        Should terminate any child processes and cleanup resources.
        """
        pass

    @property
    @abstractmethod
    def base_url(self) -> str:
        """OpenAI-compatible API base URL (e.g., http://localhost:8080/v1)."""
        pass

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Currently loaded model identifier."""
        pass

    def get_extra_body(self) -> Dict[str, Any]:
        """
        Return backend-specific extra parameters for API calls.

        Override in subclasses to provide backend-specific params.
        Default returns empty dict (standard OpenAI params only).

        Returns:
            Dict of extra parameters to pass to chat.completions.create()
        """
        return {}
