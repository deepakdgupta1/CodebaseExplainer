"""
Backend factory for LLM providers.

Creates the appropriate backend instance based on configuration.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codehierarchy.config.schema import LLMConfig
    from .base import BaseLLMBackend


def create_backend(config: "LLMConfig") -> "BaseLLMBackend":
    """
    Create an LLM backend instance based on configuration.

    Args:
        config: LLMConfig with backend field set to 'llamacpp' or 'lmstudio'.

    Returns:
        Configured backend instance ready for setup().

    Raises:
        ValueError: If backend type is unknown.

    Example:
        >>> backend = create_backend(config.llm)
        >>> backend.setup()
        >>> if backend.is_healthy():
        ...     model_id = backend.load_model()
    """
    backend_type = getattr(config, 'backend', 'lmstudio')

    if backend_type == "llamacpp":
        from .llamacpp import LlamaCppBackend
        return LlamaCppBackend(config)
    elif backend_type == "lmstudio":
        from .lmstudio import LMStudioBackend
        return LMStudioBackend(config)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
