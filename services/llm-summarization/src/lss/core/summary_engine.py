"""
Summary Engine for LLM-based summarization.

Coordinates prompt rendering, LLM generation, and response
validation for different summary types.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, AsyncGenerator, List

from ..backends.base import BaseLLMBackend
from .prompt_registry import PromptRegistry


logger = logging.getLogger(__name__)


@dataclass
class SummaryResult:
    """Result of summarization."""
    summary: str
    quality_score: float
    tokens_used: int
    provider: str
    model: str
    latency_ms: int
    cached: bool = False


class SummaryEngine:
    """
    Engine for generating summaries using LLM backends.

    Supports:
    - Multiple summary types (extractive, abstractive, hierarchical)
    - Configurable prompts and parameters
    - Quality validation
    - Caching
    - Streaming
    """

    def __init__(
        self,
        backend: BaseLLMBackend,
        prompt_registry: Optional[PromptRegistry] = None
    ) -> None:
        """
        Initialize the summary engine.

        Args:
            backend: LLM backend to use.
            prompt_registry: Prompt registry (uses default if None).
        """
        self.backend = backend
        self.prompts = prompt_registry or PromptRegistry()
        
        # Simple in-memory cache
        self._cache: Dict[str, SummaryResult] = {}

    async def summarize(
        self,
        content: str,
        summary_type: str = "abstractive",
        prompt_id: Optional[str] = None,
        context: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        use_cache: bool = True,
        **kwargs
    ) -> SummaryResult:
        """
        Generate a summary for content.

        Args:
            content: Content to summarize.
            summary_type: Type of summary (extractive/abstractive/hierarchical).
            prompt_id: Specific prompt to use (overrides summary_type).
            context: Additional context for summarization.
            max_tokens: Maximum tokens for response.
            temperature: Sampling temperature.
            use_cache: Whether to use/update cache.
            **kwargs: Additional template variables.

        Returns:
            SummaryResult with summary and metadata.
        """
        start_time = time.time()
        
        # Generate cache key
        cache_key = self._cache_key(content, summary_type, prompt_id)
        
        # Check cache
        if use_cache and cache_key in self._cache:
            result = self._cache[cache_key]
            result.cached = True
            return result
        
        # Select prompt
        if prompt_id:
            prompt_template = prompt_id
        else:
            prompt_template = self._default_prompt_for_type(summary_type)
        
        # Render prompt
        prompt = self.prompts.render(
            prompt_template,
            code=content,
            context=context or "",
            **kwargs
        )
        
        # Generate
        try:
            response = await self.backend.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            raise
        
        # Calculate quality score (simple heuristic)
        quality = self._estimate_quality(response, content)
        
        # Estimate tokens (rough: 4 chars per token)
        tokens = len(prompt) // 4 + len(response) // 4
        
        latency = int((time.time() - start_time) * 1000)
        
        result = SummaryResult(
            summary=response.strip(),
            quality_score=quality,
            tokens_used=tokens,
            provider=self.backend.provider_name,
            model=self.backend.model_id,
            latency_ms=latency,
            cached=False
        )
        
        # Update cache
        if use_cache:
            self._cache[cache_key] = result
        
        return result

    async def summarize_stream(
        self,
        content: str,
        summary_type: str = "abstractive",
        prompt_id: Optional[str] = None,
        context: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream summary generation.

        Args:
            content: Content to summarize.
            summary_type: Type of summary.
            prompt_id: Specific prompt to use.
            context: Additional context.
            max_tokens: Maximum tokens.
            temperature: Sampling temperature.
            **kwargs: Additional template variables.

        Yields:
            Text chunks as they are generated.
        """
        # Select and render prompt
        if prompt_id:
            prompt_template = prompt_id
        else:
            prompt_template = self._default_prompt_for_type(summary_type)
        
        prompt = self.prompts.render(
            prompt_template,
            code=content,
            context=context or "",
            **kwargs
        )
        
        # Stream from backend
        async for chunk in self.backend.generate_stream(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature
        ):
            yield chunk

    async def summarize_batch(
        self,
        items: List[Dict[str, Any]],
        summary_type: str = "abstractive",
        **kwargs
    ) -> List[SummaryResult]:
        """
        Summarize multiple items.

        Args:
            items: List of dicts with 'content' and optional 'context'.
            summary_type: Type of summary.
            **kwargs: Common parameters.

        Returns:
            List of SummaryResults.
        """
        results = []
        for item in items:
            result = await self.summarize(
                content=item["content"],
                context=item.get("context"),
                summary_type=summary_type,
                **kwargs
            )
            results.append(result)
        return results

    def _default_prompt_for_type(self, summary_type: str) -> str:
        """Get default prompt ID for summary type."""
        mapping = {
            "abstractive": "code_abstractive",
            "extractive": "code_extractive",
            "hierarchical": "code_hierarchical",
        }
        return mapping.get(summary_type, "code_abstractive")

    def _cache_key(
        self,
        content: str,
        summary_type: str,
        prompt_id: Optional[str]
    ) -> str:
        """Generate cache key."""
        import hashlib
        data = f"{content}:{summary_type}:{prompt_id or ''}"
        return hashlib.md5(data.encode()).hexdigest()

    def _estimate_quality(self, summary: str, original: str) -> float:
        """
        Estimate summary quality using heuristics.

        Checks:
        - Not too short
        - Not too similar to original (no copying)
        - Contains substantive content
        """
        if len(summary) < 20:
            return 0.3
        
        if len(summary) > len(original):
            return 0.4
        
        # Check not just copying
        if summary.strip() == original.strip():
            return 0.2
        
        # Basic quality estimate
        words = summary.split()
        if len(words) < 5:
            return 0.4
        
        return min(0.9, 0.5 + len(words) / 100)

    def clear_cache(self) -> None:
        """Clear the summary cache."""
        self._cache.clear()
