"""
Anthropic Claude backend for LLM summarization.
"""

import logging
from typing import Dict, Any, AsyncGenerator, Optional

import httpx

from .base import BaseLLMBackend


logger = logging.getLogger(__name__)


class AnthropicBackend(BaseLLMBackend):
    """
    Anthropic Claude API backend.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-haiku-20240307",
        timeout: float = 60.0
    ) -> None:
        """
        Initialize Anthropic backend.

        Args:
            api_key: Anthropic API key.
            model: Model identifier.
            timeout: Request timeout.
        """
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.base_url = "https://api.anthropic.com/v1"
        
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01"
                },
                timeout=self.timeout
            )
        return self._client

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        **kwargs
    ) -> str:
        """Generate text using Claude messages API."""
        client = await self._get_client()
        
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        }
        
        try:
            response = await client.post("/messages", json=payload)
            response.raise_for_status()
            data = response.json()
            return data["content"][0]["text"]
        
        except httpx.HTTPError as e:
            logger.error(f"Anthropic request failed: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream text using Claude messages API."""
        client = await self._get_client()
        
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": True
        }
        
        try:
            async with client.stream("POST", "/messages", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            import json
                            data = json.loads(line[6:])
                            if data["type"] == "content_block_delta":
                                yield data["delta"]["text"]
                        except Exception:
                            continue
        
        except httpx.HTTPError as e:
            logger.error(f"Anthropic stream failed: {e}")
            raise

    async def is_healthy(self) -> bool:
        """Check if API is accessible."""
        try:
            # Simple auth check
            client = await self._get_client()
            response = await client.post("/messages", json={
                "model": self.model,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "ping"}]
            })
            return response.status_code in (200, 400)  # 400 = valid auth, bad request
        except Exception:
            return False

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_id(self) -> str:
        return self.model

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
