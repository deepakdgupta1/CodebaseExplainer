"""
OpenAI-compatible backend for LLM summarization.

Supports OpenAI API and compatible providers like:
- OpenAI (GPT-4, GPT-3.5)
- Azure OpenAI
- Local servers (llama.cpp, vLLM)
"""

import logging
from typing import Dict, Any, AsyncGenerator, Optional

import httpx

from .base import BaseLLMBackend


logger = logging.getLogger(__name__)


class OpenAIBackend(BaseLLMBackend):
    """
    OpenAI API compatible backend.

    Works with OpenAI, Azure, and local OpenAI-compatible servers.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-3.5-turbo",
        timeout: float = 60.0
    ) -> None:
        """
        Initialize OpenAI backend.

        Args:
            api_key: API key (empty for local servers).
            base_url: API base URL.
            model: Model identifier.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
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
        """Generate text using chat completions API."""
        client = await self._get_client()
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **self.get_extra_body(),
            **kwargs
        }
        
        try:
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        
        except httpx.HTTPError as e:
            logger.error(f"OpenAI request failed: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream text using chat completions API."""
        client = await self._get_client()
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            **self.get_extra_body(),
            **kwargs
        }
        
        try:
            async with client.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            import json
                            chunk = json.loads(data)
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                        except Exception:
                            continue
        
        except httpx.HTTPError as e:
            logger.error(f"OpenAI stream failed: {e}")
            raise

    async def is_healthy(self) -> bool:
        """Check if API is accessible."""
        try:
            client = await self._get_client()
            response = await client.get("/models")
            return response.status_code == 200
        except Exception:
            return False

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_id(self) -> str:
        return self.model

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
