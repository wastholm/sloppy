"""
OpenAI-compatible API client wrapper using httpx.
Handles authentication, request/response, and error handling.
"""

import logging
from typing import Any, Optional

import httpx

from config import get_config

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Async client for OpenAI-compatible APIs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 30.0,
    ):
        cfg = get_config()
        self.api_key = api_key or cfg.openai.api_key
        self.base_url = base_url or cfg.openai.base_url
        self.model_name = model_name or cfg.model_name
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "OpenAIClient":
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Create client if it doesn't exist."""
        if self._client is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
            )
        return self._client

    async def generate_search_page(self, system_prompt: Optional[str] = None) -> str:
        """
        Generate a complete search page HTML using the configured model.
        
        Args:
            system_prompt: Optional custom system prompt. If None, uses a default.
        
        Returns:
            Complete HTML string for a search page.
        
        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """
        default_system_prompt = (
            "You are a web page generator. "
            "Generate a complete, simple search page HTML like Google or DuckDuckGo. "
            "The page should have:"
            " - A clean, minimal design with a centered search box"
            " - A search button or allow pressing Enter"
            " - Responsive layout that works on mobile and desktop"
            " - No external dependencies (inline CSS only)"
            " - A title tag"
            " - Proper HTML5 doctype"
            "Return ONLY the complete HTML, no markdown, no code blocks, no explanations."
        )
        
        prompt = system_prompt or default_system_prompt
        
        request_body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Generate a simple search page HTML."},
            ],
            "temperature": 0.7,
            "max_tokens": 2000,
        }

        client = await self._ensure_client()
        
        try:
            response = await client.post("/chat/completions", json=request_body)
            response.raise_for_status()
            data = response.json()
            
            # Extract the generated content
            if data.get("choices") and len(data["choices"]) > 0:
                content = data["choices"][0].get("message", {}).get("content", "")
                if content:
                    # Clean up any markdown code blocks if present
                    content = content.strip()
                    if content.startswith("```html"):
                        content = content[7:]
                    if content.startswith("```"):
                        content = content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()
                    return content
            
            logger.error("Unexpected response format: %s", data)
            raise ValueError("Unexpected response format from API")
            
        except httpx.HTTPStatusError as e:
            logger.error("API request failed: %s", e)
            raise

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Convenience function to get a client with global config
async def get_client() -> OpenAIClient:
    """Get an OpenAIClient instance with global configuration."""
    return OpenAIClient()
