"""
OpenAI-compatible API client wrapper using httpx.
Handles authentication, request/response, and error handling.
"""

import logging
from typing import Any, AsyncGenerator, Optional

import httpx

from config import get_config

logger = logging.getLogger(__name__)

MAX_TOKENS = 4000

class OpenAIClient:
    """Async client for OpenAI-compatible APIs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        cfg = get_config()
        self.api_key = api_key or cfg.openai.api_key
        self.base_url = base_url or cfg.openai.base_url
        self.model_name = model_name or cfg.model_name
        self.timeout = timeout if timeout is not None else cfg.timeout
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
            
            # Debug logging
            logger.debug("OpenAIClient config:")
            logger.debug("  base_url: %s", self.base_url)
            logger.debug("  api_key: %s", "**REDACTED**" if self.api_key else "(empty)")
            logger.debug("  headers: %s", headers)
            logger.debug("  model_name: %s", self.model_name)
            
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
            )
        return self._client

    async def _stream_completion(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = MAX_TOKENS,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a completion from the API, yielding content chunks.
        Handles cleanup of markdown code blocks.
        """
        client = await self._ensure_client()
        request_body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        try:
            logger.debug("Sending streaming POST request to: %s/chat/completions", self.base_url)
            logger.debug("Request body: %s", request_body)
            
            # Use stream=True for streaming responses
            async with client.stream("POST", "/chat/completions", json=request_body) as response:
                response.raise_for_status()
                
                # Buffer for handling markdown code blocks at boundaries
                buffer = ""
                first_chunk = True
                
                async for line in response.aiter_lines():
                    if line:
                        line = line.strip()
                        if line.startswith("data: ") and line != "data: [DONE]":
                            data_str = line[6:]  # Remove "data: " prefix
                            # Handle the data string - it's JSON
                            if data_str:
                                import json
                                try:
                                    data = json.loads(data_str)
                                    if data.get("choices") and len(data["choices"]) > 0:
                                        chunk = data["choices"][0].get("delta", {}).get("content", "")
                                        if chunk:
                                            buffer += chunk
                                            # Clean markdown code blocks incrementally
                                            # This is simplified - for streaming we need to be careful
                                            if first_chunk:
                                                # Clean leading markdown blocks from first chunk
                                                if buffer.startswith("```html"):
                                                    buffer = buffer[7:]
                                                elif buffer.startswith("```"):
                                                    buffer = buffer[3:]
                                                first_chunk = False
                                            
                                            # Yield available content, keeping some buffer for boundary handling
                                            # For simplicity, yield everything and let the route handler manage
                                            yield buffer
                                            buffer = ""
                                except json.JSONDecodeError:
                                    logger.warning("Failed to decode streaming JSON: %s", data_str)
                        elif line == "data: [DONE]":
                            # Final cleanup
                            if buffer:
                                if buffer.endswith("```"):
                                    buffer = buffer[:-3]
                                buffer = buffer.strip()
                                if buffer:
                                    yield buffer
                            break
        except httpx.HTTPStatusError as e:
            logger.error("Streaming API request failed: %s", e)
            raise

    async def generate_search_results_page(self, query: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a search results page HTML for a given query.
        
        Args:
            query: The search query from the user
            system_prompt: Optional custom system prompt
            
        Returns:
            Complete HTML string for a search results page.
        """
        results_system_prompt = (
            "You are a search results generator. "
            "Generate a complete HTML page showing search results for the query: "
            "'{query}'. "
            "The page should have:"
            " - A title tag mentioning 'Sloppy' and the query"
            " - A heading showing the query"
            " - A list of 5-10 relevant search results"
            " - Each result must have a clear title (as a link) and a short summary paragraph"
            " - Each result must be associated with a plausible domain name like 'reddit.com' or 'en.wikipedia.org'"
            " - Each link must use the format: href='/web/{domain}/{title}?q={query}' "
            "where {domain} is the plausible domain name, {title} is the title of the result (URL-encoded), and {query} is the search query (URL-encoded)"
            " - Results should be genuinely relevant to the query"
            " - Clean, readable layout"
            " - No external dependencies (inline CSS only)"
            " - Proper HTML5 doctype"
            "Return ONLY the complete HTML, no markdown, no code blocks, no explanations."
        )
        
        prompt = system_prompt or results_system_prompt
        
        request_body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Generate search results for: {query}"},
            ],
            "temperature": 0.7,
            "max_tokens": MAX_TOKENS,
        }

        client = await self._ensure_client()
        
        try:
            logger.debug("Sending POST request for search results to: %s/chat/completions", self.base_url)
            logger.debug("Request body: %s", request_body)
            response = await client.post("/chat/completions", json=request_body)
            logger.debug("Response status: %s", response.status_code)
            logger.debug("Response headers: %s", dict(response.headers))
            response.raise_for_status()
            data = response.json()
            
            if data.get("choices") and len(data["choices"]) > 0:
                content = data["choices"][0].get("message", {}).get("content", "")
                if content:
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

    async def generate_web_page(self, domain: str, query: str, title: str = "", system_prompt: Optional[str] = None) -> str:
        """
        Generate a web page for a specific domain, title, and query.
        
        Args:
            domain: The domain (e.g., 'reddit.com', 'en.wikipedia.org')
            query: The search query to influence page contents
            title: The title of the page
            system_prompt: Optional custom system prompt
            
        Returns:
            Complete HTML string for a web page.
        """
        page_system_prompt = (
            "You are a web page generator. "
            f"Generate a complete HTML page for domain '{domain}' with title '{title}' related to the query '{query}'. "
            "The page should have:"
            " - A title tag mentioning the domain and title"
            " - Content relevant to the domain, title, and query"
            " - If images are needed, use dummy placeholder images with descriptive alt text. "
            "Each image must have src='/img?t={alt_text}&w={width}&h={height}' "
            "where {alt_text} is the alt text (URL-encoded), {width} is the width in pixels, and {height} is the height in pixels"
            " - Clean, readable layout with proper structure"
            " - All links must use the format href='/web/{domain}/{link_title}' "
            "where {domain} is the current domain and {link_title} is derived from the link text "
            "(e.g., 'about' for an 'About Us' link, 'contact' for a 'Contact' link)"
            " - No external dependencies or real URLs"
            " - No functional JavaScript"
            " - Proper HTML5 doctype"
            "Return ONLY the complete HTML, no markdown, no code blocks, no explanations."
        )
        
        prompt = system_prompt or page_system_prompt
        
        request_body = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Generate a page for domain '{domain}' with title '{title}' about: {query}"},
            ],
            "temperature": 0.7,
            "max_tokens": MAX_TOKENS,
        }

        client = await self._ensure_client()
        
        try:
            logger.debug("Sending POST request for web page (domain=%s, title=%s, q=%s) to: %s/chat/completions", domain, title, query, self.base_url)
            logger.debug("Request body: %s", request_body)
            response = await client.post("/chat/completions", json=request_body)
            logger.debug("Response status: %s", response.status_code)
            logger.debug("Response headers: %s", dict(response.headers))
            response.raise_for_status()
            data = response.json()
            
            if data.get("choices") and len(data["choices"]) > 0:
                content = data["choices"][0].get("message", {}).get("content", "")
                if content:
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

    async def stream_web_page(
        self, 
        domain: str, 
        query: str, 
        title: str = "", 
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream a web page for a specific domain, title, and query.
        
        Args:
            domain: The domain (e.g., 'reddit.com', 'en.wikipedia.org')
            query: The search query to influence page contents
            title: The title of the page
            system_prompt: Optional custom system prompt
            
        Yields:
            Chunks of HTML content as they are streamed from the API.
        """
        page_system_prompt = (
            "You are a web page generator. "
            f"Generate a complete HTML page for domain '{domain}' with title '{title}' related to the query '{query}'. "
            "The page should have:"
            " - A title tag mentioning the domain and title"
            " - Content relevant to the domain, title, and query"
            " - If images are needed, use dummy placeholder images with descriptive alt text. "
            "Each image must have src='/img?t={alt_text}&w={width}&h={height}' "
            "where {alt_text} is the alt text (URL-encoded), {width} is the width in pixels, and {height} is the height in pixels"
            " - Clean, readable layout with proper structure"
            " - All links must use the format href='/web/{domain}/{link_title}' "
            "where {domain} is the current domain and {link_title} is derived from the link text "
            "(e.g., 'about' for an 'About Us' link, 'contact' for a 'Contact' link)"
            " - No external dependencies or real URLs"
            " - No functional JavaScript"
            " - Proper HTML5 doctype"
            "Return ONLY the complete HTML, no markdown, no code blocks, no explanations."
        )
        
        prompt = system_prompt or page_system_prompt
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Generate a page for domain '{domain}' with title '{title}' about: {query}"},
        ]
        
        async for chunk in self._stream_completion(messages, temperature=0.7, max_tokens=MAX_TOKENS):
            yield chunk

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
            " - A title tag that includes the word 'Sloppy'"
            " - A GET form with action='/web' and method='GET'"
            " - Two submit buttons: one labeled 'Search' and one labeled "
            "'I\'m Feeling Sloppy'"
            " - Responsive layout that works on mobile and desktop"
            " - No external dependencies (inline CSS only)"
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
            "max_tokens": MAX_TOKENS,
        }

        client = await self._ensure_client()
        
        try:
            logger.debug("Sending POST request to: %s/chat/completions", self.base_url)
            logger.debug("Request body: %s", request_body)
            response = await client.post("/chat/completions", json=request_body)
            logger.debug("Response status: %s", response.status_code)
            logger.debug("Response headers: %s", dict(response.headers))
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

    async def stream_search_results_page(
        self, query: str, system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream a search results page HTML for a given query.
        
        Args:
            query: The search query from the user
            system_prompt: Optional custom system prompt
            
        Yields:
            Chunks of HTML content as they are streamed from the API.
        """
        results_system_prompt = (
            "You are a search results generator. "
            "Generate a complete HTML page showing search results for the query: "
            f"'{query}'. "
            "The page should have:"
            " - A title tag mentioning 'Sloppy' and the query"
            " - A heading showing the query"
            " - A list of 5-10 relevant search results"
            " - Each result must have a clear title (as a link) and a short summary paragraph"
            " - Each result must be associated with a plausible domain name like 'reddit.com' or 'en.wikipedia.org'"
            f" - Each link must use the format: href='/web/{{domain}}/{{title}}?q={query}' "
            "where {domain} is the plausible domain name and {title} is the title of the result (URL-encoded)"
            " - Results should be genuinely relevant to the query"
            " - Clean, readable layout"
            " - No external dependencies (inline CSS only)"
            " - Proper HTML5 doctype"
            "Return ONLY the complete HTML, no markdown, no code blocks, no explanations."
        )
        
        prompt = system_prompt or results_system_prompt
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Generate search results for: {query}"},
        ]
        
        async for chunk in self._stream_completion(messages, temperature=0.7, max_tokens=MAX_TOKENS):
            yield chunk

    async def stream_search_page(self, system_prompt: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Stream a complete search page HTML.
        
        Args:
            system_prompt: Optional custom system prompt
            
        Yields:
            Chunks of HTML content as they are streamed from the API.
        """
        default_system_prompt = (
            "You are a web page generator. "
            "Generate a complete, simple search page HTML like Google or DuckDuckGo. "
            "The page should have:"
            " - A clean, minimal design with a centered search box"
            " - A logo or doodle in inline SVG format"
            " - A title tag that includes the word 'Sloppy'"
            " - A GET form with action='/web' and method='GET'"
            " - Two submit buttons: one labeled 'Search' and one labeled "
            "'I\'m Feeling Sloppy'"
            " - Responsive layout that works on mobile and desktop"
            " - No external dependencies (inline CSS only)"
            " - Proper HTML5 doctype"
            "Return ONLY the complete HTML, no markdown, no code blocks, no explanations."
        )
        
        prompt = system_prompt or default_system_prompt
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Generate a simple search page HTML."},
        ]
        
        async for chunk in self._stream_completion(messages, temperature=0.7, max_tokens=MAX_TOKENS):
            yield chunk

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Convenience function to get a client with global config
async def get_client() -> OpenAIClient:
    """Get an OpenAIClient instance with global configuration."""
    return OpenAIClient()
