"""
Sloppy - FastAPI server that generates search pages using AI.

Usage:
    uvicorn main:app --reload
    
    Or with CLI args:
    python main.py --api-key your_key --base-url https://api.openai.com/v1 --model gpt-4o-mini
"""

import argparse
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from client import OpenAIClient
from config import get_config, update_config_from_cli

logger = logging.getLogger(__name__)

# Configure logging
log_level = logging.DEBUG if os.environ.get("DEBUG", "").lower() in ("1", "true", "yes") else logging.INFO
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    cfg = get_config()
    logger.info("Starting Sloppy server")
    logger.info(f"Model: {cfg.model_name}")
    logger.info(f"OpenAI Base URL: {cfg.openai.base_url}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Sloppy server")


app = FastAPI(
    title="Sloppy",
    description="Generate web pages using AI models",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def generate_search_page(request: Request) -> HTMLResponse:
    """
    Generate a complete search page using the configured AI model.
    
    Returns:
        HTMLResponse with the generated search page.
    """
    try:
        async with OpenAIClient() as client:
            html_content = await client.generate_search_page()
            return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        logger.error(f"Failed to generate search page: {e}")
        error_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error - Sloppy</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
                h1 { color: #d32f2f; }
                p { color: #666; }
                .error { background: #ffebee; padding: 20px; border-radius: 5px; display: inline-block; }
            </style>
        </head>
        <body>
            <div class="error">
                <h1>Error Generating Page</h1>
                <p>Sorry, Sloppy couldn't generate the search page. Please check the server logs.</p>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=500)


@app.get("/web/{domain}/{title:path}")
async def web_page(domain: str, title: str, request: Request):
    """
    Generate a web page for a specific domain and title.
    
    Takes:
    - domain: Path parameter for the domain (e.g., 'reddit.com', 'en.wikipedia.org')
    - title: Path parameter for the page title
    - q: Query parameter containing the user's search query
    
    Returns a page matching the domain, title, and query.
    If STREAM=true is set, the page loads progressively.
    """
    query = request.query_params.get("q", "")
    cfg = get_config()
    
    try:
        async with OpenAIClient() as client:
            if cfg.stream:
                # Stream the response progressively
                async def generate():
                    async for chunk in client.stream_web_page(domain, query, title):
                        yield chunk
                return StreamingResponse(generate(), media_type="text/html")
            else:
                # Non-streaming: wait for complete response
                html_content = await client.generate_web_page(domain, query, title)
                return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        logger.error(f"Failed to generate web page for domain={domain}, title={title}, q={query}: {e}")
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error - Sloppy</title>
            <style>
                body {{ font-family: Arial, sans-serif; text-align: center; padding: 50px; }}
                h1 {{ color: #d32f2f; }}
                p {{ color: #666; }}
                .error {{ background: #ffebee; padding: 20px; border-radius: 5px; display: inline-block; }}
            </style>
        </head>
        <body>
            <div class="error">
                <h1>Error Generating Page</h1>
                <p>Sorry, Sloppy couldn't generate the page for {domain}/{title}. Please check the server logs.</p>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=500)


@app.get("/web", response_class=HTMLResponse)
async def search(request: Request) -> HTMLResponse:
    """
    Generate a search results page for the given query.
    
    Takes a 'q' query parameter containing the user's search query.
    Returns a page with relevant search results, each having a title and summary.
    
    Returns:
        HTMLResponse with the generated search results page.
    """
    query = request.query_params.get("q", "")
    if not query:
        error_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>No Query - Sloppy</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
                h1 { color: #d32f2f; }
                p { color: #666; }
            </style>
        </head>
        <body>
            <h1>No search query provided</h1>
            <p>Please enter a search term.</p>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=400)
    
    try:
        async with OpenAIClient() as client:
            html_content = await client.generate_search_results_page(query)
            return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        logger.error(f"Failed to generate search results: {e}")
        error_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error - Sloppy</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
                h1 { color: #d32f2f; }
                p { color: #666; }
                .error { background: #ffebee; padding: 20px; border-radius: 5px; display: inline-block; }
            </style>
        </head>
        <body>
            <div class="error">
                <h1>Error Generating Results</h1>
                <p>Sorry, Sloppy couldn't generate search results. Please check the server logs.</p>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=500)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """
    Health check endpoint.
    
    Returns:
        Simple JSON response indicating server is running.
    """
    return {"status": "ok", "message": "Sloppy is running"}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sloppy - Generate web pages using AI models"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (overrides env/config)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        help="OpenAI base URL (overrides env/config)",
    )
    parser.add_argument(
        "--model",
        type=str,
        dest="model_name",
        help="Model name to use (overrides env/config)",
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Server host (overrides env/config)",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Server port (overrides env/config)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help="Request timeout in seconds (overrides env/config)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming responses for progressive page loading",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Update config from CLI args
    update_config_from_cli(
        api_key=args.api_key,
        base_url=args.base_url,
        model_name=args.model_name,
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        stream=args.stream,
    )
    
    # Run the server
    import uvicorn
    
    cfg = get_config()
    uvicorn.run(
        "main:app",
        host=cfg.server.host,
        port=cfg.server.port,
        reload=True,
        log_level="info",
    )
