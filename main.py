"""
Sloppy - FastAPI server that generates search pages using AI.

Usage:
    uvicorn main:app --reload
    
    Or with CLI args:
    python main.py --api-key your_key --base-url https://api.openai.com/v1 --model gpt-4o-mini
"""

import argparse
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import httpx

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse

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


async def _html_response(
    client: OpenAIClient,
    stream_func: str,
    non_stream_func: str,
    *args: Any,
    **kwargs: Any,
) -> StreamingResponse | HTMLResponse:
    """
    Common helper to return either StreamingResponse or HTMLResponse based on config.
    
    Args:
        client: OpenAIClient instance
        stream_func: Method name for streaming (e.g., 'stream_search_page')
        non_stream_func: Method name for non-streaming (e.g., 'generate_search_page')
        *args: Arguments to pass to the methods
        **kwargs: Keyword arguments to pass to the methods
    
    Returns:
        StreamingResponse if STREAM=true, otherwise HTMLResponse
    """
    cfg = get_config()
    if cfg.stream:
        stream_method = getattr(client, stream_func)
        async def generate():
            async for chunk in stream_method(*args, **kwargs):
                yield chunk
        return StreamingResponse(generate(), media_type="text/html")
    else:
        non_stream_method = getattr(client, non_stream_func)
        content = await non_stream_method(*args, **kwargs)
        return HTMLResponse(content=content, status_code=200)


@app.get("/")
async def generate_search_page(request: Request):
    """
    Generate a complete search page using the configured AI model.
    
    Returns:
        HTMLResponse or StreamingResponse with the generated search page.
    """
    try:
        async with OpenAIClient() as client:
            return await _html_response(
                client,
                "stream_search_page",
                "generate_search_page",
            )
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
    
    try:
        async with OpenAIClient() as client:
            return await _html_response(
                client,
                "stream_web_page",
                "generate_web_page",
                domain,
                query,
                title,
            )
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


@app.get("/web")
async def search(request: Request):
    """
    Generate a search results page for the given query.
    
    Takes a 'q' query parameter containing the user's search query.
    Returns a page with relevant search results, each having a title and summary.
    
    Returns:
        HTMLResponse or StreamingResponse with the generated search results page.
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
            return await _html_response(
                client,
                "stream_search_results_page",
                "generate_search_results_page",
                query,
            )
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


async def _get_comfyui_models(client: httpx.AsyncClient, base_url: str) -> tuple[str | None, str | None]:
    """
    Query ComfyUI for available checkpoints and VAEs.
    Tries multiple endpoint variations since the REST API extension may not be installed.
    Returns (checkpoint_name, vae_name) - may return None if not discoverable.
    """
    # List of endpoint variations to try for checkpoints
    checkpoint_endpoints = [
        "/sdapi/v1/sd-models",
        "/models/checkpoints",
        "/models",
    ]
    
    checkpoint = None
    for endpoint in checkpoint_endpoints:
        try:
            response = await client.get(f"{base_url}{endpoint}")
            if response.status_code == 404:
                continue
            response.raise_for_status()
            models = response.json()
            if models and isinstance(models, list):
                logger.info(f"Available checkpoints via {endpoint}: {models}")
                # Prefer Z-Image Turbo, then standard SD checkpoints
                for m in models:
                    if "z-image" in m.lower() or "z_image" in m.lower() or "zimage" in m.lower() or "turbo" in m.lower():
                        checkpoint = m
                        logger.info(f"Found Z-Image Turbo checkpoint via {endpoint}: {checkpoint}")
                        break
                if checkpoint is None:
                    # Prefer standard SD checkpoints over FLUX ones
                    for m in models:
                        if any(sd_keyword in m.lower() for sd_keyword in ["dreamshaper", "pixelwave", "model-v2", "model.safetensors"]):
                            checkpoint = m
                            logger.info(f"Found SD checkpoint via {endpoint}: {checkpoint}")
                            break
                if checkpoint is None:
                    checkpoint = models[0]
                    logger.info(f"Using first checkpoint via {endpoint}: {checkpoint}")
                break
            elif models and isinstance(models, dict):
                # Some endpoints return {"checkpoints": [...]} or similar
                for key, value in models.items():
                    if isinstance(value, list) and value and key.lower() in ("checkpoints", "checkpoint", "models", "sd"):
                        logger.info(f"Available checkpoints via {endpoint} (dict key={key}): {value}")
                        # Prefer Z-Image Turbo, then standard SD checkpoints
                        for m in value:
                            if "z-image" in m.lower() or "z_image" in m.lower() or "zimage" in m.lower() or "turbo" in m.lower():
                                checkpoint = m
                                logger.info(f"Found Z-Image Turbo checkpoint via {endpoint} (dict): {checkpoint}")
                                break
                        if checkpoint is None:
                            for m in value:
                                if any(sd_keyword in m.lower() for sd_keyword in ["dreamshaper", "pixelwave", "model-v2", "model.safetensors"]):
                                    checkpoint = m
                                    logger.info(f"Found SD checkpoint via {endpoint} (dict): {checkpoint}")
                                    break
                        if checkpoint is None:
                            checkpoint = value[0]
                            logger.info(f"Using first checkpoint via {endpoint} (dict): {checkpoint}")
                        break
                if checkpoint:
                    break
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                logger.debug(f"Endpoint {endpoint} returned {e.response.status_code}")
            continue
        except Exception as e:
            logger.debug(f"Failed to parse response from {endpoint}: {e}")
            continue
    
    if checkpoint is None:
        # If we couldn't find any, try to get the default model from system info
        try:
            response = await client.get(f"{base_url}/system_stats")
            response.raise_for_status()
            stats = response.json()
            # The default checkpoint might be in the stats
            checkpoint = stats.get("sd_model_checkpoint") or stats.get("checkpoint")
            if checkpoint:
                logger.info(f"Found checkpoint from system_stats: {checkpoint}")
        except Exception:
            pass
    
    if checkpoint is None:
        logger.warning("Could not determine checkpoint from API, will return None")

    # List of endpoint variations to try for VAEs
    vae_endpoints = [
        "/sdapi/v1/vae",
        "/models/vae",
        "/models",
    ]
    
    vae = None
    for endpoint in vae_endpoints:
        try:
            response = await client.get(f"{base_url}{endpoint}")
            if response.status_code == 404:
                continue
            response.raise_for_status()
            vaes = response.json()
            if vaes and isinstance(vaes, list):
                for v in vaes:
                    if "vae" in v.lower():
                        vae = v
                        logger.info(f"Found VAE via {endpoint}: {vae}")
                        break
                if vae:
                    break
            elif vaes and isinstance(vaes, dict):
                for key, value in vaes.items():
                    if isinstance(value, list) and value:
                        for v in value:
                            if "vae" in str(v).lower():
                                vae = v
                                logger.info(f"Found VAE via {endpoint} (dict): {vae}")
                                break
                        if vae:
                            break
                if vae:
                    break
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                logger.debug(f"Endpoint {endpoint} returned {e.response.status_code}")
            continue
        except Exception as e:
            logger.debug(f"Failed to parse response from {endpoint}: {e}")
            continue
    
    if vae is None:
        try:
            response = await client.get(f"{base_url}/system_stats")
            response.raise_for_status()
            stats = response.json()
            vae = stats.get("vae") or stats.get("vae_model")
            if vae:
                logger.info(f"Found VAE from system_stats: {vae}")
        except Exception:
            pass
    
    if vae is None:
        logger.warning("Could not determine VAE from API, will return None")

    return checkpoint, vae


def _is_flux_checkpoint(checkpoint_name: str) -> bool:
    """Check if a checkpoint name appears to be a FLUX model."""
    flux_keywords = ["flux", "Flux", "FLUX", "noobai", "NoobAI", "xl-vpred", "vpred"]
    return any(keyword in checkpoint_name.lower() for keyword in flux_keywords)


async def _execute_comfyui_workflow(
    client: httpx.AsyncClient,
    base_url: str,
    prompt: str,
    width: int,
    height: int,
    checkpoint: str | None,
    vae: str | None,
) -> bytes | None:
    """
    Execute a txt2img workflow on ComfyUI and return the image bytes.
    Returns None if image generation fails or if checkpoint/vae are not provided.
    
    Handles both SD-style (CLIP) and FLUX-style (T5) models.
    Requires both checkpoint and VAE to be provided (not None).
    """
    # We need both checkpoint and VAE to build a valid workflow
    if checkpoint is None or vae is None:
        logger.warning(f"Cannot build workflow without checkpoint and VAE (checkpoint={checkpoint}, vae={vae})")
        return None
    
    # Determine model type and build appropriate workflow
    is_flux = _is_flux_checkpoint(checkpoint)
    
    if is_flux:
        logger.debug(f"Using FLUX workflow for checkpoint: {checkpoint}")
        # Try using checkpoint's own text encoder (may be T5 or CLIP)
        # For NoobAI-XL-Vpred, CheckpointLoaderSimple may output text encoder at index 1
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": checkpoint}
            },
            "2": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": vae}
            },
            "3": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": width, "height": height, "batch_size": 1}
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["1", 1], "text": prompt}
            },
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["1", 1], "text": "worst quality, low quality, blurry"}
            },
            "6": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "latent_image": ["3", 0],
                    "seed": 123456789,
                    "steps": 20,
                    "cfg": 0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                }
            },
            "7": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["6", 0], "vae": ["2", 0]}
            },
            "8": {
                "class_type": "SaveImage",
                "inputs": {"images": ["7", 0]}
            }
        }
    else:
        logger.debug(f"Using SD workflow for checkpoint: {checkpoint}")
        # SD models: use checkpoint's own CLIP encoder (index 1 from CheckpointLoaderSimple)
        workflow = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": checkpoint}
            },
            "2": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": vae}
            },
            "3": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": width, "height": height, "batch_size": 1}
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["1", 1], "text": prompt}
            },
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["1", 1], "text": "worst quality, low quality, blurry"}
            },
            "6": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "latent_image": ["3", 0],
                    "seed": 123456789,
                    "steps": 20,
                    "cfg": 7,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                }
            },
            "7": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["6", 0], "vae": ["2", 0]}
            },
            "8": {
                "class_type": "SaveImage",
                "inputs": {"images": ["7", 0]}
            }
        }
    
    logger.debug(f"Sending workflow to ComfyUI with checkpoint={checkpoint}, vae={vae}, flux={is_flux}")
    logger.debug(f"Workflow: {workflow}")
    
    response = await client.post(
        f"{base_url}/prompt",
        json=workflow
    )
    logger.debug(f"ComfyUI /prompt response status: {response.status_code}")
    
    # Log the full response for debugging
    try:
        data = response.json()
        logger.debug(f"ComfyUI /prompt response: {data}")
    except Exception:
        logger.debug(f"ComfyUI /prompt response (raw): {response.text[:500]}")
        data = {}
    
    response.raise_for_status()

    # /prompt returns a prompt_id, need to poll for result
    prompt_id = data.get("prompt_id")
    logger.info(f"ComfyUI prompt_id: {prompt_id}")
    if not prompt_id:
        return None

    # Poll for the result with a timeout
    for i in range(120):  # Try for up to 2 minutes
        await asyncio.sleep(1)
        try:
            history_resp = await client.get(
                f"{base_url}/history/{prompt_id}"
            )
            history_resp.raise_for_status()
            history_data = history_resp.json()
            if prompt_id in history_data:
                output = history_data[prompt_id].get("outputs", {})
                if output:
                    # Find the image in outputs
                    for node_id, node_output in output.items():
                        if "images" in node_output:
                            for img_data in node_output["images"]:
                                if img_data.get("type") == "output":
                                    import base64
                                    image_data = base64.b64decode(img_data["image"])
                                    return image_data
        except Exception as e:
            logger.debug(f"Poll attempt {i+1} failed: {e}")

    logger.warning(f"ComfyUI /prompt returned no image after polling")
    return None


@app.get("/img")
async def placeholder_image(request: Request):
    """
    Generate an image using ComfyUI or return an SVG placeholder.
    
    Query parameters:
    - t: The prompt/alt text for the image (URL-encoded)
    - w: Width in pixels (default: 512)
    - h: Height in pixels (default: 512)
    
    Uses ComfyUI for real image generation with a high timeout (5 minutes) to accommodate slow generation.
    Tries both the REST API endpoint (/sdapi/v1/txt2img) and the workflow endpoint (/prompt).
    Falls back to SVG placeholder if ComfyUI is unavailable.
    """
    cfg = get_config()
    
    t = request.query_params.get("t", "")
    w = request.query_params.get("w", "512")
    h = request.query_params.get("h", "512")
    
    # Set a high timeout for image generation (5 minutes)
    img_timeout = 300.0
    
    logger.info(f"Generating image with prompt: {t[:50]}... (w={w}, h={h})")
    
    try:
        async with httpx.AsyncClient(timeout=img_timeout) as client:
            # Try the REST API endpoint first (if extension is installed)
            try:
                response = await client.post(
                    f"{cfg.comfyui_url}/sdapi/v1/txt2img",
                    json={
                        "prompt": t,
                        "negative_prompt": "worst quality, low quality, blurry",
                        "width": int(w),
                        "height": int(h),
                        "steps": 20,
                        "cfg_scale": 7,
                        "sampler_name": "euler",
                        "scheduler": "normal",
                    }
                )
                response.raise_for_status()
                data = response.json()
                if data.get("images"):
                    import base64
                    image_data = base64.b64decode(data["images"][0])
                    return Response(content=image_data, media_type="image/png")
                logger.warning(f"ComfyUI REST API returned no image: {data}")
            except httpx.HTTPStatusError as e:
                if e.response.status_code != 405:
                    raise
                # 405 means REST API extension not installed, try /prompt endpoint
                logger.info("REST API not available, trying /prompt endpoint")
            
            # Query ComfyUI for available models and VAEs
            checkpoint, vae = None, None
            try:
                checkpoint, vae = await _get_comfyui_models(client, cfg.comfyui_url)
            except Exception as e:
                logger.warning(f"Could not query ComfyUI for models: {e}")
            
            # If we couldn't discover models, try common checkpoint/VAE combinations
            if checkpoint is None or vae is None:
                # List of common checkpoint and VAE combinations to try
                common_checkpoints = [
                    "v1-5-pruned-emaonly.safetensors",
                    "v1-5-pruned.safetensors",
                    "sd_v1-5.safetensors",
                    "sd-vae-ft-mse-840000.safetensors",
                    "model.safetensors",
                ]
                common_vaes = [
                    "vae-ft-mse-840000.safetensors",
                    "vae.safetensors",
                    "stabilityai/sd-vae-ft-mse",
                ]
                
                logger.info("Trying common checkpoint/VAE combinations")
                for cp in common_checkpoints:
                    for v in common_vaes:
                        logger.debug(f"Trying checkpoint={cp}, vae={v}")
                        image_data = await _execute_comfyui_workflow(
                            client, cfg.comfyui_url, t, int(w), int(h), cp, v
                        )
                        if image_data:
                            return Response(content=image_data, media_type="image/png")
            else:
                # Use the discovered models
                logger.info(f"Trying ComfyUI /prompt endpoint with discovered checkpoint={checkpoint}, vae={vae}")
                image_data = await _execute_comfyui_workflow(
                    client, cfg.comfyui_url, t, int(w), int(h), checkpoint, vae
                )
                if image_data:
                    return Response(content=image_data, media_type="image/png")
            
            logger.warning(f"ComfyUI /prompt returned no image with any checkpoint/VAE combination")
            
    except Exception as e:
        logger.error(f"Failed to generate image with ComfyUI: {e}")
    
    # Fallback: return SVG placeholder
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <rect width="{w}" height="{h}" fill="#cccccc"/>
  <text x="50%" y="50%" text-anchor="middle" dy=".3em" fill="#666" font-size="14">{t}</text>
</svg>'''
    return Response(content=svg, media_type="image/svg+xml")


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
    parser.add_argument(
        "--comfyui-url",
        type=str,
        help="ComfyUI server URL for image generation (overrides env/config)",
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
        comfyui_url=args.comfyui_url,
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
