"""Drop-in image generation for an existing Python webapp.

Endpoint shape:  GET /img?w=300&h=200&t=some+text+prompt  ->  PNG bytes

Design notes:
  - Same request = same image: the seed is derived from the params, and
    results are cached on disk. A GET-style URL gets hit repeatedly
    (reloads, thumbnails), so caching matters more than anything else here.
  - Backend is chosen by env var, no code change to switch:
      IMG_BACKEND=local   diffusers + SDXL-Turbo on local GPU (default)
      IMG_BACKEND=hf      Hugging Face Inference API (same open models)
  - Generation runs at ~512px on the long side, then the PNG is resized
    to the requested w/h. Keeps VRAM bounded no matter what clients ask.

Env vars:
  IMG_BACKEND   local | hf           (default local)
  IMG_STEPS     inference steps      (default 4)
  IMG_HF_TOKEN  HF access token     (hf backend only)
  IMG_HF_MODEL  HF model id         (default stabilityai/sdxl-turbo)
  IMG_CACHE_DIR cache location      (default ~/.cache/imggen)
"""
from __future__ import annotations

import hashlib
import io
import os
import sys
import threading
from functools import lru_cache

BACKEND = os.environ.get("IMG_BACKEND", "local")
STEPS = int(os.environ.get("IMG_STEPS", "4"))
HF_MODEL = os.environ.get("IMG_HF_MODEL", "stabilityai/sdxl-turbo")
CACHE_DIR = os.path.expanduser(os.environ.get("IMG_CACHE_DIR", "~/.cache/imggen"))

_lock = threading.Lock()  # torch pipelines are not thread-safe


# ---------------------------------------------------------------- helpers

def _clamp(v: int, lo: int = 64, hi: int = 2048) -> int:
    return min(max(int(v), lo), hi)


def _gen_size(w: int, h: int, base: int = 512) -> tuple[int, int]:
    """Long side ~512, multiples of 8, floors at 256. Diffusion-friendly."""
    scale = base / max(w, h)
    return max(256, round(w * scale / 8) * 8), max(256, round(h * scale / 8) * 8)


def _seed(prompt: str, w: int, h: int) -> int:
    raw = f"{BACKEND}|{HF_MODEL}|{STEPS}|{prompt}|{w}|{h}"
    return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "big")


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, key + ".png")


def _cache_get(key: str) -> bytes | None:
    try:
        with open(_cache_path(key), "rb") as f:
            return f.read()
    except FileNotFoundError:
        return None


def _cache_put(key: str, data: bytes) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = _cache_path(key) + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, _cache_path(key))  # atomic on POSIX


# ---------------------------------------------------------------- backends

@lru_cache(maxsize=1)
def _device() -> str:
    """Use CUDA if an NVIDIA driver/GPU is present, otherwise CPU."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dep of this backend
        raise
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _pipeline():
    import torch
    from diffusers import AutoPipelineForText2Image

    device = _device()
    # fp16 is CUDA-only; on CPU fall back to fp32 so the model actually runs.
    dtype = torch.float16 if device == "cuda" else torch.float32
    kwargs = {"torch_dtype": dtype}
    if device == "cuda":
        # The fp16 variant wheel is only published for CUDA.
        kwargs["variant"] = "fp16"
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", **kwargs
    ).to(device)
    # CPU inference is slow; use all cores to take the edge off.
    if device == "cpu":
        try:
            import os as _os
            torch.set_num_threads(max(1, _os.cpu_count() or 1))
        except Exception:
            pass
    return pipe


def _generate_local(prompt: str, w: int, h: int):
    import torch

    device = _device()
    gw, gh = _gen_size(w, h)
    with _lock:
        return _pipeline()(
            prompt=prompt,
            width=gw,
            height=gh,
            num_inference_steps=STEPS,
            guidance_scale=0.0,
            generator=torch.Generator(device).manual_seed(_seed(prompt, w, h)),
        ).images[0]


def _generate_hf(prompt: str, w: int, h: int):
    from huggingface_hub import InferenceClient

    client = InferenceClient(token=os.environ.get("IMG_HF_TOKEN"), model=HF_MODEL)
    return client.text_to_image(prompt, width=w, height=h)


# ---------------------------------------------------------------- public API

def generate(prompt: str, w: int = 512, h: int = 512) -> bytes:
    """Return PNG bytes for the prompt at w x h. Cached and deterministic."""
    from PIL import Image

    w, h = _clamp(w), _clamp(h)
    key = _seed(prompt, w, h).to_bytes(8, "big").hex()

    cached = _cache_get(key)
    if cached is not None:
        return cached

    if BACKEND == "hf":
        img = _generate_hf(prompt, w, h)
    else:
        img = _generate_local(prompt, w, h)

    if img.size != (w, h):
        img = img.resize((w, h), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, "PNG")
    data = buf.getvalue()
    _cache_put(key, data)
    return data


# ---------------------------------------------------------------- route adapters

def register_flask(app):
    """Flask: call register_flask(app) once at startup."""
    from flask import request, send_file

    @app.route("/img")
    def img():
        return send_file(
            io.BytesIO(generate(request.args.get("t", ""), request.args.get("w", 512), request.args.get("h", 512))),
            mimetype="image/png",
            max_age=86400,
        )


def register_fastapi(app):
    """FastAPI: call register_fastapi(app) once at startup."""
    from fastapi import Response

    @app.get("/img")
    def img(t: str = "", w: int = 512, h: int = 512):
        return Response(
            generate(t, w, h),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )


# ---------------------------------------------------------------- CLI test

if __name__ == "__main__":
    # python imggen.py "a cat in a hat" 300 200 > out.png
    prompt = " ".join(sys.argv[1:]) or "a cat in a hat"
    sys.stdout.buffer.write(generate(prompt, 300, 200))
