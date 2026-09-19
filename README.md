# Sloppy

Sloppy is a small web server that fakes an entire search engine using a
large language model. Every page you visit is generated on the fly by an
OpenAI-compatible chat API and streamed back to your browser:

- `/` generates a Google-style home page (search box, a silly slogan, an
  inline-SVG logo).
- `/web?q=...` generates a page of fake search results for your query.
- `/web/{domain}/{title}?q=...` generates a fake page "on" the domain you
  clicked through to, with content themed around the query and title.
- `/img?t=...&w=...&h=...` generates a real image from a text prompt, so the
  generated pages can include pictures. Results are cached on disk.

It's a toy. Nothing it shows you is real, and the "search results" link to
more generated pages, not the actual web.

## Quick start (container, recommended)

`run-container.sh` builds and runs the server in [Podman](https://podman.io/)
with the GPU passed through. By default it expects a local
[Ollama](https://ollama.com/) server on port 11434 and uses the
`gemma4:latest` model.

```bash
# 1. Have Ollama running with a model pulled:
ollama serve &
ollama pull gemma4

# 2. Build and run Sloppy (defaults to http://localhost:8000):
./run-container.sh
```

Then open <http://localhost:8000/>.

The script passes the host's NVIDIA devices into the container so image
generation runs on the GPU. On a machine without an NVIDIA GPU, image
generation automatically falls back to CPU (slow, but it works); set
`IMG_BACKEND=hf` (see below) if you'd rather not generate locally at all.

### Common overrides

Pass extra `-e` flags after the port; the script forwards them to the
container. Environment variables also work.

```bash
# Use a different model and a longer timeout:
./run-container.sh 8000 -e MODEL_NAME=llama3.2 -e TIMEOUT=60

# Point at a different OpenAI-compatible endpoint:
OPENAI_BASE_URL=https://api.openai.com/v1 OPENAI_API_KEY=sk-... ./run-container.sh

# Run on a different port:
./run-container.sh 9000
```

You can use Docker instead of Podman by building and running the image
manually with the same environment variables and device flags.

## Quick start (without a container)

```bash
pip install -r requirements.txt
OPENAI_BASE_URL=http://localhost:11434/v1 MODEL_NAME=gemma4 uvicorn main:app
```

Note: the `local` image backend pulls PyTorch with CUDA support via the
Dockerfile; running outside a container without a GPU will use CPU
inference for images (works, but slow).

## Configuration

All settings can be set via environment variables or (for the server
settings) CLI arguments to `python main.py`.

### LLM / server

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible `/v1` endpoint. Container default is Ollama: `http://localhost:11434/v1`. |
| `OPENAI_API_KEY` | _(empty)_ | Optional; local endpoints like Ollama usually don't need one. |
| `MODEL_NAME` | `gpt-4o-mini` (container: `gemma4:latest`) | Model id passed to the API. |
| `HOST` | `0.0.0.0` | Bind address. |
| `PORT` | `8000` | Listen port. |
| `STREAM` | `false` (container: `true`) | Stream generated pages to the browser as the LLM produces them. |
| `TIMEOUT` | `30.0` (container: `120`) | Seconds to wait for the LLM per request. |
| `DEBUG` | _(off)_ | Set to `true` for verbose logging. |

### Image generation (`/img`)

| Variable | Default | Notes |
|---|---|---|
| `IMG_BACKEND` | `local` | `local` = run SDXL-Turbo via [diffusers](https://github.com/huggingface/diffusers) in-process; `hf` = use the Hugging Face Inference API. |
| `IMG_STEPS` | `4` | Inference steps. Lower is faster. |
| `IMG_HF_TOKEN` | _(empty)_ | Required only for `IMG_BACKEND=hf`. |
| `IMG_HF_MODEL` | `stabilityai/sdxl-turbo` | Model id, used by the `hf` backend. |
| `IMG_CACHE_DIR` | `~/.cache/imggen` (container: `/tmp/imggen-cache`) | Where generated images are cached. Identical requests return the cached image. |

The `local` backend uses CUDA if an NVIDIA GPU is available and otherwise
falls back to CPU automatically.

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Generated search home page. |
| `GET` | `/web?q={query}` | Generated search results for `{query}`. |
| `GET` | `/web/{domain}/{title}?q={query}` | A generated page "on" `{domain}` titled `{title}`, themed around `{query}`. |
| `GET` | `/img?t={text}&w={width}&h={height}` | A generated image for the prompt `{text}` at `{width}`x`{height}`. Returns PNG (`local`/`hf`). |
| `GET` | `/health` | Health check; returns `{"status":"ok"}`. |

## Project structure

```
Sloppy/
├── main.py            # FastAPI app and route handlers
├── client.py          # OpenAI-compatible API client (streaming + non-streaming)
├── config.py          # Configuration (pydantic-settings): env vars + CLI args
├── imggen.py          # /img generation: local diffusers or Hugging Face backend
├── requirements.txt   # Python dependencies
├── Dockerfile         # Multi-stage container image (CUDA torch + app)
├── run-container.sh   # Build and run with Podman, GPU passthrough, sane defaults
└── README.md
```

## License

MIT
