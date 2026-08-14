# Sloppy

A minimal web server that uses an OpenAI-compatible API to generate and serve a complete search page.

## Quick Start

### Using Python directly

```bash
# Install dependencies
pip install -r requirements.txt

# Run with environment variables
OPENAI_API_KEY=your_key OPENAI_BASE_URL=https://api.openai.com/v1 uvicorn main:app --reload

# Or with CLI arguments
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Using Docker

```bash
# Build and run
docker build -t sloppy .
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key -e OPENAI_BASE_URL=https://api.openai.com/v1 sloppy
```

## Configuration

The server can be configured via:

1. **Environment variables** (highest priority):
   - `OPENAI_API_KEY` - Your API key
   - `OPENAI_BASE_URL` - Base URL for OpenAI-compatible API (default: `https://api.openai.com/v1`)
   - `MODEL_NAME` - Model to use (default: `gpt-4o-mini`)
   - `PORT` - Server port (default: `8000`)
   - `HOST` - Server host (default: `0.0.0.0`)

2. **Config file** (`config.yaml`):
   ```yaml
   openai:
     api_key: your_key
     base_url: https://api.openai.com/v1
   model_name: gpt-4o-mini
   server:
     host: 0.0.0.0
     port: 8000
   ```

3. **CLI arguments** (lowest priority):
   ```bash
   python main.py --api-key your_key --base-url https://api.openai.com/v1 --model gpt-4o-mini
   ```

## Usage

1. Start the server with your configuration
2. Visit `http://localhost:8000/` in your browser
3. The server will prompt the AI to generate a complete search page HTML
4. The generated page will be served to you

## API Endpoints

- `GET /` - Generate and serve a search page
- `GET /web?q={query}` - Generate a search results page for the given query
- `GET /web/{domain}/{title}?q={query}` - Generate an individual result page where `{domain}` is a domain name (e.g., `en.wikipedia.org`), `{title}` is the page title (URL-encoded, may contain slashes), and `{query}` is the search query
- `GET /health` - Health check endpoint

## Project Structure

```
Sloppy/
├── main.py           # FastAPI application
├── config.py         # Configuration management
├── client.py         # OpenAI API client wrapper
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container configuration
└── README.md         # This file
```

## License

MIT
