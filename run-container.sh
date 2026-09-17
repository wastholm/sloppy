#!/bin/bash
# Build and run Sloppy container with common options

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
# Port can be specified as:
# 1. First positional argument: ./run-container.sh 9000
# 2. PORT environment variable: PORT=9000 ./run-container.sh
# 3. Default: 8000
PORT="${1:-${PORT:-8000}}"

# Build the container
echo "Building sloppy container..."
podman build -t sloppy .

# Run with common defaults
# To override, pass additional -e flags before the image name
# Example: ./run-container.sh 9000 -e MODEL_NAME=llama3.2 -e TIMEOUT=60

exec podman run --userns keep-id --network=host \
  --device /dev/nvidia0 \
  --device /dev/nvidiactl \
  --device /dev/nvidia-modeset \
  --device /dev/nvidia-uvm \
  --device /dev/nvidia-uvm-tools \
  --device /dev/nvidia-caps \
  -p "$PORT":"$PORT" \
  -e DEBUG="${DEBUG:-true}" \
  -e MODEL_NAME="${MODEL_NAME:-gemma4:latest}" \
  -e OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:11434/v1}" \
  -e STREAM=true \
  -e TIMEOUT="${TIMEOUT:-120}" \
  -e PORT="$PORT" \
  -e IMG_BACKEND="${IMG_BACKEND:-local}" \
  -e IMG_STEPS="${IMG_STEPS:-4}" \
  -e IMG_HF_TOKEN="${IMG_HF_TOKEN:-}" \
  -e IMG_HF_MODEL="${IMG_HF_MODEL:-stabilityai/sdxl-turbo}" \
  -e IMG_CACHE_DIR="${IMG_CACHE_DIR:-/tmp/imggen-cache}" \
  "${@:2}" \
  sloppy
