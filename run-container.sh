#!/bin/bash
# Build and run Sloppy container with common options

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Build the container
echo "Building sloppy container..."
podman build -t sloppy .

# Run with common defaults
# To override, pass additional -e flags before the image name
# Example: ./run-container.sh -e MODEL_NAME=llama3.2 -e TIMEOUT=60

exec podman run --userns keep-id --network=host -p 8000:8000 \
  -e DEBUG="${DEBUG:-true}" \
  -e MODEL_NAME="${MODEL_NAME:-gemma4:latest}" \
  -e OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:11434/v1}" \
  -e STREAM=true \
  -e TIMEOUT="${TIMEOUT:-120}" \
  "$@" \
  sloppy
