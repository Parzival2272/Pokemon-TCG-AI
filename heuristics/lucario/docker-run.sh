#!/usr/bin/env bash
# Build and run the Lucario cabt sim in Linux Docker (macOS/Windows host).
set -euo pipefail

cd "$(dirname "$0")"
IMAGE="${IMAGE:-lucario-cabt}"

docker build --platform linux/amd64 -t "$IMAGE" .

# Mount source so agent edits on the host are picked up without rebuilding.
docker run --rm --platform linux/amd64 \
  -v "$PWD:/app" \
  -w /app \
  "$IMAGE" \
  python run_local.py "$@"
