#!/usr/bin/env bash
# Download the Hugging Face model on your machine, then build the image (no HF access inside Docker).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_DIR="${SCRIPT_DIR}/model"
HF_MODEL_ID="${HF_MODEL_ID:-google/flan-t5-small}"

if [ ! -d "${MODEL_DIR}" ] || [ -z "$(ls -A "${MODEL_DIR}" 2>/dev/null)" ]; then
  echo "Downloading ${HF_MODEL_ID} to ${MODEL_DIR}..."
  pip install -q transformers torch sentencepiece
  python scripts/download_model.py --model-id "${HF_MODEL_ID}" --output-dir "${MODEL_DIR}"
  echo "Download done."
else
  echo "Using existing model in ${MODEL_DIR}"
fi

echo "Building Docker image..."
docker build -f Dockerfile.preloaded -t jfrog-hf-demo:latest .
echo "Done. Run: docker run -p 8000:8000 jfrog-hf-demo:latest"
