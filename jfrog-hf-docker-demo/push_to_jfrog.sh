#!/usr/bin/env bash
#
# Build the Docker image and push it to JFrog Artifactory.
# Requires: DOCKER_IMAGE_NAME, ARTIFACTORY_URL, and Artifactory credentials.
#
# Usage:
#   export ARTIFACTORY_URL="your-instance.jfrog.io"
#   export ARTIFACTORY_REPO="docker-local"   # or your Docker repo key
#   export ARTIFACTORY_USER="your-username"
#   export ARTIFACTORY_PASSWORD="your-password-or-api-key"
#   export DOCKER_IMAGE_NAME="jfrog-hf-demo"
#   ./push_to_jfrog.sh
#
# Push existing image only (no rebuild):
#   PUSH_ONLY=1 ./push_to_jfrog.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

: "${DOCKER_IMAGE_NAME:=jfrog-hf-demo}"
: "${HF_MODEL_ID:=google/flan-t5-small}"
: "${IMAGE_TAG:=latest}"

if [ -z "${ARTIFACTORY_URL}" ] || [ -z "${ARTIFACTORY_REPO}" ]; then
  echo "Error: Set ARTIFACTORY_URL and ARTIFACTORY_REPO (and optionally ARTIFACTORY_USER / ARTIFACTORY_PASSWORD)."
  echo "Example:"
  echo "  export ARTIFACTORY_URL=mycompany.jfrog.io"
  echo "  export ARTIFACTORY_REPO=docker-local"
  exit 1
fi

REMOTE_IMAGE="${ARTIFACTORY_URL}/${ARTIFACTORY_REPO}/${DOCKER_IMAGE_NAME}:${IMAGE_TAG}"

if [ "${PUSH_ONLY}" != "1" ]; then
  if [ -d "${SCRIPT_DIR}/model" ] && [ -n "$(ls -A "${SCRIPT_DIR}/model" 2>/dev/null)" ]; then
    echo "Building image from local model (Dockerfile.preloaded)..."
    docker build -f Dockerfile.preloaded -t "${DOCKER_IMAGE_NAME}:${IMAGE_TAG}" .
  else
    echo "Building image ${DOCKER_IMAGE_NAME}:${IMAGE_TAG} (HF model: ${HF_MODEL_ID})..."
    docker build \
      --build-arg HF_MODEL_ID="${HF_MODEL_ID}" \
      -t "${DOCKER_IMAGE_NAME}:${IMAGE_TAG}" \
      -f Dockerfile \
      .
  fi
else
  echo "PUSH_ONLY=1: using existing image ${DOCKER_IMAGE_NAME}:${IMAGE_TAG}"
fi

echo "Tagging for Artifactory: ${REMOTE_IMAGE}"
docker tag "${DOCKER_IMAGE_NAME}:${IMAGE_TAG}" "${REMOTE_IMAGE}"

if [ -n "${ARTIFACTORY_USER}" ] && [ -n "${ARTIFACTORY_PASSWORD}" ]; then
  echo "Logging in to Artifactory..."
  echo "${ARTIFACTORY_PASSWORD}" | docker login "${ARTIFACTORY_URL}" -u "${ARTIFACTORY_USER}" --password-stdin
fi

echo "Pushing to ${REMOTE_IMAGE}..."
docker push "${REMOTE_IMAGE}"

echo "Done. Image pushed to ${REMOTE_IMAGE}"
