# JFrog Artifactory + Hugging Face Docker Demo

Docker image with a **Hugging Face model baked in** (no download at runtime), suitable for pushing to a JFrog Artifactory Docker repository.

## What’s included

- **FastAPI app** that loads the model from `/app/model` at startup and exposes:
  - `GET /health` – health check
  - `POST /predict` – text-to-text generation (body: `{"text": "Your prompt or question"}`)
  - `GET /docs` – Swagger UI
- **Dockerfile** that downloads the Hugging Face model at **build time** and bakes it into the image (default: **google/flan-t5-small** — small, popular FLAN-T5 for summarization, Q&A, etc.).
- **Script** to build and push the image to Artifactory.

## Install Docker (macOS)

If Docker is not installed, run in **Terminal** (you’ll be prompted for your password):

```bash
cd jfrog-hf-docker-demo
./install-docker-mac.sh
```

This installs Homebrew (if needed) and Docker Desktop. Then open **Docker Desktop** from Applications, accept the terms, and wait until it’s running before using `docker` commands.

## Build the image locally

```bash
cd jfrog-hf-docker-demo
docker build -t jfrog-hf-demo:latest .
```

Optional: use a different Hugging Face model (e.g. sentiment):

```bash
docker build --build-arg HF_MODEL_ID=distilbert-base-uncased-finetuned-sst-2-english -t jfrog-hf-demo:latest .
```
Note: switching the model also requires matching app code (e.g. sentiment vs text2text).

## Run locally

```bash
docker run -p 8000:8000 jfrog-hf-demo:latest
```

Then:

- Open http://localhost:8000/docs
- Try `POST /predict` with body:
  - `{"text": "Summarize: Machine learning is a subset of artificial intelligence."}`
  - `{"text": "Question: What is the capital of France? Answer:"}`

Example:

```bash
curl -s http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Summarize: Machine learning is a subset of artificial intelligence."}'
```
  
## Push to JFrog Artifactory

1. Log in to Artifactory:

   ```bash
   docker login solenglatest.jfrog.io
   ```

2. Tag and push the image:

   ```bash
   docker tag jfrog-hf-demo:latest solenglatest.jfrog.io/shadow-guyes/jfrog-hf-demo:latest
   docker push solenglatest.jfrog.io/shadow-guyes/jfrog-hf-demo:latest
   ```

## Pull and run from Artifactory

After pushing:

```bash
docker login your-instance.jfrog.io -u YOUR_USER -p YOUR_PASSWORD
docker pull your-instance.jfrog.io/docker-local/jfrog-hf-demo:latest
docker run -p 8000:8000 your-instance.jfrog.io/docker-local/jfrog-hf-demo:latest
```
