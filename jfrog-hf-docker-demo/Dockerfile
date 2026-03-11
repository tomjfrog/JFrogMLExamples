# Build stage: download Hugging Face model and bake it into the image
FROM python:3.11-slim AS builder

WORKDIR /build

# Install deps needed only for downloading the model
RUN pip install --no-cache-dir transformers torch sentencepiece

# Download model at build time (baked into image) — FLAN-T5 small, popular text2text
# Disable Xet since it can fail in some network environments.
ARG HF_MODEL_ID=google/flan-t5-small
ENV HF_HUB_DISABLE_XET=1
RUN mkdir -p /app/model && python -c "from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; path='/app/model'; t=AutoTokenizer.from_pretrained('${HF_MODEL_ID}'); m=AutoModelForSeq2SeqLM.from_pretrained('${HF_MODEL_ID}'); t.save_pretrained(path); m.save_pretrained(path); print('Model saved to', path)"

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy model from builder (baked in)
COPY --from=builder /app/model /app/model

# Install runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/

# No network needed at runtime; model is in the image
ENV MODEL_PATH=/app/model
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
