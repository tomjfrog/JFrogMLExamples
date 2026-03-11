"""
FastAPI app that serves a Hugging Face model baked into the Docker image.
Uses FLAN-T5 small (text-to-text): prompt in, generated text out.
Model is loaded from /app/model at startup (no download at runtime).
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


# Model path baked into the image at build time
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "100"))
model = None
tokenizer = None
device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup from the baked-in path."""
    global model, tokenizer, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
    yield
    model = None
    tokenizer = None


app = FastAPI(
    title="Hugging Face FLAN-T5 API",
    description="Text-to-text generation (summarization, Q&A) using FLAN-T5 small baked into the Docker image",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    generated_text: str


@app.get("/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded")
    inputs = tokenizer(
        req.text[:1024],
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return PredictResponse(generated_text=generated.strip())


@app.get("/")
def root():
    return {
        "message": "Hugging Face FLAN-T5 small API (model baked in image)",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict — send {\"text\": \"your prompt\"}",
    }
