
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.schemas import (
    AnalyzeRequest,
    BatchRequest,
    BatchResponse,
    HealthResponse,
    SentimentResult,
)
from app.sentiment import MODEL_NAME, analyze_text, is_model_loaded


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentiment Analysis API",
    description=(
        "Analyze English text and return positive, negative, "
        "or neutral sentiment with a confidence score."
    ),
    version="1.0.0",
)


@app.get("/api/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_loaded": is_model_loaded(),
    }


@app.post("/api/analyze", response_model=SentimentResult)
def analyze(request: AnalyzeRequest):
    try:
        return analyze_text(request.text)
    except Exception as exc:
        logger.exception("Sentiment inference failed")
        raise HTTPException(
            status_code=503,
            detail="Sentiment model is temporarily unavailable.",
        ) from exc


@app.post("/api/analyze/batch", response_model=BatchResponse)
def analyze_batch(request: BatchRequest):
    results = []

    try:
        for text in request.texts:
            results.append(analyze_text(text))

        return {"results": results}

    except Exception as exc:
        logger.exception("Batch sentiment inference failed")
        raise HTTPException(
            status_code=503,
            detail="Sentiment model is temporarily unavailable.",
        ) from exc