"""FastAPI service for merchant descriptor normalization.

Loads the base model plus the LoRA adapter once at startup. Configuration via
MERCHANT_-prefixed environment variables (see merchant_llm.config.Settings).

Run:
    python -m uvicorn merchant_llm.serve.app:app --port 8000
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from merchant_llm.config import Settings
from merchant_llm.parsing import parse_prediction

settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Imported here so the module stays importable without torch installed.
    from merchant_llm.inference import load_model

    model, tokenizer, device = load_model(
        settings.base_model, settings.adapter_or_none, settings.device
    )
    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.device = device
    yield


app = FastAPI(title="merchant-normalizer", version="0.1.0", lifespan=lifespan)


class NormalizeRequest(BaseModel):
    raw: str = Field(min_length=1, max_length=200, description="raw statement descriptor")


class NormalizeResponse(BaseModel):
    merchant_name: str | None
    category: str | None
    parsed: bool
    latency_ms: float


@app.post("/normalize", response_model=NormalizeResponse)
def normalize(request: NormalizeRequest) -> NormalizeResponse:
    from merchant_llm.inference import generate_batch

    start = time.perf_counter()
    output = generate_batch(
        app.state.model,
        app.state.tokenizer,
        [request.raw],
        app.state.device,
        max_new_tokens=settings.max_new_tokens,
    )[0]
    parsed = parse_prediction(output)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)
    return NormalizeResponse(
        merchant_name=parsed["merchant_name"] if parsed else None,
        category=parsed["category"] if parsed else None,
        parsed=parsed is not None,
        latency_ms=latency_ms,
    )


@app.get("/healthz")
def healthz() -> dict:
    return {
        "status": "ok",
        "base_model": settings.base_model,
        "adapter": settings.adapter_or_none,
        "device": getattr(app.state, "device", None),
        "max_new_tokens": settings.max_new_tokens,
    }
