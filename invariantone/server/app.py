"""
FastAPI HTTP Inference Server for InvariantOne.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException

from invariantone.model import InvariantOne
from invariantone.server.schemas import DecideRequest, DecideResponse, HealthResponse


def create_app(model: InvariantOne | None = None) -> FastAPI:
    """
    Creates and configures the FastAPI application instance.

    Args:
        model: Optional pre-loaded InvariantOne instance. If None, loaded lazily.
    """
    app = FastAPI(
        title="InvariantOne Decision Service",
        description="High-assurance permutation-equivariant natural-language decision model API.",
        version="1.0.0",
    )

    state: dict[str, Any] = {"model": model}

    def get_model() -> InvariantOne:
        if state["model"] is None:
            checkpoint_dir = os.environ.get("INVARIANTONE_CHECKPOINT_DIR", "checkpoints/invariantone-v1")
            device = os.environ.get("INVARIANTONE_DEVICE", "auto")
            state["model"] = InvariantOne.from_pretrained(
                pretrained_model_name_or_path=checkpoint_dir,
                device=device,
                strict=True,
            )
        return state["model"]

    @app.get("/v1/health", response_model=HealthResponse)
    def health_check() -> HealthResponse:
        current_model = state["model"]
        device_name = str(current_model.device) if current_model else "uninitialized"
        return HealthResponse(
            status="ok",
            model="InvariantOne",
            version="1.0.0",
            device=device_name,
            integrity_verified=True,
        )

    @app.get("/v1/info")
    def model_info() -> dict[str, Any]:
        m = get_model()
        return m.info()

    @app.post("/v1/decide", response_model=DecideResponse)
    def decide_endpoint(request: DecideRequest) -> DecideResponse:
        try:
            m = get_model()
            result = m.decide(
                state=request.state,
                question=request.question,
                options=request.options,
            )
            return DecideResponse(
                choice_index=result.choice_index,
                choice=result.choice,
                probabilities=result.probabilities,
                scores=result.scores,
                model="InvariantOne-v1",
                latency_ms=result.latency_ms,
            )
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return app


app = create_app()
