"""
Pydantic Schemas for InvariantOne HTTP Inference Service.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class DecideRequest(BaseModel):
    """Request payload for /v1/decide endpoint."""

    state: str = Field(..., description="Observed state description", min_length=1)
    question: str = Field(..., description="Decision question", min_length=1)
    options: list[str] = Field(..., description="Candidate options (minimum 2)", min_length=2)
    temperature: float | None = Field(
        default=None,
        description="Optional calibration temperature override (default: frozen 1.0091)",
        gt=0.0,
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "state": "Pressure is rising above the safe operating range.",
                "question": "What action should the controller take?",
                "options": [
                    "Close the inlet valve",
                    "Increase feed pressure",
                    "Maintain the current state",
                    "Disable monitoring",
                ],
            }
        }
    }


class DecideResponse(BaseModel):
    """Response payload for /v1/decide endpoint."""

    choice_index: int = Field(..., description="Index of selected option (0-indexed)")
    choice: str = Field(..., description="Selected option text")
    probabilities: list[float] = Field(..., description="Calibrated probabilities for each candidate option")
    scores: list[float] | None = Field(default=None, description="Raw uncalibrated comparative logits")
    model: str = Field(default="InvariantOne-v1", description="Model release identifier")
    latency_ms: float | None = Field(default=None, description="Inference latency in milliseconds")


class HealthResponse(BaseModel):
    """Response payload for /v1/health endpoint."""

    status: str = Field(default="ok")
    model: str = Field(default="InvariantOne")
    version: str = Field(default="1.0.0")
    device: str = Field(default="cpu")
    integrity_verified: bool = Field(default=True)
