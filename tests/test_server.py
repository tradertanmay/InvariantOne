"""
Tests for InvariantOne FastAPI HTTP Inference Server.
"""

from __future__ import annotations

import pytest
import torch
from starlette.testclient import TestClient

from invariantone.architectures.direct_comparative_head import DirectComparativeHead
from invariantone.config import InvariantOneConfig
from invariantone.model import InvariantOne
from invariantone.result import DecisionResult
from invariantone.server.app import create_app


class MockInvariantOneModel:
    """Mock model to test server HTTP endpoints without reloading 4B parameter weights."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.config = InvariantOneConfig()

    def decide(self, state: str, question: str, options: list[str]) -> DecisionResult:
        if len(options) < 2:
            raise ValueError("At least 2 options required")
        probs = [1.0 / len(options)] * len(options)
        scores = [0.0] * len(options)
        return DecisionResult(
            choice=options[0],
            choice_index=0,
            probabilities=probs,
            scores=scores,
            options=options,
            state=state,
            question=question,
            temperature=1.0091,
            latency_ms=12.5,
            device="cpu",
            model_version="1.0.0",
        )

    def info(self) -> dict:
        return {
            "model_name": "InvariantOne",
            "model_version": "1.0.0",
            "architecture": "D5-L4",
        }


@pytest.fixture
def client():
    mock_model = MockInvariantOneModel()
    app = create_app(model=mock_model)
    return TestClient(app)


def test_server_health_endpoint(client):
    """Verifies GET /v1/health endpoint."""
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model"] == "InvariantOne"
    assert data["version"] == "1.0.0"


def test_server_info_endpoint(client):
    """Verifies GET /v1/info endpoint."""
    response = client.get("/v1/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "InvariantOne"
    assert data["architecture"] == "D5-L4"


def test_server_decide_endpoint(client):
    """Verifies POST /v1/decide with valid payload."""
    payload = {
        "state": "Pressure is rising above safe limits.",
        "question": "What action to take?",
        "options": [
            "Close inlet valve",
            "Increase feed pressure",
            "Maintain current state",
        ],
    }
    response = client.post("/v1/decide", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["choice_index"] == 0
    assert data["choice"] == "Close inlet valve"
    assert len(data["probabilities"]) == 3
    assert data["model"] == "InvariantOne-v1"


def test_server_decide_insufficient_options(client):
    """Verifies POST /v1/decide rejects payloads with fewer than 2 options."""
    payload = {
        "state": "Pressure is rising.",
        "question": "Action?",
        "options": ["Single option"],
    }
    response = client.post("/v1/decide", json=payload)
    assert response.status_code == 422  # Pydantic validation error
