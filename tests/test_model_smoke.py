import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import app

pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    os.getenv("RUN_MODEL_SMOKE") != "1", reason="requires downloaded ONNX weights"
)]


def test_published_models_serve_a_committed_sample():
    with TestClient(app) as client:
        assert client.get("/api/ready").status_code == 200
        image = Path("figures/helmet_val_pred.jpg").read_bytes()
        response = client.post("/predict", files={"file": ("sample.jpg", image, "image/jpeg")})
        assert response.status_code == 200, response.text
        body = response.json()
        assert isinstance(body["violations"], list)
        assert isinstance(body["debug"], list)
        assert body["inference_time_sec"] >= 0
        assert isinstance(body["ocr_available"], bool)
        assert body["debug"], "sample should exercise detection, not only an empty response"
        for record in body["violations"]:
            assert record["num_riders"] >= 0
            assert 0 <= record["helmet_violations"] <= record["num_riders"]
