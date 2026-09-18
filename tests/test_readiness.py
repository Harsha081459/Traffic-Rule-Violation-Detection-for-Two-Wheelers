import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as api

pytestmark = pytest.mark.unit


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "_limiter", api._RateLimiter(20))
    monkeypatch.setattr(api, "_detector", None)
    return TestClient(api.app)


def test_liveness_does_not_masquerade_as_readiness(client):
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/ready").status_code == 503
    assert client.get("/api/info").json()["ocr_available"] is False


def test_spoofed_forwarded_header_does_not_bypass_limiter(client, monkeypatch):
    monkeypatch.setattr(api, "_limiter", api._RateLimiter(1))
    for expected, ip in [(415, "1.1.1.1"), (429, "2.2.2.2")]:
        response = client.post("/predict", files={"file": ("x.jpg", b"invalid")}, headers={"X-Forwarded-For": ip})
        assert response.status_code == expected


def test_new_rate_limit_bucket_uses_a_single_clock_snapshot(monkeypatch):
    clock = iter([100.0, 100.1])
    monkeypatch.setattr(api, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    limiter = api._RateLimiter(1)
    assert limiter.is_allowed("new-client")
    assert not limiter.is_allowed("new-client")


def test_header_only_png_is_not_a_valid_image(client):
    response = client.post("/predict", files={"file": ("x.png", b"\x89PNG\r\n\x1a\n" + bytes(30))})
    assert response.status_code == 400


def test_gif_cannot_be_disguised_as_jpeg(client):
    assert client.post("/predict", files={"file": ("x.jpg", b"GIF89a" + bytes(30))}).status_code == 415


def test_prediction_cleans_temp_file_and_reports_ocr(client, monkeypatch):
    seen = []

    def predict(path):
        assert os.path.isfile(path)
        seen.append(path)
        return {"violations": [], "debug": [], "inference_time_sec": 0.01}

    monkeypatch.setattr(api, "_detector", SimpleNamespace(predict_debug=predict, _ocr=SimpleNamespace(available=False)))
    image = cv2.imencode(".png", np.zeros((32, 32, 3), dtype=np.uint8))[1].tobytes()
    response = client.post("/predict", files={"file": ("x.png", image)})
    assert response.status_code == 200
    assert response.json()["ocr_available"] is False
    assert len(seen) == 1 and not Path(seen[0]).exists()
