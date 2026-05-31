"""
Tests for Traffic Sentinel AI FastAPI backend.

Run with: pytest tests/test_api.py -v
"""

from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Import the app after setting the ONNX env var
import os
os.environ["TV_USE_ONNX"] = "1"

from app import app, _limiter


@pytest.fixture
def client():
    """Return a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_image_jpeg():
    """Return a minimal valid JPEG image as bytes."""
    # Minimal 1x1 JPEG
    jpeg_bytes = (
        b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00'
        b'\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t'
        b'\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f'
        b'\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342'
        b'\xff\xc0\x00\x0b\x01\x01\x01\x01\x01\x01\x11\x00\xff\xc4\x00\x1f'
        b'\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00'
        b'\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5'
        b'\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01'
        b'}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"q\x142\x81'
        b'\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19'
        b'\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86'
        b'\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4'
        b'\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2'
        b'\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9'
        b'\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5'
        b'\xf6\xf7\xf8\xf9\xfa\xff\xda\x08\x01\x01\x00\x00?\x00\xfb\xd2\x8a'
        b'\x28\xa0\x0f\xff\xd9'
    )
    return jpeg_bytes


class TestHealth:
    """Health check endpoint tests."""

    @pytest.mark.unit
    def test_health_endpoint_returns_200(self, client):
        """GET /api/health should return 200 with status ok."""
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "onnx" in data
        assert data["onnx"] is True  # should be True since we set TV_USE_ONNX=1

    @pytest.mark.unit
    def test_info_endpoint(self, client):
        """GET /api/info should return model config."""
        resp = client.get("/api/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_dir" in data
        assert "use_onnx" in data
        assert data["use_onnx"] is True


class TestInputValidation:
    """File upload input validation tests."""

    @pytest.mark.unit
    def test_predict_rejects_non_image_mime_type(self, client):
        """POST /predict with non-image MIME should reject with 415."""
        resp = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code == 415
        assert "not allowed" in resp.json()["detail"].lower()

    @pytest.mark.unit
    def test_predict_rejects_empty_file(self, client):
        """POST /predict with empty file should reject with 400."""
        resp = client.post(
            "/predict",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    @pytest.mark.unit
    def test_predict_rejects_wrong_extension(self, client):
        """POST /predict with .exe renamed to .jpg should reject after magic check."""
        # A simple non-image binary
        bad_binary = b"MZ\x90\x00"  # PE header
        resp = client.post(
            "/predict",
            files={"file": ("notimage.jpg", bad_binary, "image/jpeg")},
        )
        assert resp.status_code == 415
        assert "does not look like a supported image" in resp.json()["detail"]

    @pytest.mark.unit
    def test_predict_accepts_valid_jpeg(self, client, sample_image_jpeg):
        """POST /predict with valid JPEG should proceed to inference (not reject on validation)."""
        # This will fail at inference stage (detector not loaded in unit test),
        # but should pass file validation.
        resp = client.post(
            "/predict",
            files={"file": ("sample.jpg", sample_image_jpeg, "image/jpeg")},
        )
        # In a unit test, the detector is None, so we should get 503 (service unavailable)
        # not 415/400 (validation error)
        assert resp.status_code in (503, 500)  # detector load failure, not validation failure

    @pytest.mark.unit
    def test_predict_rejects_oversized_file(self, client):
        """POST /predict with file > 15 MB should reject with 413."""
        # Simulate a large file (don't actually create 15 MB in memory)
        large_data = b"X" * (16 * 1024 * 1024)  # 16 MB
        resp = client.post(
            "/predict",
            files={"file": ("huge.jpg", large_data, "image/jpeg")},
        )
        assert resp.status_code == 413
        assert "too large" in resp.json()["detail"].lower()


class TestRateLimiting:
    """Rate limiting tests."""

    @pytest.mark.unit
    def test_rate_limiter_allows_initial_request(self):
        """First request from IP should be allowed."""
        limiter = _limiter.__class__(rpm=2)  # 2 req/min
        assert limiter.is_allowed("192.168.1.1") is True

    @pytest.mark.unit
    def test_rate_limiter_allows_under_limit(self):
        """Requests under the RPM limit should be allowed."""
        limiter = _limiter.__class__(rpm=3)  # 3 req/min
        ip = "10.0.0.1"
        # First 3 should be allowed
        assert limiter.is_allowed(ip) is True
        assert limiter.is_allowed(ip) is True
        assert limiter.is_allowed(ip) is True

    @pytest.mark.unit
    def test_rate_limiter_rejects_over_limit(self):
        """4th request from same IP in same period should be rejected."""
        limiter = _limiter.__class__(rpm=3)
        ip = "10.0.0.2"
        # Use up the bucket
        assert limiter.is_allowed(ip) is True
        assert limiter.is_allowed(ip) is True
        assert limiter.is_allowed(ip) is True
        # Next request should be rejected
        assert limiter.is_allowed(ip) is False

    @pytest.mark.unit
    def test_rate_limiter_different_ips_independent(self):
        """Different IPs should have independent rate limit buckets."""
        limiter = _limiter.__class__(rpm=2)
        assert limiter.is_allowed("1.1.1.1") is True
        assert limiter.is_allowed("1.1.1.1") is True
        assert limiter.is_allowed("1.1.1.1") is False  # IP 1 over limit

        # Different IP should still have capacity
        assert limiter.is_allowed("2.2.2.2") is True
        assert limiter.is_allowed("2.2.2.2") is True
        assert limiter.is_allowed("2.2.2.2") is False


class TestErrorHandling:
    """Error response format tests."""

    @pytest.mark.unit
    def test_validation_error_returns_json_detail(self, client):
        """All 4xx errors should return JSON with 'detail' field."""
        resp = client.post(
            "/predict",
            files={"file": ("bad.txt", b"test", "text/plain")},
        )
        assert resp.status_code == 415
        data = resp.json()
        assert isinstance(data, dict)
        assert "detail" in data
        assert isinstance(data["detail"], str)

    @pytest.mark.unit
    def test_missing_file_returns_422(self, client):
        """POST /predict without a file should return 422."""
        resp = client.post("/predict")
        assert resp.status_code == 422  # FastAPI validation error
