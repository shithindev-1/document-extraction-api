"""Routing, API-key authentication and the security middleware - no model calls involved."""

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(settings, "require_https", False)
    return TestClient(app)


def test_health_is_public_and_unprefixed(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_keys", "secret")
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_routes_live_under_api_v1(client: TestClient) -> None:
    assert client.post("/ocr").status_code == 404
    assert client.post("/ocr/leasing", json={}).status_code == 404


@pytest.mark.parametrize("path", ["/api/v1/ocr", "/api/v1/ocr/leasing"])
def test_missing_api_key_is_rejected(client: TestClient, monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    monkeypatch.setattr(settings, "api_keys", "first, second")
    assert client.post(path, json={}).status_code == 401
    assert client.post(path, json={}, headers={"X-API-Key": "wrong"}).status_code == 401


def test_valid_api_key_reaches_the_route(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_keys", "first, second")
    # Past authentication, an empty body fails request validation instead.
    assert client.post("/api/v1/ocr/leasing", json={}, headers={"X-API-Key": "second"}).status_code == 422


def test_ocr_401_keeps_the_ocr_error_envelope(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "api_keys", "secret")
    body = client.post("/api/v1/ocr").json()
    assert body["success"] is False
    assert "errorInfo" in body


@pytest.mark.parametrize("path", ["/api/v1/ocr", "/api/v1/ocr/leasing", "/api/v1/ocr/leasing/upload"])
def test_https_is_enforced_on_every_ocr_route(client: TestClient, monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    monkeypatch.setattr(settings, "require_https", True)
    assert client.post(path, json={}).status_code == 400
    forwarded = client.post(path, json={}, headers={"X-Forwarded-Proto": "https"})
    assert forwarded.status_code != 400 or "HTTPS" not in forwarded.text
