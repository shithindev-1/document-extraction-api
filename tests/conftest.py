"""Shared test setup: keep a developer's own .env from changing what the tests exercise."""

import pytest

from app.core.config import settings
from app.services.validation import _rate_events


@pytest.fixture(autouse=True)
def _auth_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # A real API_KEYS in .env would otherwise reject every test request with 401.
    monkeypatch.setattr(settings, "api_keys", "")


@pytest.fixture(autouse=True)
def _fresh_rate_limiter() -> None:
    # The limiter is process-wide; without a reset, a long test run trips its 30-per-minute cap.
    _rate_events.clear()
