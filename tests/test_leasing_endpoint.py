"""Endpoint tests for /ocr/leasing and /ocr/leasing/upload.

Gemini is stubbed out throughout. What is under test is the leasing response shape and how a batch
is judged: every document checked in full, each failure reported on its own document only, and a
separately-uploaded front and back folded into one merged entry.
"""

import io
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.main import app
from app.schemas.ocr import DATA_FIELDS, MISSING_INFO_FIELDS
from app.services.gemini import get_gemini_service

UPLOAD_URL = "/ocr/leasing/upload"


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 40), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _raw(types_present: list[str], complete: bool = True, front: bool = True, back: bool = True,
         **fields: Any) -> dict[str, Any]:
    data = dict.fromkeys(DATA_FIELDS)
    data.update(fields)
    return {"data": data, "documentComplete": complete, "documentTypesPresent": types_present,
            "frontVisible": front, "backVisible": back}


def _upload(client: TestClient, documents: list[tuple[str, str, str, str]]) -> Any:
    """documents: (filename, document_id, document_type, document_name) per file, in order."""
    files = [("file", (filename, _png(), "image/png")) for filename, *_ in documents]
    data = {
        "tenant_type": "individual",
        "document_refnumber": "ref-1",
        "document_id": [doc_id for _, doc_id, _, _ in documents],
        "document_type": [doc_type for _, _, doc_type, _ in documents],
        "document_name": [name for *_, name in documents],
    }
    return client.post(UPLOAD_URL, files=files, data=data)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.setattr(settings, "detection_enabled", False)
    monkeypatch.setattr(settings, "validation_enabled", False)
    monkeypatch.setattr(settings, "leasing_output_dir", str(tmp_path))
    return TestClient(app)


def _answer_by_filename(monkeypatch: pytest.MonkeyPatch, answers: dict[str, dict[str, Any]]) -> None:
    async def fake_extract(**kwargs: Any) -> dict[str, Any]:
        return answers[kwargs["filename"]]

    monkeypatch.setattr(get_gemini_service(), "extract", fake_extract)


def test_front_and_back_uploads_merge_into_one_document(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _answer_by_filename(monkeypatch, {
        "front.png": _raw(["national_id"], complete=False, front=True, back=False),
        "back.png": _raw(["national_id"], complete=False, front=False, back=True),
        "front_back_merged.pdf": _raw(["national_id"], National_Id="784-1985-1234567-1"),
    })

    response = _upload(client, [
        ("front.png", "1", "National Id", "NID Front"),
        ("back.png", "2", "National Id", "NID Back"),
    ])

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["error"] is None
    assert list(body["form_data"]) == MISSING_INFO_FIELDS
    assert body["form_data"]["national_id"] == "784-1985-1234567-1"
    assert "national_id" not in body["missing"]
    assert body["processing_time"].endswith(" sec")
    [document] = body["documents"]
    assert document["merged"] is True
    assert document["document_type"] == "national_id"
    assert document["error"] is None
    assert [source["document_id"] for source in document["sources"]] == ["1", "2"]
    assert all(source["pages"] == [1] for source in document["sources"])


def test_failure_is_reported_only_on_the_failing_document(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _answer_by_filename(monkeypatch, {
        "passport.png": _raw(["passport"], complete=False, front=True, back=False),
        "ejari.png": _raw(["ejari_certificate"]),
    })

    response = _upload(client, [
        ("passport.png", "1", "Passport", "passport"),
        ("ejari.png", "2", "Ejari Certificate", "ejari"),
    ])

    body = response.json()
    assert body["status"] == "failed"
    passport, ejari = body["documents"]
    assert passport["error"]
    assert ejari["error"] is None
    assert ejari["document_type_is_correct"] is True
    assert body["error"] == passport["error"]
    assert all(value is None for value in body["form_data"].values())
    assert body["missing"] == MISSING_INFO_FIELDS


def test_every_document_is_checked_even_after_one_fails(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _answer_by_filename(monkeypatch, {
        "wrong.png": _raw(["passport"]),
        "incomplete.png": _raw(["visa"], complete=False, front=True, back=False),
        "fine.png": _raw(["ejari_certificate"]),
    })

    response = _upload(client, [
        ("wrong.png", "1", "National Id", "wrong type"),
        ("incomplete.png", "2", "Visa", "lone visa"),
        ("fine.png", "3", "Ejari Certificate", "ejari"),
    ])

    wrong, incomplete, fine = response.json()["documents"]
    assert wrong["error"] and wrong["document_type_is_correct"] is False
    assert incomplete["error"] and incomplete["error"] != wrong["error"]
    assert fine["error"] is None


def test_mismatched_field_counts_are_rejected(client: TestClient) -> None:
    response = client.post(UPLOAD_URL, files=[("file", ("a.png", _png(), "image/png"))], data={
        "tenant_type": "individual", "document_refnumber": "ref-1",
        "document_id": ["1", "2"], "document_type": ["Passport"], "document_name": ["a"],
    })

    assert response.status_code == 400


def test_undownloadable_source_is_reported_in_the_response(client: TestClient) -> None:
    response = client.post("/ocr/leasing", json={
        "tenant_type": "individual",
        "document_refnumber": "ref-1",
        "sources": [{"source": "", "document_id": "16", "document_type": "National Id",
                     "document_name": "Nationalid1"}],
    })

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    [document] = body["documents"]
    assert document["document_type"] == "unknown"
    assert document["sources"][0]["pages"] == []
    assert document["error"].startswith("Failed to fetch document from ''")
    assert body["error"] == document["error"]
