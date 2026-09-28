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
from app.services import leasing
from app.services.blob_storage import BlobStorage, BlobStorageError
from app.services.gemini import get_gemini_service

UPLOAD_URL = "/api/v1/ocr/leasing/upload"


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
    # Never the real container from .env - the blob tests install their own fake.
    monkeypatch.setattr(leasing, "get_blob_storage", lambda: None)
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
        "eid_front.png": _raw(["national_id"], complete=False, front=True, back=False),
        "ejari.png": _raw(["ejari_certificate"]),
    })

    response = _upload(client, [
        ("eid_front.png", "1", "National Id", "nid front"),
        ("ejari.png", "2", "Ejari Certificate", "ejari"),
    ])

    body = response.json()
    assert body["status"] == "failed"
    nid, ejari = body["documents"]
    assert nid["error"]
    assert ejari["error"] is None
    assert ejari["document_type_is_correct"] is True
    assert body["error"] == nid["error"]
    assert all(value is None for value in body["form_data"].values())
    assert body["missing"] == MISSING_INFO_FIELDS


def test_every_document_is_checked_even_after_one_fails(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _answer_by_filename(monkeypatch, {
        "wrong.png": _raw(["passport"]),
        "incomplete.png": _raw(["national_id"], complete=False, front=True, back=False),
        "fine.png": _raw(["ejari_certificate"]),
    })

    response = _upload(client, [
        ("wrong.png", "1", "National Id", "wrong type"),
        ("incomplete.png", "2", "National Id", "lone nid front"),
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
    response = client.post("/api/v1/ocr/leasing", json={
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


# --- Blob storage -------------------------------------------------------------------------------

CONTAINER_URL = "https://acct.blob.core.windows.net/files-uat"
FOLDER = "ApplicationFiles/LeasingRequestOCR/1/4f6a387a"


class FakeBlobStorage(BlobStorage):
    """In-memory container: `blobs` is what is stored, `uploads` records every write."""

    def __init__(self, blobs: dict[str, bytes]) -> None:
        self.container_url = CONTAINER_URL
        self.blobs = dict(blobs)
        self.uploads: dict[str, tuple[bytes, str]] = {}
        self.fail_uploads = False

    async def download(self, path: str, *, max_bytes: int) -> tuple[bytes, str | None]:
        if path not in self.blobs:
            raise BlobStorageError("The specified blob does not exist.", not_found=True)
        return self.blobs[path], "application/octet-stream"

    async def upload(self, path: str, content: bytes, content_type: str) -> None:
        if self.fail_uploads:
            raise BlobStorageError("This request is not authorized to perform this operation.")
        self.uploads[path] = (content, content_type)


def _jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 40), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def blob(monkeypatch: pytest.MonkeyPatch) -> FakeBlobStorage:
    storage = FakeBlobStorage({
        f"{FOLDER}/Passport.jpg": _jpeg(),
        f"{FOLDER}/EID Front.jpg": _jpeg(),
        f"{FOLDER}/EID Back.jpg": _jpeg(),
    })
    monkeypatch.setattr(leasing, "get_blob_storage", lambda: storage)
    monkeypatch.setattr(settings, "detection_enabled", True)

    async def fake_detection(**kwargs: Any) -> list[Path]:
        # One upright crop per page, written where the real pipeline would write it.
        output_dir = Path(kwargs["settings"].detection_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pages = 2 if kwargs["mime_type"] == "application/pdf" else 1
        paths = []
        for page in range(1, pages + 1):
            path = output_dir / f"crop_page{page}.png"
            Image.new("RGB", (80, 50), "gray").save(path, format="PNG")
            paths.append(path)
        return paths

    monkeypatch.setattr(leasing, "run_detection", fake_detection)
    return storage


def _sources_request(sources: list[tuple[str, str, str, str]]) -> dict[str, Any]:
    """sources: (source, document_id, document_type, document_name) per document."""
    return {
        "tenant_type": "individual",
        "document_refnumber": "4f6a387a",
        "sources": [
            {"source": source, "document_id": doc_id, "document_type": doc_type, "document_name": name}
            for source, doc_id, doc_type, name in sources
        ],
    }


def test_blob_source_is_fetched_and_its_result_uploaded_beside_it_as_ocr(
    client: TestClient, blob: FakeBlobStorage, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _answer_by_filename(monkeypatch, {"Passport.jpg": _raw(["passport"], Passport_Number="P123")})

    body = client.post("/api/v1/ocr/leasing", json=_sources_request([
        (f"{FOLDER}/Passport.jpg", "1", "Passport", "passport"),
    ])).json()

    assert body["status"] == "success", body
    assert body["form_data"]["passport_number"] == "P123"
    assert body["documents"][0]["sources"][0]["ocr_file"] == f"{CONTAINER_URL}/{FOLDER}/Passport_ocr.jpg"
    assert list(blob.uploads) == [f"{FOLDER}/Passport_ocr.jpg"]
    content, content_type = blob.uploads[f"{FOLDER}/Passport_ocr.jpg"]
    assert content_type == "image/jpeg"
    assert Image.open(io.BytesIO(content)).format == "JPEG"


def test_a_full_url_into_the_container_is_treated_as_a_blob_path(
    client: TestClient, blob: FakeBlobStorage, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _answer_by_filename(monkeypatch, {"Passport.jpg": _raw(["passport"])})

    body = client.post("/api/v1/ocr/leasing", json=_sources_request([
        (f"{CONTAINER_URL}/{FOLDER}/Passport.jpg", "1", "Passport", "passport"),
    ])).json()

    assert body["status"] == "success", body
    assert body["documents"][0]["sources"][0]["ocr_file"] == f"{CONTAINER_URL}/{FOLDER}/Passport_ocr.jpg"


def test_merged_front_and_back_upload_one_pdf_named_after_the_front(
    client: TestClient, blob: FakeBlobStorage, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _answer_by_filename(monkeypatch, {
        "EID Front.jpg": _raw(["national_id"], complete=False, front=True, back=False),
        "EID Back.jpg": _raw(["national_id"], complete=False, front=False, back=True),
        "EID Front_EID Back_merged.pdf": _raw(["national_id"], National_Id="784-1985-1234567-1"),
    })

    body = client.post("/api/v1/ocr/leasing", json=_sources_request([
        (f"{FOLDER}/EID Front.jpg", "1", "National Id", "NID Front"),
        (f"{FOLDER}/EID Back.jpg", "2", "National Id", "NID Back"),
    ])).json()

    assert body["status"] == "success", body
    [document] = body["documents"]
    assert document["merged"] is True
    expected_path = f"{FOLDER}/EID Front_ocr.pdf"
    expected_url = f"{CONTAINER_URL}/{FOLDER}/EID%20Front_ocr.pdf"
    assert [source["ocr_file"] for source in document["sources"]] == [expected_url, expected_url]
    assert list(blob.uploads) == [expected_path]
    assert blob.uploads[expected_path][1] == "application/pdf"


def test_nothing_is_uploaded_when_the_request_fails(
    client: TestClient, blob: FakeBlobStorage, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _answer_by_filename(monkeypatch, {
        "Passport.jpg": _raw(["visa"]),  # not the passport it was declared as
    })

    body = client.post("/api/v1/ocr/leasing", json=_sources_request([
        (f"{FOLDER}/Passport.jpg", "1", "Passport", "passport"),
    ])).json()

    assert body["status"] == "failed"
    assert blob.uploads == {}
    assert body["documents"][0]["sources"][0]["ocr_file"] is None


def test_a_missing_blob_is_reported_on_its_own_document(
    client: TestClient, blob: FakeBlobStorage, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _answer_by_filename(monkeypatch, {"Passport.jpg": _raw(["passport"])})

    body = client.post("/api/v1/ocr/leasing", json=_sources_request([
        (f"{FOLDER}/Passport.jpg", "1", "Passport", "passport"),
        (f"{FOLDER}/Visa.jpg", "2", "Visa", "visa"),
    ])).json()

    assert body["status"] == "failed"
    passport, visa = body["documents"]
    assert passport["error"] is None
    assert visa["error"] == f"Failed to fetch document from '{FOLDER}/Visa.jpg': The specified blob does not exist."
    assert blob.uploads == {}


def test_a_failed_upload_fails_the_request_on_that_document(
    client: TestClient, blob: FakeBlobStorage, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _answer_by_filename(monkeypatch, {"Passport.jpg": _raw(["passport"])})
    blob.fail_uploads = True

    body = client.post("/api/v1/ocr/leasing", json=_sources_request([
        (f"{FOLDER}/Passport.jpg", "1", "Passport", "passport"),
    ])).json()

    assert body["status"] == "failed"
    assert body["documents"][0]["error"].startswith(
        f"Failed to upload processed file for '{FOLDER}/Passport.jpg'"
    )
    assert all(value is None for value in body["form_data"].values())


def test_blob_path_only_accepts_paths_inside_the_container() -> None:
    storage = FakeBlobStorage({})
    assert storage.blob_path(f"{FOLDER}/Passport.jpg") == f"{FOLDER}/Passport.jpg"
    assert storage.blob_path(f"{CONTAINER_URL}/{FOLDER}/EID%20Front.jpg?sv=x") == f"{FOLDER}/EID Front.jpg"
    assert storage.blob_path("https://acct.blob.core.windows.net/other-container/a.jpg") is None
    assert storage.blob_path("https://evil.example.com/files-uat/a.jpg") is None
    assert storage.blob_path("") is None


@pytest.mark.parametrize(("document_type", "raw_type"), [("Passport", "passport"), ("Visa", "visa")])
def test_one_sided_passport_or_visa_is_extracted_without_a_back(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, document_type: str, raw_type: str,
) -> None:
    _answer_by_filename(monkeypatch, {
        "doc.png": _raw([raw_type], complete=False, front=True, back=False, Tenant_Name_En="Jane Doe"),
    })

    body = _upload(client, [("doc.png", "1", document_type, "doc")]).json()

    assert body["status"] == "success", body
    assert body["documents"][0]["error"] is None
    assert body["form_data"]["tenant_name_en"] == "Jane Doe"


# --- Viewing an uploaded _ocr result --------------------------------------------------------------


def test_an_uploaded_ocr_file_can_be_viewed_inline(
    client: TestClient, blob: FakeBlobStorage, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _answer_by_filename(monkeypatch, {"Passport.jpg": _raw(["passport"])})
    body = client.post("/api/v1/ocr/leasing", json=_sources_request([
        (f"{FOLDER}/Passport.jpg", "1", "Passport", "passport"),
    ])).json()
    ocr_url = body["documents"][0]["sources"][0]["ocr_file"]
    path = f"{FOLDER}/Passport_ocr.jpg"
    blob.blobs[path] = blob.uploads[path][0]  # what the real container now holds

    response = client.get("/api/v1/ocr/leasing/files", params={"url": ocr_url})

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["content-disposition"].startswith("inline;")
    assert response.content == blob.uploads[path][0]

    by_path = client.get(f"/api/v1/ocr/leasing/files/{path}")
    assert by_path.status_code == 200 and by_path.content == response.content


def test_a_merged_pdf_with_spaces_in_its_name_can_be_viewed(client: TestClient, blob: FakeBlobStorage) -> None:
    blob.blobs[f"{FOLDER}/EID Front_ocr.pdf"] = b"%PDF-1.4 fake"

    response = client.get(f"/api/v1/ocr/leasing/files/{FOLDER}/EID%20Front_ocr.pdf")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content == b"%PDF-1.4 fake"


@pytest.mark.parametrize("path", [
    f"{FOLDER}/Passport.jpg",           # an original upload, not a result
    f"{FOLDER}/notes_ocr.txt",          # not a supported result type
])
def test_only_ocr_results_can_be_viewed(client: TestClient, blob: FakeBlobStorage, path: str) -> None:
    response = client.get(f"/api/v1/ocr/leasing/files/{path}")

    assert response.status_code == 400


def test_a_missing_ocr_file_is_a_404(client: TestClient, blob: FakeBlobStorage) -> None:
    response = client.get(f"/api/v1/ocr/leasing/files/{FOLDER}/Visa_ocr.jpg")

    assert response.status_code == 404
    assert "does not exist" in response.json()["detail"]


def test_a_path_that_climbs_out_of_its_folder_is_rejected(blob: FakeBlobStorage) -> None:
    # An HTTP client normalises "../" away before the request is sent; guard the function itself.
    import asyncio

    from fastapi import HTTPException

    with pytest.raises(HTTPException) as caught:
        asyncio.run(leasing.fetch_ocr_file(f"{FOLDER}/../other/Passport_ocr.jpg"))
    assert caught.value.status_code == 400


def test_cheque_and_salary_statement_are_verified_like_other_documents(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _answer_by_filename(monkeypatch, {
        "cheque.png": _raw(["cheque"], Tenant_Name_En="Jane Doe"),
        "salary.png": _raw(["salary_certificate"], Employer_Name="ACME LLC"),
        "not_a_cheque.png": _raw(["bank_statement"]),
    })

    ok = _upload(client, [
        ("cheque.png", "1", "Cheque", "cheque"),
        ("salary.png", "2", "Salary Statement", "salary"),
    ]).json()
    assert ok["status"] == "success", ok
    assert [d["document_type"] for d in ok["documents"]] == ["cheque", "salary_certificate"]
    assert ok["form_data"]["employer_name"] == "ACME LLC"

    wrong = _upload(client, [("not_a_cheque.png", "1", "Cheque", "cheque")]).json()
    assert wrong["status"] == "failed"
    assert wrong["documents"][0]["document_type_is_correct"] is False


def test_a_url_outside_the_container_cannot_be_viewed(client: TestClient, blob: FakeBlobStorage) -> None:
    response = client.get("/api/v1/ocr/leasing/files", params={"url": "https://evil.example.com/files-uat/x_ocr.jpg"})

    assert response.status_code == 400
