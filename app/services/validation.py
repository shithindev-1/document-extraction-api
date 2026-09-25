"""Upload validation (type, size, MIME, PDF/image integrity) and rate limiting."""

import logging
import re
import time
from pathlib import Path
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass
from io import BytesIO
from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader
from app.core.config import Settings

ALLOWED_FILES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".pdf": "application/pdf",
}
_rate_events: dict[str, deque[float]] = defaultdict(deque)
_safe_id = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_safe_document_type = re.compile(r"^[a-z0-9_]{1,50}$")
logger = logging.getLogger("uae_ocr")


@dataclass(frozen=True)
class ValidatedDocument:
    document_type: str
    extension: str
    mime_type: str
    page_count: int | None
    sha256: str


def request_identity(*, client_ip: str, tenant_id: str | None, user_id: str | None) -> str:
    tenant = tenant_id if tenant_id and _safe_id.fullmatch(tenant_id) else "anonymous"
    user = user_id if user_id and _safe_id.fullmatch(user_id) else "anonymous"
    return f"{client_ip}:{tenant}:{user}"


def enforce_rate_limit(identity: str, settings: Settings) -> None:
    now = time.monotonic()
    events = _rate_events[identity]
    cutoff = now - settings.rate_limit_window_seconds
    while events and events[0] <= cutoff:
        events.popleft()
    if len(events) >= settings.rate_limit_requests:
        logger.warning("OCR rate limit event identity=%s", identity)
        raise HTTPException(status_code=429, detail="Request rate limit exceeded", headers={"Retry-After": str(settings.rate_limit_window_seconds)})
    events.append(now)


def _file_error(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail=message)


def _detect_mime(content: bytes) -> str | None:
    if content.startswith(b"%PDF-"):
        return "application/pdf"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return "image/webp"
    return None


def _validate_pdf(content: bytes) -> int:
    try:
        reader = PdfReader(BytesIO(content), strict=True)
        if reader.is_encrypted:
            raise _file_error("Encrypted PDF files are not supported")
        page_count = len(reader.pages)
        if page_count < 1:
            raise _file_error("PDF contains no readable pages")
        return page_count
    except HTTPException:
        raise
    except Exception as exc:
        raise _file_error("PDF is corrupt or unreadable") from exc


def _validate_image(content: bytes, mime_type: str) -> None:
    try:
        with Image.open(BytesIO(content)) as image:
            if image.format is None:
                raise _file_error("Image format could not be verified")
            image.verify()
        with Image.open(BytesIO(content)) as image:
            image.load()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise _file_error(f"{mime_type} file is corrupt or unreadable") from exc


def validate_document_request(
    *,
    file: UploadFile,
    content: bytes,
    document_name: str,
    document_type: str,
    settings: Settings,
) -> ValidatedDocument:
    # Any document type is accepted, not a fixed KYC whitelist. The value is normalised and
    # charset-restricted because it reaches log lines and the Gemini prompt.
    normalized_type = document_type.strip().lower().replace(" ", "_").replace("-", "_")
    if not _safe_document_type.fullmatch(normalized_type):
        raise _file_error("documentType must be 1-50 characters using letters, digits, spaces, hyphens, or underscores")
    if not 1 <= len(document_name.strip()) <= 100 or any(ord(char) < 32 for char in document_name):
        raise _file_error("documentName must be 1-100 characters without control characters")
    if not content:
        raise _file_error("Uploaded file is empty")
    if len(content) > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds the configured size limit")

    filename = file.filename or ""
    extension = Path(filename).suffix.lower()
    expected_mime = ALLOWED_FILES.get(extension)
    if expected_mime is None:
        raise _file_error("Only JPG, JPEG, PNG, WEBP, and PDF files are supported")
    actual_mime = _detect_mime(content)
    if actual_mime != expected_mime or file.content_type != expected_mime:
        raise _file_error(f"This file doesn't look like a genuine {extension} file. Check the file and try again.")

    page_count: int | None = None
    if actual_mime == "application/pdf":
        page_count = _validate_pdf(content)
    else:
        _validate_image(content, actual_mime)

    # Page count says nothing about front/back completeness: both sides routinely share a single
    # page (stacked, side by side, or scanned as one continuous image), and a two-page file can
    # just as easily be the same side twice. Only the content can answer it, so that judgement
    # belongs to the extraction call rather than a short-circuit here.
    return ValidatedDocument(
        document_type=normalized_type,
        extension=extension,
        mime_type=actual_mime,
        page_count=page_count,
        sha256=hashlib.sha256(content).hexdigest(),
    )


def log_processing(started_at: float, api_status: str) -> None:
    logger.info("Gemini status=%s processing_time_sec=%.2f", api_status, time.perf_counter() - started_at)