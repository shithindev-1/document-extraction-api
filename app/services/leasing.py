"""Batch processing behind the /ocr/leasing endpoints.

Both endpoints reduce their input to the same list of items and hand it to `process_batch`, which
reuses the shared OCR pipeline (app.services.ocr_pipeline) step by step rather than
re-implementing any of it, so the leasing and /ocr endpoints cannot drift apart.

How a batch is judged:
  - Every document is checked in full before anything is decided - download, validation,
    extraction and type check, front/back pairing (per document type), self-verification - so one
    document's problem never stops the others from being checked. Each failing document carries
    its own reason in its own `error`; every document that passed has `error: null`.
  - Cross-verification runs only once every document passed on its own, and its findings go on the
    documents they name.
  - The *result* is all-or-nothing: any failure blanks `form_data`, sets `status` to "failed", and
    puts the first failing document's message in the top-level `error`. Cropping, rotation and the
    merged PDF only happen when the whole request passed.
  - The one exception to checking everything is the OCR provider itself being unavailable, which
    stops the request at that document.
  - `document_name` must be unique within a request: it is how a merged front/back result is traced
    back to its two original documents.
  - Output files (originals, crops, merged PDF) are written under `settings.leasing_output_dir`.
  - A source inside the configured blob container (a path like "ApplicationFiles/.../Passport.jpg",
    or a full URL into that container) is fetched from blob storage, and - only when the whole
    request passed - its cropped/rotated result is uploaded back into the same folder as
    `<name>_ocr.<ext>`, reported in that source's `ocr_file` as its full blob URL. One crop keeps the original's image
    format; several crops become one PDF; a merged front/back pair uploads one PDF named after the
    front. A failed upload is reported on its document and fails the request.
"""

import asyncio
import logging
import posixpath
import re
import time
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException
from PIL import Image

from app.core.config import settings
from app.core.exceptions import ValidationStopped
from app.schemas.leasing import LeasingOcrRequest, SourceDocument
from app.schemas.ocr import DATA_FIELDS, MISSING_INFO_FIELDS
from app.services.blob_storage import BlobStorage, BlobStorageError, get_blob_storage
from app.services.detection import run_detection, save_merged_document
from app.services.gemini import get_gemini_service, start_usage_tracking
from app.services.ocr_pipeline import (
    extract_and_type_check,
    log_request_usage,
    merge_extracted_data,
    resolve_incomplete_pairs,
    run_self_verification,
)
from app.services.validation import ALLOWED_FILES, validate_document_request

logger = logging.getLogger("uae_ocr.leasing")

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_CONTENT_TYPE_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "application/pdf": ".pdf",
}
# Pillow format names for an _ocr result that keeps its original's image type.
_IMAGE_FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}


def _safe_name(value: str) -> str:
    return _UNSAFE_NAME.sub("_", value.strip())[:80] or "document"


def _output_root() -> Path:
    return Path(settings.leasing_output_dir)


async def _download_source(client: httpx.AsyncClient, source: SourceDocument) -> tuple[bytes, str, str]:
    """Downloads one `source` URL. Returns (content, filename, content_type).

    filename/content_type are derived the same way a browser upload's would be, so the downloaded
    bytes can be handed straight to `validate_document_request` unchanged: the URL's own last path
    segment is the filename when it has one of the four supported extensions, falling back to the
    response's Content-Type header, then to the document's own declared name as a last resort.
    """
    try:
        response = await client.get(source.source, follow_redirects=True, timeout=60.0)
        response.raise_for_status()
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch document from '{source.source}': {exc}",
        ) from exc

    content = response.content
    if len(content) > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"'{source.document_name}' exceeds the configured size limit after download",
        )

    url_name = Path(urlparse(source.source).path).name or source.document_name
    extension = Path(url_name).suffix.lower()
    content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()

    if extension in ALLOWED_FILES:
        filename = url_name
        if content_type not in ALLOWED_FILES.values():
            content_type = ALLOWED_FILES[extension]
    elif content_type in _CONTENT_TYPE_EXTENSIONS:
        filename = f"{Path(url_name).stem or _safe_name(source.document_name)}{_CONTENT_TYPE_EXTENSIONS[content_type]}"
    else:
        # Nothing usable was reported at all - let validate_document_request reject it with its own
        # clear "only JPG/JPEG/PNG/WEBP/PDF" message rather than guessing.
        filename = url_name or f"{source.document_name}.bin"

    return content, filename, content_type


async def _download_blob(
    storage: BlobStorage, blob_path: str, source: SourceDocument,
) -> tuple[bytes, str, str]:
    """Fetches one source from blob storage. Returns (content, filename, content_type) exactly
    like `_download_source`, so the rest of the batch cannot tell the two apart."""
    try:
        content, stored_type = await storage.download(
            blob_path, max_bytes=settings.max_file_size_mb * 1024 * 1024,
        )
    except BlobStorageError as exc:
        raise HTTPException(
            status_code=502, detail=f"Failed to fetch document from '{source.source}': {exc}",
        ) from exc
    filename = posixpath.basename(blob_path) or f"{_safe_name(source.document_name)}.bin"
    extension = posixpath.splitext(filename)[1].lower()
    # The extension wins over whatever Content-Type the blob was stored with (often a generic
    # application/octet-stream); validate_document_request still checks the real bytes.
    content_type = ALLOWED_FILES.get(extension) or (stored_type or "").split(";")[0].strip().lower()
    return content, filename, content_type


def ensure_unique_document_names(names: list[str]) -> None:
    """Rejects a request that reuses a document_name - names are how a merged front/back result and
    every per-document error are traced back to the right upload, so a repeat would be ambiguous."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise HTTPException(
            status_code=400,
            detail=f"Each document_name must be unique within a request; repeated: {', '.join(duplicates)}",
        )


def _document_folder(refnumber: str, document_id: str, document_name: str) -> Path:
    folder = _output_root() / _safe_name(refnumber) / f"{_safe_name(document_id)}_{_safe_name(document_name)}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _merged_folder(refnumber: str, front_id: str, back_id: str) -> Path:
    folder = _output_root() / _safe_name(refnumber) / f"merged_{_safe_name(front_id)}_{_safe_name(back_id)}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _new_report(item: dict[str, Any]) -> dict[str, Any]:
    """One `documents` entry per source, before anything has been read. `document_type` stays
    "unknown" until the source passes validation; a merged front/back pair is folded into a single
    entry with both sources only once the whole request has succeeded."""
    return {
        "document_type": "unknown",
        "document_type_is_correct": False,
        "merged": False,
        "sources": [{
            "source": item["source"],
            "document_id": item["document_id"],
            "document_name": item["document_name"],
            "document_type": item["document_type"],
            "pages": [],
            "ocr_file": None,
        }],
        "error": None,
    }


def _first_error_message(response: dict[str, Any]) -> str:
    errors = response.get("errorInfo") or []
    if errors:
        return errors[0].get("DocumentErrorToShow") or errors[0].get("DocumentError") or "Validation failed"
    return "Validation failed"


def _response(
    *, status: str, tenant_type: str, document_refnumber: str, started_at: float,
    documents: list[dict[str, Any]], error: str | None, form_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The one response shape both endpoints return, success or failure. `form_data` keys are the
    lowercase names (tenant_name_en, ...) rather than the pipeline's own Tenant_Name_En spelling."""
    form_data = {
        key: (form_data or {}).get(field)
        for field, key in zip(DATA_FIELDS, MISSING_INFO_FIELDS)
    }
    elapsed = round(time.perf_counter() - started_at, 2)
    return {
        "status": status,
        "tenant_type": tenant_type,
        "document_refnumber": document_refnumber,
        "form_data": form_data,
        "missing": [key for key, value in form_data.items() if value is None],
        "processing_time": f"{elapsed:g} sec",
        "documents": documents,
        "error": error,
    }


def _fold_merged_pairs(
    documents_report: list[dict[str, Any]], merged_pairs: list[tuple[int, int, str]],
) -> list[dict[str, Any]]:
    """Replaces each merged front/back pair's two entries with one `merged: true` entry carrying
    both sources, placed where the earlier of the two was uploaded."""
    folded: dict[int, dict[str, Any]] = {}
    absorbed: set[int] = set()
    for front_index, back_index, document_type in merged_pairs:
        first, second = sorted((front_index, back_index))
        folded[first] = {
            "document_type": document_type,
            "document_type_is_correct": True,
            "merged": True,
            "sources": documents_report[front_index]["sources"] + documents_report[back_index]["sources"],
            "error": None,
        }
        absorbed.add(second)
    return [
        folded.get(index, report)
        for index, report in enumerate(documents_report)
        if index not in absorbed
    ]


def _indices_named(names: str, index_by_name: dict[str, int], fallback: list[int]) -> list[int]:
    """Maps an errorInfo DocumentName back to upload positions. The pipeline names a merged pair
    "<front> + <back>" and an ambiguous group "<a>, <b>, ..."; anything unrecognised falls back to
    `fallback` so an error is never silently dropped."""
    if names in index_by_name:
        return [index_by_name[names]]
    parts = [part.strip() for chunk in names.split(" + ") for part in chunk.split(", ")]
    found = [index_by_name[part] for part in parts if part in index_by_name]
    return found or fallback


async def _resolve_pairs_per_type(
    *, request_id: str, candidates: list[dict[str, Any]], index_by_name: dict[str, int],
    documents_report: list[dict[str, Any]], started_at: float,
) -> list[dict[str, Any]]:
    """Runs the pipeline's own front/back pairing once per document type, so a problem with one type's
    uploads is reported on those documents only and never stops another type from being paired.
    Returns every document that resolved cleanly (complete originals plus merged pairs)."""
    by_type: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_type.setdefault(candidate["document_type"], []).append(candidate)

    resolved: list[dict[str, Any]] = []
    for group in by_type.values():
        try:
            resolved.extend(await resolve_incomplete_pairs(
                request_id=request_id,
                candidates=group,
                validation_enabled=settings.validation_enabled,
                started_at=started_at,
            ))
        except ValidationStopped as stopped:
            message = _first_error_message(stopped.response)
            errors = stopped.response.get("errorInfo") or [{}]
            incomplete = [c for c in group if not c["is_complete"]]
            named = _indices_named(
                errors[0].get("DocumentName") or "", index_by_name,
                [index_by_name[c["document_name"]] for c in incomplete],
            )
            for index in named:
                documents_report[index]["error"] = message
            # Complete documents in this group were never the problem - they still go on to
            # self-verification. An incomplete one the error did not name (e.g. a front whose only
            # possible partner was the unpairable upload) is still incomplete: pair it alone, which
            # returns its own lone-incomplete error with no Gemini call.
            resolved.extend(c for c in group if c["is_complete"])
            for candidate in incomplete:
                index = index_by_name[candidate["document_name"]]
                if index in named:
                    continue
                try:
                    await resolve_incomplete_pairs(
                        request_id=request_id, candidates=[candidate],
                        validation_enabled=settings.validation_enabled, started_at=started_at,
                    )
                except ValidationStopped as lone:
                    documents_report[index]["error"] = _first_error_message(lone.response)
    return resolved


def _ocr_output(image_paths: list[Path], original_extension: str) -> tuple[bytes, str, str]:
    """Builds the uploaded result from a document's final cropped/rotated PNGs.

    Returns (content, extension, content_type). One crop keeps the original's own image format
    (a .jpg stays a .jpg); several crops - a PDF's pages, or two cards found in one photo - become
    one PDF, a page per crop, since a single image file cannot hold more than one.
    """
    images = [Image.open(path) for path in image_paths]
    buffer = BytesIO()
    image_format = _IMAGE_FORMATS.get(original_extension)
    if len(images) == 1 and image_format:
        image = images[0].convert("RGB") if image_format == "JPEG" else images[0]
        image.save(buffer, format=image_format, **({"quality": 95} if image_format == "JPEG" else {}))
        return buffer.getvalue(), original_extension, ALLOWED_FILES[original_extension]
    pages = [image.convert("RGB") for image in images]
    pages[0].save(buffer, format="PDF", save_all=True, append_images=pages[1:])
    return buffer.getvalue(), ".pdf", "application/pdf"


def _ocr_blob_path(blob_path: str, extension: str) -> str:
    """ApplicationFiles/.../Passport.jpg -> ApplicationFiles/.../Passport_ocr.jpg (same folder)."""
    folder, _, name = blob_path.rpartition("/")
    renamed = f"{posixpath.splitext(name)[0]}_ocr{extension}"
    return f"{folder}/{renamed}" if folder else renamed


async def _upload_ocr_result(blob_path: str, crop_paths: list[Path], merged_pdf: Path | None) -> str:
    """Uploads one document's result beside its original and returns its full blob URL."""
    storage = get_blob_storage()
    if storage is None:
        raise BlobStorageError("blob storage is not configured")
    if merged_pdf is not None:
        content, extension, content_type = merged_pdf.read_bytes(), ".pdf", "application/pdf"
    else:
        extension = posixpath.splitext(blob_path)[1].lower()
        content, extension, content_type = await asyncio.to_thread(_ocr_output, crop_paths, extension)
    target = _ocr_blob_path(blob_path, extension)
    await storage.upload(target, content, content_type)
    return storage.url(target)


_OCR_FILE_NAME = re.compile(r"_ocr\.(jpe?g|png|webp|pdf)$", re.IGNORECASE)
# A merged or multi-page result is a PDF of full-resolution crops - allow well past the upload limit.
_MAX_OCR_FILE_BYTES = 100 * 1024 * 1024


async def fetch_ocr_file(ocr_file: str) -> tuple[bytes, str, str]:
    """Reads one uploaded `<name>_ocr.<ext>` result back out of blob storage, for viewing.

    `ocr_file` is either the full blob URL a response's `ocr_file` carries, or the blob path
    inside the container. Returns (content, content_type, filename). Only `_ocr` results are
    served - never an original upload or any other blob in the container.
    """
    storage = get_blob_storage()
    if storage is None:
        raise HTTPException(status_code=503, detail="Blob storage is not configured")
    path = storage.blob_path(ocr_file)
    if path is None:
        raise HTTPException(status_code=400, detail="Only files inside the configured blob container can be viewed")
    filename = posixpath.basename(path)
    if ".." in path.split("/") or not _OCR_FILE_NAME.search(filename):
        raise HTTPException(status_code=400, detail="Only processed <name>_ocr.<ext> files can be viewed")
    try:
        content, _ = await storage.download(path, max_bytes=_MAX_OCR_FILE_BYTES)
    except BlobStorageError as exc:
        raise HTTPException(
            status_code=404 if exc.not_found else 502, detail=f"Could not read '{path}': {exc}",
        ) from exc
    return content, ALLOWED_FILES[posixpath.splitext(filename)[1].lower()], filename


def _merged_indices(document: dict[str, Any], index_by_name: dict[str, int]) -> list[int]:
    if document.get("is_merged_pair"):
        front_name, _, back_name = document["document_name"].partition(" + ")
        return [index_by_name[front_name], index_by_name[back_name]]
    return [index_by_name[document["document_name"]]]


async def process_batch(
    *, tenant_type: str, document_refnumber: str, items: list[dict[str, Any]],
    started_at: float | None = None,
) -> dict[str, Any]:
    """Everything from here down is identical for both endpoints - the only difference between
    `/ocr/leasing` and `/ocr/leasing/upload` is how `items` gets built (downloaded vs. read
    straight from the multipart upload) before this function ever runs.

    Each item in `items` is: source (str - the URL, or the uploaded filename), content (bytes),
    filename (str), content_type (str), document_id (str), document_type (str), document_name (str)
    - exactly what a real upload and a downloaded URL both reduce to once the bytes are in hand. An
    item that could not be fetched instead carries `error` (str) and no content.

    Every document is checked in full before anything is decided: validation, extraction and type
    check, front/back pairing, then self-verification. A document that fails a step gets that
    step's message in its own `error` and goes no further; the others carry on. Cross-verification
    runs only when every document passed on its own. Cropping, rotation, the merged PDF and
    `form_data` happen only when the whole request passed.
    """
    started_at = time.perf_counter() if started_at is None else started_at
    request_id = document_refnumber

    if not items:
        raise HTTPException(status_code=400, detail="At least one document must be provided")

    # document_name is assumed unique within one request - it is the only thing the pipeline's own
    # `_merge_front_back` carries forward into a merged document's own `document_name`
    # ("<front name> + <back name>"), so it is the only way this module can trace a result back to
    # the original documents without changing the pipeline itself.
    index_by_name = {item["document_name"]: index for index, item in enumerate(items)}
    documents_report = [_new_report(item) for item in items]

    def _finish(status: str, *, error: str | None, form_data: dict[str, Any] | None = None,
                documents: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return _response(
            status=status, tenant_type=tenant_type, document_refnumber=document_refnumber,
            started_at=started_at, documents=documents if documents is not None else documents_report,
            error=error, form_data=form_data,
        )

    def _failed() -> dict[str, Any]:
        first = next(report["error"] for report in documents_report if report["error"])
        return _finish("failed", error=first)

    refnumber_folder = _output_root() / _safe_name(document_refnumber)
    refnumber_folder.mkdir(parents=True, exist_ok=True)

    start_usage_tracking()

    # 1. Validation, extraction and type check - every document, whatever happened to the others.
    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        report = documents_report[index]
        if item.get("error"):
            report["error"] = item["error"]
            continue

        position = f"{index + 1}/{len(items)}"
        content, filename, content_type = item["content"], item["filename"], item["content_type"]

        folder = _document_folder(document_refnumber, item["document_id"], item["document_name"])
        (folder / f"original_{_safe_name(filename)}").write_bytes(content)

        fake_upload = SimpleNamespace(filename=filename, content_type=content_type)
        try:
            validated = validate_document_request(
                file=fake_upload,
                content=content,
                document_name=item["document_name"],
                document_type=item["document_type"],
                settings=settings,
            )
        except HTTPException as exc:
            report["error"] = str(exc.detail)
            continue

        report["document_type"] = validated.document_type
        report["sources"][0]["pages"] = list(range(1, (validated.page_count or 1) + 1))

        try:
            candidate = await extract_and_type_check(
                request_id=request_id,
                position=position,
                content=content,
                filename=filename,
                mime_type=validated.mime_type,
                sha256=validated.sha256,
                document_name=item["document_name"],
                document_type=validated.document_type,
                started_at=started_at,
            )
        except HTTPException as exc:
            # The pipeline turns a GeminiServiceError into HTTPException(502, "OCR provider unavailable").
            # Nothing else in the request can be checked without the provider, so stop here.
            report["error"] = str(exc.detail)
            return _failed()
        except ValidationStopped as stopped:
            # The only stop `_extract_and_type_check` raises is a document type mismatch.
            report["error"] = _first_error_message(stopped.response)
            continue

        report["document_type_is_correct"] = True
        candidate["_document_id"] = item["document_id"]
        candidates.append(candidate)

    # 2. Front/back pairing, per document type.
    resolved = await _resolve_pairs_per_type(
        request_id=request_id, candidates=candidates, index_by_name=index_by_name,
        documents_report=documents_report, started_at=started_at,
    )

    # 3. Self-verification of every document that got this far.
    extracted: list[dict[str, Any]] = []
    for index, document in enumerate(resolved, start=1):
        try:
            await run_self_verification(
                request_id=request_id,
                position=f"{index}/{len(resolved)}",
                document=document,
                validation_enabled=settings.validation_enabled,
                started_at=started_at,
            )
        except ValidationStopped as stopped:
            message = _first_error_message(stopped.response)
            for upload_index in _merged_indices(document, index_by_name):
                documents_report[upload_index]["error"] = message
            del document["content"]
            continue
        extracted.append(document)

    if any(report["error"] for report in documents_report):
        return _failed()

    # 4. Cross-verification - only once every document is valid on its own.
    if len(extracted) > 1 and settings.validation_enabled:
        try:
            issues = await get_gemini_service().verify_documents_cross(documents=[
                {"label": document["document_name"], "document_type": document["document_type"],
                 "data": document["data"]}
                for document in extracted
            ])
        except Exception:
            logger.exception("Cross-document verification failed request_id=%s", request_id)
            issues = []
        if issues:
            # Each finding names the documents it concerns by their label (document_name). One
            # that names none of them is attached to every document, rather than dropped.
            everyone = list(range(len(items)))
            for issue in issues:
                lowered = issue.lower()
                involved = [
                    index for name, index in index_by_name.items() if name.lower() in lowered
                ] or everyone
                for index in involved:
                    existing = documents_report[index]["error"]
                    documents_report[index]["error"] = f"{existing}; {issue}" if existing else issue
            return _failed()

    # 5. Everything passed - crop, rotate, build merged PDFs, upload _ocr results, fill form_data.
    merged_pairs: list[tuple[int, int, str]] = []
    for document in extracted:
        is_merged = bool(document.get("is_merged_pair"))
        upload_indices = _merged_indices(document, index_by_name)
        if is_merged:
            front_index, back_index = upload_indices
            merged_pairs.append((front_index, back_index, document["document_type"]))
            output_dir = _merged_folder(
                document_refnumber, items[front_index]["document_id"], items[back_index]["document_id"],
            )
        else:
            output_dir = _document_folder(document_refnumber, document["_document_id"], document["document_name"])

        saved_paths: list[Path] | None = None
        merged_pdf: Path | None = None
        if settings.detection_enabled:
            per_document_settings = settings.model_copy(update={"detection_output_dir": str(output_dir)})
            saved_paths = await run_detection(
                content=document["content"],
                filename=document["filename"],
                mime_type=document["mime_type"],
                document_type=document["document_type"],
                sha256=document["sha256"],
                settings=per_document_settings,
                gemini_service=get_gemini_service(),
            )
            if is_merged and saved_paths:
                merged_pdf = await asyncio.to_thread(
                    save_merged_document,
                    saved_paths,
                    output_dir=output_dir,
                    filename=document["filename"],
                    sha256=document["sha256"],
                )
        del document["content"]

        # A merged pair uploads one PDF beside the front's original; the back's source row points
        # at that same file.
        blob_path = items[upload_indices[0]].get("blob_path")
        if blob_path and saved_paths:
            try:
                ocr_path = await _upload_ocr_result(blob_path, saved_paths, merged_pdf)
            except BlobStorageError as exc:
                message = f"Failed to upload processed file for '{blob_path}': {exc}"
                for index in upload_indices:
                    documents_report[index]["error"] = message
                continue
            for index in upload_indices:
                documents_report[index]["sources"][0]["ocr_file"] = ocr_path

    if any(report["error"] for report in documents_report):
        return _failed()

    merged_data = merge_extracted_data(extracted)
    log_request_usage(request_id, "success", settings.detection_enabled)
    return _finish(
        "success", error=None, form_data=merged_data,
        documents=_fold_merged_pairs(documents_report, merged_pairs),
    )


async def process_sources(request: LeasingOcrRequest) -> dict[str, Any]:
    """POST /ocr/leasing: download every source URL, then run the batch.

    A source that cannot be downloaded gets the reason in its own `error`; every other source is
    still downloaded and checked in full, exactly as if it had been uploaded.
    """
    ensure_unique_document_names([source.document_name for source in request.sources])
    started_at = time.perf_counter()
    items: list[dict[str, Any]] = [
        {
            "source": source.source, "document_id": source.document_id,
            "document_type": source.document_type, "document_name": source.document_name,
        }
        for source in request.sources
    ]
    storage = get_blob_storage()
    async with httpx.AsyncClient() as client:
        for source, item in zip(request.sources, items):
            blob_path = storage.blob_path(source.source) if storage else None
            try:
                if blob_path:
                    content, filename, content_type = await _download_blob(storage, blob_path, source)
                    item["blob_path"] = blob_path
                elif not source.source.lower().startswith(("http://", "https://")):
                    reason = (
                        "not a file path inside the configured blob container, or an http(s) URL"
                        if storage else
                        "blob storage is not configured, so only a full http(s) URL can be used"
                    )
                    raise HTTPException(
                        status_code=400, detail=f"Failed to fetch document from '{source.source}': {reason}",
                    )
                else:
                    content, filename, content_type = await _download_source(client, source)
            except HTTPException as exc:
                item["error"] = str(exc.detail)
                continue
            item.update(content=content, filename=filename, content_type=content_type)

    return await process_batch(
        tenant_type=request.tenant_type, document_refnumber=request.document_refnumber, items=items,
        started_at=started_at,
    )
