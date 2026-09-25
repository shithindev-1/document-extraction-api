"""The OCR pipeline shared by every endpoint.

Extraction and type check, front/back pairing and merging, self- and cross-verification, crop and
rotation, and folding every document's fields into one result. `/ocr` drives the whole thing via
`run_ocr_batch`; the `/ocr/leasing` endpoints reuse the individual steps.
"""

import asyncio
import hashlib
import logging
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request, UploadFile
from PIL import Image
from pypdf import PdfReader, PdfWriter

from app.core.config import settings
from app.core.exceptions import ValidationStopped
from app.schemas.ocr import DATA_FIELDS, FRONT_BACK_TYPES, MISSING_INFO_FIELDS, ErrorInfo, OcrResponse
from app.services.detection import run_detection, save_merged_document
from app.services.gemini import GeminiServiceError, get_gemini_service, log_usage_summary
from app.services.post_processing import (
    ambiguous_pair_response,
    document_type_matches,
    incomplete_response,
    normalize_response,
    pair_mismatch_response,
)
from app.services.validation import validate_document_request

logger = logging.getLogger("uae_ocr")


def log_request_usage(request_id: str, status: str, detection_enabled: bool) -> None:
    """Record this request's token usage, per call and per pipeline.

    The structured record goes to usage.log; the line here keeps the same totals visible in the
    lifecycle log next to the request it belongs to. One record per request no matter how many
    documents it carried: the usage accumulator is scoped to the request, so it reports what
    extracting (and validating) every one of them together spent.
    """
    summary = log_usage_summary(request_id, status, detection_enabled)
    total = summary["request_total"]
    logger.info(
        "OCR usage request_id=%s status=%s calls=%d extraction=%d crop=%d rotation=%d validation=%d "
        "input_tokens=%d output_tokens=%d total_tokens=%d",
        request_id, status, total["calls"],
        summary["extraction"]["total_tokens"], summary["crop"]["total_tokens"],
        summary["rotation"]["total_tokens"], summary["validation"]["total_tokens"],
        total["input_tokens"], total["output_tokens"], total["total_tokens"],
    )


def issue_errors(document_name: str, document_filename: str, issues: list[str]) -> list[dict[str, str]]:
    """Turn plain problem sentences from a verification call into ordinary errorInfo entries.

    Nothing more elaborate than that: no status, no check name, no separate field - the existing
    DocumentError/DocumentErrorToShow shape every other /ocr rejection already uses.
    """
    return [
        {
            "DocumentName": document_name,
            "DocumentFileName": document_filename,
            "DocumentError": issue,
            "DocumentErrorToShow": issue,
        }
        for issue in issues
    ]


def merge_extracted_data(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold every document's own fields into one map, first non-null value in upload order.

    Each document only ever supplies the fields its own type actually carries - the extraction
    prompt already restricts a National ID to National_Id, a passport to Passport_Number, a visa to
    Visa_Number, and so on - so uploading one of each simply lands their fields in the same map
    without colliding. A genuine disagreement about the same field between two documents is
    cross-verification's job to catch before this ever runs; by the time it does, every document
    has already been confirmed to agree with the others on anything they share.
    """
    merged: dict[str, Any] = dict.fromkeys(DATA_FIELDS)
    for document in documents:
        for field, value in document["data"].items():
            if merged.get(field) is None and value is not None:
                merged[field] = value
    return merged


async def extract_and_type_check(
    *,
    request_id: str,
    position: str,
    content: bytes,
    filename: str,
    mime_type: str,
    sha256: str,
    document_name: str,
    document_type: str,
    started_at: float,
) -> dict[str, Any]:
    """Extraction plus the type-match check - shared by a real upload and a synthetic merged file.

    Completeness is reported here, never enforced: the caller decides whether an incomplete
    front/back document might be paired with another incomplete upload of the same type elsewhere
    in the batch (see `resolve_incomplete_pairs`), instead of rejecting it outright the moment it's
    seen. Only a provider error (hard failure, its own status code) or a type mismatch - neither
    fixable by pairing - stops anything here.
    """
    try:
        raw_response = await get_gemini_service().extract(
            content=content,
            filename=filename,
            content_type=mime_type,
            document_name=document_name,
            document_type=document_type,
        )
    except GeminiServiceError as exc:
        del content
        logger.error(
            "OCR validation status=error request_id=%s document=%s processing_time_sec=%.2f",
            request_id, position, time.perf_counter() - started_at,
        )
        raise HTTPException(status_code=502, detail="OCR provider unavailable") from exc

    if not document_type_matches(document_type, raw_response.get("documentTypesPresent")):
        del content
        logger.warning(
            "OCR validation status=type_mismatch request_id=%s document=%s selected=%s detected=%s "
            "processing_time_sec=%.2f",
            request_id, position, document_type, raw_response.get("documentTypesPresent"),
            time.perf_counter() - started_at,
        )
        # normalize_response drops the extracted values and attaches the mismatch error, so nothing
        # read off the wrong document reaches the caller.
        raise ValidationStopped(normalize_response(
            raw_response,
            document_name=document_name,
            document_filename=filename,
            document_type=document_type,
            default_reference_id=request_id,
        ))

    return {
        "document_name": document_name,
        "document_type": document_type,
        "data": raw_response.get("data") or {},
        "content": content,
        "mime_type": mime_type,
        "sha256": sha256,
        "filename": filename,
        "is_complete": bool(raw_response.get("documentComplete", True)),
        "front_visible": bool(raw_response.get("frontVisible", True)),
        "back_visible": bool(raw_response.get("backVisible", True)),
    }


async def validate_and_extract_one(
    *,
    request: Request,
    request_id: str,
    position: str,
    upload: UploadFile,
    document_name: str,
    document_type: str,
    started_at: float,
) -> dict[str, Any]:
    """File validation plus extraction for one real uploaded file.

    A file-validation problem or a provider error raises HTTPException, exactly as it always has -
    a hard failure with its own status code. A type mismatch raises `ValidationStopped`. Neither
    self-verification nor detection run here - see `run_self_verification` and the batch detection
    loop in `run_ocr_batch`, both of which only ever run once every uploaded document has been
    resolved into a final, complete list (originals plus any merged front/back pairs).
    """
    request.state.document_name = document_name
    request.state.document_filename = upload.filename or document_name
    content = await upload.read()
    try:
        validated = validate_document_request(
            file=upload,
            content=content,
            document_name=document_name,
            document_type=document_type,
            settings=settings,
        )
    except HTTPException as exc:
        logger.warning(
            "OCR validation status=rejected request_id=%s document=%s status_code=%d detail=%s",
            request_id, position, exc.status_code, exc.detail,
        )
        del content
        raise
    logger.info(
        "OCR reference=%s document=%s document_type=%s file_name=%s file_size=%d mime_type=%s sha256=%s",
        request_id,
        position,
        validated.document_type,
        upload.filename or "unnamed",
        len(content),
        validated.mime_type,
        validated.sha256,
    )
    return await extract_and_type_check(
        request_id=request_id,
        position=position,
        content=content,
        filename=upload.filename or document_name,
        mime_type=validated.mime_type,
        sha256=validated.sha256,
        document_name=document_name,
        document_type=validated.document_type,
        started_at=started_at,
    )


def _build_merged_pdf(first_content: bytes, first_mime: str, second_content: bytes, second_mime: str) -> bytes:
    """Combines two uploaded files - each an image or a PDF - into one multi-page PDF, in order.

    Uses only what this project already depends on: pypdf to read/write PDF pages, Pillow to turn a
    single image into one PDF page first when a side was uploaded as a JPG/PNG/WEBP rather than a
    PDF. No new dependency.
    """
    writer = PdfWriter()
    for content, mime_type in ((first_content, first_mime), (second_content, second_mime)):
        if mime_type == "application/pdf":
            writer.append(PdfReader(BytesIO(content)))
        else:
            page_buffer = BytesIO()
            Image.open(BytesIO(content)).convert("RGB").save(page_buffer, format="PDF")
            page_buffer.seek(0)
            writer.append(PdfReader(page_buffer))
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


async def _merge_front_back(
    *,
    request_id: str,
    front: dict[str, Any],
    back: dict[str, Any],
    document_type: str,
    validation_enabled: bool,
    started_at: float,
) -> dict[str, Any]:
    """Confirms and builds one merged document from two separately-uploaded, complementary sides.

    The confirmation is Gemini looking at both images and judging whether they are genuinely the
    same document, not a hardcoded rule - the same reasoning every other validation decision in
    this service already uses - so two different people's incomplete uploads landing in the same
    batch are never silently merged just because their document types and missing sides happen to
    line up. Raises `ValidationStopped` if that check comes back negative, or if the merged file
    still comes back incomplete on its own full re-extraction.
    """
    combined_name = f"{front['document_name']} + {back['document_name']}"
    combined_filename = f"{front['filename']} + {back['filename']}"

    if validation_enabled:
        same_document = await get_gemini_service().verify_document_pair(
            first_content=front["content"],
            first_mime=front["mime_type"],
            second_content=back["content"],
            second_mime=back["mime_type"],
            document_type=document_type,
        )
        if not same_document:
            logger.warning(
                "OCR validation status=pair_mismatch request_id=%s documents=%s",
                request_id, combined_filename,
            )
            raise ValidationStopped(pair_mismatch_response(
                document_name=combined_name,
                document_filename=combined_filename,
                document_type=document_type,
                reference_id=request_id,
            ))

    merged_content = _build_merged_pdf(front["content"], front["mime_type"], back["content"], back["mime_type"])
    merged_sha256 = hashlib.sha256(merged_content).hexdigest()
    merged_filename = f"{Path(front['filename']).stem}_{Path(back['filename']).stem}_merged.pdf"

    merged = await extract_and_type_check(
        request_id=request_id,
        position=combined_filename,
        content=merged_content,
        filename=merged_filename,
        mime_type="application/pdf",
        sha256=merged_sha256,
        document_name=combined_name,
        document_type=document_type,
        started_at=started_at,
    )
    if not merged["is_complete"]:
        logger.warning(
            "OCR validation status=incomplete request_id=%s document=%s reason=merge_still_incomplete",
            request_id, merged_filename,
        )
        del merged["content"]
        raise ValidationStopped(incomplete_response(
            document_name=combined_name,
            document_filename=combined_filename,
            document_type=document_type,
            reference_id=request_id,
        ))
    merged["is_merged_pair"] = True
    return merged


async def resolve_incomplete_pairs(
    *,
    request_id: str,
    candidates: list[dict[str, Any]],
    validation_enabled: bool,
    started_at: float,
) -> list[dict[str, Any]]:
    """Turns extracted candidates into the final list of complete documents left to verify.

    A candidate that is already complete (or whose type has no front/back requirement) passes
    through unchanged. An incomplete front/back candidate is grouped with others of the same
    document type: exactly one complementary pair - one showing only the front, the other only the
    back - is merged via `_merge_front_back`. A lone incomplete upload, or a group that cannot be
    resolved unambiguously (three or more, or two that show the same side), stops the whole request
    with a clear errorInfo instead of guessing which front belongs with which back.
    """
    resolved: list[dict[str, Any]] = []
    incomplete_by_type: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        if candidate["is_complete"] or candidate["document_type"] not in FRONT_BACK_TYPES:
            resolved.append(candidate)
        else:
            incomplete_by_type.setdefault(candidate["document_type"], []).append(candidate)

    for document_type, group in incomplete_by_type.items():
        if len(group) == 1:
            # Nothing else in the batch even claims to be this type, let alone shows a
            # complementary side - this is an ordinary lone incomplete document, exactly as if it
            # had been the only upload, regardless of which side it happens to show.
            candidate = group[0]
            logger.warning(
                "OCR validation status=incomplete request_id=%s document=%s",
                request_id, candidate["filename"],
            )
            raise ValidationStopped(incomplete_response(
                document_name=candidate["document_name"],
                document_filename=candidate["filename"],
                document_type=document_type,
                reference_id=request_id,
            ))

        pairable: list[dict[str, Any]] = []
        unpairable: list[dict[str, Any]] = []
        for candidate in group:
            # A candidate showing neither side (or, if the model ever reports it, both) has nothing
            # for another upload to complete - it is always its own, ordinary incomplete-document
            # error, never a pairing candidate.
            (pairable if candidate["front_visible"] != candidate["back_visible"] else unpairable).append(candidate)

        if unpairable:
            candidate = unpairable[0]
            logger.warning(
                "OCR validation status=incomplete request_id=%s document=%s",
                request_id, candidate["filename"],
            )
            raise ValidationStopped(incomplete_response(
                document_name=candidate["document_name"],
                document_filename=candidate["filename"],
                document_type=document_type,
                reference_id=request_id,
            ))

        front_candidates = [c for c in pairable if c["front_visible"]]
        back_candidates = [c for c in pairable if c["back_visible"]]
        if len(pairable) != 2 or len(front_candidates) != 1 or len(back_candidates) != 1:
            logger.warning(
                "OCR validation status=ambiguous_pair request_id=%s document_type=%s count=%d",
                request_id, document_type, len(pairable),
            )
            names = ", ".join(candidate["document_name"] for candidate in pairable)
            filenames = ", ".join(candidate["filename"] for candidate in pairable)
            raise ValidationStopped(ambiguous_pair_response(
                document_name=names,
                document_filename=filenames,
                document_type=document_type,
                count=len(pairable),
                reference_id=request_id,
            ))

        resolved.append(await _merge_front_back(
            request_id=request_id,
            front=front_candidates[0],
            back=back_candidates[0],
            document_type=document_type,
            validation_enabled=validation_enabled,
            started_at=started_at,
        ))

    return resolved


async def run_self_verification(
    *,
    request_id: str,
    position: str,
    document: dict[str, Any],
    validation_enabled: bool,
    started_at: float,
) -> None:
    """Self-verification for one already-complete document (original or merged front/back pair).

    Self-verification is Gemini looking at the document again, not a Python rule engine:
    completeness, required fields, formats, date logic, expiry, and internal consistency are all
    judgment calls it can make directly from the pixels. Only the call's own operational failure is
    swallowed, exactly like detection's - never fail an otherwise-good extraction because the check
    itself could not run; a genuine finding is never swallowed, and raises `ValidationStopped`.
    """
    issues: list[str] = []
    if validation_enabled:
        try:
            issues = await get_gemini_service().verify_document(
                content=document["content"],
                media_type=document["mime_type"],
                document_type=document["document_type"],
                today=datetime.now(timezone.utc).date(),
            )
        except Exception:
            logger.exception(
                "Self-verification failed request_id=%s document=%s", request_id, position,
            )
    else:
        logger.info(
            "Self-verification skipped request_id=%s document=%s reason=disabled",
            request_id, position,
        )

    if issues:
        logger.warning(
            "OCR validation status=self_verification_failed request_id=%s document=%s issues=%d "
            "processing_time_sec=%.2f",
            request_id, position, len(issues), time.perf_counter() - started_at,
        )
        raise ValidationStopped(OcrResponse(
            ocrReferenceId=request_id,
            errorInfo=[ErrorInfo.model_validate(error) for error in issue_errors(
                document["document_name"], document["filename"], issues,
            )],
        ).model_dump())

    logger.info(
        "OCR validation status=passed request_id=%s document=%s processing_time_sec=%.2f",
        request_id, position, time.perf_counter() - started_at,
    )


async def run_ocr_batch(
    *,
    request: Request,
    request_id: str,
    file: list[UploadFile],
    documentName: list[str],
    documentType: list[str],
    detection_enabled: bool,
    validation_enabled: bool,
    started_at: float,
) -> dict[str, Any]:
    """POST /ocr's whole pipeline for an already-shape-checked batch - see that route's docstring
    for the behaviour. Returns the one OcrResponse body (success, or blank `data` with the reason in
    `errorInfo`); a file-validation or provider failure raises HTTPException instead.
    """
    extracted: list[dict[str, Any]] = []
    try:
        # Sequentially, in upload order: the documents share one Gemini client, one provider-side
        # rate budget, and one usage accumulator. A hard failure or a type mismatch still stops
        # everything immediately, exactly as before - only an incomplete front/back document no
        # longer stops the request on the spot, since a later upload in this same batch might be
        # its missing other side.
        candidates: list[dict[str, Any]] = []
        for index, (upload, name, requested_type) in enumerate(zip(file, documentName, documentType), start=1):
            position = f"{index}/{len(file)}"
            candidates.append(await validate_and_extract_one(
                request=request,
                request_id=request_id,
                position=position,
                upload=upload,
                document_name=name,
                document_type=requested_type,
                started_at=started_at,
            ))

        resolved = await resolve_incomplete_pairs(
            request_id=request_id,
            candidates=candidates,
            validation_enabled=validation_enabled,
            started_at=started_at,
        )

        for index, document in enumerate(resolved, start=1):
            position = f"{index}/{len(resolved)}"
            await run_self_verification(
                request_id=request_id,
                position=position,
                document=document,
                validation_enabled=validation_enabled,
                started_at=started_at,
            )
            extracted.append(document)
    except ValidationStopped as stopped:
        log_request_usage(request_id, "validation_failed", detection_enabled)
        return stopped.response

    if len(extracted) > 1 and validation_enabled:
        # By construction, every document reaching this point already passed its own type match,
        # completeness check, and self-verification - cross-verification only ever compares
        # documents already confirmed individually sound.
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
            logger.warning(
                "OCR cross-verification failed request_id=%s documents=%d issues=%d",
                request_id, len(extracted), len(issues),
            )
            log_request_usage(request_id, "validation_failed", detection_enabled)
            # No single document to blame - the conflict is between two or more of them - so the
            # combined list of the names involved stands in for DocumentName/DocumentFileName.
            # None of these documents is cropped or rotated: detection only ever runs below, once
            # the whole batch - including this check - has cleared.
            names = ", ".join(document["document_name"] for document in extracted)
            return OcrResponse(
                ocrReferenceId=request_id,
                errorInfo=[ErrorInfo.model_validate(error) for error in issue_errors(names, names, issues)],
            ).model_dump()

    # Only now - every document individually validated, and the batch cross-verified if there was
    # more than one - does the crop/rotate pipeline run. Nothing before this point ever reaches it,
    # so a request that ends up answering with errorInfo never had a document cropped or rotated
    # for nothing.
    if detection_enabled:
        for document in extracted:
            saved_paths = await run_detection(
                content=document["content"],
                filename=document["filename"],
                mime_type=document["mime_type"],
                document_type=document["document_type"],
                sha256=document["sha256"],
                settings=settings,
                gemini_service=get_gemini_service(),
            )
            if document.get("is_merged_pair") and saved_paths:
                # The crop pipeline above already produced one clean, upright PNG per side of this
                # merged document; stitch those into the one combined file the person who uploaded
                # the two separate sides actually wants to download, next to the crop output on
                # disk, found by its `_merged.pdf` suffix.
                await asyncio.to_thread(
                    save_merged_document,
                    saved_paths,
                    output_dir=Path(settings.detection_output_dir),
                    filename=document["filename"],
                    sha256=document["sha256"],
                )
    else:
        logger.info(
            "Detection pipeline skipped request_id=%s reason=disabled (extraction only) documents=%d",
            request_id, len(extracted),
        )
    for document in extracted:
        del document["content"]

    merged_data = merge_extracted_data(extracted)
    response = OcrResponse(
        ocrReferenceId=request_id,
        data=merged_data,
        missingInfo=[
            missing_name
            for field, missing_name in zip(DATA_FIELDS, MISSING_INFO_FIELDS)
            if merged_data[field] is None
        ],
    ).model_dump()
    log_request_usage(request_id, "success", detection_enabled)
    logger.info(
        "OCR request complete request_id=%s documents=%d processing_time_sec=%.2f",
        request_id, len(extracted), time.perf_counter() - started_at,
    )
    return response
