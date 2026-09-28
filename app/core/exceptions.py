"""Pipeline stop signal and the app-wide exception handlers.

/ocr answers every error in its own errorInfo response shape; every other route keeps FastAPI's
standard `{"detail": ...}` body.
"""

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.services.post_processing import error_response

logger = logging.getLogger("uae_ocr")

API_V1_PREFIX = "/api/v1"
OCR_PATH = f"{API_V1_PREFIX}/ocr"
LEASING_PATH = f"{API_V1_PREFIX}/ocr/leasing"


class ValidationStopped(Exception):
    """Raised the moment any document fails type-matching, completeness, or self-verification.

    Carries the exact response the request should answer with - always blank `data` plus the
    reason in `errorInfo` - so the whole request can stop in one place (`run_ocr_batch`'s catch)
    instead of every caller of `validate_and_extract_one` needing its own copy of that logic.
    One document's problem is the whole request's problem: there is only ever one response object,
    so there is nowhere a failure could be reported that would let the rest keep going.
    """

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response


async def ocr_http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    if request.url.path != OCR_PATH:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers)
    logger.warning("OCR request rejected status_code=%d detail=%s", exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(
            document_name=getattr(request.state, "document_name", "unknown"),
            document_filename=getattr(request.state, "document_filename", "unknown"),
            message=exc.detail,
        ),
        headers=exc.headers,
    )


async def ocr_request_validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    if request.url.path != OCR_PATH:
        return JSONResponse(status_code=422, content={"detail": exc.errors()})
    # loc/type only, never "input" — a field's raw submitted value could be arbitrarily large
    # or, for the file field, binary content, and has no place in a text log.
    fields = [{"loc": err.get("loc"), "type": err.get("type")} for err in exc.errors()]
    logger.warning("OCR request rejected status_code=422 reason=malformed_request fields=%s", fields)
    return JSONResponse(
        status_code=422,
        content=error_response(
            document_name=getattr(request.state, "document_name", "unknown"),
            document_filename=getattr(request.state, "document_filename", "unknown"),
            message="Required fields are missing or malformed",
        ),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = request.headers.get("x-request-id", "unknown")
    logger.exception("OCR unhandled error request_id=%s error_type=%s", request_id, type(exc).__name__)
    if request.url.path != OCR_PATH:
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
    return JSONResponse(
        status_code=500,
        content=error_response(
            document_name=getattr(request.state, "document_name", "unknown"),
            document_filename=getattr(request.state, "document_filename", "unknown"),
            message="Internal server error",
        ),
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(HTTPException, ocr_http_exception_handler)
    app.add_exception_handler(RequestValidationError, ocr_request_validation_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
