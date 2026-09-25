"""HTTPS enforcement and per-client rate limiting for POST /ocr."""

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.exceptions import OCR_PATH
from app.services.post_processing import error_response
from app.services.validation import enforce_rate_limit, request_identity


def _unknown_document_error(status_code: int, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    # No multipart body has been parsed yet at this point (middleware runs before the
    # route handler), so there is no real documentName/filename to report.
    return JSONResponse(
        status_code=status_code,
        content=error_response(document_name="unknown", document_filename="unknown", message=message),
        headers=headers,
    )


class OcrSecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == OCR_PATH:
            if settings.require_https and request.headers.get("x-forwarded-proto", request.url.scheme).lower() != "https":
                return _unknown_document_error(400, "HTTPS is required")
            client_ip = request.client.host if request.client else "unknown"
            identity = request_identity(
                client_ip=client_ip,
                tenant_id=request.headers.get("x-tenant-id"),
                user_id=request.headers.get("x-user-id"),
            )
            try:
                enforce_rate_limit(identity, settings)
            except HTTPException as exc:
                return _unknown_document_error(exc.status_code, exc.detail, exc.headers)
        return await call_next(request)
