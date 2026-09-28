"""HTTPS enforcement and per-client rate limiting for the OCR endpoints (/api/v1/ocr and below)."""

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings
from app.core.exceptions import OCR_PATH
from app.services.post_processing import error_response
from app.services.validation import enforce_rate_limit, request_identity


def _is_ocr_route(path: str) -> bool:
    return path == OCR_PATH or path.startswith(f"{OCR_PATH}/")


def _rejection(path: str, status_code: int, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    if path != OCR_PATH:
        return JSONResponse(status_code=status_code, content={"detail": message}, headers=headers)
    # No multipart body has been parsed yet at this point (middleware runs before the
    # route handler), so there is no real documentName/filename to report.
    return JSONResponse(
        status_code=status_code,
        content=error_response(document_name="unknown", document_filename="unknown", message=message),
        headers=headers,
    )


class OcrSecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if _is_ocr_route(path):
            if settings.require_https and request.headers.get("x-forwarded-proto", request.url.scheme).lower() != "https":
                return _rejection(path, 400, "HTTPS is required")
            client_ip = request.client.host if request.client else "unknown"
            identity = request_identity(
                client_ip=client_ip,
                tenant_id=request.headers.get("x-tenant-id"),
                user_id=request.headers.get("x-user-id"),
            )
            try:
                enforce_rate_limit(identity, settings)
            except HTTPException as exc:
                return _rejection(path, exc.status_code, exc.detail, exc.headers)
        return await call_next(request)
