"""Application entry point.

    uvicorn app.main:app --reload
"""

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import OcrSecurityMiddleware


def create_app() -> FastAPI:
    configure_logging(settings)

    application = FastAPI(title="UAE OCR Development API", version="1.0.0")
    application.add_middleware(OcrSecurityMiddleware)
    register_exception_handlers(application)
    application.include_router(api_router)
    return application


app = create_app()
