"""Application entry point.

    python -m app                        # production / Windows service (host and port from .env)
    uvicorn app.main:app --reload        # local development
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.router import api_router
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import OcrSecurityMiddleware
from app.services.retention import run_retention_sweeps


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    sweeper = asyncio.create_task(run_retention_sweeps(settings)) if settings.output_retention_hours > 0 else None
    try:
        yield
    finally:
        if sweeper:
            sweeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sweeper


def create_app() -> FastAPI:
    configure_logging(settings)

    docs = settings.docs_enabled
    application = FastAPI(
        title="UAE OCR API",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if docs else None,
        redoc_url="/redoc" if docs else None,
        openapi_url="/openapi.json" if docs else None,
    )
    application.add_middleware(OcrSecurityMiddleware)
    if settings.cors_origin_list:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )
    register_exception_handlers(application)
    application.include_router(api_router)
    return application


app = create_app()
