"""Every route the service exposes, collected into one router."""

from fastapi import APIRouter

from app.api.routes import health, leasing, ocr

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(ocr.router)
api_router.include_router(leasing.router)
