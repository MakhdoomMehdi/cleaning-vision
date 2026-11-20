from fastapi import APIRouter

from app.api.v1 import detection, inspections, references

# Versioned API router. The reviews module is registered here when its phase adds it.
api_router = APIRouter(prefix="/api/v1")
api_router.include_router(detection.router)
api_router.include_router(references.router)
api_router.include_router(inspections.router)
