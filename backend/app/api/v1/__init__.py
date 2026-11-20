from fastapi import APIRouter

from app.api.v1 import inspections, references

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(references.router)
api_router.include_router(inspections.router)
