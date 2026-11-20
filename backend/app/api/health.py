from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from app import __version__
from app.core.config import get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str
    environment: str
    timestamp: datetime
    openai_configured: bool
    openai_model: str
    yolo_model_path: str


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness plus non-secret configuration info (never returns the API key)."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=__version__,
        environment=settings.app_env,
        timestamp=datetime.now(timezone.utc),
        openai_configured=settings.openai_configured,
        openai_model=settings.openai_model,
        yolo_model_path=settings.yolo_model_path,
    )
