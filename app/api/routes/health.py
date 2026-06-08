from fastapi import APIRouter

from app.core.config import settings
from app.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check():
    model_exists = settings.model_path.exists()
    return HealthResponse(
        status="healthy" if model_exists else "degraded",
        service=settings.app_name,
        version=settings.version,
        model_ready=model_exists,
        model_path=str(settings.model_path),
    )


@router.get("/")
def root():
    return {"message": settings.app_name, "docs": "/docs", "health": "/health"}
