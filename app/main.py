from fastapi import FastAPI

from app.api.routes import health, monitoring, pipeline, prediction
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    description="MLOps-style backend for loan approval: ingestion, preparation, training, evaluation, inference, and drift monitoring.",
    version=settings.version,
)

app.include_router(health.router)
app.include_router(prediction.router)
app.include_router(pipeline.router)
app.include_router(monitoring.router)
