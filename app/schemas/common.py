from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_ready: bool = False
    model_path: str | None = None
