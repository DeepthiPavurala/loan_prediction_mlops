"""Centralized test configuration for backend integration and Docker tests."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


MODEL_PATH = Path("artifacts/models/loan_model_pipeline.joblib")


@dataclass(frozen=True)
class BackendTestConfig:
    compose_file: str = os.getenv("PYTEST_COMPOSE_FILE", "docker-compose.yml")
    service_name: str = os.getenv("PYTEST_SERVICE_NAME", "loan_prediction_backend")
    container_name: str = os.getenv("PYTEST_CONTAINER_NAME", "loan_prediction_backend")

    base_url: str = os.getenv("TEST_BACKEND_URL", "http://127.0.0.1:9000")
    health_path: str = os.getenv("TEST_BACKEND_HEALTH_PATH", "/health")

    startup_timeout_seconds: int = int(os.getenv("TEST_BACKEND_STARTUP_TIMEOUT", "120"))
    poll_interval_seconds: float = float(os.getenv("TEST_BACKEND_POLL_INTERVAL", "2"))

    keep_containers: bool = _env_bool("PYTEST_KEEP_DOCKER", False)
    skip_docker: bool = _env_bool("PYTEST_SKIP_DOCKER", True)

    run_mode: str = "local"

    @property
    def model_available(self) -> bool:
        if self.run_mode == "docker":
            return True
        return MODEL_PATH.exists()

    def with_docker_enabled(self, enabled: bool) -> BackendTestConfig:
        mode = "docker" if enabled else "local"
        return replace(self, skip_docker=not enabled, run_mode=mode)


backend_config = BackendTestConfig()
