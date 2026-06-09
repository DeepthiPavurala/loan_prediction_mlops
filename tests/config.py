from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "loan_model_pipeline.joblib"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.lower() in {"1", "true", "yes", "y"}


@dataclass(frozen=True)
class BackendTestConfig:
    compose_file: str = os.getenv("PYTEST_COMPOSE_FILE", "docker-compose.yml")
    service_name: str = os.getenv("PYTEST_SERVICE_NAME", "loan_prediction_backend")
    container_name: str = os.getenv("PYTEST_CONTAINER_NAME", "loan_prediction_backend")

    base_url: str = os.getenv("TEST_BACKEND_URL", "http://127.0.0.1:9000")
    health_path: str = os.getenv("TEST_BACKEND_HEALTH_PATH", "/health")

    startup_timeout_seconds: int = int(os.getenv("TEST_BACKEND_STARTUP_TIMEOUT", "120"))
    poll_interval_seconds: float = float(os.getenv("TEST_BACKEND_POLL_INTERVAL", "2"))

    external_backend: bool = _env_bool("PYTEST_EXTERNAL_BACKEND", default=False)
    keep_containers: bool = _env_bool("PYTEST_KEEP_DOCKER", default=False)

    run_mode: str = "local"
    is_ci: bool = _env_bool("CI", default=False)

    @property
    def compose_files(self) -> list[str]:
        files = [self.compose_file]

        if self.is_ci:
            files.append("docker-compose.ci.yml")

        return files

    @property
    def model_available(self) -> bool:
        if self.run_mode == "docker":
            return True

        return MODEL_PATH.exists()

    def with_docker_enabled(self, enabled: bool) -> BackendTestConfig:
        mode = "docker" if enabled else "local"
        return replace(self, run_mode=mode)


backend_config = BackendTestConfig()
