from __future__ import annotations

import logging
import time
from pathlib import Path

import requests
from python_on_whales import DockerClient

from tests.config import BackendTestConfig

logger = logging.getLogger(__name__)

CI_DIND_BASE_URL = "http://docker:9000"


class DockerOrchestrator:
    def __init__(self, config: BackendTestConfig) -> None:
        self.config = config
        self._project_root = Path(__file__).resolve().parents[1]
        self.docker: DockerClient | None = None

    def _create_docker_client(self) -> DockerClient:
        return DockerClient(
            compose_files=[self._project_root / compose_file for compose_file in self.config.compose_files],
            compose_project_directory=self._project_root,
        )

    def _ensure_docker(self) -> DockerClient:
        if self.docker is None:
            raise RuntimeError("DockerClient not initialized. Call backend_start() first.")
        return self.docker

    def _remove_existing_container(self) -> None:
        containers = self._ensure_docker().container.list(
            all=True,
            filters={"name": self.config.container_name},
        )

        for container in containers:
            logger.info("Removing existing container: %s", container.name)
            container.stop()
            container.remove()

    def _build_image(self) -> None:
        logger.info("Building backend image via docker compose...")
        self._ensure_docker().compose.build(services=[self.config.service_name])

    def _start_service(self) -> None:
        logger.info("Starting backend service via docker compose...")
        self._ensure_docker().compose.up(
            services=[self.config.service_name],
            detach=True,
        )

    def _wait_for_health(self) -> None:
        base_url = CI_DIND_BASE_URL if self.config.is_ci else self.config.base_url
        health_url = f"{base_url}{self.config.health_path}"

        logger.info("Waiting for backend health at %s", health_url)

        started_at = time.time()

        while time.time() - started_at < self.config.startup_timeout_seconds:
            try:
                response = requests.get(health_url, timeout=3)

                if response.status_code == 200:
                    logger.info(
                        "Backend is healthy after %.1f seconds",
                        time.time() - started_at,
                    )
                    return

            except requests.RequestException:
                time.sleep(self.config.poll_interval_seconds)

        logs = self._ensure_docker().compose.logs(
            services=[self.config.service_name],
            tail=100,
        )

        raise TimeoutError(
            f"Backend did not become healthy within {self.config.startup_timeout_seconds} seconds.\nContainer logs:\n{logs}"
        )

    def _update_base_url_for_ci(self) -> None:
        if not self.config.is_ci:
            return

        object.__setattr__(self.config, "base_url", CI_DIND_BASE_URL)
        logger.info("CI mode: set base_url -> %s", self.config.base_url)

    def backend_start(self) -> None:
        if self.config.external_backend:
            logger.info("external_backend=True, skipping Docker lifecycle")
            return

        logger.info("Starting backend via docker compose...")

        self.docker = self._create_docker_client()

        self._remove_existing_container()
        self._build_image()
        self._start_service()
        self._wait_for_health()
        self._update_base_url_for_ci()

    def backend_stop(self) -> None:
        if self.config.external_backend:
            return

        if self.docker is None:
            return

        logger.info("Stopping backend via docker compose...")
        self._ensure_docker().compose.down(remove_orphans=True, volumes=True)
