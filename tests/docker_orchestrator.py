"""Docker orchestration for pytest integration tests.

Manages the backend container lifecycle: build, start, health-poll, stop.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

import requests

from tests.config import BackendTestConfig

logger = logging.getLogger(__name__)


class DockerOrchestrator:
    def __init__(self, config: BackendTestConfig) -> None:
        self.config = config
        self._project_root = Path(__file__).resolve().parents[1]

    def _run(self, command: list[str], check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            command,
            cwd=self._project_root,
            check=check,
            text=True,
            capture_output=True,
        )

    def _remove_existing_container(self) -> None:
        result = self._run(
            ["docker", "ps", "-a", "--filter", f"name={self.config.container_name}", "-q"],
            check=False,
        )

        container_ids = result.stdout.strip().splitlines()

        for container_id in container_ids:
            self._run(["docker", "stop", container_id], check=False)
            self._run(["docker", "rm", container_id], check=False)

    def _build_image(self) -> None:
        try:
            self._run(
                ["docker", "compose", "-f", self.config.compose_file, "build", self.config.service_name]
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Docker build failed:\n\nSTDOUT:\n{exc.output}\n\nSTDERR:\n{exc.stderr}"
            ) from exc

    def _start_service(self) -> None:
        self._run(
            [
                "docker",
                "compose",
                "-f",
                self.config.compose_file,
                "up",
                "-d",
                self.config.service_name,
            ]
        )

    def _wait_for_health(self) -> None:
        health_url = f"{self.config.base_url}{self.config.health_path}"
        logger.info("Waiting for backend health at %s", health_url)

        started_at = time.time()

        while time.time() - started_at < self.config.startup_timeout_seconds:
            try:
                response = requests.get(health_url, timeout=3)
                if response.status_code == 200:
                    logger.info("Backend is healthy")
                    return
            except requests.RequestException:
                time.sleep(self.config.poll_interval_seconds)

        logs = self._run(["docker", "logs", self.config.container_name], check=False)
        raise TimeoutError(
            f"Backend did not become healthy within {self.config.startup_timeout_seconds} seconds.\n"
            f"Container logs:\n{logs.stdout}\n{logs.stderr}"
        )

    def backend_start(self) -> None:
        logger.info("Starting backend via docker compose...")
        self._remove_existing_container()
        self._build_image()
        self._start_service()
        self._wait_for_health()

    def backend_stop(self) -> None:
        logger.info("Stopping backend via docker compose...")
        self._run(
            ["docker", "compose", "-f", self.config.compose_file, "down"],
            check=False,
        )
