from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests import config as test_config
from tests.loan_prediction.request_client import RequestClient, Response


class InProcessRequestClient:
    """Adapter around FastAPI TestClient.

    Exposes the same interface as RequestClient (get/post/patch/delete
    returning Response) so all downstream code works with either client.
    """

    def __init__(self, test_client: TestClient) -> None:
        self.test_client = test_client

    def get(self, path: str, params: dict[str, Any] | None = None) -> Response:
        resp = self.test_client.get(path, params=params)
        return Response(status_code=resp.status_code, body=resp.json())

    def post(self, path: str, json: dict[str, Any] | list | None = None) -> Response:
        resp = self.test_client.post(path, json=json)
        return Response(status_code=resp.status_code, body=resp.json())

    def patch(self, path: str, json: dict[str, Any] | None = None) -> Response:
        resp = self.test_client.patch(path, json=json)
        return Response(status_code=resp.status_code, body=resp.json())

    def delete(self, path: str) -> Response:
        resp = self.test_client.delete(path)
        return Response(status_code=resp.status_code, body=resp.json())


@pytest.fixture(scope="session")
def client(request):
    """Dual-mode client fixture.

    Default: in-process via FastAPI TestClient (no Docker needed).
    With --run-docker: real HTTP client against the Docker container.
    """
    run_docker = bool(request.config.getoption("--run-docker"))

    if run_docker:
        yield RequestClient(test_config.backend_config.base_url)
        return

    with TestClient(app) as test_client:
        yield InProcessRequestClient(test_client)
