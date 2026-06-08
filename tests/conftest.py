# from __future__ import annotations
#
# from pathlib import Path
#
# import pytest
#
# from tests import config as test_config
# from tests.data.loan_payloads import valid_loan_payload as _valid_loan_payload
# from tests.docker_orchestrator import DockerOrchestrator
# from tests.loan_prediction.loan_prediction_service import LoanPredictionService
#
# Path("reports").mkdir(parents=True, exist_ok=True)
# pytest_plugins = [
#     "tests.fixtures.client_fixtures",
# ]
#
# dm = DockerOrchestrator(test_config.backend_config)
#
#
# def _should_run_docker(config: pytest.Config) -> bool:
#     return bool(config.getoption("--run-docker", default=False))
#
#
# def _is_xdist_master(config: pytest.Config) -> bool:
#     return not hasattr(config, "workerinput")
#
#
# def pytest_addoption(parser: pytest.Parser) -> None:
#     parser.addoption(
#         "--run-docker",
#         action="store_true",
#         default=False,
#         help="Start Docker-backed backend before running tests.",
#     )
#
#
# def pytest_configure(config: pytest.Config) -> None:
#     test_config.backend_config = test_config.backend_config.with_docker_enabled(_should_run_docker(config))
#
#     if not _should_run_docker(config):
#         return
#
#     if _is_xdist_master(config):
#         dm.backend_start()
#
#
# def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
#     if _should_run_docker(config):
#         return
#
#     skip_docker = pytest.mark.skip(reason="use --run-docker to run Docker-backed tests")
#     for item in items:
#         if "docker" in item.keywords:
#             item.add_marker(skip_docker)
#
#
# def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
#     if not _should_run_docker(session.config):
#         return
#
#     if test_config.backend_config.keep_containers:
#         return
#
#     if _is_xdist_master(session.config):
#         dm.backend_stop()
#
#
# @pytest.fixture(scope="session")
# def valid_loan_payload():
#     return _valid_loan_payload()
#
#
# @pytest.fixture(scope="session")
# def loan_service(client):
#     return LoanPredictionService(client)
#
#
# requires_model = pytest.mark.skipif(
#     not test_config.backend_config.model_available,
#     reason="Model artifact not found. Run: python scripts/run_pipeline.py",
# )

from __future__ import annotations

from pathlib import Path

import pytest

from tests import config as test_config
from tests.data.loan_payloads import valid_loan_payload as _valid_loan_payload
from tests.docker_orchestrator import DockerOrchestrator
from tests.loan_prediction.loan_prediction_service import LoanPredictionService

Path("reports").mkdir(parents=True, exist_ok=True)
pytest_plugins = [
    "tests.fixtures.client_fixtures",
]

dm = DockerOrchestrator(test_config.backend_config)


def _should_run_docker(config: pytest.Config) -> bool:
    return bool(config.getoption("--run-docker", default=False))


def _is_xdist_master(config: pytest.Config) -> bool:
    return not hasattr(config, "workerinput")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-docker",
        action="store_true",
        default=False,
        help="Start Docker-backed backend before running tests.",
    )


def pytest_configure(config: pytest.Config) -> None:
    test_config.backend_config = test_config.backend_config.with_docker_enabled(_should_run_docker(config))

    if not _should_run_docker(config):
        return

    if _is_xdist_master(config):
        dm.backend_start()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _should_run_docker(config):
        return

    skip_docker = pytest.mark.skip(reason="use --run-docker to run Docker-backed tests")
    for item in items:
        if "docker" in item.keywords:
            item.add_marker(skip_docker)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not _should_run_docker(session.config):
        return

    if test_config.backend_config.keep_containers:
        return

    if _is_xdist_master(session.config):
        dm.backend_stop()


@pytest.fixture(scope="session")
def valid_loan_payload():
    return _valid_loan_payload()


@pytest.fixture(scope="session")
def loan_service(client):
    return LoanPredictionService(client)


requires_model = pytest.mark.skipif(
    not test_config.backend_config.model_available,
    reason="Model artifact not found. Run: python scripts/run_pipeline.py",
)
