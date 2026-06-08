import pytest


@pytest.mark.docker
def test_pipeline_ingest_endpoint(loan_service):
    result = loan_service.run_pipeline_step("ingest")

    assert result["status"] == "success"


@pytest.mark.docker
def test_pipeline_prepare_endpoint(loan_service):
    result = loan_service.run_pipeline_step("prepare")

    assert result["status"] == "success"


@pytest.mark.docker
def test_pipeline_train_endpoint(loan_service):
    result = loan_service.run_pipeline_step("train")

    assert result["status"] == "success"


@pytest.mark.docker
def test_pipeline_evaluate_endpoint(loan_service):
    result = loan_service.run_pipeline_step("evaluate")

    assert result["status"] == "success"


@pytest.mark.docker
def test_full_pipeline_runs_all_steps(loan_service):
    result = loan_service.run_full_pipeline()

    assert result["status"] == "success"
    assert set(result["steps"].keys()) == {"ingestion", "preparation", "training", "evaluation"}
