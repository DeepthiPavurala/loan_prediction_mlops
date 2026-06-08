import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.mark.contract
def test_openapi_schema_is_available():
    with TestClient(app) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200

    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema


@pytest.mark.contract
def test_openapi_schema_contains_predict_endpoint():
    with TestClient(app) as client:
        response = client.get("/openapi.json")

    schema = response.json()
    assert "/predict" in schema["paths"]
    assert "post" in schema["paths"]["/predict"]


@pytest.mark.contract
def test_openapi_schema_contains_batch_predict_endpoint():
    with TestClient(app) as client:
        response = client.get("/openapi.json")

    schema = response.json()
    assert "/predict/batch" in schema["paths"]
    assert "post" in schema["paths"]["/predict/batch"]


@pytest.mark.contract
def test_openapi_schema_contains_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/openapi.json")

    schema = response.json()
    assert "/health" in schema["paths"]
    assert "get" in schema["paths"]["/health"]
