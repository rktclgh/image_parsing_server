from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.status import ModelStatus
from app.services.model_state import ModelState, get_model_state


def build_client(model_state: ModelState | None = None) -> TestClient:
    app = create_app()
    if model_state is not None:
        app.dependency_overrides[get_model_state] = lambda: model_state
    return TestClient(app)


def test_healthz_returns_service_identity_when_model_not_loaded():
    client = build_client(ModelState(status=ModelStatus.NOT_LOADED))

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "image-parsing-server",
        "version": "v1",
    }


def test_readyz_returns_503_until_model_is_ready_or_degraded():
    client = build_client(ModelState(status=ModelStatus.NOT_LOADED, detail="model not loaded"))

    response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "model_status": "not_loaded",
        "detail": "model not loaded",
    }


def test_readyz_returns_200_when_model_is_ready():
    client = build_client(ModelState(status=ModelStatus.READY))

    response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_status": "ready",
        "detail": None,
    }


def test_readyz_returns_200_when_model_is_degraded():
    client = build_client(ModelState(status=ModelStatus.DEGRADED, detail="fallback mode"))

    response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_status": "degraded",
        "detail": "fallback mode",
    }


def test_model_status_returns_stable_model_contract():
    client = build_client(ModelState(status=ModelStatus.READY))

    response = client.get("/v1/model/status")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model_id": "google/gemma-4-E4B-it",
        "quantization": "8bit",
        "max_concurrent_generations": 1,
        "loaded": True,
        "detail": None,
    }


def test_parse_full_is_not_part_of_stable_v1_status_module():
    client = build_client()

    response = client.post("/v1/parse/full")

    assert response.status_code == 404
