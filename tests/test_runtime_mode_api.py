from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.schemas.status import ModelStatus
from app.services.model_state import ModelState, get_model_state


def test_runtime_mode_status_uses_environment_backed_settings():
    client = TestClient(create_app(settings=Settings(vlm_mode="cold")))

    response = client.get("/v1/model/status")

    assert response.status_code == 200
    assert response.json()["runtime_mode"] == "cold"
    assert response.json()["load_on_startup"] is False
    assert response.json()["unload_after_request"] is False


def test_runtime_mode_api_switches_resident_to_cold_without_restart():
    app = create_app(settings=Settings(vlm_mode="resident"))
    app.dependency_overrides[get_model_state] = lambda: ModelState(
        status=ModelStatus.NOT_LOADED,
        detail="model not loaded",
    )
    client = TestClient(app)

    before = client.get("/readyz")
    response = client.patch(
        "/v1/model/runtime-mode",
        json={"mode": "cold", "unload_after_request": True},
    )
    after = client.get("/readyz")
    status = client.get("/v1/model/status")

    assert before.status_code == 503
    assert response.status_code == 200
    assert response.json() == {
        "runtime_mode": "cold",
        "load_on_startup": False,
        "unload_after_request": True,
    }
    assert after.status_code == 200
    assert status.json()["runtime_mode"] == "cold"
    assert status.json()["load_on_startup"] is False
    assert status.json()["unload_after_request"] is True


def test_runtime_mode_api_accepts_hot_alias_for_resident_mode():
    client = TestClient(create_app(settings=Settings(vlm_mode="cold")))

    response = client.patch("/v1/model/runtime-mode", json={"mode": "hot"})

    assert response.status_code == 200
    assert response.json() == {
        "runtime_mode": "resident",
        "load_on_startup": True,
        "unload_after_request": False,
    }


def test_runtime_mode_api_rejects_unknown_mode():
    client = TestClient(create_app())

    response = client.patch("/v1/model/runtime-mode", json={"mode": "warm"})

    assert response.status_code == 422
