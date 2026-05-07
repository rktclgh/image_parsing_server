from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


class FakeLoader:
    def __init__(self) -> None:
        self.loaded = False
        self.load_calls = 0
        self.unload_calls = 0

    def load(self):
        self.load_calls += 1
        self.loaded = True
        return self

    def unload(self) -> None:
        self.unload_calls += 1
        self.loaded = False


def test_app_lifespan_loads_resident_runtime_on_startup_and_unloads_on_shutdown():
    loader = FakeLoader()
    app = create_app(
        settings=Settings(vlm_mode="resident"),
        model_loader=loader,
    )

    with TestClient(app) as client:
        response = client.get("/v1/model/status")

        assert response.status_code == 200
        assert loader.load_calls == 1
        assert response.json()["status"] == "ready"
        assert response.json()["loaded"] is True

    assert loader.unload_calls == 1
    assert loader.loaded is False


def test_app_lifespan_keeps_cold_runtime_unloaded_but_process_ready():
    loader = FakeLoader()
    app = create_app(
        settings=Settings(vlm_mode="cold"),
        model_loader=loader,
    )

    with TestClient(app) as client:
        status_response = client.get("/v1/model/status")
        ready_response = client.get("/readyz")

    assert loader.load_calls == 0
    assert status_response.status_code == 200
    assert status_response.json()["runtime_mode"] == "cold"
    assert status_response.json()["status"] == "not_loaded"
    assert status_response.json()["loaded"] is False
    assert ready_response.status_code == 200


def test_app_lifespan_unloads_when_resident_startup_fails():
    class FailingLoader(FakeLoader):
        def load(self):
            self.load_calls += 1
            raise RuntimeError("startup failed")

    loader = FailingLoader()
    app = create_app(
        settings=Settings(vlm_mode="resident"),
        model_loader=loader,
    )

    try:
        with TestClient(app):
            raise AssertionError("lifespan startup should fail")
    except RuntimeError as exc:
        assert str(exc) == "startup failed"

    assert loader.load_calls == 1
    assert loader.unload_calls == 1


def test_runtime_mode_api_updates_actual_runtime_snapshot():
    loader = FakeLoader()
    app = create_app(
        settings=Settings(vlm_mode="resident", model_load_on_startup=False),
        model_loader=loader,
    )

    with TestClient(app) as client:
        before = client.get("/readyz")
        response = client.patch(
            "/v1/model/runtime-mode",
            json={"mode": "cold", "unload_after_request": True},
        )
        after = client.get("/readyz")
        status_response = client.get("/v1/model/status")

    runtime = app.state.vlm_runtime
    assert before.status_code == 503
    assert response.status_code == 200
    assert response.json() == {
        "runtime_mode": "cold",
        "load_on_startup": False,
        "unload_after_request": True,
    }
    assert runtime.mode == "cold"
    assert runtime.unload_after_request is True
    assert after.status_code == 200
    assert status_response.json()["runtime_mode"] == "cold"
    assert status_response.json()["status"] == "not_loaded"
