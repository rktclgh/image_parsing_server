from app.model.runtime import VLMRuntime


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


def test_resident_runtime_loads_on_startup_when_configured():
    loader = FakeLoader()
    runtime = VLMRuntime(loader=loader, mode="resident", load_on_startup=True)

    runtime.startup()

    assert loader.load_calls == 1
    assert runtime.loaded is True
    assert runtime.status.status == "ready"


def test_cold_runtime_does_not_load_on_startup_by_default():
    loader = FakeLoader()
    runtime = VLMRuntime(loader=loader, mode="cold", load_on_startup=False)

    runtime.startup()

    assert loader.load_calls == 0
    assert runtime.loaded is False
    assert runtime.status.status == "not_loaded"
    assert runtime.process_ready is True


def test_cold_runtime_is_process_ready_while_loading():
    loader = FakeLoader()
    runtime = VLMRuntime(loader=loader, mode="cold", load_on_startup=False)

    runtime.mark_loading()

    assert runtime.status.status == "loading"
    assert runtime.process_ready is True


def test_cold_runtime_loads_on_demand_and_can_unload_after_request():
    loader = FakeLoader()
    runtime = VLMRuntime(
        loader=loader,
        mode="cold",
        load_on_startup=False,
        unload_after_request=True,
    )

    runtime.ensure_loaded()
    runtime.after_request()

    assert loader.load_calls == 1
    assert loader.unload_calls == 1
    assert runtime.loaded is False
    assert runtime.status.status == "not_loaded"
