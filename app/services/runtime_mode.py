from dataclasses import dataclass
from threading import Lock
from typing import Literal

from fastapi import Depends, Request

from app.core.config import Settings, get_settings
from app.model.runtime import VLMRuntime

RuntimeMode = Literal["resident", "cold"]


@dataclass(frozen=True)
class RuntimeModeSnapshot:
    mode: RuntimeMode
    load_on_startup: bool
    unload_after_request: bool


class RuntimeModeController:
    def __init__(self, settings: Settings, *, runtime: VLMRuntime | None = None) -> None:
        self._lock = Lock()
        self._runtime = runtime
        self._snapshot = RuntimeModeSnapshot(
            mode=settings.vlm_mode,
            load_on_startup=settings.model_load_on_startup,
            unload_after_request=settings.unload_after_request,
        )

    @property
    def snapshot(self) -> RuntimeModeSnapshot:
        return self._snapshot

    def update(
        self,
        *,
        mode: RuntimeMode,
        load_on_startup: bool | None = None,
        unload_after_request: bool | None = None,
    ) -> RuntimeModeSnapshot:
        with self._lock:
            self._snapshot = RuntimeModeSnapshot(
                mode=mode,
                load_on_startup=(
                    load_on_startup
                    if load_on_startup is not None
                    else _default_load_on_startup(mode)
                ),
                unload_after_request=(
                    unload_after_request
                    if unload_after_request is not None
                    else self._snapshot.unload_after_request
                ),
            )
            if self._runtime is not None:
                self._runtime.configure(
                    mode=self._snapshot.mode,
                    load_on_startup=self._snapshot.load_on_startup,
                    unload_after_request=self._snapshot.unload_after_request,
                )
            return self._snapshot


def create_runtime_mode_controller(settings: Settings) -> RuntimeModeController:
    return RuntimeModeController(settings)


def get_runtime_mode_controller(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> RuntimeModeController:
    controller = getattr(request.app.state, "runtime_mode_controller", None)
    if controller is None:
        runtime = getattr(request.app.state, "vlm_runtime", None)
        controller = RuntimeModeController(settings, runtime=runtime)
        request.app.state.runtime_mode_controller = controller
    return controller


def _default_load_on_startup(mode: RuntimeMode) -> bool:
    return mode == "resident"
