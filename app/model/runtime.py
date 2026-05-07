from dataclasses import dataclass
from typing import Literal, Protocol

from app.schemas.status import ModelStatus


class Loader(Protocol):
    loaded: bool

    def load(self):
        ...

    def unload(self) -> None:
        ...


@dataclass(frozen=True)
class RuntimeStatus:
    status: ModelStatus
    detail: str | None = None


class VLMRuntime:
    def __init__(
        self,
        *,
        loader: Loader,
        mode: Literal["resident", "cold"],
        load_on_startup: bool,
        unload_after_request: bool = False,
        idle_ttl_seconds: int = 0,
    ) -> None:
        if mode not in {"resident", "cold"}:
            raise ValueError("mode must be one of: cold, resident")
        if idle_ttl_seconds < 0:
            raise ValueError("idle_ttl_seconds must be >= 0")

        self._loader = loader
        self.mode = mode
        self.load_on_startup = load_on_startup
        self.unload_after_request = unload_after_request
        self.idle_ttl_seconds = idle_ttl_seconds
        self._status = RuntimeStatus(ModelStatus.NOT_LOADED, "model not loaded")

    @property
    def loaded(self) -> bool:
        return self._loader.loaded

    @property
    def status(self) -> RuntimeStatus:
        if self.loaded:
            return RuntimeStatus(ModelStatus.READY)
        return self._status

    @property
    def process_ready(self) -> bool:
        if self.loaded:
            return True
        return self.mode == "cold" and self.status.status is ModelStatus.NOT_LOADED

    def startup(self) -> None:
        if self.load_on_startup:
            self.ensure_loaded()

    def ensure_loaded(self) -> None:
        if self.loaded:
            self._status = RuntimeStatus(ModelStatus.READY)
            return

        self._status = RuntimeStatus(ModelStatus.LOADING, "model loading")
        try:
            self._loader.load()
        except Exception as exc:
            self._status = RuntimeStatus(ModelStatus.ERROR, str(exc))
            raise
        self._status = RuntimeStatus(ModelStatus.READY)

    def after_request(self) -> None:
        if self.mode == "cold" and self.unload_after_request:
            self.unload()

    def unload(self) -> None:
        self._loader.unload()
        self._status = RuntimeStatus(ModelStatus.NOT_LOADED, "model not loaded")
