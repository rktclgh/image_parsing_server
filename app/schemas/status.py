from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class ServiceStatus(StrEnum):
    OK = "ok"
    NOT_READY = "not_ready"


class ModelStatus(StrEnum):
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    READY = "ready"
    DEGRADED = "degraded"
    ERROR = "error"


class HealthResponse(BaseModel):
    status: ServiceStatus = ServiceStatus.OK
    service: str
    version: str


class ReadyResponse(BaseModel):
    status: ServiceStatus
    model_status: ModelStatus
    detail: str | None = None


class ModelStatusResponse(BaseModel):
    status: ModelStatus
    model_id: str
    quantization: str
    runtime_mode: Literal["resident", "cold"]
    load_on_startup: bool
    max_concurrent_generations: int = Field(ge=1)
    loaded: bool
    detail: str | None = None
