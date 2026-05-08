from dataclasses import dataclass

from fastapi import Request

from app.schemas.status import ModelStatus


@dataclass(frozen=True)
class ModelState:
    status: ModelStatus = ModelStatus.NOT_LOADED
    detail: str | None = None

    @property
    def loaded(self) -> bool:
        return self.status in {ModelStatus.READY, ModelStatus.DEGRADED}


_MODEL_STATE = ModelState()


def get_model_state(request: Request) -> ModelState:
    runtime = getattr(request.app.state, "vlm_runtime", None)
    if runtime is None:
        return _MODEL_STATE

    runtime_status = runtime.status
    return ModelState(
        status=runtime_status.status,
        detail=runtime_status.detail,
    )
