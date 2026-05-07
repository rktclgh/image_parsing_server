from dataclasses import dataclass

from app.schemas.status import ModelStatus


@dataclass(frozen=True)
class ModelState:
    status: ModelStatus = ModelStatus.NOT_LOADED
    detail: str | None = None

    @property
    def loaded(self) -> bool:
        return self.status in {ModelStatus.READY, ModelStatus.DEGRADED}


_MODEL_STATE = ModelState()


def get_model_state() -> ModelState:
    return _MODEL_STATE
