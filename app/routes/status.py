from fastapi import APIRouter, Depends, Response, status

from app.core.config import Settings, get_settings
from app.schemas.status import (
    HealthResponse,
    ModelStatus,
    ModelStatusResponse,
    ReadyResponse,
    ServiceStatus,
)
from app.services.model_state import ModelState, get_model_state

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse)
def healthz(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status=ServiceStatus.OK,
        service=settings.service_name,
        version=settings.service_version,
    )


@router.get("/readyz", response_model=ReadyResponse)
def readyz(
    response: Response,
    settings: Settings = Depends(get_settings),
    model_state: ModelState = Depends(get_model_state),
) -> ReadyResponse:
    is_ready = _is_process_ready(settings=settings, model_state=model_state)
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadyResponse(
        status=ServiceStatus.OK if is_ready else ServiceStatus.NOT_READY,
        model_status=model_state.status,
        detail=model_state.detail,
    )


@router.get("/v1/model/status", response_model=ModelStatusResponse)
def model_status(
    settings: Settings = Depends(get_settings),
    model_state: ModelState = Depends(get_model_state),
) -> ModelStatusResponse:
    return ModelStatusResponse(
        status=model_state.status,
        model_id=settings.model_id,
        quantization=settings.quantization,
        runtime_mode=settings.vlm_mode,
        load_on_startup=settings.model_load_on_startup,
        max_concurrent_generations=settings.max_concurrent_generations,
        loaded=model_state.loaded,
        detail=model_state.detail,
    )


def _is_process_ready(*, settings: Settings, model_state: ModelState) -> bool:
    if model_state.loaded:
        return True
    return settings.vlm_mode == "cold" and model_state.status in {
        ModelStatus.NOT_LOADED,
        ModelStatus.LOADING,
    }
