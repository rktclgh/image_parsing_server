from fastapi import APIRouter, Depends, Response, status

from app.core.config import Settings, get_settings
from app.schemas.status import (
    HealthResponse,
    ModelStatus,
    ModelStatusResponse,
    ReadyResponse,
    RuntimeModeResponse,
    RuntimeModeUpdateRequest,
    ServiceStatus,
)
from app.services.model_state import ModelState, get_model_state
from app.services.runtime_mode import RuntimeModeController, get_runtime_mode_controller

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
    model_state: ModelState = Depends(get_model_state),
    runtime_mode: RuntimeModeController = Depends(get_runtime_mode_controller),
) -> ReadyResponse:
    is_ready = _is_process_ready(
        runtime_mode=runtime_mode.snapshot.mode,
        model_state=model_state,
    )
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
    runtime_mode: RuntimeModeController = Depends(get_runtime_mode_controller),
) -> ModelStatusResponse:
    runtime_snapshot = runtime_mode.snapshot
    return ModelStatusResponse(
        status=model_state.status,
        model_id=settings.model_id,
        quantization=settings.quantization,
        runtime_mode=runtime_snapshot.mode,
        load_on_startup=runtime_snapshot.load_on_startup,
        unload_after_request=runtime_snapshot.unload_after_request,
        max_concurrent_generations=settings.max_concurrent_generations,
        loaded=model_state.loaded,
        detail=model_state.detail,
    )


@router.patch("/v1/model/runtime-mode", response_model=RuntimeModeResponse)
def update_runtime_mode(
    request: RuntimeModeUpdateRequest,
    runtime_mode: RuntimeModeController = Depends(get_runtime_mode_controller),
) -> RuntimeModeResponse:
    snapshot = runtime_mode.update(
        mode=request.mode,
        load_on_startup=request.load_on_startup,
        unload_after_request=request.unload_after_request,
    )
    return RuntimeModeResponse(
        runtime_mode=snapshot.mode,
        load_on_startup=snapshot.load_on_startup,
        unload_after_request=snapshot.unload_after_request,
    )


def _is_process_ready(*, runtime_mode: str, model_state: ModelState) -> bool:
    if model_state.loaded:
        return True
    return runtime_mode == "cold" and model_state.status in {
        ModelStatus.NOT_LOADED,
        ModelStatus.LOADING,
    }
