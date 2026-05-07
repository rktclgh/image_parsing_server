from fastapi import APIRouter, Depends, Response, status

from app.core.config import Settings, get_settings
from app.schemas.status import HealthResponse, ModelStatusResponse, ReadyResponse, ServiceStatus
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
    model_state: ModelState = Depends(get_model_state),
) -> ReadyResponse:
    is_ready = model_state.loaded
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
        max_concurrent_generations=settings.max_concurrent_generations,
        loaded=model_state.loaded,
        detail=model_state.detail,
    )
