from fastapi import APIRouter, Depends, File, Request, UploadFile

from app.core.config import Settings, get_settings
from app.model.gemma4_provider import Gemma4CompactProvider
from app.schemas.compact import CompactParseResponse
from app.services.compact_parse import CompactParseService, VLMCompactProvider

router = APIRouter()


def get_vlm_provider(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> VLMCompactProvider | None:
    runtime = getattr(request.app.state, "vlm_runtime", None)
    loader = getattr(request.app.state, "model_loader", None)
    generation_lock = getattr(request.app.state, "generation_lock", None)
    if runtime is None or loader is None:
        return None
    return Gemma4CompactProvider(
        runtime=runtime,
        loader=loader,
        generation_lock=generation_lock,
        generate_kwargs={
            "max_new_tokens": settings.generation_max_new_tokens,
            "do_sample": settings.generation_do_sample,
        },
    )


def get_compact_parse_service(
    settings: Settings = Depends(get_settings),
    vlm_provider: VLMCompactProvider | None = Depends(get_vlm_provider),
) -> CompactParseService:
    return CompactParseService(settings=settings, vlm_provider=vlm_provider)


@router.post("/v1/parse/compact", response_model=CompactParseResponse)
async def parse_compact(
    file: UploadFile = File(...),
    service: CompactParseService = Depends(get_compact_parse_service),
) -> CompactParseResponse:
    data = await file.read()
    return await service.parse_image(data)
