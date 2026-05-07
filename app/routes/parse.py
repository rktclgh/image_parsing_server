from fastapi import APIRouter, Depends, File, UploadFile

from app.core.config import Settings, get_settings
from app.schemas.compact import CompactParseResponse
from app.services.compact_parse import CompactParseService

router = APIRouter()


def get_compact_parse_service(
    settings: Settings = Depends(get_settings),
) -> CompactParseService:
    return CompactParseService(settings=settings)


@router.post("/v1/parse/compact", response_model=CompactParseResponse)
async def parse_compact(
    file: UploadFile = File(...),
    service: CompactParseService = Depends(get_compact_parse_service),
) -> CompactParseResponse:
    data = await file.read()
    return await service.parse_image(data)
