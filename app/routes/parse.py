from uuid import uuid4

from fastapi import APIRouter, Depends, File, UploadFile

from app.core.config import Settings, get_settings
from app.parsers.deterministic import analyze_image
from app.schemas.compact import ColorSwatch, CompactParseResponse, CompactStyleProfile, ParseMetadata
from app.schemas.deterministic import DeterministicAnalysis, PaletteColor

router = APIRouter()


@router.post("/v1/parse/compact", response_model=CompactParseResponse)
async def parse_compact(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
) -> CompactParseResponse:
    data = await file.read()
    analysis = analyze_image(data, settings=settings)
    return _compact_response_from_analysis(analysis)


def _compact_response_from_analysis(analysis: DeterministicAnalysis) -> CompactParseResponse:
    return CompactParseResponse(
        request_id=str(uuid4()),
        metadata=ParseMetadata(
            width=analysis.metadata.width,
            height=analysis.metadata.height,
            mime_type=analysis.metadata.mime_type,
            file_size_bytes=analysis.metadata.file_size_bytes,
            alpha_present=analysis.metadata.alpha_present,
        ),
        style=CompactStyleProfile(
            summary="deterministic image analysis",
            composition=[f"aspect_ratio:{analysis.aspect_ratio:.4f}"],
            palette=[
                _color_swatch_from_palette_color(color, index)
                for index, color in enumerate(analysis.palette)
            ],
        ),
    )


def _color_swatch_from_palette_color(color: PaletteColor, index: int) -> ColorSwatch:
    return ColorSwatch(
        hex=color.hex,
        ratio=color.ratio,
        role="dominant" if index == 0 else "accent",
    )
