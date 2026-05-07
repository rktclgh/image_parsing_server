from collections.abc import Awaitable, Callable
from uuid import uuid4

from fastapi.concurrency import run_in_threadpool

from app.core.config import Settings, get_settings
from app.parsers.deterministic import analyze_image
from app.parsers.fusion import fuse_deterministic_with_vlm
from app.schemas.compact import (
    ColorSwatch,
    CompactParseResponse,
    CompactStyleProfile,
    ParseMetadata,
)
from app.schemas.deterministic import DeterministicAnalysis, PaletteColor
from app.schemas.vlm import VLMCompactOutput

VLMCompactProvider = Callable[
    [bytes, CompactParseResponse],
    Awaitable[VLMCompactOutput | None],
]


class CompactParseService:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        request_id_factory: Callable[[], str] | None = None,
        vlm_provider: VLMCompactProvider | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._request_id_factory = request_id_factory or (lambda: str(uuid4()))
        self._vlm_provider = vlm_provider

    async def parse_image(self, data: bytes) -> CompactParseResponse:
        analysis = await run_in_threadpool(analyze_image, data, settings=self._settings)
        deterministic = self._compact_response_from_analysis(analysis)
        if self._vlm_provider is None:
            return deterministic

        try:
            vlm_output = await self._vlm_provider(data, deterministic)
            return fuse_deterministic_with_vlm(deterministic, vlm_output)
        except Exception:
            return _with_warning(deterministic, "vlm enrichment failed")

    def _compact_response_from_analysis(
        self,
        analysis: DeterministicAnalysis,
    ) -> CompactParseResponse:
        return CompactParseResponse(
            request_id=self._request_id_factory(),
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


def _with_warning(response: CompactParseResponse, warning: str) -> CompactParseResponse:
    warnings = list(response.warnings)
    if warning not in warnings:
        warnings.append(warning)
    return response.model_copy(update={"warnings": warnings}, deep=True)
