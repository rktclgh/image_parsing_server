from io import BytesIO
import asyncio

from PIL import Image
import pytest

from app.core.config import Settings
from app.core.errors import AppError
from app.schemas.errors import ErrorCode
from app.schemas.vlm import VLMCompactOutput
from app.services.compact_parse import CompactParseService


def _palette_image_bytes() -> bytes:
    image = Image.new("RGB", (3, 1))
    image.putdata([(255, 0, 0), (255, 0, 0), (0, 0, 255)])
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_compact_parse_service_builds_deterministic_response_contract():
    data = _palette_image_bytes()
    service = CompactParseService(
        settings=Settings(),
        request_id_factory=lambda: "req-fixed",
    )

    response = asyncio.run(service.parse_image(data))

    assert response.request_id == "req-fixed"
    assert response.metadata.width == 3
    assert response.metadata.height == 1
    assert response.metadata.mime_type == "image/png"
    assert response.metadata.file_size_bytes == len(data)
    assert response.style.summary == "deterministic image analysis"
    assert response.style.composition == ["aspect_ratio:3.0000"]
    assert [(swatch.hex, swatch.role) for swatch in response.style.palette] == [
        ("#ff0000", "dominant"),
        ("#0000ff", "accent"),
    ]
    assert response.elements == []
    assert response.warnings == []


def test_compact_parse_service_fuses_optional_vlm_compact_output():
    data = _palette_image_bytes()

    async def vlm_provider(_data, deterministic):
        assert deterministic.style.palette
        return VLMCompactOutput.model_validate(
            {
                "asset_type": "logo",
                "style": {
                    "summary": "VLM style read",
                    "visual_tone": ["playful"],
                    "typography": ["rounded"],
                    "composition": ["centered mark"],
                    "palette": [{"hex": "#123456", "role": "model"}],
                },
                "warnings": ["vlm low confidence"],
            }
        )

    service = CompactParseService(
        settings=Settings(),
        request_id_factory=lambda: "req-fused",
        vlm_provider=vlm_provider,
    )

    response = asyncio.run(service.parse_image(data))

    assert response.request_id == "req-fused"
    assert response.style.summary == "VLM style read"
    assert response.style.visual_tone == ["playful"]
    assert response.style.typography == ["rounded"]
    assert response.style.composition == ["centered mark"]
    assert [swatch.hex for swatch in response.style.palette] == ["#ff0000", "#0000ff"]
    assert response.warnings == ["vlm low confidence"]


def test_compact_parse_service_falls_back_when_optional_vlm_provider_fails():
    data = _palette_image_bytes()

    async def vlm_provider(_data, _deterministic):
        raise RuntimeError("model unavailable")

    service = CompactParseService(
        settings=Settings(),
        request_id_factory=lambda: "req-fallback",
        vlm_provider=vlm_provider,
    )

    response = asyncio.run(service.parse_image(data))

    assert response.request_id == "req-fallback"
    assert response.style.summary == "deterministic image analysis"
    assert [swatch.hex for swatch in response.style.palette] == ["#ff0000", "#0000ff"]
    assert response.warnings == ["vlm enrichment failed"]


@pytest.mark.parametrize(
    "error_code",
    [
        ErrorCode.PARSER_BUSY,
        ErrorCode.MODEL_NOT_READY,
        ErrorCode.VLM_TIMEOUT,
        ErrorCode.VLM_OOM,
        ErrorCode.VLM_INVALID_JSON,
    ],
)
def test_compact_parse_service_propagates_typed_vlm_errors(error_code):
    data = _palette_image_bytes()

    async def vlm_provider(_data, _deterministic):
        raise AppError(
            error_code,
            "typed VLM failure",
            status_code=503,
            details={"component": "vlm_provider"},
        )

    service = CompactParseService(
        settings=Settings(),
        request_id_factory=lambda: "req-busy",
        vlm_provider=vlm_provider,
    )

    with pytest.raises(AppError) as exc_info:
        asyncio.run(service.parse_image(data))

    error = exc_info.value
    assert error.error_code == error_code
    assert error.status_code == 503
    assert error.details == {"component": "vlm_provider"}
