from app.schemas.compact import CompactParseRequest, CompactParseResponse, ParsedElement, ParseMetadata
from app.schemas.errors import ErrorCode
from app.schemas.status import ModelStatus, ModelStatusResponse


def test_error_code_enum_contains_v1_direct_error_contract():
    assert {code.value for code in ErrorCode} == {
        "PARSER_BUSY",
        "INVALID_IMAGE",
        "UNSUPPORTED_MEDIA_TYPE",
        "IMAGE_TOO_LARGE",
        "IMAGE_DECODE_FAILED",
        "MODEL_NOT_READY",
        "VLM_TIMEOUT",
        "VLM_OOM",
        "VLM_INVALID_JSON",
        "INTERNAL_ERROR",
    }


def test_model_status_response_serializes_contract():
    response = ModelStatusResponse(
        status=ModelStatus.READY,
        model_id="google/gemma-4-E4B-it",
        quantization="8bit",
        max_concurrent_generations=1,
        loaded=True,
    )

    assert response.model_dump(mode="json") == {
        "status": "ready",
        "model_id": "google/gemma-4-E4B-it",
        "quantization": "8bit",
        "max_concurrent_generations": 1,
        "loaded": True,
        "detail": None,
    }


def test_compact_parse_contract_excludes_raw_debug_by_default():
    request = CompactParseRequest(project_id="project-1", image_ref="upload://image.png")
    response = CompactParseResponse(
        request_id="req-789",
        metadata=ParseMetadata(width=100, height=200, mime_type="image/png"),
        elements=[ParsedElement(kind="text", bbox=[0, 0, 50, 20], text="Hello")],
    )

    assert request.model_dump(mode="json") == {
        "project_id": "project-1",
        "image_ref": "upload://image.png",
        "options": {},
    }
    assert "raw_vlm_output" not in response.model_dump(mode="json")
    assert response.elements[0].bbox == [0, 0, 50, 20]
