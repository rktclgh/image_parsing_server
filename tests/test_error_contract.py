from app.core.errors import AppError
from app.schemas.errors import ErrorCode, ErrorResponse
import pytest


def test_error_response_serializes_stable_shape():
    response = ErrorResponse(
        error_code=ErrorCode.PARSER_BUSY,
        message="Parser is busy",
        request_id="req-123",
        details={"retry_after_seconds": 3},
    )

    assert response.model_dump(mode="json") == {
        "error_code": "PARSER_BUSY",
        "message": "Parser is busy",
        "request_id": "req-123",
        "details": {"retry_after_seconds": 3},
    }


def test_app_error_converts_to_error_response():
    error = AppError(
        ErrorCode.INVALID_IMAGE,
        "Image is invalid",
        status_code=422,
        details={"field": "file"},
    )

    response = error.to_response(request_id="req-456")

    assert error.status_code == 422
    assert response == ErrorResponse(
        error_code=ErrorCode.INVALID_IMAGE,
        message="Image is invalid",
        request_id="req-456",
        details={"field": "file"},
    )


def test_error_response_rejects_unsafe_detail_keys():
    with pytest.raises(ValueError, match="unsafe error detail keys"):
        ErrorResponse(
            error_code=ErrorCode.INTERNAL_ERROR,
            message="Internal error",
            details={"traceback": "hidden", "raw_vlm_output": "hidden"},
        )
