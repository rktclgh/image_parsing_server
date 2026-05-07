from enum import StrEnum
from typing import TypeAlias

from pydantic import BaseModel, Field, field_validator


class ErrorCode(StrEnum):
    PARSER_BUSY = "PARSER_BUSY"
    INVALID_IMAGE = "INVALID_IMAGE"
    UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    IMAGE_DECODE_FAILED = "IMAGE_DECODE_FAILED"
    MODEL_NOT_READY = "MODEL_NOT_READY"
    VLM_TIMEOUT = "VLM_TIMEOUT"
    VLM_OOM = "VLM_OOM"
    VLM_INVALID_JSON = "VLM_INVALID_JSON"
    INTERNAL_ERROR = "INTERNAL_ERROR"


SafeErrorValue: TypeAlias = str | int | float | bool | None
SafeErrorDetails: TypeAlias = dict[str, SafeErrorValue | list[SafeErrorValue]]

UNSAFE_DETAIL_KEYWORDS = (
    "exception",
    "prompt",
    "raw",
    "stack",
    "trace",
    "traceback",
    "vlm",
)


class ErrorResponse(BaseModel):
    error_code: ErrorCode
    message: str
    request_id: str | None = None
    details: SafeErrorDetails = Field(default_factory=dict)

    @field_validator("details")
    @classmethod
    def reject_unsafe_detail_keys(cls, details: SafeErrorDetails) -> SafeErrorDetails:
        unsafe_keys = [
            key
            for key in details
            if any(keyword in key.lower() for keyword in UNSAFE_DETAIL_KEYWORDS)
        ]
        if unsafe_keys:
            keys = ", ".join(sorted(unsafe_keys))
            raise ValueError(f"unsafe error detail keys: {keys}")
        return details
