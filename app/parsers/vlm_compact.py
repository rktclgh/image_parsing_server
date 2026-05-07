import json
import re

from pydantic import ValidationError

from app.core.errors import AppError
from app.schemas.errors import ErrorCode
from app.schemas.vlm import VLMCompactOutput

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def parse_vlm_compact_output(model_text: str) -> VLMCompactOutput:
    """Parse compact VLM JSON text without importing model/runtime packages."""
    try:
        payload = _extract_json_object(model_text)
        data = json.loads(payload)
        return VLMCompactOutput.model_validate(data)
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
        raise AppError(
            ErrorCode.VLM_INVALID_JSON,
            "VLM output did not match the compact JSON contract.",
            status_code=502,
            details={"reason": exc.__class__.__name__},
        ) from exc


def _extract_json_object(model_text: str) -> str:
    if not isinstance(model_text, str) or not model_text.strip():
        raise ValueError("empty model output")

    fenced = _JSON_FENCE_RE.search(model_text)
    if fenced:
        return fenced.group(1).strip()

    start = model_text.find("{")
    if start == -1:
        raise ValueError("missing JSON object")

    decoder = json.JSONDecoder()
    _, end = decoder.raw_decode(model_text[start:])
    return model_text[start : start + end]
