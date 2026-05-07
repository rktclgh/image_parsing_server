import pytest

from app.core.errors import AppError
from app.parsers.vlm_compact import parse_vlm_compact_output
from app.schemas.errors import ErrorCode
from app.schemas.vlm import VLMCompactOutput


VALID_COMPACT_JSON = {
    "asset_type": "logo",
    "content": {
        "subject": "wordmark with circular icon",
        "text": ["ACME"],
        "objects": ["wordmark", "circle icon"],
    },
    "style": {
        "summary": "Minimal technology brand mark",
        "visual_tone": ["clean", "precise"],
        "typography": ["geometric sans"],
        "composition": ["centered horizontal lockup"],
        "palette": [
            {"hex": "#111111", "ratio": 0.7, "role": "foreground"},
            {"hex": "#00AEEF", "ratio": 0.3, "role": "accent"},
        ],
    },
    "elements": [
        {
            "kind": "logo",
            "bbox": [4, 8, 120, 64],
            "text": "ACME",
            "confidence": 0.86,
        }
    ],
    "warnings": ["low resolution preview"],
}


def test_parse_vlm_compact_output_accepts_valid_json():
    parsed = parse_vlm_compact_output(__import__("json").dumps(VALID_COMPACT_JSON))

    assert isinstance(parsed, VLMCompactOutput)
    assert parsed.asset_type == "logo"
    assert parsed.content.subject == "wordmark with circular icon"
    assert parsed.style.palette[1].hex == "#00AEEF"
    assert parsed.elements[0].kind == "logo"


def test_parse_vlm_compact_output_extracts_fenced_json():
    fenced = "Here is the compact parse:\n```json\n" + __import__("json").dumps(VALID_COMPACT_JSON) + "\n```"

    parsed = parse_vlm_compact_output(fenced)

    assert parsed.asset_type == "logo"
    assert parsed.warnings == ["low resolution preview"]


def test_parse_vlm_compact_output_rejects_invalid_json_without_raw_output():
    with pytest.raises(AppError) as exc_info:
        parse_vlm_compact_output("```json\n{not valid json}\n```")

    error = exc_info.value
    assert error.error_code == ErrorCode.VLM_INVALID_JSON
    assert error.status_code == 502
    assert "raw" not in error.details
    assert "vlm" not in error.details
    assert "{not valid json}" not in error.message


def test_compact_parse_response_does_not_expose_raw_vlm_output_field():
    parsed = parse_vlm_compact_output(__import__("json").dumps(VALID_COMPACT_JSON))

    assert "raw_vlm_output" not in parsed.model_dump(mode="json")
