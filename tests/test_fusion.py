from app.parsers.fusion import fuse_deterministic_with_vlm
from app.schemas.compact import (
    ColorSwatch,
    CompactParseResponse,
    CompactStyleProfile,
    ParsedElement,
    ParseMetadata,
)
from app.schemas.vlm import VLMCompactOutput


def test_fusion_preserves_deterministic_metadata_and_palette_precedence():
    deterministic = CompactParseResponse(
        request_id="req-1",
        metadata=ParseMetadata(width=320, height=180, mime_type="image/png"),
        style=CompactStyleProfile(
            summary="deterministic summary",
            palette=[ColorSwatch(hex="#FFFFFF", ratio=0.6, role="background")],
        ),
        elements=[ParsedElement(kind="logo", bbox=(0, 0, 120, 80), text="detected")],
    )
    vlm = VLMCompactOutput.model_validate(
        {
            "asset_type": "branding",
            "content": {"subject": "launch badge", "text": ["Launch"]},
            "style": {
                "summary": "VLM interpretive summary",
                "visual_tone": ["energetic"],
                "typography": ["bold condensed"],
                "composition": ["radial badge"],
                "palette": [{"hex": "#FF0000", "ratio": 1.0, "role": "accent"}],
            },
            "elements": [{"kind": "text", "bbox": [10, 20, 90, 40], "text": "Launch"}],
        }
    )

    fused = fuse_deterministic_with_vlm(deterministic, vlm)

    assert fused.metadata == deterministic.metadata
    assert [swatch.hex for swatch in fused.style.palette] == ["#FFFFFF"]
    assert fused.style.summary == "VLM interpretive summary"
    assert fused.style.visual_tone == ["energetic"]
    assert fused.style.typography == ["bold condensed"]
    assert fused.style.composition == ["radial badge"]
    assert fused.elements == deterministic.elements


def test_fusion_uses_vlm_palette_only_when_deterministic_palette_missing():
    deterministic = CompactParseResponse(
        request_id="req-2",
        metadata=ParseMetadata(width=200, height=200, mime_type="image/jpeg"),
    )
    vlm = VLMCompactOutput.model_validate(
        {
            "asset_type": "character",
            "content": {"subject": "mascot"},
            "style": {"palette": [{"hex": "#123456", "role": "primary"}]},
        }
    )

    fused = fuse_deterministic_with_vlm(deterministic, vlm)

    assert [swatch.hex for swatch in fused.style.palette] == ["#123456"]
    assert fused.metadata.mime_type == "image/jpeg"
