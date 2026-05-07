from app.schemas.compact import CompactParseResponse, CompactStyleProfile
from app.schemas.vlm import VLMCompactOutput


def fuse_deterministic_with_vlm(
    deterministic: CompactParseResponse,
    vlm: VLMCompactOutput | None,
) -> CompactParseResponse:
    if vlm is None:
        return deterministic.model_copy(deep=True)

    deterministic_style = deterministic.style
    vlm_style = vlm.style
    palette = deterministic_style.palette or vlm_style.palette

    fused_style = CompactStyleProfile(
        summary=vlm_style.summary or deterministic_style.summary,
        visual_tone=vlm_style.visual_tone or deterministic_style.visual_tone,
        typography=vlm_style.typography or deterministic_style.typography,
        composition=vlm_style.composition or deterministic_style.composition,
        palette=palette,
    )

    warnings = list(deterministic.warnings)
    for warning in vlm.warnings:
        if warning not in warnings:
            warnings.append(warning)

    return deterministic.model_copy(
        update={
            "style": fused_style,
            "elements": deterministic.elements or vlm.elements,
            "warnings": warnings,
        },
        deep=True,
    )
