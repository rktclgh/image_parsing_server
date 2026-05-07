from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.compact import ColorSwatch, ParsedElement


class VLMContentLayer(BaseModel):
    subject: str | None = None
    text: list[str] = Field(default_factory=list)
    objects: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class VLMStyleLayer(BaseModel):
    summary: str | None = None
    visual_tone: list[str] = Field(default_factory=list)
    typography: list[str] = Field(default_factory=list)
    composition: list[str] = Field(default_factory=list)
    palette: list[ColorSwatch] = Field(default_factory=list)


class VLMCompactOutput(BaseModel):
    asset_type: Literal[
        "logo",
        "character",
        "branding",
        "ui",
        "illustration",
        "photo",
        "document",
        "unknown",
    ] = "unknown"
    content: VLMContentLayer = Field(default_factory=VLMContentLayer)
    style: VLMStyleLayer = Field(default_factory=VLMStyleLayer)
    elements: list[ParsedElement] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
