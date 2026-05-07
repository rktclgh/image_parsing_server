from typing import Any, Literal

from pydantic import BaseModel, Field


class CompactParseRequest(BaseModel):
    project_id: str | None = None
    image_ref: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class ParseMetadata(BaseModel):
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    mime_type: str
    file_size_bytes: int | None = Field(default=None, ge=0)
    alpha_present: bool | None = None


class ColorSwatch(BaseModel):
    hex: str
    ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    role: str | None = None


class ParsedElement(BaseModel):
    kind: Literal["text", "logo", "icon", "illustration", "shape", "photo", "unknown"]
    bbox: list[int] = Field(min_length=4, max_length=4)
    text: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class CompactStyleProfile(BaseModel):
    summary: str | None = None
    visual_tone: list[str] = Field(default_factory=list)
    typography: list[str] = Field(default_factory=list)
    composition: list[str] = Field(default_factory=list)
    palette: list[ColorSwatch] = Field(default_factory=list)


class CompactParseResponse(BaseModel):
    request_id: str
    metadata: ParseMetadata
    style: CompactStyleProfile = Field(default_factory=CompactStyleProfile)
    elements: list[ParsedElement] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
