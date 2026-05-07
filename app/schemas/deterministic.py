from pydantic import BaseModel, Field


class ImageMetadata(BaseModel):
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    mime_type: str
    file_size_bytes: int = Field(ge=0)
    alpha_present: bool


class PaletteColor(BaseModel):
    hex: str
    ratio: float = Field(ge=0.0, le=1.0)


class DeterministicAnalysis(BaseModel):
    metadata: ImageMetadata
    palette: list[PaletteColor]
    non_transparent_bbox: list[int] | None = None
    dominant_color_count: int = Field(ge=0)
    aspect_ratio: float = Field(gt=0.0)
