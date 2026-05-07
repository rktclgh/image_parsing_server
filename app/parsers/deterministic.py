from collections import Counter
from collections.abc import Iterable, Iterator

from PIL import Image

from app.core.config import Settings
from app.parsers.image_validation import decode_image_upload
from app.schemas.deterministic import DeterministicAnalysis, PaletteColor


def analyze_image(
    data: bytes,
    *,
    settings: Settings | None = None,
    palette_size: int = 8,
) -> DeterministicAnalysis:
    decoded = decode_image_upload(data, settings=settings)
    image = decoded.image
    rgba_image = image if image.mode == "RGBA" else image.convert("RGBA")
    visible_pixels = _visible_rgb_pixels(rgba_image)
    palette = _extract_palette(visible_pixels, palette_size=palette_size)
    bbox = _non_transparent_bbox(rgba_image) if decoded.metadata.alpha_present else None

    return DeterministicAnalysis(
        metadata=decoded.metadata,
        palette=palette,
        non_transparent_bbox=bbox,
        dominant_color_count=len(palette),
        aspect_ratio=decoded.metadata.width / decoded.metadata.height,
    )


def _visible_rgb_pixels(image: Image.Image) -> Iterator[tuple[int, int, int]]:
    for red, green, blue, alpha in _iter_rgba_pixels(image):
        if alpha > 0:
            yield red, green, blue


def _iter_rgba_pixels(image: Image.Image):
    get_flattened_data = getattr(image, "get_flattened_data", None)
    if get_flattened_data is not None:
        return get_flattened_data()
    return image.getdata()


def _extract_palette(
    pixels: Iterable[tuple[int, int, int]],
    *,
    palette_size: int,
) -> list[PaletteColor]:
    if palette_size <= 0:
        return []

    total = 0
    counts: Counter[tuple[int, int, int]] = Counter()
    for rgb in pixels:
        counts[rgb] += 1
        total += 1

    if total == 0:
        return []

    ranked = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )

    return [
        PaletteColor(hex=_to_hex(rgb), ratio=count / total)
        for rgb, count in ranked[:palette_size]
    ]


def _non_transparent_bbox(image: Image.Image) -> list[int] | None:
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    return list(bbox) if bbox else None


def _to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
