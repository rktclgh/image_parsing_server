from io import BytesIO

from PIL import Image

from app.parsers.deterministic import analyze_image


def _png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_analyze_image_extracts_palette_ratios_from_opaque_pixels():
    image = Image.new("RGB", (4, 1))
    image.putdata(
        [
            (255, 0, 0),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
        ]
    )

    analysis = analyze_image(_png_bytes(image), palette_size=3)

    assert [(color.hex, color.ratio) for color in analysis.palette] == [
        ("#ff0000", 0.5),
        ("#0000ff", 0.25),
        ("#00ff00", 0.25),
    ]
    assert analysis.dominant_color_count == 3
    assert analysis.aspect_ratio == 4.0
    assert analysis.metadata.alpha_present is False
    assert analysis.non_transparent_bbox is None


def test_analyze_image_ignores_fully_transparent_pixels_and_reports_alpha_bbox():
    image = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    image.putpixel((1, 1), (255, 0, 0, 255))
    image.putpixel((2, 1), (255, 0, 0, 128))
    image.putpixel((1, 2), (0, 0, 255, 255))

    analysis = analyze_image(_png_bytes(image), palette_size=2)

    assert analysis.metadata.alpha_present is True
    assert analysis.non_transparent_bbox == [1, 1, 3, 3]
    assert [(color.hex, color.ratio) for color in analysis.palette] == [
        ("#ff0000", 2 / 3),
        ("#0000ff", 1 / 3),
    ]
    assert analysis.dominant_color_count == 2


def test_analyze_image_handles_fully_transparent_images_consistently():
    image = Image.new("RGBA", (2, 2), (255, 0, 0, 0))

    analysis = analyze_image(_png_bytes(image))

    assert analysis.metadata.alpha_present is True
    assert analysis.non_transparent_bbox is None
    assert analysis.palette == []
    assert analysis.dominant_color_count == 0
