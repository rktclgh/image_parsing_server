from io import BytesIO

import pytest
from PIL import Image

from app.core.config import Settings
from app.core.errors import AppError
from app.parsers.image_validation import validate_image_upload
from app.schemas.errors import ErrorCode


def _image_bytes(mode: str, size: tuple[int, int], color, image_format: str) -> bytes:
    image = Image.new(mode, size, color)
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("image_format", "expected_mime"),
    [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")],
)
def test_validate_image_upload_accepts_supported_media(image_format, expected_mime):
    data = _image_bytes("RGB", (3, 2), (10, 20, 30), image_format)

    metadata = validate_image_upload(data)

    assert metadata.width == 3
    assert metadata.height == 2
    assert metadata.mime_type == expected_mime
    assert metadata.file_size_bytes == len(data)
    assert metadata.alpha_present is False


def test_validate_image_upload_rejects_unsupported_media_type():
    data = _image_bytes("RGB", (2, 2), (10, 20, 30), "GIF")

    with pytest.raises(AppError) as exc_info:
        validate_image_upload(data)

    assert exc_info.value.error_code == ErrorCode.UNSUPPORTED_MEDIA_TYPE


def test_validate_image_upload_enforces_upload_byte_limit_before_decode():
    data = _image_bytes("RGB", (2, 2), (10, 20, 30), "PNG")
    settings = Settings(max_upload_bytes=len(data) - 1)

    with pytest.raises(AppError) as exc_info:
        validate_image_upload(data, settings=settings)

    assert exc_info.value.error_code == ErrorCode.IMAGE_TOO_LARGE


def test_validate_image_upload_enforces_decoded_pixel_limit():
    data = _image_bytes("RGB", (2, 2), (10, 20, 30), "PNG")
    settings = Settings(max_decoded_pixels=3)

    with pytest.raises(AppError) as exc_info:
        validate_image_upload(data, settings=settings)

    assert exc_info.value.error_code == ErrorCode.IMAGE_TOO_LARGE


def test_validate_image_upload_maps_pillow_decompression_bomb_to_app_error(monkeypatch):
    data = _image_bytes("RGB", (2, 2), (10, 20, 30), "PNG")

    def raise_decompression_bomb(_buffer):
        raise Image.DecompressionBombError("too many pixels")

    monkeypatch.setattr("app.parsers.image_validation.Image.open", raise_decompression_bomb)

    with pytest.raises(AppError) as exc_info:
        validate_image_upload(data)

    assert exc_info.value.error_code == ErrorCode.IMAGE_TOO_LARGE
    assert exc_info.value.status_code == 413


def test_validate_image_upload_reports_decode_failures_for_corrupt_supported_header():
    data = b"\x89PNG\r\n\x1a\nnot-a-real-png"

    with pytest.raises(AppError) as exc_info:
        validate_image_upload(data)

    assert exc_info.value.error_code == ErrorCode.IMAGE_DECODE_FAILED


def test_validate_image_upload_rejects_empty_payload_as_invalid_image():
    with pytest.raises(AppError) as exc_info:
        validate_image_upload(b"")

    assert exc_info.value.error_code == ErrorCode.INVALID_IMAGE


def test_validate_image_upload_reports_alpha_presence():
    data = _image_bytes("RGBA", (2, 2), (10, 20, 30, 128), "PNG")

    metadata = validate_image_upload(data)

    assert metadata.alpha_present is True


def test_validate_image_upload_normalizes_exif_orientation():
    image = Image.new("RGB", (2, 3), (10, 20, 30))
    exif = Image.Exif()
    exif[274] = 6
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif)

    metadata = validate_image_upload(buffer.getvalue())

    assert metadata.width == 3
    assert metadata.height == 2
