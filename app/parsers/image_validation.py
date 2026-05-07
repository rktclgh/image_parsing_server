from dataclasses import dataclass
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.schemas.deterministic import ImageMetadata
from app.schemas.errors import ErrorCode


SUPPORTED_FORMAT_MIME_TYPES = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
}

SUPPORTED_SIGNATURES = (
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"RIFF",
)


@dataclass(frozen=True)
class DecodedImage:
    image: Image.Image
    metadata: ImageMetadata


def validate_image_upload(
    data: bytes,
    *,
    settings: Settings | None = None,
) -> ImageMetadata:
    return decode_image_upload(data, settings=settings).metadata


def decode_image_upload(
    data: bytes,
    *,
    settings: Settings | None = None,
) -> DecodedImage:
    active_settings = settings or get_settings()
    file_size_bytes = len(data)

    if file_size_bytes == 0:
        raise _app_error(ErrorCode.INVALID_IMAGE, "Image upload is empty", status_code=422)

    if file_size_bytes > active_settings.max_upload_bytes:
        raise _app_error(
            ErrorCode.IMAGE_TOO_LARGE,
            "Image upload exceeds the configured byte limit",
            status_code=413,
            details={
                "file_size_bytes": file_size_bytes,
                "max_upload_bytes": active_settings.max_upload_bytes,
            },
        )

    try:
        with Image.open(BytesIO(data)) as opened:
            image_format = opened.format
            if image_format not in SUPPORTED_FORMAT_MIME_TYPES:
                raise _app_error(
                    ErrorCode.UNSUPPORTED_MEDIA_TYPE,
                    "Only PNG, JPEG, and WebP images are supported",
                    status_code=415,
                    details={"detected_format": image_format or "unknown"},
                )

            _enforce_pixel_limit(
                opened.width,
                opened.height,
                max_decoded_pixels=active_settings.max_decoded_pixels,
            )
            image = ImageOps.exif_transpose(opened)
            image.load()
    except AppError:
        raise
    except Image.DecompressionBombError as exc:
        raise _app_error(
            ErrorCode.IMAGE_TOO_LARGE,
            "Image exceeds the configured decoded pixel limit",
            status_code=413,
        ) from exc
    except UnidentifiedImageError as exc:
        error_code = (
            ErrorCode.IMAGE_DECODE_FAILED
            if _has_supported_signature(data)
            else ErrorCode.INVALID_IMAGE
        )
        raise _app_error(
            error_code,
            "Image could not be decoded",
            status_code=422,
        ) from exc
    except OSError as exc:
        raise _app_error(
            ErrorCode.IMAGE_DECODE_FAILED,
            "Image could not be decoded",
            status_code=422,
        ) from exc

    width, height = image.size
    metadata = ImageMetadata(
        width=width,
        height=height,
        mime_type=SUPPORTED_FORMAT_MIME_TYPES[image_format],
        file_size_bytes=file_size_bytes,
        alpha_present=_has_alpha(image),
    )
    return DecodedImage(image=image, metadata=metadata)


def _has_supported_signature(data: bytes) -> bool:
    if data.startswith(SUPPORTED_SIGNATURES[:2]):
        return True
    return data.startswith(b"RIFF") and data[8:12] == b"WEBP"


def _has_alpha(image: Image.Image) -> bool:
    return "A" in image.getbands() or "transparency" in image.info


def _enforce_pixel_limit(
    width: int,
    height: int,
    *,
    max_decoded_pixels: int,
) -> None:
    decoded_pixels = width * height
    if decoded_pixels > max_decoded_pixels:
        raise _app_error(
            ErrorCode.IMAGE_TOO_LARGE,
            "Image exceeds the configured decoded pixel limit",
            status_code=413,
            details={
                "decoded_pixels": decoded_pixels,
                "max_decoded_pixels": max_decoded_pixels,
            },
        )


def _app_error(
    error_code: ErrorCode,
    message: str,
    *,
    status_code: int,
    details: dict[str, str | int] | None = None,
) -> AppError:
    return AppError(error_code, message, status_code=status_code, details=details)
