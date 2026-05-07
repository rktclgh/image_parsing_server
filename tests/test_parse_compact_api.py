from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app.main import create_app
from app.routes.parse import get_compact_parse_service
from app.schemas.compact import CompactParseResponse, CompactStyleProfile, ParseMetadata


def _image_bytes(mode: str, size: tuple[int, int], color, image_format: str) -> bytes:
    image = Image.new(mode, size, color)
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def _palette_image_bytes() -> bytes:
    image = Image.new("RGB", (3, 1))
    image.putdata([(255, 0, 0), (255, 0, 0), (0, 0, 255)])
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_parse_compact_returns_deterministic_image_summary():
    client = TestClient(create_app())
    data = _palette_image_bytes()

    response = client.post(
        "/v1/parse/compact",
        files={"file": ("palette.png", data, "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"]
    assert body["metadata"] == {
        "width": 3,
        "height": 1,
        "mime_type": "image/png",
        "file_size_bytes": len(data),
        "alpha_present": False,
    }
    assert body["style"]["summary"] == "deterministic image analysis"
    assert body["style"]["visual_tone"] == []
    assert body["style"]["typography"] == []
    assert body["style"]["composition"] == ["aspect_ratio:3.0000"]
    assert body["style"]["palette"][0] == {
        "hex": "#ff0000",
        "ratio": pytest.approx(2 / 3),
        "role": "dominant",
    }
    assert body["style"]["palette"][1] == {
        "hex": "#0000ff",
        "ratio": pytest.approx(1 / 3),
        "role": "accent",
    }
    assert body["elements"] == []
    assert body["warnings"] == []


def test_parse_compact_delegates_to_compact_parse_service():
    app = create_app()
    captured = {}

    class FakeCompactParseService:
        async def parse_image(self, data: bytes) -> CompactParseResponse:
            captured["data"] = data
            return CompactParseResponse(
                request_id="req-fake",
                metadata=ParseMetadata(width=9, height=4, mime_type="image/png"),
                style=CompactStyleProfile(summary="fake service response"),
            )

    app.dependency_overrides[get_compact_parse_service] = FakeCompactParseService
    client = TestClient(app)
    data = _palette_image_bytes()

    response = client.post(
        "/v1/parse/compact",
        files={"file": ("palette.png", data, "image/png")},
    )

    assert captured["data"] == data
    assert response.status_code == 200
    assert response.json()["request_id"] == "req-fake"
    assert response.json()["style"]["summary"] == "fake service response"


def test_parse_compact_reports_app_errors_with_stable_shape():
    client = TestClient(create_app())

    response = client.post(
        "/v1/parse/compact",
        files={"file": ("empty.png", b"", "image/png")},
        headers={"x-request-id": "req-test"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "error_code": "INVALID_IMAGE",
        "message": "Image upload is empty",
        "request_id": "req-test",
        "details": {},
    }


def test_parse_compact_rejects_unsupported_image_types():
    client = TestClient(create_app())
    data = _image_bytes("RGB", (2, 2), (10, 20, 30), "GIF")

    response = client.post(
        "/v1/parse/compact",
        files={"file": ("animated.gif", data, "image/gif")},
    )

    assert response.status_code == 415
    assert response.json()["error_code"] == "UNSUPPORTED_MEDIA_TYPE"
