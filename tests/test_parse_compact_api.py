import json
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app.core.config import Settings
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


def test_parse_compact_falls_back_to_deterministic_image_summary_when_vlm_unavailable():
    client = TestClient(
        create_app(settings=Settings(vlm_mode="cold"), model_loader=FailingGemmaLoader())
    )
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
    assert body["warnings"] == ["vlm enrichment failed"]


def test_parse_compact_wires_app_runtime_to_gemma_provider():
    processor = FakeProcessor(
        generated_text=json.dumps(
            {
                "asset_type": "document",
                "style": {
                    "summary": "VLM style read",
                    "visual_tone": ["crisp"],
                    "typography": ["bold sans"],
                    "composition": ["split layout"],
                },
                "warnings": ["vlm low confidence"],
            }
        )
    )
    loader = FakeGemmaLoader(
        processor=processor,
        model=FakeModel(),
    )
    app = create_app(
        settings=Settings(vlm_mode="cold"),
        model_loader=loader,
    )
    data = _palette_image_bytes()

    with TestClient(app) as client:
        response = client.post(
            "/v1/parse/compact",
            files={"file": ("palette.png", data, "image/png")},
        )

    assert response.status_code == 200
    body = response.json()
    assert loader.load_calls == 1
    assert processor.messages[0]["content"][0]["type"] == "image"
    assert body["style"]["summary"] == "VLM style read"
    assert body["style"]["visual_tone"] == ["crisp"]
    assert body["style"]["typography"] == ["bold sans"]
    assert body["style"]["composition"] == ["split layout"]
    assert body["warnings"] == ["vlm low confidence"]


def test_parse_compact_returns_parser_busy_when_generation_lock_is_held():
    loader = FakeGemmaLoader(
        processor=FakeProcessor(
            generated_text=json.dumps(
                {
                    "asset_type": "document",
                    "style": {"summary": "should not generate while busy"},
                }
            )
        ),
        model=FakeModel(),
    )
    app = create_app(
        settings=Settings(vlm_mode="cold", max_concurrent_generations=1),
        model_loader=loader,
    )
    data = _palette_image_bytes()

    with TestClient(app) as client:
        lease = app.state.generation_lock.acquire(blocking=False)
        try:
            response = client.post(
                "/v1/parse/compact",
                files={"file": ("palette.png", data, "image/png")},
                headers={"x-request-id": "req-busy"},
            )
        finally:
            lease.release()

    assert response.status_code == 503
    assert response.json() == {
        "error_code": "PARSER_BUSY",
        "message": "parser is busy",
        "request_id": "req-busy",
        "details": {"retry_after_seconds": 1},
    }
    assert loader.load_calls == 0


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


class FailingGemmaLoader:
    model = None
    processor = None

    @property
    def loaded(self):
        return False

    def load(self):
        raise RuntimeError("fake model unavailable")

    def unload(self):
        return None


class FakeGemmaLoader:
    def __init__(self, *, processor, model):
        self._processor = processor
        self._model = model
        self.processor = None
        self.model = None
        self.load_calls = 0
        self.unload_calls = 0

    @property
    def loaded(self):
        return self.processor is not None and self.model is not None

    def load(self):
        self.load_calls += 1
        self.processor = self._processor
        self.model = self._model
        return self

    def unload(self):
        self.unload_calls += 1
        self.processor = None
        self.model = None


class FakeProcessor:
    def __init__(self, *, generated_text: str):
        self.generated_text = generated_text
        self.messages = None
        self.decoded_ids = None

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt,
        tokenize,
        return_dict,
        return_tensors,
    ):
        self.messages = messages
        assert add_generation_prompt is True
        assert tokenize is True
        assert return_dict is True
        assert return_tensors == "pt"
        return FakeInputs({"input_ids": [[1]]})

    def batch_decode(self, generated_ids, *, skip_special_tokens):
        self.decoded_ids = generated_ids
        assert skip_special_tokens is True
        return [self.generated_text]


class FakeInputs(dict):
    def to(self, device):
        assert device is not None
        return self


class FakeModel:
    device = "cuda:0"

    def generate(self, **inputs):
        assert inputs == {"input_ids": [[1]]}
        return [[1, 2]]
