import asyncio
import json
from base64 import b64encode
from unittest.mock import ANY

import pytest

from app.core.errors import AppError
from app.model.gemma4_provider import (
    Gemma4CompactProvider,
    build_compact_messages,
    image_bytes_to_data_url,
)
from app.schemas.compact import (
    ColorSwatch,
    CompactParseResponse,
    CompactStyleProfile,
    ParseMetadata,
)
from app.schemas.errors import ErrorCode


PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-preview"


def _deterministic_response() -> CompactParseResponse:
    return CompactParseResponse(
        request_id="req-provider",
        metadata=ParseMetadata(
            width=800,
            height=600,
            mime_type="image/png",
            file_size_bytes=len(PNG_BYTES),
            alpha_present=True,
        ),
        style=CompactStyleProfile(
            summary="deterministic image analysis",
            composition=["aspect_ratio:1.3333"],
            palette=[
                ColorSwatch(hex="#111111", ratio=0.6, role="dominant"),
                ColorSwatch(hex="#ffffff", ratio=0.4, role="accent"),
            ],
        ),
    )


def test_image_bytes_to_data_url_uses_detected_mime_and_base64_payload():
    data_url = image_bytes_to_data_url(PNG_BYTES, mime_type="image/png")

    assert data_url.startswith("data:image/png;base64,")
    assert "fake-preview" not in data_url
    assert data_url == "data:image/png;base64," + b64encode(PNG_BYTES).decode("ascii")


def test_build_compact_messages_places_image_before_text_with_context_hints():
    messages = build_compact_messages(
        image_bytes=PNG_BYTES,
        deterministic=_deterministic_response(),
        prompt_text="Return compact JSON only.",
    )

    assert messages == [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": "data:image/png;base64,iVBORw0KGgpmYWtlLXByZXZpZXc=",
                },
                {
                    "type": "text",
                    "text": ANY,
                },
            ],
        }
    ]
    text = messages[0]["content"][1]["text"]
    assert text.startswith("Return compact JSON only.")
    assert "width: 800" in text
    assert "height: 600" in text
    assert "mime_type: image/png" in text
    assert "alpha_present: true" in text
    assert "#111111" in text
    assert "#ffffff" in text


def test_gemma4_compact_provider_generates_and_parses_compact_output():
    loader = FakeLoader(
        processor=FakeProcessor(
            generated_text=json.dumps(
                {
                    "asset_type": "logo",
                    "style": {"summary": "soft cloud terminal icon"},
                    "warnings": ["low contrast text"],
                }
            )
        ),
        model=FakeModel(),
    )
    runtime = FakeRuntime(loader)
    provider = Gemma4CompactProvider(runtime=runtime, loader=loader, prompt_text="Analyze.")

    output = asyncio.run(provider(PNG_BYTES, _deterministic_response()))

    assert output.asset_type == "logo"
    assert output.style.summary == "soft cloud terminal icon"
    assert output.warnings == ["low contrast text"]
    assert runtime.ensure_loaded_calls == 1
    assert runtime.after_request_calls == 1
    assert loader.processor.messages[0]["content"][0]["type"] == "image"
    assert loader.processor.messages[0]["content"][1]["type"] == "text"
    assert loader.model.generated_inputs == {"input_ids": [1]}


def test_gemma4_compact_provider_rejects_missing_model_binding_safely():
    loader = FakeLoader(processor=FakeProcessor(generated_text="{}"), model=None)
    provider = Gemma4CompactProvider(
        runtime=FakeRuntime(loader),
        loader=loader,
        prompt_text="Analyze.",
    )

    with pytest.raises(AppError) as exc_info:
        asyncio.run(provider(PNG_BYTES, _deterministic_response()))

    error = exc_info.value
    assert error.error_code == ErrorCode.MODEL_NOT_READY
    assert error.status_code == 503
    assert "raw" not in error.details
    assert "prompt" not in error.details


class FakeRuntime:
    def __init__(self, loader):
        self.loader = loader
        self.ensure_loaded_calls = 0
        self.after_request_calls = 0

    def ensure_loaded(self):
        self.ensure_loaded_calls += 1

    def after_request(self):
        self.after_request_calls += 1


class FakeLoader:
    def __init__(self, *, processor, model):
        self.processor = processor
        self.model = model


class FakeProcessor:
    def __init__(self, *, generated_text: str):
        self.generated_text = generated_text
        self.messages = None

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
        return FakeInputs({"input_ids": [1]})

    def batch_decode(self, generated_ids, *, skip_special_tokens):
        assert generated_ids == [[2]]
        assert skip_special_tokens is True
        return [self.generated_text]


class FakeInputs(dict):
    def to(self, _device):
        return self


class FakeModel:
    device = "cuda:0"

    def __init__(self):
        self.generated_inputs = None

    def generate(self, **inputs):
        self.generated_inputs = inputs
        return [[2]]
