import asyncio
import json
import threading
from base64 import b64encode
from unittest.mock import ANY

import pytest

from app.core.errors import AppError
from app.model.generation_lock import GenerationLock
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
    caller_thread_id = threading.get_ident()

    output = asyncio.run(provider(PNG_BYTES, _deterministic_response()))

    assert output.asset_type == "logo"
    assert output.style.summary == "soft cloud terminal icon"
    assert output.warnings == ["low contrast text"]
    assert runtime.ensure_loaded_calls == 1
    assert runtime.after_request_calls == 1
    assert runtime.ensure_thread_ids[0] != caller_thread_id
    assert runtime.after_request_thread_ids[0] != caller_thread_id
    assert loader.processor.messages[0]["content"][0]["type"] == "image"
    assert loader.processor.messages[0]["content"][1]["type"] == "text"
    assert loader.model.generated_inputs == {"input_ids": [[1]]}
    assert loader.processor.decoded_ids == [[2]]


def test_gemma4_compact_provider_decodes_only_new_tokens_after_prompt():
    loader = FakeLoader(
        processor=FakeProcessor(
            generated_text=json.dumps(
                {
                    "asset_type": "branding",
                    "style": {"summary": "trimmed assistant output"},
                }
            ),
            input_ids=[[101, 102]],
        ),
        model=FakeModel(generated_ids=[[101, 102, 201, 202]]),
    )
    provider = Gemma4CompactProvider(
        runtime=FakeRuntime(loader),
        loader=loader,
        prompt_text="Analyze with {JSON schema example}.",
    )

    output = asyncio.run(provider(PNG_BYTES, _deterministic_response()))

    assert output.asset_type == "branding"
    assert output.style.summary == "trimmed assistant output"
    assert loader.processor.decoded_ids == [[201, 202]]


def test_gemma4_compact_provider_does_not_move_inputs_without_model_device():
    loader = FakeLoader(
        processor=FakeProcessor(
            generated_text=json.dumps(
                {
                    "asset_type": "illustration",
                    "style": {"summary": "model without device attr"},
                }
            )
        ),
        model=FakeModelWithoutDevice(),
    )
    provider = Gemma4CompactProvider(
        runtime=FakeRuntime(loader),
        loader=loader,
        prompt_text="Analyze.",
    )

    output = asyncio.run(provider(PNG_BYTES, _deterministic_response()))

    assert output.asset_type == "illustration"
    assert output.style.summary == "model without device attr"


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


def test_gemma4_compact_provider_rejects_when_generation_lock_is_busy():
    loader = FakeLoader(
        processor=FakeProcessor(generated_text="{}"),
        model=FakeModel(),
    )
    runtime = FakeRuntime(loader)
    generation_lock = GenerationLock(max_concurrent_generations=1)
    held_lease = generation_lock.acquire(blocking=False)
    provider = Gemma4CompactProvider(
        runtime=runtime,
        loader=loader,
        prompt_text="Analyze.",
        generation_lock=generation_lock,
    )

    try:
        with pytest.raises(AppError) as exc_info:
            asyncio.run(provider(PNG_BYTES, _deterministic_response()))
    finally:
        held_lease.release()

    error = exc_info.value
    assert error.error_code == ErrorCode.PARSER_BUSY
    assert error.status_code == 503
    assert error.details == {"retry_after_seconds": 1}
    assert runtime.ensure_loaded_calls == 0
    assert runtime.after_request_calls == 0
    assert loader.model.generated_inputs is None


def test_gemma4_compact_provider_releases_generation_lock_after_request():
    loader = FakeLoader(
        processor=FakeProcessor(
            generated_text=json.dumps(
                {
                    "asset_type": "logo",
                    "style": {"summary": "lock released after generation"},
                }
            )
        ),
        model=FakeModel(),
    )
    generation_lock = GenerationLock(max_concurrent_generations=1)
    provider = Gemma4CompactProvider(
        runtime=FakeRuntime(loader),
        loader=loader,
        prompt_text="Analyze.",
        generation_lock=generation_lock,
    )

    asyncio.run(provider(PNG_BYTES, _deterministic_response()))

    lease = generation_lock.acquire(blocking=False)
    try:
        assert lease.acquired is True
    finally:
        lease.release()


def test_gemma4_compact_provider_releases_generation_lock_when_model_not_ready():
    loader = FakeLoader(processor=FakeProcessor(generated_text="{}"), model=None)
    generation_lock = GenerationLock(max_concurrent_generations=1)
    provider = Gemma4CompactProvider(
        runtime=FakeRuntime(loader),
        loader=loader,
        prompt_text="Analyze.",
        generation_lock=generation_lock,
    )

    with pytest.raises(AppError) as exc_info:
        asyncio.run(provider(PNG_BYTES, _deterministic_response()))

    assert exc_info.value.error_code == ErrorCode.MODEL_NOT_READY
    lease = generation_lock.acquire(blocking=False)
    try:
        assert lease.acquired is True
    finally:
        lease.release()


def test_gemma4_compact_provider_calls_after_request_when_model_load_fails():
    loader = FakeLoader(
        processor=FakeProcessor(generated_text="{}"),
        model=FakeModel(),
    )
    runtime = FakeRuntime(loader, ensure_error=TimeoutError("load took too long"))
    generation_lock = GenerationLock(max_concurrent_generations=1)
    provider = Gemma4CompactProvider(
        runtime=runtime,
        loader=loader,
        prompt_text="Analyze.",
        generation_lock=generation_lock,
    )

    with pytest.raises(AppError) as exc_info:
        asyncio.run(provider(PNG_BYTES, _deterministic_response()))

    error = exc_info.value
    assert error.error_code == ErrorCode.VLM_TIMEOUT
    assert error.status_code == 504
    assert error.details == {"stage": "load"}
    assert runtime.ensure_loaded_calls == 1
    assert runtime.after_request_calls == 1

    lease = generation_lock.acquire(blocking=False)
    try:
        assert lease.acquired is True
    finally:
        lease.release()


def test_gemma4_compact_provider_releases_generation_lock_when_output_is_invalid():
    loader = FakeLoader(
        processor=FakeProcessor(generated_text="not json"),
        model=FakeModel(),
    )
    generation_lock = GenerationLock(max_concurrent_generations=1)
    provider = Gemma4CompactProvider(
        runtime=FakeRuntime(loader),
        loader=loader,
        prompt_text="Analyze.",
        generation_lock=generation_lock,
    )

    with pytest.raises(AppError) as exc_info:
        asyncio.run(provider(PNG_BYTES, _deterministic_response()))

    assert exc_info.value.error_code == ErrorCode.VLM_INVALID_JSON
    lease = generation_lock.acquire(blocking=False)
    try:
        assert lease.acquired is True
    finally:
        lease.release()


def test_gemma4_compact_provider_maps_generation_timeout_safely():
    loader = FakeLoader(
        processor=FakeProcessor(generated_text="{}"),
        model=FakeModel(generate_error=TimeoutError("generation timed out")),
    )
    provider = Gemma4CompactProvider(
        runtime=FakeRuntime(loader),
        loader=loader,
        prompt_text="Analyze.",
    )

    with pytest.raises(AppError) as exc_info:
        asyncio.run(provider(PNG_BYTES, _deterministic_response()))

    error = exc_info.value
    assert error.error_code == ErrorCode.VLM_TIMEOUT
    assert error.status_code == 504
    assert error.details == {"stage": "generate"}
    assert "generation timed out" not in error.message


def test_gemma4_compact_provider_maps_cuda_oom_safely():
    loader = FakeLoader(
        processor=FakeProcessor(generated_text="{}"),
        model=FakeModel(generate_error=RuntimeError("CUDA out of memory: private details")),
    )
    provider = Gemma4CompactProvider(
        runtime=FakeRuntime(loader),
        loader=loader,
        prompt_text="Analyze.",
    )

    with pytest.raises(AppError) as exc_info:
        asyncio.run(provider(PNG_BYTES, _deterministic_response()))

    error = exc_info.value
    assert error.error_code == ErrorCode.VLM_OOM
    assert error.status_code == 503
    assert error.details == {"stage": "generate"}
    assert "private details" not in error.message
    assert "raw" not in error.details
    assert "stack" not in error.details


def test_gemma4_compact_provider_preserves_primary_error_when_after_request_fails():
    loader = FakeLoader(
        processor=FakeProcessor(generated_text="{}"),
        model=FakeModel(generate_error=TimeoutError("generation timed out")),
    )
    runtime = FakeRuntime(loader, after_request_error=RuntimeError("cleanup failed"))
    provider = Gemma4CompactProvider(
        runtime=runtime,
        loader=loader,
        prompt_text="Analyze.",
    )

    with pytest.raises(AppError) as exc_info:
        asyncio.run(provider(PNG_BYTES, _deterministic_response()))

    error = exc_info.value
    assert error.error_code == ErrorCode.VLM_TIMEOUT
    assert error.status_code == 504
    assert error.details == {"stage": "generate"}
    assert runtime.after_request_calls == 1


def test_gemma4_compact_provider_does_not_rewrap_app_errors():
    loader = FakeLoader(processor=FakeProcessor(generated_text="not json"), model=FakeModel())
    provider = Gemma4CompactProvider(
        runtime=FakeRuntime(loader),
        loader=loader,
        prompt_text="Analyze.",
    )

    with pytest.raises(AppError) as exc_info:
        asyncio.run(provider(PNG_BYTES, _deterministic_response()))

    error = exc_info.value
    assert error.error_code == ErrorCode.VLM_INVALID_JSON
    assert error.__cause__ is not error


class FakeRuntime:
    def __init__(self, loader, *, ensure_error=None, after_request_error=None):
        self.loader = loader
        self.ensure_error = ensure_error
        self.after_request_error = after_request_error
        self.ensure_loaded_calls = 0
        self.after_request_calls = 0
        self.ensure_thread_ids = []
        self.after_request_thread_ids = []

    def ensure_loaded(self):
        self.ensure_loaded_calls += 1
        self.ensure_thread_ids.append(threading.get_ident())
        if self.ensure_error is not None:
            raise self.ensure_error

    def after_request(self):
        self.after_request_calls += 1
        self.after_request_thread_ids.append(threading.get_ident())
        if self.after_request_error is not None:
            raise self.after_request_error


class FakeLoader:
    def __init__(self, *, processor, model):
        self.processor = processor
        self.model = model


class FakeProcessor:
    def __init__(
        self,
        *,
        generated_text: str,
        input_ids=None,
    ):
        self.generated_text = generated_text
        self.input_ids = input_ids or [[1]]
        self.decoded_ids = None
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
        return FakeInputs({"input_ids": self.input_ids})

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

    def __init__(self, *, generated_ids=None, generate_error=None):
        self.generated_ids = generated_ids or [[1, 2]]
        self.generate_error = generate_error
        self.generated_inputs = None

    def generate(self, **inputs):
        if self.generate_error is not None:
            raise self.generate_error
        self.generated_inputs = inputs
        return self.generated_ids


class FakeModelWithoutDevice:
    def __init__(self):
        self.generated_inputs = None

    def generate(self, **inputs):
        self.generated_inputs = inputs
        return [[1, 2]]
