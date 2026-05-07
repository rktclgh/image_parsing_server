from __future__ import annotations

import asyncio
from base64 import b64encode
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from app.core.errors import AppError
from app.parsers.vlm_compact import parse_vlm_compact_output
from app.schemas.compact import CompactParseResponse
from app.schemas.errors import ErrorCode
from app.schemas.vlm import VLMCompactOutput

DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "prompts" / "design_style_parser_compact_v1.txt"
)


class Gemma4Runtime(Protocol):
    def ensure_loaded(self) -> None:
        ...

    def after_request(self) -> None:
        ...


class Gemma4LoaderBinding(Protocol):
    model: Any
    processor: Any


def image_bytes_to_data_url(image_bytes: bytes, *, mime_type: str) -> str:
    payload = b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def build_compact_messages(
    *,
    image_bytes: bytes,
    deterministic: CompactParseResponse,
    prompt_text: str | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": image_bytes_to_data_url(
                        image_bytes,
                        mime_type=deterministic.metadata.mime_type,
                    ),
                },
                {
                    "type": "text",
                    "text": _build_prompt_text(
                        prompt_text if prompt_text is not None else _read_default_prompt(),
                        deterministic,
                    ),
                },
            ],
        }
    ]


class Gemma4CompactProvider:
    def __init__(
        self,
        *,
        runtime: Gemma4Runtime,
        loader: Gemma4LoaderBinding,
        prompt_text: str | None = None,
        generate_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self._runtime = runtime
        self._loader = loader
        self._prompt_text = prompt_text
        self._generate_kwargs = dict(generate_kwargs or {})

    async def __call__(
        self,
        image_bytes: bytes,
        deterministic: CompactParseResponse,
    ) -> VLMCompactOutput:
        self._runtime.ensure_loaded()
        try:
            return await asyncio.to_thread(
                self._generate_and_parse,
                image_bytes,
                deterministic,
            )
        finally:
            self._runtime.after_request()

    def _generate_and_parse(
        self,
        image_bytes: bytes,
        deterministic: CompactParseResponse,
    ) -> VLMCompactOutput:
        model = self._loader.model
        processor = self._loader.processor
        if model is None or processor is None:
            raise AppError(
                ErrorCode.MODEL_NOT_READY,
                "model is not ready",
                status_code=503,
                details={"component": "gemma4_provider"},
            )

        messages = build_compact_messages(
            image_bytes=image_bytes,
            deterministic=deterministic,
            prompt_text=self._prompt_text,
        )
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(getattr(model, "device", None))

        generated_ids = model.generate(**inputs, **self._generate_kwargs)
        new_token_ids = _trim_prompt_tokens(generated_ids, inputs)
        decoded = processor.batch_decode(new_token_ids, skip_special_tokens=True)
        if not decoded:
            raise AppError(
                ErrorCode.VLM_INVALID_JSON,
                "model returned no decodable output",
                status_code=502,
                details={"reason": "empty_decoded_output"},
            )
        return parse_vlm_compact_output(decoded[0])


def _build_prompt_text(base_prompt: str, deterministic: CompactParseResponse) -> str:
    context = [
        "Deterministic preview context:",
        f"- width: {deterministic.metadata.width}",
        f"- height: {deterministic.metadata.height}",
        f"- mime_type: {deterministic.metadata.mime_type}",
    ]
    if deterministic.metadata.file_size_bytes is not None:
        context.append(f"- file_size_bytes: {deterministic.metadata.file_size_bytes}")
    if deterministic.metadata.alpha_present is not None:
        alpha_present = str(deterministic.metadata.alpha_present).lower()
        context.append(f"- alpha_present: {alpha_present}")
    if deterministic.style.summary:
        context.append(f"- deterministic_summary: {deterministic.style.summary}")
    if deterministic.style.composition:
        context.append("- composition_hints: " + ", ".join(deterministic.style.composition))
    if deterministic.style.palette:
        palette = ", ".join(
            _format_palette_swatch(swatch) for swatch in deterministic.style.palette
        )
        context.append(f"- palette_hints: {palette}")

    return base_prompt.rstrip() + "\n\n" + "\n".join(context)


def _format_palette_swatch(swatch) -> str:
    parts = [swatch.hex]
    if swatch.role:
        parts.append(f"role={swatch.role}")
    if swatch.ratio is not None:
        parts.append(f"ratio={swatch.ratio:.4f}")
    return " ".join(parts)


def _trim_prompt_tokens(generated_ids, inputs):
    input_ids = _input_ids_from_model_inputs(inputs)
    if input_ids is None:
        return generated_ids

    try:
        return [
            output_ids[len(prompt_ids) :]
            for prompt_ids, output_ids in zip(input_ids, generated_ids, strict=False)
        ]
    except TypeError:
        return generated_ids


def _input_ids_from_model_inputs(inputs):
    if isinstance(inputs, Mapping):
        return inputs.get("input_ids")
    if hasattr(inputs, "get"):
        return inputs.get("input_ids")
    return None


def _read_default_prompt() -> str:
    return DEFAULT_PROMPT_PATH.read_text(encoding="utf-8")
