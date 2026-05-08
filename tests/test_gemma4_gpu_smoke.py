import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from app.core.config import Settings
from app.model.gemma4_loader import Gemma4Loader, Gemma4LoaderConfig
from app.model.gemma4_provider import Gemma4CompactProvider
from app.model.generation_lock import GenerationLock
from app.model.runtime import VLMRuntime
from app.services.compact_parse import CompactParseService


pytestmark = pytest.mark.gpu


def _gpu_smoke_enabled() -> bool:
    return os.getenv("IMAGE_PARSER_RUN_GPU_SMOKE") == "1"


@pytest.mark.skipif(
    not _gpu_smoke_enabled(),
    reason="set IMAGE_PARSER_RUN_GPU_SMOKE=1 to run the real Gemma GPU smoke test",
)
def test_gemma4_e4b_8bit_real_gpu_compact_parse_smoke():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("bitsandbytes")
    pytest.importorskip("accelerate")
    pytest.importorskip("torchvision")

    image_path = _gpu_smoke_image_path()
    if not image_path.exists():
        pytest.skip(f"GPU smoke image not found: {image_path}")

    image_bytes = image_path.read_bytes()
    deterministic = asyncio.run(
        CompactParseService(
            settings=Settings(vlm_mode="cold", model_load_on_startup=False),
            request_id_factory=lambda: "req-gpu-smoke",
        ).parse_image(image_bytes)
    )
    loader = Gemma4Loader(
        Gemma4LoaderConfig(
            model_id=os.getenv("IMAGE_PARSER_GPU_MODEL_ID", "google/gemma-4-E4B-it"),
            quantization="8bit",
        )
    )
    runtime = VLMRuntime(
        loader=loader,
        mode="cold",
        load_on_startup=False,
        unload_after_request=False,
    )
    provider = Gemma4CompactProvider(
        runtime=runtime,
        loader=loader,
        generation_lock=GenerationLock(max_concurrent_generations=1),
        generate_kwargs={"max_new_tokens": 900, "do_sample": False},
    )

    started = time.perf_counter()
    before_vram_mib = _nvidia_used_memory_mib()
    max_seconds = float(os.getenv("IMAGE_PARSER_GPU_SMOKE_MAX_SECONDS", "180"))
    try:
        output = asyncio.run(asyncio.wait_for(provider(image_bytes, deterministic), timeout=max_seconds))
        elapsed_seconds = time.perf_counter() - started
        loaded_vram_mib = _nvidia_used_memory_mib()
        loaded_vram_delta_mib = loaded_vram_mib - before_vram_mib

        payload = output.model_dump(mode="json")
        semantic_text = json.dumps(payload, ensure_ascii=False).lower()
        assert output.asset_type in {"logo", "branding", "ui", "illustration", "unknown"}
        assert output.style.summary
        assert output.content.subject or output.content.objects or output.elements
        assert "purple" in semantic_text or "#8" in semantic_text or "#9" in semantic_text
        assert any(token in semantic_text for token in ("cloud", "blob", "rounded", "icon"))
        assert any(token in semantic_text for token in ("arrow", "caret", "chevron", "angle", "code"))
        assert "raw_vlm_output" not in payload
        assert "prompt" not in semantic_text
        assert loader.loaded is True

        max_baseline_vram_mib = int(os.getenv("IMAGE_PARSER_GPU_SMOKE_MAX_BASELINE_VRAM_MIB", "1500"))
        max_loaded_vram_delta_mib = int(
            os.getenv("IMAGE_PARSER_GPU_SMOKE_MAX_LOADED_VRAM_DELTA_MIB", "12500")
        )
        assert elapsed_seconds < max_seconds
        assert before_vram_mib <= max_baseline_vram_mib
        assert loaded_vram_delta_mib <= max_loaded_vram_delta_mib
        print(
            "GPU smoke:",
            {
                "elapsed_seconds": round(elapsed_seconds, 2),
                "before_vram_mib": before_vram_mib,
                "loaded_vram_mib": loaded_vram_mib,
                "loaded_vram_delta_mib": loaded_vram_delta_mib,
                "asset_type": output.asset_type,
                "summary": output.style.summary,
            },
        )
    finally:
        runtime.unload()

    assert loader.loaded is False


def _nvidia_used_memory_mib() -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    first_value = result.stdout.strip().splitlines()[0]
    return int(first_value.strip())


def _gpu_smoke_image_path() -> Path:
    image_path = os.getenv("IMAGE_PARSER_GPU_SMOKE_IMAGE")
    if not image_path:
        pytest.skip("set IMAGE_PARSER_GPU_SMOKE_IMAGE to a local design image")
    return Path(image_path).expanduser().resolve()
