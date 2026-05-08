import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture(scope="module")
def smoke_module():
    script_path = Path(__file__).parents[1] / "scripts" / "gpu_http_smoke.py"
    spec = importlib.util.spec_from_file_location("gpu_http_smoke", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_nvidia_device_id_defaults_to_first_gpu(smoke_module, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert smoke_module._nvidia_device_id() == "0"


def test_nvidia_device_id_uses_first_visible_device(smoke_module, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    assert smoke_module._nvidia_device_id() == "2"


def test_contains_any_term_matches_case_insensitively(smoke_module):
    assert smoke_module._contains_any_term("Soft Rounded ICON", ["icon"])
    assert not smoke_module._contains_any_term("Soft Rounded ICON", ["triangle"])


def test_parse_args_documents_default_codex_fixture_expectations(smoke_module, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["gpu_http_smoke.py"])

    args = smoke_module._parse_args()

    assert args.expected_width == 640
    assert args.expected_height == 640
    assert args.expected_color_term == ["purple", "#8", "#9"]
    assert args.expected_shape_term == ["cloud", "blob", "rounded", "icon"]
    assert args.max_new_tokens == 900
    assert args.do_sample is False


def test_parse_args_accepts_generation_overrides(smoke_module, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["gpu_http_smoke.py", "--max-new-tokens", "64", "--do-sample"],
    )

    args = smoke_module._parse_args()

    assert args.max_new_tokens == 64
    assert args.do_sample is True


def test_response_leak_check_allows_words_like_prompted(smoke_module, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["gpu_http_smoke.py"])
    args = smoke_module._parse_args()
    payload = {
        "metadata": {"width": 640, "height": 640},
        "style": {"summary": "A prompted purple cloud icon with rounded edges."},
        "warnings": [],
    }

    smoke_module._assert_compact_parse_response(payload, args=args)


def test_response_leak_check_rejects_top_level_prompt_key(smoke_module, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["gpu_http_smoke.py"])
    args = smoke_module._parse_args()
    payload = {
        "metadata": {"width": 640, "height": 640},
        "style": {"summary": "A purple cloud icon with rounded edges."},
        "warnings": [],
        "prompt": "internal prompt text",
    }

    with pytest.raises(RuntimeError, match="leaked raw VLM or prompt"):
        smoke_module._assert_compact_parse_response(payload, args=args)
