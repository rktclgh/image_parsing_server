import importlib.util
from pathlib import Path
import sys


def _load_smoke_module():
    script_path = Path(__file__).parents[1] / "scripts" / "gpu_http_smoke.py"
    spec = importlib.util.spec_from_file_location("gpu_http_smoke", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_nvidia_device_id_defaults_to_first_gpu(monkeypatch):
    module = _load_smoke_module()
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert module._nvidia_device_id() == "0"


def test_nvidia_device_id_uses_first_visible_device(monkeypatch):
    module = _load_smoke_module()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    assert module._nvidia_device_id() == "2"


def test_contains_any_term_matches_case_insensitively():
    module = _load_smoke_module()

    assert module._contains_any_term("Soft Rounded ICON", ["icon"])
    assert not module._contains_any_term("Soft Rounded ICON", ["triangle"])


def test_parse_args_documents_default_codex_fixture_expectations(monkeypatch):
    module = _load_smoke_module()
    monkeypatch.setattr(sys, "argv", ["gpu_http_smoke.py"])

    args = module._parse_args()

    assert args.expected_width == 640
    assert args.expected_height == 640
    assert args.expected_color_term == ["purple", "#8", "#9"]
    assert args.expected_shape_term == ["cloud", "blob", "rounded", "icon"]
