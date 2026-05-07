import pytest

from app.model.gemma4_loader import Gemma4Loader, Gemma4LoaderConfig


def test_gemma4_loader_config_defaults_to_e4b_8bit_without_nf4():
    config = Gemma4LoaderConfig()

    assert config.model_id == "google/gemma-4-E4B-it"
    assert config.quantization == "8bit"
    assert config.load_in_4bit is False
    assert config.bnb_4bit_quant_type is None


@pytest.mark.parametrize("quantization", ["4bit", "nf4", "fp16"])
def test_gemma4_loader_config_rejects_quantization_other_than_8bit_or_none(quantization):
    with pytest.raises(ValueError, match="quantization must be one of"):
        Gemma4LoaderConfig(quantization=quantization)


def test_gemma4_loader_load_boundary_is_lazy_and_patchable(monkeypatch):
    calls = []

    def fake_load_transformers_model(config):
        calls.append(config)
        return object(), object()

    monkeypatch.setattr("app.model.gemma4_loader._load_transformers_model", fake_load_transformers_model)
    loader = Gemma4Loader(Gemma4LoaderConfig(model_id="local/test-model", quantization="none"))

    loaded = loader.load()

    assert loaded is loader
    assert loader.loaded is True
    assert calls == [loader.config]


def test_gemma4_loader_load_serializes_concurrent_calls(monkeypatch):
    calls = []

    def fake_load_transformers_model(config):
        calls.append(config)
        return object(), object()

    monkeypatch.setattr("app.model.gemma4_loader._load_transformers_model", fake_load_transformers_model)
    loader = Gemma4Loader()

    assert loader.load() is loader
    assert loader.load() is loader
    assert calls == [loader.config]
