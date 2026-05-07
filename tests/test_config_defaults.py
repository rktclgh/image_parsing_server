from app.core.config import Settings


def test_settings_defaults_match_parser_contract():
    settings = Settings()

    assert settings.model_id == "google/gemma-4-E4B-it"
    assert settings.quantization == "8bit"
    assert settings.max_upload_bytes == 20 * 1024 * 1024
    assert settings.max_decoded_pixels == 16_000_000
    assert settings.expose_raw_vlm_output is False
    assert settings.max_concurrent_generations == 1
    assert settings.service_version == "v1"


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("IMAGE_PARSER_MODEL_ID", "local/test-model")
    monkeypatch.setenv("IMAGE_PARSER_QUANTIZATION", "none")
    monkeypatch.setenv("IMAGE_PARSER_MAX_UPLOAD_BYTES", "1024")
    monkeypatch.setenv("IMAGE_PARSER_EXPOSE_RAW_VLM_OUTPUT", "true")

    settings = Settings()

    assert settings.model_id == "local/test-model"
    assert settings.quantization == "none"
    assert settings.max_upload_bytes == 1024
    assert settings.expose_raw_vlm_output is True
