from functools import lru_cache
from typing import Any, Literal

from pydantic import Field
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="IMAGE_PARSER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    service_name: str = "image-parsing-server"
    service_version: str = "v1"

    model_id: str = "google/gemma-4-E4B-it"
    quantization: Literal["8bit", "none"] = "8bit"
    vlm_mode: Literal["resident", "cold"] = "resident"
    model_load_on_startup: bool = True
    model_idle_ttl_seconds: int = Field(default=0, ge=0)
    unload_after_request: bool = False
    max_concurrent_generations: int = Field(default=1, ge=1)

    max_upload_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    max_decoded_pixels: int = Field(default=16_000_000, ge=1)
    request_timeout_seconds: int = Field(default=60, ge=1)
    generation_max_new_tokens: int = Field(default=900, ge=1)
    generation_do_sample: bool = False

    expose_raw_vlm_output: bool = False
    debug_enabled: bool = False

    @model_validator(mode="before")
    @classmethod
    def apply_runtime_mode_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "model_load_on_startup" in data:
            return data

        resolved_data = dict(data)
        resolved_data["model_load_on_startup"] = (
            resolved_data.get("vlm_mode", "resident") == "resident"
        )
        return resolved_data


@lru_cache
def get_settings() -> Settings:
    return Settings()
