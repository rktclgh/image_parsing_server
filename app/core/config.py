from functools import lru_cache
from typing import Literal

from pydantic import Field
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
    max_concurrent_generations: int = Field(default=1, ge=1)

    max_upload_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    max_decoded_pixels: int = Field(default=16_000_000, ge=1)
    request_timeout_seconds: int = Field(default=60, ge=1)

    expose_raw_vlm_output: bool = False
    debug_enabled: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
