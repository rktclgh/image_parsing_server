from dataclasses import dataclass
from typing import Literal

DEFAULT_MODEL_ID = "google/gemma-4-E4B-it"
SUPPORTED_QUANTIZATIONS = {"8bit", "none"}


@dataclass(frozen=True)
class Gemma4LoaderConfig:
    model_id: str = DEFAULT_MODEL_ID
    quantization: Literal["8bit", "none"] | str = "8bit"

    def __post_init__(self) -> None:
        if self.quantization not in SUPPORTED_QUANTIZATIONS:
            allowed = ", ".join(sorted(SUPPORTED_QUANTIZATIONS))
            raise ValueError(f"quantization must be one of: {allowed}")

    @property
    def load_in_4bit(self) -> bool:
        return False

    @property
    def bnb_4bit_quant_type(self) -> None:
        return None


class Gemma4Loader:
    def __init__(self, config: Gemma4LoaderConfig | None = None) -> None:
        self.config = config or Gemma4LoaderConfig()
        self.model = None
        self.processor = None

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def load(self) -> "Gemma4Loader":
        if self.loaded:
            return self
        self.model, self.processor = _load_transformers_model(self.config)
        return self


def _load_transformers_model(config: Gemma4LoaderConfig):
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required to load Gemma 4; install the gpu extra"
        ) from exc

    model_kwargs = {"device_map": "auto"}
    if config.quantization == "8bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    processor = AutoProcessor.from_pretrained(config.model_id)
    model = AutoModelForImageTextToText.from_pretrained(config.model_id, **model_kwargs)
    return model, processor
