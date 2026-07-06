"""Runtime configuration (pydantic-settings). No torch imports here."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class Settings(BaseSettings):
    """Serving configuration; override with MERCHANT_-prefixed env vars."""

    model_config = SettingsConfigDict(env_prefix="MERCHANT_")

    base_model: str = DEFAULT_MODEL
    adapter_path: str = "models/adapter"
    device: str = "auto"
    max_new_tokens: int = 64

    @property
    def adapter_or_none(self) -> str | None:
        """Empty string or "none" disables the adapter (serves the base model)."""
        if not self.adapter_path or self.adapter_path.lower() == "none":
            return None
        return self.adapter_path
