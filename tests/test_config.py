from merchant_llm.config import DEFAULT_MODEL, Settings


def test_defaults():
    settings = Settings()
    assert settings.base_model == DEFAULT_MODEL
    assert settings.adapter_path == "models/adapter"
    assert settings.adapter_or_none == "models/adapter"
    assert settings.max_new_tokens == 64


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("MERCHANT_MAX_NEW_TOKENS", "32")
    monkeypatch.setenv("MERCHANT_ADAPTER_PATH", "none")
    settings = Settings()
    assert settings.max_new_tokens == 32
    assert settings.adapter_or_none is None
