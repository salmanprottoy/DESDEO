"""Tests for the server-owned grounded-chat model registry."""

import pytest

from desdeo.api.llm_config import (
    CloudConsentRequiredError,
    UnknownProviderError,
    available_model_profiles,
    default_profile_id,
    get_llm_configuration,
)

MODEL_ENV_VARS = [
    "XNIMBUS_LOCAL_MODEL",
    "XNIMBUS_LOCAL_MODELS",
    "OPENAI_MODEL",
    "OPENAI_MODELS",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_MODELS",
    "GOOGLE_AI_MODEL",
    "GOOGLE_AI_MODELS",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_MODELS",
    "GROQ_MODEL",
    "GROQ_MODELS",
]
KEY_ENV_VARS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY",
    "GROQ_API_KEY",
]


@pytest.fixture(autouse=True)
def clear_model_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep registry tests independent of developer credentials."""
    for name in [*MODEL_ENV_VARS, *KEY_ENV_VARS, "XNIMBUS_LLM_DEFAULT_PROFILE"]:
        monkeypatch.delenv(name, raising=False)


def test_registry_lists_only_explicitly_configured_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """A credential alone or a model alone never creates a cloud profile."""
    monkeypatch.setenv("XNIMBUS_LOCAL_MODELS", "qwen3:8b,gemma4:12b")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-test")

    profiles = available_model_profiles()

    assert [profile.profile_id for profile in profiles] == [
        "local:qwen3:8b",
        "local:gemma4:12b",
        "anthropic:claude-test",
    ]
    assert default_profile_id(profiles) == "local:qwen3:8b"


def test_registry_supports_balanced_six_provider_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """The research registry supports local plus five explicitly configured cloud providers."""
    monkeypatch.setenv("XNIMBUS_LOCAL_MODEL", "local-test")
    for key in KEY_ENV_VARS:
        if key != "GEMINI_API_KEY":
            monkeypatch.setenv(key, "secret")
    monkeypatch.setenv("OPENAI_MODEL", "openai-test")
    monkeypatch.setenv("ANTHROPIC_MODEL", "anthropic-test")
    monkeypatch.setenv("GOOGLE_AI_MODEL", "google-test")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-test")
    monkeypatch.setenv("GROQ_MODEL", "groq-test")

    providers = {profile.provider for profile in available_model_profiles()}

    assert providers == {"local", "openai", "anthropic", "google_ai", "deepseek", "groq"}


def test_cloud_profile_requires_per_turn_consent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configured credentials do not authorize silently sending a DM snapshot."""
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")

    with pytest.raises(CloudConsentRequiredError):
        get_llm_configuration("openai:gpt-test")

    client, profile = get_llm_configuration("openai:gpt-test", cloud_consent=True)
    assert client is not None
    assert profile is not None
    assert profile.deployment == "cloud"


def test_default_cannot_be_a_cloud_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local-first policy cannot be bypassed by a server cloud default."""
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("XNIMBUS_LLM_DEFAULT_PROFILE", "openai:gpt-test")

    with pytest.raises(UnknownProviderError):
        default_profile_id()


def test_unknown_profile_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clients cannot submit arbitrary provider or model identifiers."""
    monkeypatch.setenv("XNIMBUS_LOCAL_MODEL", "qwen3:8b")

    with pytest.raises(UnknownProviderError):
        get_llm_configuration("local:not-configured")
