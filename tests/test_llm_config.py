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
    for name in [
        *MODEL_ENV_VARS,
        *KEY_ENV_VARS,
        "XNIMBUS_LLM_DEFAULT_PROFILE",
        "GOOGLE_AI_TIMEOUT_SECONDS",
        "XNIMBUS_LOCAL_THINKING_CONTROL",
    ]:
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


def test_registry_disables_thinking_for_local_thinking_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Qwen 3 reserves its bounded output budget for final schema-constrained claims."""
    monkeypatch.setenv("XNIMBUS_LOCAL_MODELS", "qwen3.5:9b,gemma4:12b,qwen2.5:7b-instruct,llama3.1:8b")

    profiles = {profile.model_id: profile for profile in available_model_profiles()}

    assert profiles["qwen3.5:9b"].decoding_parameters["reasoning_effort"] == "none"
    assert profiles["gemma4:12b"].decoding_parameters["reasoning_effort"] == "none"
    assert "reasoning_effort" not in profiles["qwen2.5:7b-instruct"].decoding_parameters
    assert "reasoning_effort" not in profiles["llama3.1:8b"].decoding_parameters


def test_registry_uses_chat_template_thinking_control_for_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    """A vLLM backend gets enable_thinking=False instead of Ollama's reasoning_effort."""
    monkeypatch.setenv("XNIMBUS_LOCAL_MODELS", "qwen3.5-9b,gemma4-12b,llama3.1-8b")
    monkeypatch.setenv("XNIMBUS_LOCAL_THINKING_CONTROL", "chat_template")

    profiles = {profile.model_id: profile for profile in available_model_profiles()}

    for thinking_model in ("qwen3.5-9b", "gemma4-12b"):
        assert profiles[thinking_model].decoding_parameters["enable_thinking"] is False
        assert "reasoning_effort" not in profiles[thinking_model].decoding_parameters
    assert "enable_thinking" not in profiles["llama3.1-8b"].decoding_parameters


def test_registry_gives_gpt_oss_low_reasoning_and_room_for_the_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """gpt-oss always reasons, so it gets the lowest effort and an output budget that leaves room for the JSON."""
    monkeypatch.setenv("XNIMBUS_LOCAL_MODELS", "gpt-oss-120b,qwen3.5-27b")
    monkeypatch.setenv("XNIMBUS_LOCAL_THINKING_CONTROL", "chat_template")

    profiles = {profile.model_id: profile for profile in available_model_profiles()}

    gpt_oss = profiles["gpt-oss-120b"].decoding_parameters
    assert gpt_oss["reasoning_effort"] == "low"
    assert gpt_oss["max_tokens"] == 4000
    assert "enable_thinking" not in gpt_oss
    assert profiles["qwen3.5-27b"].decoding_parameters["max_tokens"] == 800


def test_registry_supports_balanced_six_provider_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """The research registry supports local plus five explicitly configured cloud providers."""
    monkeypatch.setenv("XNIMBUS_LOCAL_MODEL", "local-test")
    for key in KEY_ENV_VARS:
        if key != "GEMINI_API_KEY":
            monkeypatch.setenv(key, "secret")
    monkeypatch.setenv("OPENAI_MODEL", "openai-test")
    monkeypatch.setenv("ANTHROPIC_MODEL", "anthropic-test")
    monkeypatch.setenv("GOOGLE_AI_MODEL", "google-test")
    monkeypatch.setenv("GOOGLE_AI_TIMEOUT_SECONDS", "75")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-test")
    monkeypatch.setenv("GROQ_MODEL", "groq-test")

    providers = {profile.provider for profile in available_model_profiles()}

    assert providers == {"local", "openai", "anthropic", "google_ai", "deepseek", "groq"}
    google = next(profile for profile in available_model_profiles() if profile.provider == "google_ai")
    assert google.decoding_parameters["timeout_seconds"] == 75.0
    anthropic = next(profile for profile in available_model_profiles() if profile.provider == "anthropic")
    assert "temperature" not in anthropic.decoding_parameters
    assert anthropic.decoding_parameters["max_tokens"] == 4000
    openai = next(profile for profile in available_model_profiles() if profile.provider == "openai")
    assert openai.decoding_parameters["temperature"] == 0
    assert openai.decoding_parameters["max_tokens"] == 800


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
