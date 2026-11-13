"""Server-owned registry for selectable grounded-chat model profiles."""

import os

from desdeo.api.anthropic import AnthropicClient
from desdeo.api.deepseek import DEEPSEEK_BASE_URL, DeepSeekClient
from desdeo.api.google_ai import GoogleAIClient
from desdeo.api.grounded_chat import DeploymentType, GroundedLLMClient, ModelProfile, ProviderResponseError
from desdeo.api.openai_compatible import OpenAICompatibleClient

LOCAL_BASE_URL = "http://localhost:11434/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

ProviderError = ProviderResponseError


class UnknownProviderError(ValueError):
    """Raised when a requested model profile is not configured."""


class CloudConsentRequiredError(ValueError):
    """Raised when a cloud profile is requested without explicit per-turn consent."""


def available_model_profiles() -> list[ModelProfile]:
    """Return only profiles whose model and required credential are configured."""
    profiles: list[ModelProfile] = []
    profiles.extend(
        _profiles(
            provider="local",
            label="Local",
            models=_model_ids("XNIMBUS_LOCAL"),
            deployment="local",
            response_format="json_schema",
            timeout_seconds=float(os.environ.get("XNIMBUS_LOCAL_TIMEOUT_SECONDS", "90")),
        )
    )
    if os.environ.get("OPENAI_API_KEY"):
        profiles.extend(
            _profiles(
                provider="openai",
                label="OpenAI",
                models=_model_ids("OPENAI"),
                deployment="cloud",
                response_format="json_schema",
                timeout_seconds=60,
            )
        )
    if os.environ.get("ANTHROPIC_API_KEY"):
        profiles.extend(
            _profiles(
                provider="anthropic",
                label="Anthropic",
                models=_model_ids("ANTHROPIC"),
                deployment="cloud",
                response_format="json_schema",
                timeout_seconds=60,
            )
        )
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        profiles.extend(
            _profiles(
                provider="google_ai",
                label="Google Gemini",
                models=_model_ids("GOOGLE_AI"),
                deployment="cloud",
                response_format="json_schema",
                timeout_seconds=60,
            )
        )
    if os.environ.get("DEEPSEEK_API_KEY"):
        profiles.extend(
            _profiles(
                provider="deepseek",
                label="DeepSeek",
                models=_model_ids("DEEPSEEK"),
                deployment="cloud",
                response_format="json_object",
                timeout_seconds=60,
            )
        )
    if os.environ.get("GROQ_API_KEY"):
        profiles.extend(
            _profiles(
                provider="groq",
                label="Groq",
                models=_model_ids("GROQ"),
                deployment="cloud",
                response_format="json_object",
                temperature=1e-8,
                timeout_seconds=30,
            )
        )
    return profiles


def default_profile_id(profiles: list[ModelProfile] | None = None) -> str | None:
    """Return an explicitly configured local default, otherwise the first local profile."""
    configured = profiles if profiles is not None else available_model_profiles()
    local_profiles = {profile.profile_id: profile for profile in configured if profile.deployment == "local"}
    requested = os.environ.get("XNIMBUS_LLM_DEFAULT_PROFILE")
    if requested:
        if requested not in local_profiles:
            raise UnknownProviderError(
                "XNIMBUS_LLM_DEFAULT_PROFILE must name a configured local profile; cloud profiles require consent."
            )
        return requested
    return next(iter(local_profiles), None)


def get_llm_configuration(
    profile_id: str | None = None,
    *,
    cloud_consent: bool = False,
) -> tuple[GroundedLLMClient | None, ModelProfile | None]:
    """Resolve a configured profile and build its adapter without exposing credentials."""
    profiles = available_model_profiles()
    selected_id = profile_id or default_profile_id(profiles)
    if selected_id is None:
        return None, None
    profile = next((candidate for candidate in profiles if candidate.profile_id == selected_id), None)
    if profile is None:
        raise UnknownProviderError(f"Model profile {selected_id!r} is not configured on this server.")
    if profile.deployment == "cloud" and not cloud_consent:
        raise CloudConsentRequiredError(
            f"Model profile {selected_id!r} sends the evidence snapshot to {profile.provider}; "
            "explicit consent is required."
        )
    return _build_client(profile), profile


def _profiles(
    *,
    provider: str,
    label: str,
    models: list[str],
    deployment: DeploymentType,
    response_format: str,
    temperature: float = 0,
    timeout_seconds: float = 30,
) -> list[ModelProfile]:
    return [
        ModelProfile(
            profile_id=f"{provider}:{model_id}",
            label=f"{label} · {model_id}",
            provider=provider,
            model_id=model_id,
            deployment=deployment,
            capabilities=["structured_output", "grounded_chat"],
            decoding_parameters={
                "temperature": temperature,
                "max_tokens": 800,
                "response_format": response_format,
                "timeout_seconds": timeout_seconds,
            },
        )
        for model_id in models
    ]


def _model_ids(prefix: str) -> list[str]:
    raw = os.environ.get(f"{prefix}_MODELS") or os.environ.get(f"{prefix}_MODEL", "")
    return list(dict.fromkeys(model.strip() for model in raw.split(",") if model.strip()))


def _build_client(profile: ModelProfile) -> GroundedLLMClient:
    if profile.provider == "local":
        return OpenAICompatibleClient(
            api_key=os.environ.get("XNIMBUS_LOCAL_API_KEY", "ollama"),
            base_url=os.environ.get("XNIMBUS_LOCAL_BASE_URL", LOCAL_BASE_URL),
            timeout_seconds=float(profile.decoding_parameters["timeout_seconds"]),
        )
    if profile.provider == "openai":
        return OpenAICompatibleClient(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL", OPENAI_BASE_URL),
            timeout_seconds=float(profile.decoding_parameters["timeout_seconds"]),
        )
    if profile.provider == "anthropic":
        return AnthropicClient(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            base_url=os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            timeout_seconds=float(profile.decoding_parameters["timeout_seconds"]),
        )
    if profile.provider == "google_ai":
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ["GEMINI_API_KEY"]
        return GoogleAIClient(
            api_key=api_key,
            base_url=os.environ.get("GOOGLE_AI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"),
            timeout_seconds=float(profile.decoding_parameters["timeout_seconds"]),
        )
    if profile.provider == "deepseek":
        return DeepSeekClient(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.environ.get("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL),
            timeout_seconds=float(profile.decoding_parameters["timeout_seconds"]),
        )
    if profile.provider == "groq":
        return OpenAICompatibleClient(
            api_key=os.environ["GROQ_API_KEY"],
            base_url=os.environ.get("GROQ_BASE_URL", GROQ_BASE_URL),
            timeout_seconds=float(profile.decoding_parameters["timeout_seconds"]),
        )
    raise UnknownProviderError(f"Unsupported provider {profile.provider!r}.")
