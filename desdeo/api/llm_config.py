"""Select the language model provider for grounded XNIMBUS chat from the environment."""

import os

from desdeo.api.deepseek import DeepSeekProviderError, deepseek_configuration
from desdeo.api.google_ai import GoogleAIProviderError, google_ai_configuration
from desdeo.api.grounded_chat import GroundedLLMClient, ModelProfile

PROVIDER_ENV = "XNIMBUS_LLM_PROVIDER"
DEFAULT_PROVIDER = "deepseek"

_ALIASES = {
    "deepseek": "deepseek",
    "google": "google_ai",
    "google_ai": "google_ai",
    "gemini": "google_ai",
}

ProviderError = (DeepSeekProviderError, GoogleAIProviderError)
"""Exceptions a provider adapter raises when it cannot return a valid answer."""


class UnknownProviderError(ValueError):
    """Raised when ``XNIMBUS_LLM_PROVIDER`` names a provider that does not exist."""


def configured_provider() -> str:
    """Return the canonical provider name, raising for an unknown one instead of guessing."""
    raw = os.environ.get(PROVIDER_ENV, DEFAULT_PROVIDER).strip().casefold()
    if raw not in _ALIASES:
        raise UnknownProviderError(
            f"{PROVIDER_ENV}={raw!r} is not supported. Use one of: {', '.join(sorted(_ALIASES))}."
        )
    return _ALIASES[raw]


def get_llm_configuration() -> tuple[GroundedLLMClient | None, ModelProfile | None]:
    """Build the configured client and profile, or ``(None, None)`` when its API key is missing."""
    provider = configured_provider()
    if provider == "google_ai":
        return google_ai_configuration()
    return deepseek_configuration()
