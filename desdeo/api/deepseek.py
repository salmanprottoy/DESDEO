"""DeepSeek compatibility wrapper for grounded XNIMBUS answers."""

import os

from desdeo.api.grounded_chat import ModelGeneration, ModelProfile, ProviderResponseError
from desdeo.api.openai_compatible import OpenAICompatibleClient, OpenAICompatibleProviderError

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekProviderError(ProviderResponseError):
    """Raised when DeepSeek cannot return a valid grounded answer."""


class DeepSeekClient(OpenAICompatibleClient):
    """Call DeepSeek through its OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEEPSEEK_BASE_URL,
        timeout_seconds: float = 30.0,
    ):
        """Create a DeepSeek client with its official endpoint by default."""
        super().__init__(api_key=api_key, base_url=base_url, timeout_seconds=timeout_seconds)

    def generate(self, *, prompt: str, profile: ModelProfile) -> ModelGeneration:
        """Generate one answer while preserving the provider-specific error type."""
        try:
            return super().generate(prompt=prompt, profile=profile)
        except OpenAICompatibleProviderError as error:
            raise DeepSeekProviderError(str(error)) from error


def deepseek_configuration() -> tuple[DeepSeekClient | None, ModelProfile | None]:
    """Build the explicitly pinned DeepSeek profile used by evaluation scripts."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    model_id = os.environ.get("DEEPSEEK_MODEL")
    if not api_key or not model_id:
        return None, None
    return (
        DeepSeekClient(api_key=api_key, base_url=DEEPSEEK_BASE_URL),
        ModelProfile(
            profile_id=f"deepseek:{model_id}",
            label=f"DeepSeek · {model_id}",
            provider="deepseek",
            model_id=model_id,
            deployment="cloud",
            capabilities=["structured_output", "grounded_chat"],
            decoding_parameters={"temperature": 0, "max_tokens": 800, "response_format": "json_object"},
        ),
    )
