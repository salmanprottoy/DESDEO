"""DeepSeek adapter for structured, evidence-grounded XNIMBUS answers."""

import json
import os
from typing import Any

import httpx
from pydantic import ValidationError

from desdeo.api.grounded_chat import GroundedAnswer, GroundedLLMClient, ModelProfile

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"


class DeepSeekProviderError(RuntimeError):
    """Raised when DeepSeek cannot return a valid grounded answer."""


class DeepSeekClient(GroundedLLMClient):
    """Call DeepSeek's OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEEPSEEK_BASE_URL,
        timeout_seconds: float = 30.0,
    ):
        """Create a client using an environment-provided API key."""
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    def generate(self, *, prompt: str, profile: ModelProfile) -> GroundedAnswer:
        """Generate and validate one structured answer from DeepSeek."""
        payload = {
            "model": profile.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer only from the supplied DESDEO evidence. "
                        "Return JSON with exactly these fields: answer, claims, "
                        "uncertainty, follow_up_questions. Each claim must "
                        "include evidence_ids. Use uncertainty='insufficient' "
                        "when the evidence does not answer the question. "
                        "The response must contain the word JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        try:
            response = httpx.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            body: dict[str, Any] = response.json()
            content = body["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content:
                raise DeepSeekProviderError("DeepSeek returned empty structured content.")
            return GroundedAnswer.model_validate(json.loads(content))
        except (httpx.HTTPError, KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise DeepSeekProviderError("DeepSeek did not return a valid response.") from error
        except ValidationError as error:
            raise DeepSeekProviderError("DeepSeek returned an invalid grounded answer.") from error


def deepseek_configuration() -> tuple[DeepSeekClient | None, ModelProfile | None]:
    """Build the configured DeepSeek client, or safe abstention configuration."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return None, None
    model_id = os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
    return (
        DeepSeekClient(api_key=api_key),
        ModelProfile(
            profile_id=f"deepseek-{model_id}",
            label=f"DeepSeek ({model_id})",
            provider="deepseek",
            model_id=model_id,
            capabilities=["structured_output", "grounded_chat"],
        ),
    )
