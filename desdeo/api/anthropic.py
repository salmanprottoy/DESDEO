"""Anthropic Messages API adapter for grounded XNIMBUS answers."""

import json
from time import perf_counter
from typing import Any

import httpx
from pydantic import ValidationError

from desdeo.api.grounded_chat import (
    SYSTEM_INSTRUCTION,
    GroundedAnswer,
    GroundedLLMClient,
    ModelGeneration,
    ModelProfile,
    ProviderResponseError,
)

ANTHROPIC_BASE_URL = "https://api.anthropic.com"


class AnthropicProviderError(ProviderResponseError):
    """Raised when Anthropic cannot return a schema-valid answer."""


class AnthropicClient(GroundedLLMClient):
    """Call Anthropic's Messages API with JSON-schema structured output."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = ANTHROPIC_BASE_URL,
        timeout_seconds: float = 30.0,
    ):
        """Create a client using a server-side API key."""
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    def generate(self, *, prompt: str, profile: ModelProfile) -> ModelGeneration:
        """Generate and validate one structured candidate answer."""
        payload = {
            "model": profile.model_id,
            "max_tokens": profile.decoding_parameters.get("max_tokens", 800),
            "temperature": profile.decoding_parameters.get("temperature", 0),
            "system": SYSTEM_INSTRUCTION,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": GroundedAnswer.model_json_schema(),
                }
            },
        }
        started = perf_counter()
        try:
            response = httpx.post(
                f"{self._base_url}/v1/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            body: dict[str, Any] = response.json()
            content_blocks = body["content"]
            content = next(block["text"] for block in content_blocks if block.get("type") == "text")
            answer = GroundedAnswer.model_validate(json.loads(content))
            usage = body.get("usage")
            return ModelGeneration(
                answer=answer,
                latency_ms=round((perf_counter() - started) * 1000),
                token_usage=usage if isinstance(usage, dict) else {},
            )
        except (httpx.HTTPError, KeyError, StopIteration, TypeError, json.JSONDecodeError) as error:
            raise AnthropicProviderError("Anthropic did not return a valid response.") from error
        except ValidationError as error:
            raise AnthropicProviderError("Anthropic returned an invalid grounded answer.") from error
