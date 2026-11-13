"""OpenAI-compatible adapter used by local, OpenAI, DeepSeek, and Groq profiles."""

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


class OpenAICompatibleProviderError(ProviderResponseError):
    """Raised when an OpenAI-compatible endpoint returns an invalid response."""


class OpenAICompatibleClient(GroundedLLMClient):
    """Call a server-owned OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float = 30.0,
    ):
        """Create a client from server-side configuration."""
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    def generate(self, *, prompt: str, profile: ModelProfile) -> ModelGeneration:
        """Generate and validate one structured candidate answer."""
        response_format = profile.decoding_parameters.get("response_format", "json_object")
        if response_format == "json_schema":
            format_payload: dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "xnimbus_grounded_answer",
                    "strict": True,
                    "schema": GroundedAnswer.model_json_schema(),
                },
            }
        else:
            format_payload = {"type": "json_object"}

        payload = {
            "model": profile.model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
            "response_format": format_payload,
            "temperature": profile.decoding_parameters.get("temperature", 0),
            "max_tokens": profile.decoding_parameters.get("max_tokens", 800),
        }
        started = perf_counter()
        try:
            response = httpx.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            body: dict[str, Any] = response.json()
            content = body["choices"][0]["message"]["content"]
            answer = GroundedAnswer.model_validate(json.loads(_require_content(content)))
            usage = body.get("usage")
            return ModelGeneration(
                answer=answer,
                latency_ms=round((perf_counter() - started) * 1000),
                token_usage=usage if isinstance(usage, dict) else {},
            )
        except OpenAICompatibleProviderError:
            raise
        except httpx.TimeoutException as error:
            raise OpenAICompatibleProviderError(f"Provider timed out after {self._timeout:g} seconds.") from error
        except httpx.HTTPStatusError as error:
            raise OpenAICompatibleProviderError(f"Provider returned HTTP {error.response.status_code}.") from error
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise OpenAICompatibleProviderError(
                f"Provider returned malformed structured content: {type(error).__name__}."
            ) from error
        except httpx.HTTPError as error:
            raise OpenAICompatibleProviderError(f"Provider request failed: {type(error).__name__}.") from error
        except ValidationError as error:
            details = "; ".join(
                f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}" for item in error.errors()[:5]
            )
            raise OpenAICompatibleProviderError(f"Provider returned an invalid grounded answer: {details}") from error


def _require_content(content: object) -> str:
    """Return non-empty provider content or raise a normalized provider error."""
    if not isinstance(content, str) or not content:
        raise OpenAICompatibleProviderError("Provider returned empty structured content.")
    return content
