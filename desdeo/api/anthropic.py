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

# Structured outputs reject numeric, string-length, and complex array constraints. They are removed from
# the schema sent to the API; the full Pydantic model still validates the response locally.
_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
)


class AnthropicProviderError(ProviderResponseError):
    """Raised when Anthropic cannot return a schema-valid answer."""


def structured_output_schema() -> dict[str, Any]:
    """Return the grounded-answer JSON schema without constraints structured outputs reject."""
    return _strip_unsupported(GroundedAnswer.model_json_schema())


def _strip_unsupported(node: Any) -> Any:
    if isinstance(node, dict):
        return {key: _strip_unsupported(value) for key, value in node.items() if key not in _UNSUPPORTED_SCHEMA_KEYS}
    if isinstance(node, list):
        return [_strip_unsupported(item) for item in node]
    return node


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
        """Generate and validate one structured candidate answer.

        Sampling parameters such as ``temperature`` are deliberately not sent: current Claude models
        reject them with HTTP 400. Reproducibility is instead recorded through the pinned model ID,
        prompt version, and fact-sheet hash in the audit log.
        """
        payload = {
            "model": profile.model_id,
            "max_tokens": profile.decoding_parameters.get("max_tokens", 4000),
            "system": SYSTEM_INSTRUCTION,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": structured_output_schema(),
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
            _check_stop_reason(body.get("stop_reason"))
            content = next(block["text"] for block in body["content"] if block.get("type") == "text")
            answer = GroundedAnswer.model_validate(json.loads(content))
            usage = body.get("usage")
            return ModelGeneration(
                answer=answer,
                latency_ms=round((perf_counter() - started) * 1000),
                token_usage=usage if isinstance(usage, dict) else {},
            )
        except AnthropicProviderError:
            raise
        except httpx.TimeoutException as error:
            raise AnthropicProviderError(f"Anthropic timed out after {self._timeout:g} seconds.") from error
        except httpx.HTTPStatusError as error:
            detail = _response_error_detail(error.response)
            raise AnthropicProviderError(f"Anthropic returned HTTP {error.response.status_code}: {detail}") from error
        except httpx.HTTPError as error:
            raise AnthropicProviderError(f"Anthropic request failed: {type(error).__name__}.") from error
        except (KeyError, StopIteration, TypeError, json.JSONDecodeError) as error:
            raise AnthropicProviderError(
                f"Anthropic returned malformed structured content: {type(error).__name__}."
            ) from error
        except ValidationError as error:
            details = "; ".join(
                f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}" for item in error.errors()[:5]
            )
            raise AnthropicProviderError(f"Anthropic returned an invalid grounded answer: {details}") from error


def _check_stop_reason(stop_reason: object) -> None:
    """Report truncated or declined output explicitly instead of as a JSON parse failure."""
    if stop_reason == "refusal":
        raise AnthropicProviderError("Anthropic declined the request (stop_reason=refusal).")
    if stop_reason == "max_tokens":
        raise AnthropicProviderError("Anthropic output was truncated at max_tokens.")


def _response_error_detail(response: httpx.Response) -> str:
    """Extract a bounded provider error message without exposing credentials."""
    try:
        error = response.json().get("error", {})
        message = error.get("message") if isinstance(error, dict) else None
    except (json.JSONDecodeError, TypeError, AttributeError):
        message = None
    return str(message)[:500] if message else "no provider detail"
