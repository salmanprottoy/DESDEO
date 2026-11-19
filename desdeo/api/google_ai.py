"""Google AI Gemini adapter for structured, evidence-grounded XNIMBUS answers."""

import json
import os
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

GOOGLE_AI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GoogleAIProviderError(ProviderResponseError):
    """Raised when Google AI cannot return a valid grounded answer."""


class GoogleAIClient(GroundedLLMClient):
    """Call Google's Gemini generate-content endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = GOOGLE_AI_BASE_URL,
        timeout_seconds: float = 30.0,
    ):
        """Create a client using an environment-provided API key."""
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    def generate(self, *, prompt: str, profile: ModelProfile) -> ModelGeneration:
        """Generate and validate one structured answer from Google AI."""
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "answer": {"type": "STRING"},
                        "claims": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "text": {"type": "STRING"},
                                    "claim_type": {
                                        "type": "STRING",
                                        "enum": [
                                            "objective_value",
                                            "variable_value",
                                            "problem_context",
                                            "preference",
                                            "solver_status",
                                            "lagrange_multiplier",
                                            "tradeoff",
                                            "active_objective",
                                            "explanation_availability",
                                            "comparison",
                                            "experiment_summary",
                                            "iteration",
                                            "preference_history",
                                            "experiment_solution",
                                            "history_comparison",
                                        ],
                                    },
                                    "evidence_ids": {
                                        "type": "ARRAY",
                                        "items": {"type": "STRING"},
                                    },
                                },
                                "required": ["text", "claim_type", "evidence_ids"],
                            },
                        },
                        "uncertainty": {
                            "type": "STRING",
                            "enum": ["none", "partial", "insufficient"],
                        },
                        "follow_up_questions": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"},
                        },
                    },
                    "required": ["answer", "claims", "uncertainty", "follow_up_questions"],
                },
                "temperature": profile.decoding_parameters.get("temperature", 0),
                "maxOutputTokens": profile.decoding_parameters.get("max_tokens", 800),
            },
        }
        started = perf_counter()
        try:
            response = httpx.post(
                f"{self._base_url}/models/{profile.model_id}:generateContent",
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self._api_key,
                },
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            body: dict[str, Any] = response.json()
            content = body["candidates"][0]["content"]["parts"][0]["text"]
            if not isinstance(content, str) or not content:
                raise GoogleAIProviderError("Google AI returned empty structured content.")
            answer = GroundedAnswer.model_validate(json.loads(content))
            usage = body.get("usageMetadata")
            return ModelGeneration(
                answer=answer,
                latency_ms=round((perf_counter() - started) * 1000),
                token_usage=usage if isinstance(usage, dict) else {},
            )
        except httpx.TimeoutException as error:
            raise GoogleAIProviderError(f"Google AI timed out after {self._timeout:g} seconds.") from error
        except httpx.HTTPStatusError as error:
            detail = _response_error_detail(error.response)
            raise GoogleAIProviderError(f"Google AI returned HTTP {error.response.status_code}: {detail}") from error
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise GoogleAIProviderError(
                f"Google AI returned malformed structured content: {type(error).__name__}."
            ) from error
        except httpx.HTTPError as error:
            raise GoogleAIProviderError(f"Google AI request failed: {type(error).__name__}.") from error
        except ValidationError as error:
            details = "; ".join(
                f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}" for item in error.errors()[:5]
            )
            raise GoogleAIProviderError(f"Google AI returned an invalid grounded answer: {details}") from error


def _response_error_detail(response: httpx.Response) -> str:
    """Extract a bounded provider error message without exposing credentials."""
    try:
        error = response.json().get("error", {})
        message = error.get("message") if isinstance(error, dict) else None
    except (json.JSONDecodeError, TypeError):
        message = None
    return str(message)[:500] if message else "no provider detail"


def google_ai_configuration() -> tuple[GoogleAIClient | None, ModelProfile | None]:
    """Build the explicitly pinned Google AI profile used by evaluation scripts."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    model_id = os.environ.get("GOOGLE_AI_MODEL")
    if not api_key or not model_id:
        return None, None
    return (
        GoogleAIClient(api_key=api_key),
        ModelProfile(
            profile_id=f"google_ai:{model_id}",
            label=f"Google Gemini · {model_id}",
            provider="google_ai",
            model_id=model_id,
            deployment="cloud",
            capabilities=["structured_output", "grounded_chat"],
            decoding_parameters={"temperature": 0, "max_tokens": 800, "response_format": "json_schema"},
        ),
    )
