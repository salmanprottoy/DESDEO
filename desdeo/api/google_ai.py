"""Google AI Gemini adapter for structured, evidence-grounded XNIMBUS answers."""

import json
import os
from typing import Any

import httpx
from pydantic import ValidationError

from desdeo.api.grounded_chat import GroundedAnswer, GroundedLLMClient, ModelProfile

GOOGLE_AI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GOOGLE_AI_MODEL = "gemini-2.5-flash"


class GoogleAIProviderError(RuntimeError):
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

    def generate(self, *, prompt: str, profile: ModelProfile) -> GroundedAnswer:
        """Generate and validate one structured answer from Google AI."""
        payload = {
            "systemInstruction": {
                "parts": [
                    {
                        "text": (
                            "Answer only from the supplied DESDEO evidence. "
                            "Return JSON with exactly these fields: answer, claims, "
                            "uncertainty, follow_up_questions. Each claim must "
                            "include evidence_ids. Use uncertainty='insufficient' "
                            "when the evidence does not answer the question."
                        )
                    }
                ]
            },
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
                                    "evidence_ids": {
                                        "type": "ARRAY",
                                        "items": {"type": "STRING"},
                                    },
                                },
                                "required": ["text", "evidence_ids"],
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
                "temperature": 0,
            },
        }
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
            return GroundedAnswer.model_validate(json.loads(content))
        except (httpx.HTTPError, KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise GoogleAIProviderError("Google AI did not return a valid response.") from error
        except ValidationError as error:
            raise GoogleAIProviderError("Google AI returned an invalid grounded answer.") from error


def google_ai_configuration() -> tuple[GoogleAIClient | None, ModelProfile | None]:
    """Build the configured Google AI client, or safe abstention configuration."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None, None
    model_id = os.environ.get("GOOGLE_AI_MODEL", DEFAULT_GOOGLE_AI_MODEL)
    return (
        GoogleAIClient(api_key=api_key),
        ModelProfile(
            profile_id=f"google-ai-{model_id}",
            label=f"Google AI ({model_id})",
            provider="google_ai",
            model_id=model_id,
            capabilities=["structured_output", "grounded_chat"],
        ),
    )
