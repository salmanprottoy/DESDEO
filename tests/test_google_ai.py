"""Tests for the Google AI Gemini grounded-response adapter."""

import httpx
import pytest

from desdeo.api.google_ai import GoogleAIClient, GoogleAIProviderError, google_ai_configuration
from desdeo.api.grounded_chat import ModelProfile


def _profile() -> ModelProfile:
    return ModelProfile(
        profile_id="google-ai-test",
        label="Google AI test",
        provider="google_ai",
        model_id="gemini-2.5-flash",
        capabilities=["structured_output"],
    )


def test_missing_google_ai_key_keeps_safe_abstention(monkeypatch: pytest.MonkeyPatch) -> None:
    """No provider is configured when the Google AI API key is absent."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert google_ai_configuration() == (None, None)


def test_google_ai_client_parses_structured_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """The adapter converts Gemini JSON output into the shared answer schema."""
    response = httpx.Response(
        200,
        json={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": (
                                    '{"answer":"The value is 5.2.","claims":'
                                    '[{"text":"The value is 5.2.","evidence_ids":["f1"]}],'
                                    '"uncertainty":"none","follow_up_questions":[]}'
                                )
                            }
                        ]
                    }
                }
            ]
        },
        request=httpx.Request(
            "POST",
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        ),
    )
    captured: dict[str, object] = {}

    def fake_post(*args: object, **kwargs: object) -> httpx.Response:
        captured.update(kwargs)
        return response

    monkeypatch.setattr(httpx, "post", fake_post)

    answer = GoogleAIClient(api_key="test-key").generate(prompt="JSON evidence", profile=_profile())

    assert answer.uncertainty == "none"
    assert answer.claims[0].evidence_ids == ["f1"]
    generation_config = captured["json"]["generationConfig"]  # type: ignore[index]
    assert generation_config["responseSchema"]["properties"]["claims"]["items"]["required"] == [  # type: ignore[index]
        "text",
        "evidence_ids",
    ]


def test_google_ai_client_rejects_invalid_structured_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed Gemini output becomes an explicit provider error."""
    response = httpx.Response(
        200,
        json={"candidates": [{"content": {"parts": [{"text": "not-json"}]}}]},
        request=httpx.Request(
            "POST",
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        ),
    )
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: response)

    with pytest.raises(GoogleAIProviderError):
        GoogleAIClient(api_key="test-key").generate(prompt="JSON evidence", profile=_profile())
