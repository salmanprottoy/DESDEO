"""Tests for the DeepSeek configuration and structured response boundary."""

import httpx
import pytest

from desdeo.api.deepseek import DeepSeekClient, DeepSeekProviderError, deepseek_configuration
from desdeo.api.grounded_chat import ModelProfile


def _profile() -> ModelProfile:
    return ModelProfile(
        profile_id="deepseek-test",
        label="DeepSeek test",
        provider="deepseek",
        model_id="deepseek-chat",
        capabilities=["structured_output"],
    )


def test_missing_deepseek_key_keeps_safe_abstention(monkeypatch: pytest.MonkeyPatch) -> None:
    """No provider is configured when the API key is absent."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    assert deepseek_configuration() == (None, None)


def test_deepseek_client_parses_structured_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """The adapter converts DeepSeek JSON output into the shared answer schema."""
    response = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"answer":"The value is 5.2.","claims":'
                            '[{"text":"The value is 5.2.","evidence_ids":["f1"]}],'
                            '"uncertainty":"none","follow_up_questions":[]}'
                        )
                    }
                }
            ]
        },
        request=httpx.Request("POST", "https://api.deepseek.com/chat/completions"),
    )
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: response)
    client = DeepSeekClient(api_key="test-key")

    answer = client.generate(prompt="JSON evidence", profile=_profile())

    assert answer.uncertainty == "none"
    assert answer.claims[0].evidence_ids == ["f1"]


def test_deepseek_client_rejects_invalid_structured_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed provider output becomes an explicit provider error."""
    response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "not-json"}}]},
        request=httpx.Request("POST", "https://api.deepseek.com/chat/completions"),
    )
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: response)

    with pytest.raises(DeepSeekProviderError):
        DeepSeekClient(api_key="test-key").generate(prompt="JSON evidence", profile=_profile())
