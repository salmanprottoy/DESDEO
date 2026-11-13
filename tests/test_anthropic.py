"""Tests for the Anthropic grounded-response adapter."""

import httpx
import pytest

from desdeo.api.anthropic import AnthropicClient, AnthropicProviderError
from desdeo.api.grounded_chat import ModelProfile


def _profile() -> ModelProfile:
    return ModelProfile(
        profile_id="anthropic:test",
        label="Anthropic test",
        provider="anthropic",
        model_id="claude-test",
        deployment="cloud",
        capabilities=["structured_output"],
        decoding_parameters={"max_tokens": 800},
    )


def test_anthropic_client_parses_structured_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """The adapter parses a JSON-schema Messages API response and captures usage."""
    response = httpx.Response(
        200,
        json={
            "content": [
                {
                    "type": "text",
                    "text": (
                        '{"answer":"f_1 is 5.2.","claims":'
                        '[{"text":"f_1 is 5.2.","claim_type":"objective_value",'
                        '"evidence_ids":["f1"]}],"uncertainty":"none","follow_up_questions":[]}'
                    ),
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 20},
        },
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: response)

    generation = AnthropicClient(api_key="secret").generate(prompt="evidence", profile=_profile())

    assert generation.answer.claims[0].claim_type == "objective_value"
    assert generation.token_usage == {"input_tokens": 10, "output_tokens": 20}


def test_anthropic_client_rejects_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed provider content becomes a normalized provider error."""
    response = httpx.Response(
        200,
        json={"content": [{"type": "text", "text": "not-json"}]},
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: response)

    with pytest.raises(AnthropicProviderError):
        AnthropicClient(api_key="secret").generate(prompt="evidence", profile=_profile())
