"""Tests for the Anthropic grounded-response adapter."""

import json

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


def _ok_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "stop_reason": "end_turn",
            "content": [
                {"type": "thinking", "thinking": ""},
                {
                    "type": "text",
                    "text": '{"answer":"","claims":[],"uncertainty":"insufficient","follow_up_questions":[]}',
                },
            ],
        },
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )


def test_anthropic_request_omits_sampling_and_unsupported_schema_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Current Claude models reject temperature and array-length constraints in structured outputs."""
    captured: dict = {}

    def fake_post(*args, **kwargs):
        captured.update(kwargs["json"])
        return _ok_response()

    monkeypatch.setattr(httpx, "post", fake_post)
    profile = _profile().model_copy(update={"decoding_parameters": {"max_tokens": 4000, "temperature": 0}})

    generation = AnthropicClient(api_key="secret").generate(prompt="evidence", profile=profile)

    assert generation.answer.uncertainty == "insufficient"
    assert "temperature" not in captured
    assert captured["max_tokens"] == 4000
    assert "minItems" not in json.dumps(captured["output_config"]["format"]["schema"])


def test_anthropic_http_error_keeps_status_and_provider_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 400 must be diagnosable in the audit log instead of collapsing into a generic error."""
    response = httpx.Response(
        400,
        json={"type": "error", "error": {"type": "invalid_request_error", "message": "bad field"}},
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: response)

    with pytest.raises(AnthropicProviderError, match="HTTP 400: bad field"):
        AnthropicClient(api_key="secret").generate(prompt="evidence", profile=_profile())


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal"])
def test_anthropic_truncated_or_refused_output_is_a_provider_error(
    monkeypatch: pytest.MonkeyPatch, stop_reason: str
) -> None:
    """Truncated or declined output is reported explicitly rather than as a JSON parse failure."""
    response = httpx.Response(
        200,
        json={"stop_reason": stop_reason, "content": [{"type": "text", "text": '{"answer":'}]},
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: response)

    with pytest.raises(AnthropicProviderError, match=stop_reason):
        AnthropicClient(api_key="secret").generate(prompt="evidence", profile=_profile())
