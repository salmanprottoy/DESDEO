"""Tests for OpenAI-compatible local and hosted chat adapters."""

import httpx
import pytest

from desdeo.api.grounded_chat import ModelProfile
from desdeo.api.openai_compatible import OpenAICompatibleClient


def test_openai_compatible_client_forwards_configured_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Thinking models can reserve output tokens for the required final JSON answer."""
    captured: dict[str, object] = {}
    response = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"answer":"f_1 is 1.0.","claims":'
                            '[{"text":"f_1 is 1.0.","claim_type":"objective_value",'
                            '"evidence_ids":["f1"]}],"uncertainty":"none","follow_up_questions":[]}'
                        )
                    }
                }
            ]
        },
        request=httpx.Request("POST", "http://localhost:11434/v1/chat/completions"),
    )

    def fake_post(*args: object, **kwargs: object) -> httpx.Response:
        captured.update(kwargs)
        return response

    monkeypatch.setattr(httpx, "post", fake_post)
    profile = ModelProfile(
        profile_id="local:qwen3.5:9b",
        label="Qwen",
        provider="local",
        model_id="qwen3.5:9b",
        capabilities=["structured_output"],
        decoding_parameters={
            "temperature": 0,
            "max_tokens": 800,
            "response_format": "json_schema",
            "reasoning_effort": "none",
        },
    )

    OpenAICompatibleClient(api_key="ollama", base_url="http://localhost:11434/v1").generate(
        prompt="Evidence JSON: []",
        profile=profile,
    )

    assert captured["json"]["reasoning_effort"] == "none"  # type: ignore[index]
