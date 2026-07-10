from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent import auxiliary_client as auxiliary


def _response(text: str = "fallback ok") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def test_sync_fallback_uses_provider_id_and_strips_incompatible_reasoning(
    monkeypatch,
) -> None:
    client = MagicMock()
    client.base_url = "https://api.anthropic.com/v1"
    client.chat.completions.create.return_value = _response()
    built = []

    def fake_build(provider, model, messages, **kwargs):
        built.append((provider, model, messages, kwargs))
        return {"model": model, "messages": messages, "extra_body": kwargs["extra_body"]}

    monkeypatch.setattr(auxiliary, "_build_call_kwargs", fake_build)
    result = auxiliary._call_fallback_candidate_sync(
        client,
        "claude-3-5-haiku-latest",
        provider_id="anthropic",
        source_label="fallback_chain[0](anthropic)",
        task="compression",
        messages=[{"role": "user", "content": "compress"}],
        temperature=None,
        max_tokens=None,
        tools=None,
        effective_timeout=30,
        effective_extra_body={
            "reasoning": {"enabled": True, "effort": "high"},
            "safe": "preserved",
        },
    )

    assert result.choices[0].message.content == "fallback ok"
    assert built[0][0] == "anthropic"
    assert built[0][3]["extra_body"] == {"safe": "preserved"}


@pytest.mark.asyncio
async def test_async_fallback_uses_provider_id_and_strips_incompatible_reasoning(
    monkeypatch,
) -> None:
    client = MagicMock()
    client.base_url = "https://api.anthropic.com/v1"
    client.chat.completions.create = AsyncMock(return_value=_response())
    built = []

    def fake_build(provider, model, messages, **kwargs):
        built.append((provider, model, messages, kwargs))
        return {"model": model, "messages": messages, "extra_body": kwargs["extra_body"]}

    monkeypatch.setattr(auxiliary, "_build_call_kwargs", fake_build)
    result = await auxiliary._call_fallback_candidate_async(
        client,
        "claude-3-5-haiku-latest",
        provider_id="anthropic",
        source_label="fallback_chain[0](anthropic)",
        task="compression",
        messages=[{"role": "user", "content": "compress"}],
        temperature=None,
        max_tokens=None,
        tools=None,
        effective_timeout=30,
        effective_extra_body={
            "reasoning": {"enabled": True, "effort": "high"},
            "safe": "preserved",
        },
    )

    assert result.choices[0].message.content == "fallback ok"
    assert built[0][0] == "anthropic"
    assert built[0][3]["extra_body"] == {"safe": "preserved"}


@pytest.mark.parametrize(
    "source,expected",
    [
        ("fallback_chain[0](openrouter)", "openrouter"),
        ("main-agent(anthropic)", "anthropic"),
        ("openai-codex", "openai-codex"),
    ],
)
def test_fallback_source_label_is_separate_from_provider_id(
    source: str, expected: str
) -> None:
    assert auxiliary._fallback_provider_id(source) == expected


def test_failed_fallback_attempt_is_not_logged_as_used(monkeypatch) -> None:
    route_log = MagicMock()
    monkeypatch.setattr(auxiliary, "_log_gpt56_auxiliary_route", route_log)
    auxiliary._report_fallback_attempt(
        task="compression",
        policy_spec=None,
        provider_id="anthropic",
        source_label="fallback_chain[0](anthropic)",
        succeeded=False,
    )

    route_log.assert_not_called()


def test_successful_fallback_reports_actual_provider_and_source(monkeypatch) -> None:
    route_log = MagicMock()
    monkeypatch.setattr(auxiliary, "_log_gpt56_auxiliary_route", route_log)
    auxiliary._report_fallback_attempt(
        task="compression",
        policy_spec=None,
        provider_id="anthropic",
        source_label="fallback_chain[0](anthropic)",
        succeeded=True,
    )

    route_log.assert_called_once_with(
        "compression",
        None,
        provider="anthropic",
        fallback_used=True,
        fallback_source="fallback_chain[0](anthropic)",
    )
