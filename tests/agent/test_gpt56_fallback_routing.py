from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent import auxiliary_client as auxiliary


def _response(text: str = "fallback ok") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


def _enabled_routing_config() -> dict:
    return {
        "delegation": {
            "gpt56_routing": {
                "enabled": True,
                "contract": "gpt56-routing-v3",
                "provider": "openai-codex",
                "max_children": 3,
                "max_depth": 1,
            }
        }
    }


def _same_fallback_retry_case(kind: str) -> tuple[Exception, dict]:
    if kind == "transient":
        return ConnectionError("connection reset"), {}
    if kind == "temperature":
        return RuntimeError("Unsupported parameter: temperature is not supported"), {
            "temperature": 0.2
        }
    if kind == "max_tokens":
        return RuntimeError("Unsupported parameter: max_tokens is not supported"), {
            "max_tokens": 100
        }
    raise AssertionError(f"unknown retry kind: {kind}")


def _assert_actual_fallback_report(report: MagicMock) -> None:
    report.assert_called_once()
    reported = report.call_args.kwargs
    assert reported["task"] == "compression"
    assert reported["policy_spec"] is not None
    assert reported["provider_id"] == "anthropic"
    assert reported["source_label"] == "fallback_chain[0](anthropic)"
    assert reported["succeeded"] is True


@pytest.mark.parametrize("retry_kind", ["transient", "temperature", "max_tokens"])
def test_sync_initial_fallback_reports_once_after_successful_same_candidate_retry(
    monkeypatch, retry_kind: str
) -> None:
    first_error, call_options = _same_fallback_retry_case(retry_kind)
    fallback_client = MagicMock()
    fallback_client.base_url = "https://api.anthropic.com/v1"
    fallback_client.chat.completions.create.side_effect = [
        first_error,
        _response("retry ok"),
    ]
    report = MagicMock()

    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda: _enabled_routing_config()
    )
    monkeypatch.setattr(auxiliary, "_get_cached_client", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(
        auxiliary,
        "_try_configured_fallback_for_unavailable_client",
        lambda *args, **kwargs: (
            fallback_client,
            "claude-3-5-haiku-latest",
            "fallback_chain[0](anthropic)",
        ),
    )
    monkeypatch.setattr(auxiliary, "_transient_retry_count", lambda: 1)
    monkeypatch.setattr(auxiliary, "_TRANSIENT_RETRY_BACKOFF_BASE", 0.0)
    monkeypatch.setattr(auxiliary, "_report_fallback_attempt", report)

    result = auxiliary.call_llm(
        task="compression",
        messages=[{"role": "user", "content": "compress"}],
        **call_options,
    )

    assert result.choices[0].message.content == "retry ok"
    assert fallback_client.chat.completions.create.call_count == 2
    _assert_actual_fallback_report(report)


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_kind", ["transient", "temperature", "max_tokens"])
async def test_async_initial_fallback_reports_once_after_successful_same_candidate_retry(
    monkeypatch, retry_kind: str
) -> None:
    first_error, call_options = _same_fallback_retry_case(retry_kind)
    fallback_sync = MagicMock()
    fallback_async = MagicMock()
    fallback_async.base_url = "https://api.anthropic.com/v1"
    fallback_async.chat.completions.create = AsyncMock(
        side_effect=[first_error, _response("retry ok")]
    )
    report = MagicMock()

    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda: _enabled_routing_config()
    )
    monkeypatch.setattr(auxiliary, "_get_cached_client", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(
        auxiliary,
        "_try_configured_fallback_for_unavailable_client",
        lambda *args, **kwargs: (
            fallback_sync,
            "claude-3-5-haiku-latest",
            "fallback_chain[0](anthropic)",
        ),
    )
    monkeypatch.setattr(
        auxiliary,
        "_to_async_client",
        lambda *args, **kwargs: (fallback_async, "claude-3-5-haiku-latest"),
    )
    monkeypatch.setattr(auxiliary, "_report_fallback_attempt", report)

    result = await auxiliary.async_call_llm(
        task="compression",
        messages=[{"role": "user", "content": "compress"}],
        **call_options,
    )

    assert result.choices[0].message.content == "retry ok"
    assert fallback_async.chat.completions.create.await_count == 2
    _assert_actual_fallback_report(report)


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
