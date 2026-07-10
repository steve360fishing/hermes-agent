from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent import auxiliary_client as auxiliary
from hermes_cli.gpt56_routing import AUXILIARY_ROUTES


def _enabled_config() -> dict:
    return {
        "delegation": {
            "gpt56_routing": {
                "enabled": True,
                "contract": "gpt56-routing-v3",
                "provider": "openai-codex",
                "max_children": 3,
                "max_depth": 1,
            }
        },
        "auxiliary": {
            "compression": {
                "provider": "openrouter",
                "model": "legacy-model",
                "extra_body": {"reasoning": {"effort": "low"}},
            }
        },
    }


def _disabled_config() -> dict:
    return {
        "delegation": {"gpt56_routing": {"enabled": False}},
        "auxiliary": {
            "compression": {
                "provider": "openrouter",
                "model": "legacy-model",
            }
        },
    }


def _routed_client(*, async_mode: bool = False):
    client = MagicMock()
    client.base_url = "https://chatgpt.com/backend-api/codex"
    client.api_key = "oauth-token"
    response = MagicMock()
    if async_mode:
        client.chat.completions.create = AsyncMock(return_value=response)
    else:
        client.chat.completions.create.return_value = response
    return client, response


@pytest.mark.parametrize(
    "task,model,effort",
    [(task, spec.model, spec.effort) for task, spec in AUXILIARY_ROUTES.items()],
)
def test_enabled_policy_routes_every_classified_auxiliary_slot(
    task: str, model: str, effort: str
) -> None:
    with patch("hermes_cli.config.load_config", return_value=_enabled_config()):
        provider, resolved_model, base_url, api_key, api_mode = (
            auxiliary._resolve_task_provider_model(task=task)
        )
        extra_body = auxiliary._apply_gpt56_auxiliary_reasoning(task, {})

    assert provider == "openai-codex"
    assert resolved_model == model
    assert base_url is None
    assert api_key is None
    assert api_mode == "codex_responses"
    assert extra_body["reasoning"] == {"enabled": True, "effort": effort}


def test_operator_policy_replaces_ordinary_explicit_call_hints() -> None:
    with patch("hermes_cli.config.load_config", return_value=_enabled_config()):
        result = auxiliary._resolve_task_provider_model(
            task="compression",
            provider="custom",
            model="explicit-model",
            base_url="https://example.invalid/v1",
            api_key="test-only",
        )

    assert result == (
        "openai-codex",
        "gpt-5.6-luna",
        None,
        None,
        "codex_responses",
    )


def test_operator_policy_replaces_ordinary_explicit_reasoning_hint() -> None:
    client = MagicMock()
    client.base_url = "https://chatgpt.com/backend-api/codex"
    client.api_key = "oauth-token"
    client.chat.completions.create.return_value = MagicMock()

    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), patch(
        "agent.auxiliary_client._get_cached_client",
        return_value=(client, "gpt-5.6-luna"),
    ):
        auxiliary.call_llm(
            task="compression",
            provider="custom",
            model="explicit-model",
            base_url="https://example.invalid/v1",
            api_key="test-only",
            messages=[{"role": "user", "content": "compress"}],
            extra_body={"reasoning": {"effort": "high"}},
        )

    assert client.chat.completions.create.call_args.kwargs["extra_body"]["reasoning"] == {
        "enabled": True,
        "effort": "medium",
    }


def test_obsolete_session_search_is_explicitly_ignored_by_policy() -> None:
    with patch("hermes_cli.config.load_config", return_value=_enabled_config()):
        assert auxiliary._get_gpt56_auxiliary_spec("session_search") is None


def test_unknown_named_auxiliary_slot_fails_closed_when_policy_enabled() -> None:
    with patch("hermes_cli.config.load_config", return_value=_enabled_config()):
        with pytest.raises(ValueError, match="unclassified auxiliary task"):
            auxiliary._resolve_task_provider_model(task="new_unclassified_slot")


@pytest.mark.parametrize(
    "field,value",
    [
        ("contract", "wrong-contract"),
        ("provider", "openrouter"),
        ("max_children", 99),
        ("max_depth", 99),
    ],
)
def test_auxiliary_policy_rejects_contract_or_bound_mismatch(field, value) -> None:
    config = _enabled_config()
    config["delegation"]["gpt56_routing"][field] = value

    with patch("hermes_cli.config.load_config", return_value=config):
        with pytest.raises(ValueError, match="Invalid delegation.gpt56_routing"):
            auxiliary._get_gpt56_auxiliary_spec("approval")


@pytest.mark.parametrize(
    "overrides",
    [
        {"provider": "openrouter"},
        {"model": "anthropic/claude-haiku-4.5"},
        {"base_url": "https://example.invalid/v1"},
        {"api_key": "test-only"},
    ],
)
def test_protected_route_rejects_noncanonical_explicit_overrides(overrides) -> None:
    with patch("hermes_cli.config.load_config", return_value=_enabled_config()):
        with pytest.raises(ValueError, match="Protected GPT-5.6 auxiliary route"):
            auxiliary._resolve_task_provider_model(task="approval", **overrides)


def test_protected_route_accepts_exact_canonical_explicit_values() -> None:
    with patch("hermes_cli.config.load_config", return_value=_enabled_config()):
        result = auxiliary._resolve_task_provider_model(
            task="approval",
            provider="openai-codex",
            model="gpt-5.6-sol",
        )

    assert result == (
        "openai-codex",
        "gpt-5.6-sol",
        None,
        None,
        "codex_responses",
    )


def test_protected_route_rejects_noncanonical_api_mode() -> None:
    with patch("hermes_cli.config.load_config", return_value=_enabled_config()):
        with pytest.raises(ValueError, match="requires api_mode='codex_responses'"):
            auxiliary.call_llm(
                task="approval",
                api_mode="chat_completions",
                messages=[{"role": "user", "content": "approve?"}],
            )


@pytest.mark.parametrize(
    "base_url,api_key",
    [
        ("https://openrouter.ai/api/v1", "oauth-token"),
        ("https://chatgpt.com/backend-api/codex", ""),
    ],
)
def test_sync_routed_client_rejects_noncanonical_runtime_without_secret_leak(
    base_url: str,
    api_key: str,
) -> None:
    client, _ = _routed_client()
    client.base_url = base_url
    client.api_key = api_key
    secret = "must-not-appear-in-errors"

    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), patch(
        "agent.auxiliary_client._get_cached_client",
        return_value=(client, "gpt-5.6-luna"),
    ):
        with pytest.raises(ValueError, match="failed closed") as exc_info:
            auxiliary.call_llm(
                task="compression",
                api_key=secret,
                messages=[{"role": "user", "content": "compress"}],
            )

    assert secret not in str(exc_info.value)
    client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_async_routed_client_rejects_noncanonical_endpoint() -> None:
    client, _ = _routed_client(async_mode=True)
    client.base_url = "https://chatgpt.com/backend-api/not-codex"

    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), patch(
        "agent.auxiliary_client._get_cached_client",
        return_value=(client, "gpt-5.6-terra"),
    ):
        with pytest.raises(ValueError, match="failed closed"):
            await auxiliary.async_call_llm(
                task="moa_reference",
                messages=[{"role": "user", "content": "inspect"}],
            )

    client.chat.completions.create.assert_not_awaited()


def test_sync_public_helper_validates_routed_client_runtime() -> None:
    client, _ = _routed_client()
    client.api_key = ""

    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(client, "gpt-5.6-luna"),
    ):
        with pytest.raises(ValueError, match="failed closed"):
            auxiliary.get_text_auxiliary_client("compression")


def test_async_public_helper_exposes_and_logs_route_metadata_parity(caplog) -> None:
    client, _ = _routed_client(async_mode=True)

    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(client, "gpt-5.6-luna"),
    ), caplog.at_level("INFO", logger="agent.auxiliary_client"):
        binding = auxiliary.get_async_text_auxiliary_client("compression")

    resolved_client, resolved_model = binding
    assert resolved_client is client
    assert resolved_model == "gpt-5.6-luna"
    assert binding.decision.policy_spec.route_id == "explorer"
    assert binding.decision.policy_spec.effort == "medium"
    assert "task=compression" in caplog.text
    assert "fallback_used=false" in caplog.text


def test_sync_call_uses_one_disabled_policy_snapshot() -> None:
    client, response = _routed_client()

    def cached(_provider, model, **_kwargs):
        return client, model

    calls = 0

    def drifting_config(*_args):
        nonlocal calls
        calls += 1
        return _disabled_config() if calls == 1 else _enabled_config()

    with patch("agent.auxiliary_client._load_auxiliary_config_snapshot", side_effect=drifting_config) as loader, patch(
        "agent.auxiliary_client._get_cached_client", side_effect=cached
    ):
        result = auxiliary.call_llm(
            task="compression",
            messages=[{"role": "user", "content": "compress"}],
            timeout=5,
        )

    assert result is response
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "legacy-model"
    assert "reasoning" not in (kwargs.get("extra_body") or {})
    assert loader.call_count == 1


def test_sync_call_uses_one_enabled_policy_snapshot() -> None:
    client, response = _routed_client()

    def cached(_provider, model, **_kwargs):
        return client, model

    calls = 0

    def drifting_config(*_args):
        nonlocal calls
        calls += 1
        return _enabled_config() if calls == 1 else _disabled_config()

    with patch("agent.auxiliary_client._load_auxiliary_config_snapshot", side_effect=drifting_config) as loader, patch(
        "agent.auxiliary_client._get_cached_client", side_effect=cached
    ):
        result = auxiliary.call_llm(
            task="compression",
            messages=[{"role": "user", "content": "compress"}],
            timeout=5,
        )

    assert result is response
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-5.6-luna"
    assert kwargs["extra_body"]["reasoning"] == {
        "enabled": True,
        "effort": "medium",
    }
    assert loader.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first_config,expected_model,expected_effort",
    [
        (_disabled_config(), "legacy-model", None),
        (_enabled_config(), "gpt-5.6-luna", "medium"),
    ],
)
async def test_async_call_uses_one_policy_snapshot(
    first_config: dict,
    expected_model: str,
    expected_effort: str | None,
) -> None:
    client, response = _routed_client(async_mode=True)
    alternate = (
        _enabled_config()
        if first_config["delegation"]["gpt56_routing"]["enabled"] is False
        else _disabled_config()
    )

    def cached(_provider, model, **_kwargs):
        return client, model

    calls = 0

    def drifting_config(*_args):
        nonlocal calls
        calls += 1
        return first_config if calls == 1 else alternate

    with patch(
        "agent.auxiliary_client._load_auxiliary_config_snapshot",
        side_effect=drifting_config,
    ) as loader, patch(
        "agent.auxiliary_client._get_cached_client", side_effect=cached
    ):
        result = await auxiliary.async_call_llm(
            task="compression",
            messages=[{"role": "user", "content": "compress"}],
            timeout=5,
        )

    assert result is response
    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["model"] == expected_model
    if expected_effort is None:
        assert "reasoning" not in (kwargs.get("extra_body") or {})
    else:
        assert kwargs["extra_body"]["reasoning"]["effort"] == expected_effort
    assert loader.call_count == 1


def test_sync_call_enforces_route_reasoning_and_reports_no_fallback(caplog) -> None:
    client = MagicMock()
    client.base_url = "https://chatgpt.com/backend-api/codex"
    client.api_key = "oauth-token"
    response = MagicMock()
    client.chat.completions.create.return_value = response

    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), patch(
        "agent.auxiliary_client._get_cached_client",
        return_value=(client, "gpt-5.6-luna"),
    ), caplog.at_level("INFO", logger="agent.auxiliary_client"):
        result = auxiliary.call_llm(
            task="compression",
            messages=[{"role": "user", "content": "compress"}],
            extra_body={"reasoning": {"effort": "high"}},
        )

    assert result is response
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-5.6-luna"
    assert kwargs["extra_body"]["reasoning"] == {
        "enabled": True,
        "effort": "medium",
    }
    assert "fallback_used=false" in caplog.text


def test_unavailable_initial_client_reports_routed_fallback(caplog) -> None:
    fallback_client = MagicMock()
    fallback_client.base_url = "https://openrouter.ai/api/v1"
    response = MagicMock()
    fallback_client.chat.completions.create.return_value = response

    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), patch(
        "agent.auxiliary_client._get_cached_client", return_value=(None, None)
    ), patch(
        "agent.auxiliary_client._try_configured_fallback_for_unavailable_client",
        return_value=(fallback_client, "fallback-model", "openrouter"),
    ), caplog.at_level("INFO", logger="agent.auxiliary_client"):
        result = auxiliary.call_llm(
            task="compression",
            messages=[{"role": "user", "content": "compress"}],
        )

    assert result is response
    assert "provider=openrouter fallback_used=true" in caplog.text


def test_vision_initial_fallback_reports_routed_fallback(caplog) -> None:
    fallback_client = MagicMock()
    fallback_client.base_url = "https://openrouter.ai/api/v1"
    response = MagicMock()
    fallback_client.chat.completions.create.return_value = response

    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), patch(
        "agent.auxiliary_client.resolve_vision_provider_client",
        side_effect=[
            ("openai-codex", None, "gpt-5.6-terra"),
            ("openrouter", fallback_client, "fallback-vision-model"),
        ],
    ), caplog.at_level("INFO", logger="agent.auxiliary_client"):
        result = auxiliary.call_llm(
            task="vision",
            messages=[{"role": "user", "content": "inspect"}],
        )

    assert result is response
    assert "provider=openrouter fallback_used=true" in caplog.text


def test_vision_adapter_does_not_restore_explicit_endpoint_hints() -> None:
    client = MagicMock()
    client.base_url = "https://chatgpt.com/backend-api/codex"
    client.api_key = "oauth-token"
    client.chat.completions.create.return_value = MagicMock()
    resolver = MagicMock(return_value=("openai-codex", client, "gpt-5.6-terra"))

    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), patch(
        "agent.auxiliary_client.resolve_vision_provider_client", resolver
    ):
        auxiliary.call_llm(
            task="vision",
            provider="custom",
            model="expensive-model",
            base_url="https://example.invalid/v1",
            api_key="test-only",
            messages=[{"role": "user", "content": "inspect"}],
        )

    resolver_kwargs = resolver.call_args.kwargs
    assert {key: resolver_kwargs[key] for key in (
        "provider", "model", "base_url", "api_key", "async_mode"
    )} == {
        "provider": "openai-codex",
        "model": "gpt-5.6-terra",
        "base_url": None,
        "api_key": None,
        "async_mode": False,
    }
    assert resolver_kwargs["_route_decision"].model == "gpt-5.6-terra"


@pytest.mark.asyncio
async def test_async_call_enforces_route_model_and_reasoning(caplog) -> None:
    client = MagicMock()
    client.base_url = "https://chatgpt.com/backend-api/codex"
    client.api_key = "oauth-token"
    response = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)

    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), patch(
        "agent.auxiliary_client._get_cached_client",
        return_value=(client, "gpt-5.6-terra"),
    ), caplog.at_level("INFO", logger="agent.auxiliary_client"):
        result = await auxiliary.async_call_llm(
            task="moa_reference",
            messages=[{"role": "user", "content": "inspect"}],
        )

    assert result is response
    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["model"] == "gpt-5.6-terra"
    assert kwargs["extra_body"]["reasoning"] == {
        "enabled": True,
        "effort": "medium",
    }
    assert "fallback_used=false" in caplog.text


@pytest.mark.asyncio
async def test_async_unavailable_initial_client_reports_routed_fallback(caplog) -> None:
    fallback_sync = MagicMock()
    fallback_async = MagicMock()
    fallback_async.base_url = "https://openrouter.ai/api/v1"
    response = MagicMock()
    fallback_async.chat.completions.create = AsyncMock(return_value=response)

    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), patch(
        "agent.auxiliary_client._get_cached_client", return_value=(None, None)
    ), patch(
        "agent.auxiliary_client._try_configured_fallback_for_unavailable_client",
        return_value=(fallback_sync, "fallback-model", "openrouter"),
    ), patch(
        "agent.auxiliary_client._to_async_client",
        return_value=(fallback_async, "fallback-model"),
    ), caplog.at_level("INFO", logger="agent.auxiliary_client"):
        result = await auxiliary.async_call_llm(
            task="compression",
            messages=[{"role": "user", "content": "compress"}],
        )

    assert result is response
    assert "provider=openrouter fallback_used=true" in caplog.text


@pytest.mark.asyncio
async def test_async_vision_initial_fallback_reports_routed_fallback(caplog) -> None:
    fallback_client = MagicMock()
    fallback_client.base_url = "https://openrouter.ai/api/v1"
    response = MagicMock()
    fallback_client.chat.completions.create = AsyncMock(return_value=response)

    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), patch(
        "agent.auxiliary_client.resolve_vision_provider_client",
        side_effect=[
            ("openai-codex", None, "gpt-5.6-terra"),
            ("openrouter", fallback_client, "fallback-vision-model"),
        ],
    ), caplog.at_level("INFO", logger="agent.auxiliary_client"):
        result = await auxiliary.async_call_llm(
            task="vision",
            messages=[{"role": "user", "content": "inspect"}],
        )

    assert result is response
    assert "provider=openrouter fallback_used=true" in caplog.text


@pytest.mark.asyncio
async def test_async_vision_adapter_does_not_restore_explicit_endpoint_hints() -> None:
    client = MagicMock()
    client.base_url = "https://chatgpt.com/backend-api/codex"
    client.api_key = "oauth-token"
    client.chat.completions.create = AsyncMock(return_value=MagicMock())
    resolver = MagicMock(return_value=("openai-codex", client, "gpt-5.6-terra"))

    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), patch(
        "agent.auxiliary_client.resolve_vision_provider_client", resolver
    ):
        await auxiliary.async_call_llm(
            task="vision",
            provider="custom",
            model="expensive-model",
            base_url="https://example.invalid/v1",
            api_key="test-only",
            messages=[{"role": "user", "content": "inspect"}],
        )

    resolver_kwargs = resolver.call_args.kwargs
    assert {key: resolver_kwargs[key] for key in (
        "provider", "model", "base_url", "api_key", "async_mode"
    )} == {
        "provider": "openai-codex",
        "model": "gpt-5.6-terra",
        "base_url": None,
        "api_key": None,
        "async_mode": True,
    }
    assert resolver_kwargs["_route_decision"].model == "gpt-5.6-terra"


def test_protected_auxiliary_route_does_not_try_unavailable_client_fallback() -> None:
    fallback = MagicMock()
    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), patch(
        "agent.auxiliary_client._get_cached_client", return_value=(None, None)
    ), patch(
        "agent.auxiliary_client._try_configured_fallback_for_unavailable_client",
        fallback,
    ):
        with pytest.raises(RuntimeError, match="Protected GPT-5.6 auxiliary route"):
            auxiliary.call_llm(
                task="approval",
                messages=[{"role": "user", "content": "approve?"}],
            )

    fallback.assert_not_called()


@pytest.mark.parametrize("token", ["", "   ", object(), 123])
def test_canonical_runtime_rejects_non_string_or_blank_tokens(token) -> None:
    from hermes_cli.gpt56_routing import (
        RoutingPolicyError,
        route_spec,
        validate_codex_route_runtime,
    )

    with pytest.raises(RoutingPolicyError, match="token"):
        validate_codex_route_runtime(
            route_spec("worker", reason="bounded implementation"),
            provider="openai-codex",
            model="gpt-5.6-terra",
            api_mode="codex_responses",
            base_url="https://chatgpt.com/backend-api/codex",
            api_key=token,
        )
