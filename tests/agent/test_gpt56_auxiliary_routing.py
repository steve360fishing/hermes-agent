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
    client.base_url = "https://example.invalid/v1"
    client.chat.completions.create.return_value = MagicMock()

    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), patch(
        "agent.auxiliary_client._get_cached_client",
        return_value=(client, "explicit-model"),
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


def test_sync_call_enforces_route_reasoning_and_reports_no_fallback(caplog) -> None:
    client = MagicMock()
    client.base_url = "https://chatgpt.com/backend-api/codex"
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

    assert resolver.call_args.kwargs == {
        "provider": "openai-codex",
        "model": "gpt-5.6-terra",
        "base_url": None,
        "api_key": None,
        "async_mode": False,
    }


@pytest.mark.asyncio
async def test_async_call_enforces_route_model_and_reasoning(caplog) -> None:
    client = MagicMock()
    client.base_url = "https://chatgpt.com/backend-api/codex"
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

    assert resolver.call_args.kwargs == {
        "provider": "openai-codex",
        "model": "gpt-5.6-terra",
        "base_url": None,
        "api_key": None,
        "async_mode": True,
    }


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
