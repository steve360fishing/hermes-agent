from __future__ import annotations

import json
from types import SimpleNamespace

from agent.agent_runtime_helpers import restore_primary_runtime
from gateway import run as gateway_run
from gateway.run import (
    GatewayRunner,
    _apply_pre_agent_route_provenance,
    _try_resolve_fallback_provider,
)


def test_pre_agent_fallback_preserves_configured_primary(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_HEALTH_PATH", str(tmp_path / "health.json"))
    agent = SimpleNamespace(
        model="minimax/minimax-m3",
        provider="openrouter",
        session_id="route-truth-session",
        max_tokens=8000,
        _primary_runtime={
            "model": "minimax/minimax-m3",
            "provider": "openrouter",
        },
    )
    route = {
        "route_provenance": {
            "primary_model": "gpt-5.6-sol",
            "primary_provider": "openai-codex",
            "requested_reasoning": "high",
            "route_state": "fallback",
            "safe_fallback_reason": "subscription_auth_failed",
        }
    }

    _apply_pre_agent_route_provenance(agent, route)

    assert agent._primary_runtime["model"] == "gpt-5.6-sol"
    assert agent._primary_runtime["provider"] == "openai-codex"
    assert agent._fallback_activated is True
    assert agent._pre_agent_fallback is True
    assert agent._requested_reasoning == "high"
    assert agent._fallback_reason == "subscription_auth_failed"

    agent._fallback_index = 4
    assert restore_primary_runtime(agent) is False
    assert agent.model == "minimax/minimax-m3"
    assert agent.provider == "openrouter"
    assert agent._fallback_index == 0


def test_completed_turn_metadata_records_route_truth(monkeypatch):
    stored = {
        "model": "gpt-5.6-sol",
        "model_config": json.dumps({"existing": True}),
    }

    class DB:
        def get_session(self, session_id):
            assert session_id == "session-1"
            return dict(stored)

        def update_session_meta(self, session_id, model_config, *, model):
            stored["model"] = model
            stored["model_config"] = model_config

    runner = SimpleNamespace(_session_db=SimpleNamespace(_db=DB()))
    agent = SimpleNamespace(
        model="minimax/minimax-m3",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_mode="openai",
        reasoning_config={"effort": "high"},
        _requested_reasoning="high",
        _fallback_activated=True,
        _fallback_reason="subscription_rate_limited",
    )

    GatewayRunner._sync_session_model_from_agent(runner, "session-1", agent)

    runtime = json.loads(stored["model_config"])["gateway_runtime"]
    assert stored["model"] == "minimax/minimax-m3"
    assert runtime["provider"] == "openrouter"
    assert runtime["requested_reasoning"] == "high"
    assert runtime["route_state"] == "fallback"
    assert runtime["safe_fallback_reason"] == "subscription_rate_limited"
    assert runtime["completed_at"]


def test_secondary_pre_agent_route_uses_direct_api_high_reasoning(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_GATEWAY_HEALTH_PATH", str(tmp_path / "health.json"))
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "agent": {
                "reasoning_overrides": {
                    "gpt-5.6-luna": "high"
                }
            }
        },
    )
    agent = SimpleNamespace(
        model="gpt-5.6-luna",
        provider="openai-api",
        session_id="secondary-route-truth",
        max_tokens=8000,
        _primary_runtime={
            "model": "gpt-5.6-luna",
            "provider": "openai-api",
        },
    )
    route = {
        "route_provenance": {
            "primary_model": "gpt-5.6-sol",
            "primary_provider": "openai-codex",
            "requested_reasoning": "high",
            "route_state": "fallback",
            "safe_fallback_reason": "minimax_unavailable",
            "fallback_model": "gpt-5.6-luna",
            "fallback_provider": "openai-api",
        }
    }

    _apply_pre_agent_route_provenance(agent, route)

    assert agent.provider == "openai-api"
    assert agent.model == "gpt-5.6-luna"
    assert agent.reasoning_config == {"enabled": True, "effort": "high"}
    assert agent._primary_runtime == {
        "model": "gpt-5.6-sol",
        "provider": "openai-codex",
    }


def test_pre_agent_resolution_fail_closes_on_grok_chain(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "fallback_providers:\n"
        "  - provider: openrouter\n"
        "    model: x-ai/grok-4.5\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    assert (
        _try_resolve_fallback_provider(
            primary_model="gpt-5.6-sol",
            primary_provider="openai-codex",
            requested_reasoning="high",
        )
        is None
    )
