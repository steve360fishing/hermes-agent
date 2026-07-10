from __future__ import annotations

from types import SimpleNamespace

from agent.openrouter_fallback_guard import (
    OPENROUTER_FALLBACK_MODEL,
    apply_openrouter_fallback_notice,
    fallback_cap_message_if_exhausted,
    openrouter_fallback_activation_allowed,
    record_openrouter_fallback_activation,
    record_gateway_primary_route,
    restore_openrouter_fallback_state,
)


def _agent(**overrides):
    values = {
        "provider": "openrouter",
        "model": OPENROUTER_FALLBACK_MODEL,
        "_fallback_activated": True,
        "max_tokens": 8000,
        "session_id": "fallback-session",
        "_primary_runtime": {
            "provider": "openai-codex",
            "model": "gpt-5.6-sol",
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_gpt55_is_blocked_while_configured_openrouter_fallbacks_remain_compatible(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HERMES_GATEWAY_HEALTH_PATH", str(tmp_path / "health.json"))
    agent = _agent(_fallback_activated=False)

    allowed, message = openrouter_fallback_activation_allowed(
        agent, "openrouter", "openai/gpt-5.5"
    )
    assert allowed is False
    assert "explicit-only" in message

    allowed, message = openrouter_fallback_activation_allowed(
        agent, "openrouter", "anthropic/claude-sonnet-4.6"
    )
    assert allowed is True
    assert message == ""

    allowed, message = openrouter_fallback_activation_allowed(
        agent, "openrouter", OPENROUTER_FALLBACK_MODEL
    )
    assert allowed is True
    assert message == ""


def test_fallback_is_visible_and_stops_at_turn_cap(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_GATEWAY_HEALTH_PATH", str(tmp_path / "health.json"))
    monkeypatch.setenv("HERMES_OPENROUTER_AUTO_FALLBACK_MAX_TURNS", "1")
    monkeypatch.setenv("HERMES_OPENROUTER_AUTO_FALLBACK_MAX_OUTPUT_TOKENS", "1200")
    agent = _agent()

    record_openrouter_fallback_activation(agent, reason="subscription_unavailable")
    assert agent.max_tokens == 1200

    response, changed = apply_openrouter_fallback_notice(agent, "continuity response")
    assert changed is True
    assert response.startswith("OPENROUTER FALLBACK ACTIVE")
    assert "GPT-5.6 subscription access failed" in response

    cap_message = fallback_cap_message_if_exhausted(agent)
    assert cap_message is not None
    assert "spend cap reached" in cap_message


def test_unrelated_primary_session_cannot_reset_fallback_cap(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HERMES_GATEWAY_HEALTH_PATH", str(tmp_path / "health.json"))
    monkeypatch.setenv("HERMES_OPENROUTER_AUTO_FALLBACK_MAX_TURNS", "1")
    fallback_agent = _agent()
    record_openrouter_fallback_activation(fallback_agent)
    apply_openrouter_fallback_notice(fallback_agent, "continuity response")
    assert fallback_cap_message_if_exhausted(fallback_agent) is not None

    healthy_agent = _agent(
        provider="openai-codex",
        model="gpt-5.6-sol",
        _fallback_activated=False,
        session_id="healthy-session",
    )
    apply_openrouter_fallback_notice(healthy_agent, "healthy response")

    assert fallback_cap_message_if_exhausted(fallback_agent) is not None


def test_health_write_failure_does_not_disable_in_memory_cap(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HERMES_GATEWAY_HEALTH_PATH", str(tmp_path / "health.json"))
    monkeypatch.setenv("HERMES_OPENROUTER_AUTO_FALLBACK_MAX_TURNS", "1")
    monkeypatch.setattr(
        "agent.openrouter_fallback_guard._write_health", lambda *_args: False
    )
    agent = _agent()

    record_openrouter_fallback_activation(agent)
    apply_openrouter_fallback_notice(agent, "continuity response")

    assert fallback_cap_message_if_exhausted(agent) is not None


def test_non_gpt56_primary_is_not_mislabeled(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_GATEWAY_HEALTH_PATH", str(tmp_path / "health.json"))
    agent = _agent(
        _primary_runtime={"provider": "anthropic", "model": "claude-opus-4.7"}
    )

    record_openrouter_fallback_activation(agent)
    response, changed = apply_openrouter_fallback_notice(agent, "generic fallback")

    assert changed is False
    assert response == "generic fallback"


def test_cap_survives_primary_retry_until_primary_response_succeeds(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HERMES_GATEWAY_HEALTH_PATH", str(tmp_path / "health.json"))
    monkeypatch.setenv("HERMES_OPENROUTER_AUTO_FALLBACK_MAX_TURNS", "1")
    agent = _agent()
    record_openrouter_fallback_activation(agent)
    apply_openrouter_fallback_notice(agent, "continuity response")
    assert fallback_cap_message_if_exhausted(agent) is not None

    restore_openrouter_fallback_state(agent)
    agent.provider = "openai-codex"
    agent.model = "gpt-5.6-sol"
    agent._fallback_activated = False

    allowed, _ = openrouter_fallback_activation_allowed(
        agent, "openrouter", OPENROUTER_FALLBACK_MODEL
    )
    assert allowed is False

    record_gateway_primary_route(agent)
    allowed, _ = openrouter_fallback_activation_allowed(
        agent, "openrouter", OPENROUTER_FALLBACK_MODEL
    )
    assert allowed is True


def test_primary_success_clears_local_cap_when_other_session_owns_health(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HERMES_GATEWAY_HEALTH_PATH", str(tmp_path / "health.json"))
    monkeypatch.setenv("HERMES_OPENROUTER_AUTO_FALLBACK_MAX_TURNS", "1")
    first = _agent(session_id="first")
    record_openrouter_fallback_activation(first)
    apply_openrouter_fallback_notice(first, "first continuity response")
    assert fallback_cap_message_if_exhausted(first) is not None

    second = _agent(session_id="second")
    record_openrouter_fallback_activation(second)

    restore_openrouter_fallback_state(first)
    first.provider = "openai-codex"
    first.model = "gpt-5.6-sol"
    first._fallback_activated = False
    record_gateway_primary_route(first)

    allowed, _ = openrouter_fallback_activation_allowed(
        first, "openrouter", OPENROUTER_FALLBACK_MODEL
    )
    assert allowed is True


def test_runtime_integration_points_remain_wired() -> None:
    from pathlib import Path

    root = Path(__file__).parents[2]
    chat_helpers = (root / "agent/chat_completion_helpers.py").read_text(encoding="utf-8")
    turn_finalizer = (root / "agent/turn_finalizer.py").read_text(encoding="utf-8")
    gateway = (root / "gateway/run.py").read_text(encoding="utf-8")

    assert "openrouter_fallback_activation_allowed" in chat_helpers
    assert "record_openrouter_fallback_activation" in chat_helpers
    assert "is_emergency_openrouter_fallback_active" in turn_finalizer
    assert "apply_openrouter_fallback_notice" in gateway
    assert "fallback_cap_message_if_exhausted" in gateway
