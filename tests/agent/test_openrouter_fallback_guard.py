from __future__ import annotations

from types import SimpleNamespace

from agent.error_classifier import FailoverReason
from agent.openrouter_fallback_guard import (
    CONTINUITY_FALLBACK_ROUTES,
    OPENROUTER_FALLBACK_MODEL,
    OPENROUTER_FALLBACK_NOTICE,
    PRIMARY_ROUTE_RESTORED_NOTICE,
    SECONDARY_FALLBACK_MODEL,
    SECONDARY_FALLBACK_NOTICE,
    apply_openrouter_fallback_notice,
    continuity_fallback_tier,
    fallback_cap_message_if_exhausted,
    fallback_notice_from_text,
    is_continuity_fallback_active,
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


def _secondary_agent(**overrides):
    values = {
        "provider": "openai-api",
        "model": SECONDARY_FALLBACK_MODEL,
        "_fallback_activated": True,
        "max_tokens": 8000,
        "session_id": "secondary-fallback-session",
        "_primary_runtime": {
            "provider": "openai-codex",
            "model": "gpt-5.6-sol",
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_incident_fallback_is_exactly_minimax_m3_via_openrouter() -> None:
    assert OPENROUTER_FALLBACK_MODEL == "minimax/minimax-m3"
    assert CONTINUITY_FALLBACK_ROUTES == (
        ("openrouter", "minimax/minimax-m3"),
        ("openai-api", "gpt-5.6-luna"),
    )


def test_cached_fallback_cap_rechecks_primary_before_blocking() -> None:
    from agent.agent_runtime_helpers import (
        fallback_cap_message_after_primary_eligibility,
    )

    agent = _agent(_openrouter_fallback_turns=99)

    def restore_primary() -> bool:
        agent._fallback_activated = False
        agent.provider = "openai-codex"
        agent.model = "gpt-5.6-sol"
        return True

    agent._restore_primary_runtime = restore_primary

    assert fallback_cap_message_after_primary_eligibility(agent) is None


def test_incident_grok_fallback_rejects_transport_timeout() -> None:
    agent = _agent(_fallback_activated=False)

    allowed, message = openrouter_fallback_activation_allowed(
        agent,
        "openrouter",
        OPENROUTER_FALLBACK_MODEL,
        reason=FailoverReason.timeout,
    )

    assert allowed is False
    assert "timeout" in message.lower()


def test_gpt56_primary_rejects_every_unexpected_openrouter_fallback(
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
    assert allowed is False
    assert "openrouter/minimax/minimax-m3 followed by" in message
    assert "openai-api/gpt-5.6-luna" in message

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
    assert response.startswith(OPENROUTER_FALLBACK_NOTICE)
    assert "FALLBACK ACTIVE: MiniMax M3 through OpenRouter" in response
    assert "GPT-5.6 Sol through the subscription is currently unavailable" in response

    health = __import__("json").loads(
        (tmp_path / "health.json").read_text(encoding="utf-8")
    )
    assert health["status"] == "degraded"
    assert health["active_provider"] == "openrouter"
    assert health["active_model"] == "minimax/minimax-m3"
    assert "MiniMax M3" in health["last_failure"]["safe_summary"]

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


def test_secondary_fallback_is_visible_on_every_response(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_GATEWAY_HEALTH_PATH", str(tmp_path / "health.json"))
    agent = _secondary_agent()

    record_openrouter_fallback_activation(agent, reason="minimax_unavailable")
    response, changed = apply_openrouter_fallback_notice(agent, "secondary response")

    assert changed is True
    assert response == f"{SECONDARY_FALLBACK_NOTICE}\n\nsecondary response"
    assert is_continuity_fallback_active(agent) is True
    assert continuity_fallback_tier(agent.provider, agent.model) == 2
    assert fallback_notice_from_text(response) == SECONDARY_FALLBACK_NOTICE

    repeated, changed = apply_openrouter_fallback_notice(agent, response)
    assert changed is False
    assert repeated == response


def test_protected_fallback_chain_is_exact_and_ordered(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_GATEWAY_HEALTH_PATH", str(tmp_path / "health.json"))
    exact_chain = [
        {"provider": "openrouter", "model": OPENROUTER_FALLBACK_MODEL},
        {"provider": "openai-api", "model": SECONDARY_FALLBACK_MODEL},
    ]
    agent = _agent(
        _fallback_activated=False,
        _fallback_chain=exact_chain,
        _fallback_index=1,
    )

    allowed, message = openrouter_fallback_activation_allowed(
        agent, "openrouter", OPENROUTER_FALLBACK_MODEL
    )
    assert allowed is True
    assert message == ""

    agent._fallback_index = 2
    allowed, message = openrouter_fallback_activation_allowed(
        agent, "openai-api", SECONDARY_FALLBACK_MODEL
    )
    assert allowed is True
    assert message == ""

    for invalid_chain in (
        exact_chain[:1],
        list(reversed(exact_chain)),
        [*exact_chain, {"provider": "openrouter", "model": "x-ai/grok-4.5"}],
        [*exact_chain, "invalid-entry"],
    ):
        agent._fallback_chain = invalid_chain
        allowed, message = openrouter_fallback_activation_allowed(
            agent, "openrouter", OPENROUTER_FALLBACK_MODEL
        )
        assert allowed is False
        assert "must contain exactly" in message

    agent._fallback_chain = exact_chain
    agent._fallback_index = 2
    allowed, message = openrouter_fallback_activation_allowed(
        agent, "openrouter", OPENROUTER_FALLBACK_MODEL
    )
    assert allowed is False
    assert "order violation" in message


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


def test_primary_restoration_notice_emits_once_for_same_session(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("HERMES_GATEWAY_HEALTH_PATH", str(tmp_path / "health.json"))
    fallback = _agent()
    record_openrouter_fallback_activation(fallback)
    apply_openrouter_fallback_notice(fallback, "continuity response")

    fallback.provider = "openai-codex"
    fallback.model = "gpt-5.6-sol"
    fallback._fallback_activated = False
    fallback._openrouter_fallback_notice_required = False

    response, changed = apply_openrouter_fallback_notice(fallback, "primary response")
    assert changed is True
    assert response.startswith(PRIMARY_ROUTE_RESTORED_NOTICE)

    response, changed = apply_openrouter_fallback_notice(fallback, "next response")
    assert changed is False
    assert response == "next response"


def test_runtime_integration_points_remain_wired() -> None:
    from pathlib import Path

    root = Path(__file__).parents[2]
    chat_helpers = (root / "agent/chat_completion_helpers.py").read_text(encoding="utf-8")
    turn_finalizer = (root / "agent/turn_finalizer.py").read_text(encoding="utf-8")
    gateway = (root / "gateway/run.py").read_text(encoding="utf-8")

    assert "openrouter_fallback_activation_allowed" in chat_helpers
    assert "record_openrouter_fallback_activation" in chat_helpers
    assert "is_continuity_fallback_active" in turn_finalizer
    assert "apply_openrouter_fallback_notice" in gateway
    assert "fallback_cap_message_after_primary_eligibility" in gateway
