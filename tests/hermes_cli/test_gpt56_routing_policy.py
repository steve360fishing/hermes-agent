from __future__ import annotations

import pytest


def _routing():
    return __import__("hermes_cli.gpt56_routing", fromlist=["*"])


def test_canonical_route_contract_uses_least_capable_sufficient_tiers() -> None:
    routing = _routing()

    assert routing.route_spec("main").as_contract() == {
        "route_id": "main",
        "model_alias": "gpt56_sol",
        "effort": "max",
        "reason": "primary controller",
        "fanout_depth": 0,
        "fallback_used": False,
        "provider": "openai-codex",
    }
    assert routing.route_spec("explorer").model == "gpt-5.6-luna"
    assert routing.route_spec("worker").model == "gpt-5.6-terra"
    assert routing.route_spec("reviewer").effort == "high"
    assert routing.route_spec("expert").protected is True


def test_unknown_or_stale_routes_fail_closed() -> None:
    routing = _routing()

    with pytest.raises(routing.RoutingPolicyError, match="unknown route"):
        routing.route_spec("cheap")
    for stale in ("gpt-5.5", "gpt-5.6-sol-pro", "gpt-5.6-unknown"):
        with pytest.raises(routing.RoutingPolicyError, match="unsupported model"):
            routing.validate_model_effort(stale, "max")


def test_effort_capabilities_are_model_specific() -> None:
    routing = _routing()

    routing.validate_model_effort("gpt-5.6-sol", "ultra")
    routing.validate_model_effort("gpt-5.6-terra", "ultra")
    routing.validate_model_effort("gpt-5.6-luna", "max")
    with pytest.raises(routing.RoutingPolicyError, match="does not support effort"):
        routing.validate_model_effort("gpt-5.6-luna", "ultra")


def test_auxiliary_slots_match_the_accepted_policy() -> None:
    routing = _routing()

    expected = {
        "skills_hub": ("gpt-5.6-luna", "low"),
        "mcp": ("gpt-5.6-luna", "low"),
        "title_generation": ("gpt-5.6-luna", "low"),
        "tts_audio_tags": ("gpt-5.6-luna", "low"),
        "monitor": ("gpt-5.6-luna", "low"),
        "profile_describer": ("gpt-5.6-luna", "low"),
        "web_extract": ("gpt-5.6-luna", "medium"),
        "compression": ("gpt-5.6-luna", "medium"),
        "triage_specifier": ("gpt-5.6-terra", "medium"),
        "coding-qa-worker": ("gpt-5.6-terra", "medium"),
        "moa_reference": ("gpt-5.6-terra", "medium"),
        "vision": ("gpt-5.6-terra", "high"),
        "kanban_decomposer": ("gpt-5.6-terra", "high"),
        "approval": ("gpt-5.6-sol", "max"),
        "curator": ("gpt-5.6-sol", "max"),
        "background_review": ("gpt-5.6-sol", "max"),
        "moa_aggregator": ("gpt-5.6-sol", "max"),
    }
    assert set(routing.AUXILIARY_ROUTES) == set(expected)
    assert "session_search" not in routing.AUXILIARY_ROUTES
    for task, pair in expected.items():
        spec = routing.auxiliary_spec(task)
        assert (spec.model, spec.effort) == pair


def test_ultra_requires_independent_explicitly_routed_leaf_workstreams() -> None:
    routing = _routing()
    valid = [
        {"goal": "inventory files", "route": "explorer", "reason": "mechanical inventory", "independent": True},
        {"goal": "run deterministic tests", "route": "worker", "reason": "bounded verification", "independent": True},
        {"goal": "review edge cases", "route": "reviewer", "reason": "independent QA", "independent": True},
    ]

    resolved = routing.validate_ultra_tasks(valid, parent_depth=0)
    assert [item.route_id for item in resolved] == ["explorer", "worker", "reviewer"]
    assert all(item.fanout_depth == 1 for item in resolved)

    with pytest.raises(routing.RoutingPolicyError, match="two or three"):
        routing.validate_ultra_tasks(valid[:1], parent_depth=0)
    with pytest.raises(routing.RoutingPolicyError, match="top-level parent"):
        routing.validate_ultra_tasks(valid, parent_depth=1)
    with pytest.raises(routing.RoutingPolicyError, match="independent=true"):
        routing.validate_ultra_tasks([{**valid[0], "independent": False}, valid[1]], parent_depth=0)
    with pytest.raises(routing.RoutingPolicyError, match="explicit route"):
        routing.validate_ultra_tasks([{k: v for k, v in valid[0].items() if k != "route"}, valid[1]], parent_depth=0)


def test_protected_routes_never_inherit_fallback() -> None:
    routing = _routing()

    assert routing.route_spec("main").allow_fallback is False
    assert routing.route_spec("expert").allow_fallback is False
    assert routing.auxiliary_spec("approval").allow_fallback is False
    assert routing.auxiliary_spec("vision").allow_fallback is True


def test_operator_config_validator_is_shared_and_fail_closed() -> None:
    routing = _routing()
    valid = {
        "enabled": True,
        "contract": routing.ROUTING_CONTRACT,
        "provider": routing.PRIMARY_PROVIDER,
        "max_children": routing.MAX_CHILDREN,
        "max_depth": routing.MAX_DEPTH,
    }

    assert routing.validate_operator_config(valid) is True
    assert routing.validate_operator_config({"enabled": False}) is False
    with pytest.raises(routing.RoutingPolicyError, match="provider must be"):
        routing.validate_operator_config({**valid, "provider": "openrouter"})
