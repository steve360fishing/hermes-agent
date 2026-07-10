from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hermes_cli.gpt56_routing import route_spec
from tools import delegate_tool as delegation


def _enabled_config(**overrides):
    config = {
        "max_concurrent_children": 99,
        "max_async_children": 99,
        "max_spawn_depth": 99,
        "orchestrator_enabled": True,
        "max_iterations": 4,
        "gpt56_routing": {
            "enabled": True,
            "contract": "gpt56-routing-v3",
            "provider": "openai-codex",
            "max_children": 3,
            "max_depth": 1,
        },
    }
    config.update(overrides)
    return config


def _parent_agent(**overrides):
    values = {
        "_delegate_depth": 0,
        "_active_children": [],
        "_active_children_lock": None,
        "_interrupt_requested": False,
        "_memory_manager": None,
        "_session_db": None,
        "session_id": "parent",
        "session_estimated_cost_usd": 0.0,
        "session_cost_source": "none",
        "session_cost_status": "unknown",
        "model": "gpt-5.6-sol",
        "provider": "openai-codex",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_key": "parent-token-must-not-leak",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_enabled_policy_hard_caps_parallelism_and_depth(monkeypatch) -> None:
    monkeypatch.setattr(delegation, "_load_config", lambda: _enabled_config())

    assert delegation._get_max_concurrent_children() == 3
    assert delegation._get_max_async_children() == 3
    assert delegation._get_max_spawn_depth() == 1
    assert delegation._get_orchestrator_enabled() is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("contract", "wrong-contract"),
        ("provider", "openrouter"),
        ("max_children", 99),
        ("max_depth", 2),
    ],
)
def test_enabled_policy_rejects_contract_or_bound_mismatch(field, value) -> None:
    cfg = _enabled_config()
    cfg["gpt56_routing"][field] = value

    with pytest.raises(ValueError, match="Invalid delegation.gpt56_routing"):
        delegation._resolve_task_route_specs(
            [{"goal": "inventory", "route": "explorer", "reason": "mechanical"}],
            cfg=cfg,
            top_route=None,
            top_reason=None,
            mode="standard",
            parent_depth=0,
        )


def test_non_mapping_operator_policy_fails_closed() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        delegation._gpt56_routing_enabled({"gpt56_routing": "enabled"})


def test_routed_background_batches_reserve_one_global_slot_per_child() -> None:
    from tools import async_delegation

    async_delegation._reset_for_tests()
    release = threading.Event()

    def runner():
        release.wait(timeout=5)
        return {
            "results": [{"status": "completed"}] * 3,
            "total_duration_seconds": 0,
        }

    try:
        first = async_delegation.dispatch_async_delegation_batch(
            goals=["a", "b", "c"],
            context=None,
            toolsets=None,
            role="leaf",
            model=None,
            session_key="test",
            runner=runner,
            max_async_children=3,
            capacity_units=3,
        )
        second = async_delegation.dispatch_async_delegation_batch(
            goals=["d", "e", "f"],
            context=None,
            toolsets=None,
            role="leaf",
            model=None,
            session_key="test",
            runner=runner,
            max_async_children=3,
            capacity_units=3,
        )

        assert first["status"] == "dispatched"
        assert second["status"] == "rejected"
        assert "3 active and 3 requested" in second["error"]
    finally:
        release.set()
        deadline = time.monotonic() + 5
        while async_delegation.active_count() and time.monotonic() < deadline:
            time.sleep(0.01)
        async_delegation._reset_for_tests()


def test_standard_tasks_resolve_independently_and_unknown_defaults_upward() -> None:
    tasks = [
        {"goal": "inventory", "route": "explorer", "reason": "mechanical"},
        {"goal": "review", "route": "reviewer", "reason": "QA"},
        {"goal": "ambiguous"},
    ]

    specs = delegation._resolve_task_route_specs(
        tasks,
        cfg=_enabled_config(),
        top_route=None,
        top_reason=None,
        mode="standard",
        parent_depth=0,
    )

    assert [(spec.model, spec.effort) for spec in specs] == [
        ("gpt-5.6-luna", "low"),
        ("gpt-5.6-terra", "high"),
        ("gpt-5.6-sol", "max"),
    ]
    assert specs[-1].route_id == "expert"


def test_explicit_route_requires_reason_and_disabled_policy_rejects_route_fields() -> None:
    with pytest.raises(ValueError, match="requires a routing reason"):
        delegation._resolve_task_route_specs(
            [{"goal": "inventory", "route": "explorer"}],
            cfg=_enabled_config(),
            top_route=None,
            top_reason=None,
            mode="standard",
            parent_depth=0,
        )

    with pytest.raises(ValueError, match="route selection is disabled"):
        delegation._resolve_task_route_specs(
            [{"goal": "inventory", "route": "explorer", "reason": "mechanical"}],
            cfg={"gpt56_routing": {"enabled": False}},
            top_route=None,
            top_reason=None,
            mode="standard",
            parent_depth=0,
        )


def test_protected_route_drops_parent_fallback_and_reports_activation() -> None:
    parent = SimpleNamespace(_fallback_chain=[{"provider": "openrouter", "model": "anthropic/claude-haiku-4.5"}])
    assert delegation._fallback_chain_for_route(parent, route_spec("expert")) is None
    assert delegation._fallback_chain_for_route(parent, route_spec("worker")) == parent._fallback_chain

    child = SimpleNamespace(
        _gpt56_route_metadata=route_spec("worker", reason="bounded work").as_contract(),
        _fallback_activated=True,
        provider="openrouter",
    )
    metadata = delegation._child_routing_contract(child)
    assert metadata["fallback_used"] is True
    assert metadata["provider"] == "openrouter"


def test_delegate_schema_exposes_per_task_routes_and_ultra_mode() -> None:
    properties = delegation.DELEGATE_TASK_SCHEMA["parameters"]["properties"]
    task_properties = properties["tasks"]["items"]["properties"]

    assert properties["mode"]["enum"] == ["standard", "ultra"]
    assert properties["route"]["enum"] == ["explorer", "worker", "reviewer", "expert"]
    assert task_properties["route"]["enum"] == properties["route"]["enum"]
    assert "reason" in task_properties
    assert "independent" in task_properties


def test_live_dispatch_forwards_route_reason_mode_and_batch_defaults(monkeypatch) -> None:
    """The real AIAgent dispatch shim must not discard GPT-5.6 controls."""
    import run_agent

    captured = {}

    def fake_delegate(**kwargs):
        captured.update(kwargs)
        return "{}"

    monkeypatch.setattr(delegation, "delegate_task", fake_delegate)
    parent = _parent_agent()
    tasks = [
        {"goal": "inventory", "independent": True},
        {"goal": "review", "independent": True},
    ]

    run_agent.AIAgent._dispatch_delegate_task(
        parent,
        {
            "tasks": tasks,
            "context": "shared batch context",
            "role": "leaf",
            "route": "reviewer",
            "reason": "independent QA workstreams",
            "mode": "ultra",
        },
    )

    assert captured["tasks"] == tasks
    assert captured["context"] == "shared batch context"
    assert captured["role"] == "leaf"
    assert captured["route"] == "reviewer"
    assert captured["reason"] == "independent QA workstreams"
    assert captured["mode"] == "ultra"
    assert captured["background"] is True


def test_live_dispatch_reaches_ultra_validation(monkeypatch) -> None:
    """An invalid Ultra batch must fail before credentials or children exist."""
    import run_agent

    monkeypatch.setattr(delegation, "_load_config", lambda: _enabled_config())

    def fail_if_credentials_are_resolved(*_args, **_kwargs):
        pytest.fail("Ultra validation was bypassed by the live dispatch path")

    monkeypatch.setattr(
        delegation,
        "_resolve_delegation_credentials",
        fail_if_credentials_are_resolved,
    )
    result = run_agent.AIAgent._dispatch_delegate_task(
        _parent_agent(),
        {
            "tasks": [
                {"goal": "inventory", "route": "explorer", "reason": "mechanical"},
                {"goal": "review", "route": "reviewer", "reason": "QA"},
            ],
            "mode": "ultra",
        },
    )

    assert "must declare independent=true" in result


def test_delegate_task_builds_each_child_with_its_own_model_and_effort(monkeypatch) -> None:
    cfg = _enabled_config()
    monkeypatch.setattr(delegation, "_load_config", lambda: cfg)
    credential_configs = []
    built = []

    def fake_credentials(task_cfg, _parent):
        credential_configs.append(dict(task_cfg))
        return {
            "model": task_cfg.get("model"),
            "provider": task_cfg.get("provider"),
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "child-token",
            "api_mode": "codex_responses",
            "command": None,
            "args": None,
        }

    def fake_build(**kwargs):
        built.append(kwargs)
        spec = kwargs["routing_spec"]
        return SimpleNamespace(
            model=kwargs["model"],
            provider=kwargs["override_provider"],
            _delegate_role="leaf",
            _gpt56_route_metadata=spec.as_contract(),
            _fallback_activated=False,
            session_id=f"child-{kwargs['task_index']}",
            session_estimated_cost_usd=0.0,
        )

    def fake_run(task_index, goal, child, parent_agent):
        del goal, parent_agent
        return {
            "task_index": task_index,
            "status": "completed",
            "summary": "ok",
            "api_calls": 1,
            "duration_seconds": 0,
            "routing": delegation._child_routing_contract(child),
            "_child_role": "leaf",
        }

    monkeypatch.setattr(delegation, "_resolve_delegation_credentials", fake_credentials)
    monkeypatch.setattr(delegation, "_build_child_agent", fake_build)
    monkeypatch.setattr(delegation, "_run_single_child", fake_run)
    monkeypatch.setattr(delegation, "_apply_summary_budget", lambda *_: None)

    parent = SimpleNamespace(
        _delegate_depth=0,
        _active_children=[],
        _active_children_lock=None,
        _interrupt_requested=False,
        _memory_manager=None,
        _session_db=None,
        session_id="parent",
        session_estimated_cost_usd=0.0,
        session_cost_source="none",
        session_cost_status="unknown",
        model="gpt-5.6-sol",
    )
    payload = json.loads(
        delegation.delegate_task(
            tasks=[
                {"goal": "inventory", "route": "explorer", "reason": "mechanical"},
                {"goal": "review", "route": "reviewer", "reason": "QA"},
            ],
            background=False,
            parent_agent=parent,
        )
    )

    assert [item["model"] for item in credential_configs] == [
        "gpt-5.6-luna",
        "gpt-5.6-terra",
    ]
    assert [item["reasoning_effort"] for item in credential_configs] == ["low", "high"]
    assert [item["routing_spec"].route_id for item in built] == ["explorer", "reviewer"]
    assert [item["routing"]["route_id"] for item in payload["results"]] == ["explorer", "reviewer"]


def test_batch_tasks_inherit_top_level_context_and_routing_defaults(monkeypatch) -> None:
    cfg = _enabled_config()
    monkeypatch.setattr(delegation, "_load_config", lambda: cfg)
    built = []

    monkeypatch.setattr(
        delegation,
        "_resolve_delegation_credentials",
        lambda task_cfg, _parent: {
            "model": task_cfg.get("model"),
            "provider": task_cfg.get("provider"),
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "child-token",
            "api_mode": "codex_responses",
            "command": None,
            "args": None,
        },
    )

    def fake_build(**kwargs):
        built.append(kwargs)
        spec = kwargs["routing_spec"]
        return SimpleNamespace(
            model=kwargs["model"],
            provider=kwargs["override_provider"],
            _delegate_role="leaf",
            _gpt56_route_metadata=spec.as_contract(),
            _fallback_activated=False,
            session_id=f"child-{kwargs['task_index']}",
            session_estimated_cost_usd=0.0,
        )

    monkeypatch.setattr(delegation, "_build_child_agent", fake_build)
    monkeypatch.setattr(
        delegation,
        "_run_single_child",
        lambda task_index, goal, child, parent_agent: {
            "task_index": task_index,
            "status": "completed",
            "summary": "ok",
            "api_calls": 1,
            "duration_seconds": 0,
            "routing": delegation._child_routing_contract(child),
            "_child_role": "leaf",
        },
    )
    monkeypatch.setattr(delegation, "_apply_summary_budget", lambda *_: None)

    delegation.delegate_task(
        tasks=[
            {"goal": "one"},
            {"goal": "two", "context": "task-specific context"},
        ],
        context="shared batch context",
        role="leaf",
        route="worker",
        reason="bounded implementation",
        background=False,
        parent_agent=_parent_agent(),
    )

    assert [item["context"] for item in built] == [
        "shared batch context",
        "task-specific context",
    ]
    assert [item["routing_spec"].route_id for item in built] == ["worker", "worker"]


def test_routed_delegation_ignores_stale_custom_endpoint_and_credentials(monkeypatch) -> None:
    """Canonical GPT routes resolve fresh Codex auth, never stale delegation auth."""
    cfg = _enabled_config(
        base_url="https://stale-proxy.example/v1",
        api_key="stale-proxy-key",
        api_mode="chat_completions",
    )
    monkeypatch.setattr(delegation, "_load_config", lambda: cfg)
    built = []
    runtime_calls = []

    def fake_runtime_provider(**kwargs):
        runtime_calls.append(kwargs)
        return {
            "model": kwargs.get("target_model"),
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "fresh-codex-token",
            "api_mode": "codex_responses",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        fake_runtime_provider,
    )

    def fake_build(**kwargs):
        built.append(kwargs)
        spec = kwargs["routing_spec"]
        return SimpleNamespace(
            model=kwargs["model"],
            provider=kwargs["override_provider"],
            _delegate_role="leaf",
            _gpt56_route_metadata=spec.as_contract(),
            _fallback_activated=False,
            session_id="child",
            session_estimated_cost_usd=0.0,
        )

    monkeypatch.setattr(delegation, "_build_child_agent", fake_build)
    monkeypatch.setattr(
        delegation,
        "_run_single_child",
        lambda task_index, goal, child, parent_agent: {
            "task_index": task_index,
            "status": "completed",
            "summary": "ok",
            "api_calls": 1,
            "duration_seconds": 0,
            "routing": delegation._child_routing_contract(child),
            "_child_role": "leaf",
        },
    )
    monkeypatch.setattr(delegation, "_apply_summary_budget", lambda *_: None)

    delegation.delegate_task(
        goal="bounded implementation",
        route="worker",
        reason="routine code change",
        background=False,
        parent_agent=_parent_agent(
            provider="custom",
            base_url="https://parent-proxy.example/v1",
            api_key="parent-secret",
        ),
    )

    assert runtime_calls == [
        {"requested": "openai-codex", "target_model": "gpt-5.6-terra"}
    ]
    assert built[0]["override_provider"] == "openai-codex"
    assert built[0]["override_base_url"] == "https://chatgpt.com/backend-api/codex"
    assert built[0]["override_api_key"] == "fresh-codex-token"
    assert built[0]["override_api_mode"] == "codex_responses"
    assert built[0]["override_api_key"] not in {"stale-proxy-key", "parent-secret"}


@pytest.mark.parametrize(
    "base_url",
    [
        "",
        "https://parent-proxy.example/v1",
        "https://chatgpt.com:444/backend-api/codex",
        "https://chatgpt.com/backend-api/codex?upstream=custom",
    ],
)
def test_routed_credentials_reject_missing_or_noncanonical_endpoint(base_url) -> None:
    with pytest.raises(ValueError, match="credential resolution failed closed"):
        delegation._validate_routed_credentials(
            {
                "model": "gpt-5.6-terra",
                "provider": "openai-codex",
                "base_url": base_url,
                "api_key": "resolved-token",
                "api_mode": "codex_responses",
            },
            route_spec("worker", reason="bounded work"),
        )


def test_routed_capacity_rejection_never_runs_children_synchronously(monkeypatch) -> None:
    cfg = _enabled_config()
    monkeypatch.setattr(delegation, "_load_config", lambda: cfg)
    child = MagicMock()
    child.model = "gpt-5.6-terra"
    child.provider = "openai-codex"
    child.session_id = "not-started-child"
    child._delegate_role = "leaf"
    child._gpt56_route_metadata = route_spec("worker", reason="bounded work").as_contract()
    child._fallback_activated = False
    child.session_estimated_cost_usd = 0.0

    monkeypatch.setattr(
        delegation,
        "_resolve_delegation_credentials",
        lambda *_args, **_kwargs: {
            "model": "gpt-5.6-terra",
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "child-token",
            "api_mode": "codex_responses",
            "command": None,
            "args": None,
        },
    )
    monkeypatch.setattr(delegation, "_build_child_agent", lambda **_kwargs: child)

    def fail_if_child_runs(*_args, **_kwargs):
        pytest.fail("capacity-rejected routed child ran synchronously")

    monkeypatch.setattr(delegation, "_run_single_child", fail_if_child_runs)
    monkeypatch.setattr(
        "gateway.session_context.async_delivery_supported",
        lambda: True,
    )
    monkeypatch.setattr(
        "tools.async_delegation.dispatch_async_delegation_batch",
        lambda **_kwargs: {"status": "rejected", "error": "capacity reached"},
    )

    payload = json.loads(
        delegation.delegate_task(
            goal="bounded work",
            route="worker",
            reason="routine implementation",
            background=True,
            parent_agent=_parent_agent(),
        )
    )

    assert payload["status"] == "rejected"
    assert "capacity reached" in payload["error"]
    child.close.assert_called_once()


def test_legacy_capacity_rejection_preserves_synchronous_fallback(monkeypatch) -> None:
    """The new hard ceiling applies only to canonical routed delegation."""
    cfg = {
        "max_concurrent_children": 3,
        "max_iterations": 4,
        "gpt56_routing": {"enabled": False},
    }
    monkeypatch.setattr(delegation, "_load_config", lambda: cfg)
    child = MagicMock()
    child.model = "legacy-model"
    child.provider = "legacy-provider"
    child.session_id = "legacy-child"
    child._delegate_role = "leaf"
    child.session_estimated_cost_usd = 0.0
    calls = []

    monkeypatch.setattr(
        delegation,
        "_resolve_delegation_credentials",
        lambda *_args, **_kwargs: {
            "model": None,
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "command": None,
            "args": None,
        },
    )
    monkeypatch.setattr(delegation, "_build_child_agent", lambda **_kwargs: child)

    def fake_run(task_index, goal, child, parent_agent):
        calls.append(goal)
        return {
            "task_index": task_index,
            "status": "completed",
            "summary": "legacy result",
            "api_calls": 1,
            "duration_seconds": 0,
            "_child_role": "leaf",
        }

    monkeypatch.setattr(delegation, "_run_single_child", fake_run)
    monkeypatch.setattr(delegation, "_apply_summary_budget", lambda *_: None)
    monkeypatch.setattr(
        "gateway.session_context.async_delivery_supported",
        lambda: True,
    )
    monkeypatch.setattr(
        "tools.async_delegation.dispatch_async_delegation_batch",
        lambda **_kwargs: {"status": "rejected", "error": "capacity reached"},
    )

    payload = json.loads(
        delegation.delegate_task(
            goal="legacy work",
            background=True,
            parent_agent=_parent_agent(model="legacy-model", provider="legacy-provider"),
        )
    )

    assert calls == ["legacy work"]
    assert payload["results"][0]["summary"] == "legacy result"
    assert "SYNCHRONOUSLY" in payload["note"]


@pytest.mark.parametrize("interrupt", [False, True])
def test_synthetic_batch_results_keep_each_child_routing_metadata(
    monkeypatch, interrupt: bool
) -> None:
    cfg = _enabled_config()
    monkeypatch.setattr(delegation, "_load_config", lambda: cfg)

    def fake_credentials(task_cfg, _parent):
        return {
            "model": task_cfg.get("model"),
            "provider": task_cfg.get("provider"),
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "child-token",
            "api_mode": "codex_responses",
            "command": None,
            "args": None,
        }

    def fake_build(**kwargs):
        spec = kwargs["routing_spec"]
        return SimpleNamespace(
            model=kwargs["model"],
            provider=kwargs["override_provider"],
            _delegate_role="leaf",
            _gpt56_route_metadata=spec.as_contract(),
            _fallback_activated=False,
            session_id=f"child-{kwargs['task_index']}",
            session_estimated_cost_usd=0.0,
        )

    release = threading.Event()

    def fake_run(*_args, **_kwargs):
        if interrupt:
            release.wait(timeout=2)
            return {"status": "completed"}
        raise RuntimeError("synthetic child failure")

    monkeypatch.setattr(delegation, "_resolve_delegation_credentials", fake_credentials)
    monkeypatch.setattr(delegation, "_build_child_agent", fake_build)
    monkeypatch.setattr(delegation, "_run_single_child", fake_run)
    monkeypatch.setattr(delegation, "_apply_summary_budget", lambda *_: None)

    parent = SimpleNamespace(
        _delegate_depth=0,
        _active_children=[],
        _active_children_lock=None,
        _interrupt_requested=interrupt,
        _memory_manager=None,
        _session_db=None,
        session_id="parent",
        session_estimated_cost_usd=0.0,
        session_cost_source="none",
        session_cost_status="unknown",
        model="gpt-5.6-sol",
    )
    timer = threading.Timer(0.05, release.set) if interrupt else None
    if timer is not None:
        timer.start()
    try:
        payload = json.loads(
            delegation.delegate_task(
                tasks=[
                    {"goal": "inventory", "route": "explorer", "reason": "mechanical"},
                    {"goal": "review", "route": "reviewer", "reason": "QA"},
                ],
                background=False,
                parent_agent=parent,
            )
        )
    finally:
        release.set()
        if timer is not None:
            timer.join(timeout=1)

    assert [item["routing"]["route_id"] for item in payload["results"]] == [
        "explorer",
        "reviewer",
    ]
    expected_status = "interrupted" if interrupt else "error"
    assert {item["status"] for item in payload["results"]} == {expected_status}
