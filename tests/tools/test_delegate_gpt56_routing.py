from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

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
            "base_url": None,
            "api_key": None,
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
            "base_url": None,
            "api_key": None,
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
