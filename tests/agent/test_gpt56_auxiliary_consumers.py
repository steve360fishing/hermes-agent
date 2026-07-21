from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clear_auxiliary_client_cache():
    from agent.auxiliary_client import shutdown_cached_clients

    shutdown_cached_clients()
    yield
    shutdown_cached_clients()


def _enabled_config() -> dict:
    return {
        "model": {
            "provider": "openrouter",
            "default": "legacy-parent-model",
        },
        "delegation": {
            "gpt56_routing": {
                "enabled": True,
                "contract": "gpt56-routing-v3",
                "provider": "openai-codex",
                "max_children": 3,
                "max_depth": 1,
            }
        },
    }


def _disabled_config() -> dict:
    return {
        "model": {
            "provider": "openrouter",
            "default": "legacy-parent-model",
        },
        "delegation": {"gpt56_routing": {"enabled": False}},
    }


def _background_parent() -> SimpleNamespace:
    return SimpleNamespace(
        provider="openrouter",
        model="legacy-parent-model",
        _current_main_runtime=lambda: {
            "api_key": "legacy-key",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
        },
    )


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_curator_real_fork_enforces_protected_sol_max_and_reports_no_fallback(
    caplog,
) -> None:
    from agent import curator

    captured: dict = {}

    class _ReviewAgent:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs
            self.provider = kwargs.get("provider")
            self.model = kwargs.get("model")
            self.api_mode = kwargs.get("api_mode")
            self.base_url = kwargs.get("base_url")
            self.api_key = kwargs.get("api_key")
            self._memory_nudge_interval = 1
            self._skill_nudge_interval = 1
            self._memory_write_origin = "assistant_tool"
            self._session_messages = []
            self._fallback_activated = False

        def run_conversation(self, **kwargs):
            return {"final_response": "no changes"}

        def close(self):
            pass

    resolved_runtime = {
        "provider": "openai-codex",
        "api_key": "oauth-token",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_mode": "codex_responses",
    }
    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        return_value=resolved_runtime,
    ) as resolve_runtime, patch("run_agent.AIAgent", _ReviewAgent), caplog.at_level(
        "INFO", logger="agent.curator"
    ):
        meta = curator._run_llm_review("review the skill library")

    assert resolve_runtime.call_args.kwargs["requested"] == "openai-codex"
    assert resolve_runtime.call_args.kwargs["target_model"] == "gpt-5.6-sol"
    assert captured["kwargs"]["provider"] == "openai-codex"
    assert captured["kwargs"]["model"] == "gpt-5.6-sol"
    assert captured["kwargs"]["reasoning_config"] == {
        "enabled": True,
        "effort": "max",
    }
    assert captured["kwargs"]["fallback_model"] == []
    assert meta["route_id"] == "expert"
    assert meta["model_alias"] == "gpt56_sol"
    assert meta["effort"] == "max"
    assert meta["fallback_used"] is False
    assert "task=curator" in caplog.text
    assert "fallback_used=false" in caplog.text


def test_curator_invalid_enabled_policy_fails_closed_before_agent_construction() -> None:
    from agent import curator

    invalid = _enabled_config()
    invalid["delegation"]["gpt56_routing"]["provider"] = "openrouter"
    agent_factory = MagicMock()

    with patch("hermes_cli.config.load_config", return_value=invalid), patch(
        "run_agent.AIAgent", agent_factory
    ):
        meta = curator._run_llm_review("review the skill library")

    agent_factory.assert_not_called()
    assert "Invalid delegation.gpt56_routing" in meta["error"]


def test_curator_config_read_failure_fails_closed_before_agent_construction() -> None:
    from agent import curator

    agent_factory = MagicMock()
    with patch(
        "hermes_cli.config.load_config",
        side_effect=RuntimeError("config unreadable"),
    ), patch("run_agent.AIAgent", agent_factory):
        meta = curator._run_llm_review("review the skill library")

    agent_factory.assert_not_called()
    assert "config" in meta["error"].lower()


@pytest.mark.parametrize(
    "base_url,api_key",
    [
        ("https://openrouter.ai/api/v1", "oauth-token"),
        ("https://chatgpt.com/backend-api/codex", ""),
    ],
)
def test_curator_rejects_noncanonical_endpoint_or_missing_token(
    base_url: str,
    api_key: str,
) -> None:
    from agent import curator

    agent_factory = MagicMock()
    resolved = {
        "provider": "openai-codex",
        "api_key": api_key,
        "base_url": base_url,
        "api_mode": "codex_responses",
    }
    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        return_value=resolved,
    ), patch("run_agent.AIAgent", agent_factory):
        meta = curator._run_llm_review("review the skill library")

    agent_factory.assert_not_called()
    assert "failed closed" in meta["error"]


def test_curator_rejects_runtime_provider_substitution_before_construction() -> None:
    from agent import curator

    agent_factory = MagicMock()
    substituted = {
        "provider": "openrouter",
        "api_key": "wrong-provider-key",
        "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions",
    }
    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        return_value=substituted,
    ), patch("run_agent.AIAgent", agent_factory):
        meta = curator._run_llm_review("review the skill library")

    agent_factory.assert_not_called()
    assert "not canonical" in meta["error"]


def test_background_review_real_fork_enforces_protected_sol_max_and_reports_no_fallback(
    caplog,
) -> None:
    from agent import background_review

    captured: dict = {}

    class _Parent:
        provider = "openrouter"
        model = "legacy-parent-model"
        platform = "test"
        session_id = "parent-session"
        enabled_toolsets = ["memory", "skills"]
        disabled_toolsets = []
        _credential_pool = object()
        _memory_store = None
        _memory_enabled = False
        _user_profile_enabled = False
        _cached_system_prompt = "parent prompt"
        session_start = SimpleNamespace()
        memory_notifications = "on"
        background_review_callback = None

        def _current_main_runtime(self):
            return {
                "api_key": "legacy-key",
                "base_url": "https://openrouter.ai/api/v1",
                "api_mode": "chat_completions",
            }

        def _safe_print(self, *_args, **_kwargs):
            pass

        def _emit_auxiliary_failure(self, *_args, **_kwargs):
            pass

    class _ReviewAgent:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs
            self.provider = kwargs.get("provider")
            self.model = kwargs.get("model")
            self.api_mode = kwargs.get("api_mode")
            self.base_url = kwargs.get("base_url")
            self.api_key = kwargs.get("api_key")
            self._memory_write_origin = None
            self._memory_write_context = None
            self._memory_store = None
            self._memory_enabled = None
            self._user_profile_enabled = None
            self._memory_nudge_interval = None
            self._skill_nudge_interval = None
            self._cached_system_prompt = None
            self.session_start = None
            self.session_id = None
            self.suppress_status_output = None
            self._session_messages = []
            self._fallback_activated = False

        def run_conversation(self, *args, **kwargs):
            return {"final_response": "reviewed"}

        def shutdown_memory_provider(self):
            pass

        def close(self):
            pass

    resolved_runtime = {
        "provider": "openai-codex",
        "api_key": "oauth-token",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_mode": "codex_responses",
    }
    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        return_value=resolved_runtime,
    ), patch("run_agent.AIAgent", _ReviewAgent), patch(
        "model_tools.get_tool_definitions", return_value=[]
    ), caplog.at_level("INFO", logger="agent.background_review"):
        runtime = background_review._resolve_review_runtime(_Parent())
        background_review._run_review_in_thread(
            _Parent(), [], "review memory and skills"
        )

    assert runtime["route_id"] == "expert"
    assert runtime["model_alias"] == "gpt56_sol"
    assert runtime["effort"] == "max"
    assert runtime["provider"] == "openai-codex"
    assert runtime["fallback_used"] is False
    assert captured["kwargs"]["provider"] == "openai-codex"
    assert captured["kwargs"]["model"] == "gpt-5.6-sol"
    assert captured["kwargs"]["reasoning_config"] == {
        "enabled": True,
        "effort": "max",
    }
    assert captured["kwargs"]["fallback_model"] == []
    assert captured["kwargs"]["credential_pool"] is None
    assert "task=background_review" in caplog.text
    assert "fallback_used=false" in caplog.text


def test_background_review_rejects_runtime_provider_substitution() -> None:
    from agent import background_review

    parent = SimpleNamespace(
        provider="openrouter",
        model="legacy-parent-model",
        _current_main_runtime=lambda: {
            "api_key": "legacy-key",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
        },
    )
    substituted = {
        "provider": "openrouter",
        "api_key": "wrong-provider-key",
        "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions",
    }
    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        return_value=substituted,
    ):
        with pytest.raises(RuntimeError, match="not canonical"):
            background_review._resolve_review_runtime(parent)


def test_background_review_config_read_failure_fails_closed() -> None:
    from agent import background_review

    with patch(
        "hermes_cli.config.load_config",
        side_effect=RuntimeError("config unreadable"),
    ):
        with pytest.raises(RuntimeError, match="config"):
            background_review._resolve_review_runtime(_background_parent())


@pytest.mark.parametrize(
    "base_url,api_key",
    [
        ("https://chatgpt.com/backend-api/not-codex", "oauth-token"),
        ("https://chatgpt.com/backend-api/codex", ""),
    ],
)
def test_background_review_rejects_noncanonical_endpoint_or_missing_token(
    base_url: str,
    api_key: str,
) -> None:
    from agent import background_review

    resolved = {
        "provider": "openai-codex",
        "api_key": api_key,
        "base_url": base_url,
        "api_mode": "codex_responses",
    }
    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        return_value=resolved,
    ):
        with pytest.raises(RuntimeError, match="failed closed"):
            background_review._resolve_review_runtime(_background_parent())


def test_goal_judge_real_caller_remains_available_on_protected_sol_max(caplog) -> None:
    from hermes_cli import goals

    client = MagicMock()
    client.base_url = "https://chatgpt.com/backend-api/codex"
    client.api_key = "oauth-token"
    client.chat.completions.create.return_value = _response(
        json.dumps({"done": True, "reason": "all acceptance checks passed"})
    )

    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(client, "gpt-5.6-sol"),
    ), caplog.at_level("INFO", logger="agent.auxiliary_client"):
        verdict, reason, parse_failed, wait_directive, transport_failed = goals.judge_goal(
            "finish the task",
            "All acceptance checks passed.",
        )

    assert verdict == "done"
    assert reason == "all acceptance checks passed"
    assert parse_failed is False
    assert wait_directive is None
    assert transport_failed is False
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-5.6-sol"
    assert kwargs["extra_body"]["reasoning"] == {
        "enabled": True,
        "effort": "max",
    }
    assert "task=goal_judge" in caplog.text
    assert "fallback_used=false" in caplog.text


def test_goal_judge_uses_one_enabled_policy_snapshot() -> None:
    from hermes_cli import goals

    client = MagicMock()
    client.base_url = "https://chatgpt.com/backend-api/codex"
    client.api_key = "oauth-token"
    client.chat.completions.create.return_value = _response(
        json.dumps({"done": True, "reason": "verified"})
    )

    calls = 0

    def drifting_config(*_args):
        nonlocal calls
        calls += 1
        return _enabled_config() if calls == 1 else _disabled_config()

    with patch(
        "agent.auxiliary_client._load_auxiliary_config_snapshot",
        side_effect=drifting_config,
    ) as loader, patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(client, "gpt-5.6-sol"),
    ):
        verdict, reason, parse_failed, wait_directive, transport_failed = goals.judge_goal(
            "finish the task",
            "Verified evidence is attached.",
        )

    assert verdict == "done"
    assert reason == "verified"
    assert parse_failed is False
    assert wait_directive is None
    assert transport_failed is False
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-5.6-sol"
    assert kwargs["extra_body"]["reasoning"] == {
        "enabled": True,
        "effort": "max",
    }
    assert loader.call_count == 1


@pytest.mark.asyncio
async def test_teams_call_real_consumer_stays_provider_locked_without_gpt_alias(
    tmp_path,
    caplog,
) -> None:
    from plugins.teams_pipeline.models import TeamsMeetingRef
    from plugins.teams_pipeline.pipeline import TeamsMeetingPipeline
    from plugins.teams_pipeline.store import TeamsPipelineStore

    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=_response(
            json.dumps(
                {
                    "summary": "Provider-owned summary",
                    "key_decisions": [],
                    "action_items": [],
                    "risks": [],
                    "confidence": "high",
                    "confidence_notes": "native route",
                }
            )
        )
    )
    pipeline = TeamsMeetingPipeline(
        graph_client=object(),
        store=TeamsPipelineStore(tmp_path / "teams-store.json"),
    )

    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), patch(
        "agent.auxiliary_client._get_cached_client",
        return_value=(client, "provider-owned-model"),
    ), caplog.at_level("INFO", logger="agent.auxiliary_client"):
        payload = await pipeline._generate_summary_payload(
            resolved_meeting=TeamsMeetingRef(
                meeting_id="meeting-1",
                metadata={"subject": "Weekly sync"},
            ),
            transcript_text="A sufficiently detailed meeting transcript.",
            artifacts=[],
        )

    assert payload.summary == "Provider-owned summary"
    assert client.chat.completions.create.await_count == 1
    assert client.chat.completions.create.await_args.kwargs["model"] == (
        "provider-owned-model"
    )
    assert "task=call provider_locked=true owner=teams_pipeline complexity=medium" in caplog.text
    assert "GPT-5.6 auxiliary route task=call" not in caplog.text
