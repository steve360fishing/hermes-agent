from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


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
    assert "canonical provider" in meta["error"]


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
        _credential_pool = None
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
        with pytest.raises(RuntimeError, match="canonical provider"):
            background_review._resolve_review_runtime(parent)


def test_goal_judge_real_caller_remains_available_on_protected_sol_max(caplog) -> None:
    from hermes_cli import goals

    client = MagicMock()
    client.chat.completions.create.return_value = _response(
        json.dumps({"done": True, "reason": "all acceptance checks passed"})
    )

    with patch("hermes_cli.config.load_config", return_value=_enabled_config()), patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(client, "gpt-5.6-sol"),
    ), caplog.at_level("INFO", logger="agent.auxiliary_client"):
        verdict, reason, parse_failed, wait_directive = goals.judge_goal(
            "finish the task",
            "All acceptance checks passed.",
        )

    assert verdict == "done"
    assert reason == "all acceptance checks passed"
    assert parse_failed is False
    assert wait_directive is None
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-5.6-sol"
    assert kwargs["extra_body"]["reasoning"] == {
        "enabled": True,
        "effort": "max",
    }
    assert "task=goal_judge" in caplog.text
    assert "fallback_used=false" in caplog.text


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
