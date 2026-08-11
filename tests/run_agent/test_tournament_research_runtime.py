"""End-to-end conversation coverage for non-sticky tournament chat handling."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None), finish_reason="stop")],
        model="test/model", usage=None,
    )


def _agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test", base_url="https://example.test/v1",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._disable_streaming = True
    return agent


def test_publication_request_fails_safely_and_next_turn_has_no_sticky_contract():
    agent = _agent()
    agent.client.chat.completions.create.side_effect = [_response("unverified standings"), _response("normal answer")]
    streamed, persisted = [], []
    agent.stream_delta_callback = streamed.append
    agent._persist_session = lambda messages, _history: persisted.append(list(messages))
    agent._save_trajectory = lambda *_args: None
    agent._cleanup_task_resources = lambda *_args: None

    first = agent.run_conversation("publish tournament standings")
    provider_tools = agent.client.chat.completions.create.call_args_list[0].kwargs["tools"]
    assert any(
        tool.get("function", {}).get("name") == "tournament_truth_gate"
        for tool in provider_tools
    )
    assert "public tournament copy was not released" in first["final_response"].lower()
    assert first["tournament_intent"]["state"] == "publication_request"
    assert first["tournament_intent"]["code"] == "receipt_and_release_approval_required"
    assert "unverified standings" not in str(persisted[-1])

    normal = agent.run_conversation("show me search results")
    assert normal["final_response"] == "normal answer"
    assert agent._tool_guardrails.before_call("terminal", {}).action == "allow"


def test_private_tournament_chat_does_not_require_an_external_action_receipt():
    agent = _agent()
    agent.client.chat.completions.create.return_value = _response("unverified private standings")
    streamed, persisted = [], []
    agent.stream_delta_callback = streamed.append
    agent._persist_session = lambda messages, _history: persisted.append(list(messages))
    agent._save_trajectory = lambda *_args: None
    agent._cleanup_task_resources = lambda *_args: None
    prompt = (
        "Privately answer the current tournament standings. If the current-year route receipt is "
        "unavailable, state the missing evidence precisely. Do not create a file, public artifact, newsletter copy, "
        "post, publish, or send anything outside this chat."
    )

    result = agent.run_conversation(prompt)

    assert result["final_response"] == "unverified private standings"
    assert result.get("tournament_intent") is None
    assert "unverified private standings" in str(persisted[-1])


def test_negation_reset_with_real_publication_request_remains_fail_closed():
    agent = _agent()
    agent.client.chat.completions.create.return_value = _response("unverified public standings")
    persisted = []
    agent._persist_session = lambda messages, _history: persisted.append(list(messages))
    agent._save_trajectory = lambda *_args: None
    agent._cleanup_task_resources = lambda *_args: None

    result = agent.run_conversation(
        "Do not publish the stale tournament standings, please post the current standings to the website."
    )

    assert "public tournament copy was not released" in result["final_response"].lower()
    assert result["tournament_intent"]["state"] == "publication_request"
    assert "unverified public standings" not in str(persisted[-1])


def test_missing_truth_gate_persists_one_safe_recoverable_response(monkeypatch):
    from tools.registry import registry

    agent = _agent()
    persisted = []
    agent._persist_session = lambda messages, _history: persisted.append(list(messages))
    agent._save_trajectory = lambda *_args: None
    agent._cleanup_task_resources = lambda *_args: None
    original_get_entry = registry.get_entry
    monkeypatch.setattr(
        registry,
        "get_entry",
        lambda name: None if name == "tournament_truth_gate" else original_get_entry(name),
    )

    result = agent.run_conversation("Create a public tournament Story naming winners.")

    assert result["turn_exit_reason"] == "truth_gate_unavailable"
    assert result["api_calls"] == 0
    assert len(persisted) == 1
    assert persisted[0][-1]["content"] == result["final_response"]
    assert "no public copy or external action was released" in result["final_response"].lower()
    assert agent._tournament_intent_contract is None
