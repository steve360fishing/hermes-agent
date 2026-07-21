"""End-to-end conversation regression coverage for tournament stream containment."""

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


def test_tournament_blocked_candidate_never_streams_or_persists_and_next_normal_turn_is_normal():
    agent = _agent()
    agent.client.chat.completions.create.side_effect = [_response("unverified standings"), _response("normal answer")]
    streamed, persisted = [], []
    agent.stream_delta_callback = streamed.append
    agent._persist_session = lambda messages, _history: persisted.append(list(messages))
    agent._save_trajectory = lambda *_args: None
    agent._cleanup_task_resources = lambda *_args: None

    blocked = agent.run_conversation("publish tournament standings")
    assert blocked["final_response"].startswith("PUBLIC_ARTIFACT_BLOCKED:")
    assert "unverified standings" not in streamed
    assert "unverified standings" not in str(persisted[-1])

    normal = agent.run_conversation("show me search results")
    assert normal["final_response"] == "normal answer"
    assert agent._tool_guardrails.before_call("terminal", {}).action == "allow"


def test_private_canary_negated_public_terms_return_exact_route_hold_without_leakage():
    agent = _agent()
    agent.client.chat.completions.create.return_value = _response("unverified private standings")
    streamed, persisted = [], []
    agent.stream_delta_callback = streamed.append
    agent._persist_session = lambda messages, _history: persisted.append(list(messages))
    agent._save_trajectory = lambda *_args: None
    agent._cleanup_task_resources = lambda *_args: None
    prompt = (
        "Privately answer the current tournament standings. If the current-year route receipt is "
        "unavailable, reply with ROUTE_HOLD. Do not create a file, public artifact, newsletter copy, "
        "post, publish, or send anything outside this chat."
    )

    result = agent.run_conversation(prompt)

    assert result["final_response"] == "ROUTE_HOLD"
    assert result["tournament_research"]["code"] == "receipt_missing_or_consumed"
    assert streamed == []
    assert "unverified private standings" not in str(persisted[-1])
    assert persisted[-1][-1]["content"] == "ROUTE_HOLD"


def test_affirmative_public_action_after_negation_uses_public_blocker():
    agent = _agent()
    agent.client.chat.completions.create.return_value = _response("unverified public standings")
    persisted = []
    agent._persist_session = lambda messages, _history: persisted.append(list(messages))
    agent._save_trajectory = lambda *_args: None
    agent._cleanup_task_resources = lambda *_args: None

    result = agent.run_conversation(
        "Do not publish the stale tournament standings, please post the current standings to the website."
    )

    assert result["final_response"].startswith("PUBLIC_ARTIFACT_BLOCKED:")
    assert "unverified public standings" not in str(persisted[-1])
