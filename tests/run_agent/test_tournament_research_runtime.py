"""End-to-end coverage for non-blocking tournament fact-check advisories."""

import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("LOCALAPPDATA", tempfile.gettempdir())
os.environ.setdefault("USERPROFILE", tempfile.gettempdir())

from agent.turn_origin import TurnOrigin, TurnProvenance
from gateway.run import _mint_gateway_turn_provenance
from run_agent import AIAgent


def _response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None), finish_reason="stop")],
        model="test/model",
        usage=None,
    )


def _tool_response(name, arguments, call_id="call-write"):
    tool_call = SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[tool_call]), finish_reason="tool_calls")],
        model="test/model",
        usage=None,
    )


def _agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test", base_url="https://example.test/v1", quiet_mode=True,
            skip_context_files=True, skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._disable_streaming = True
    agent.platform = "telegram"
    agent._chat_id = "chat-1"
    agent._thread_id = None
    agent._gateway_session_key = "agent:test:telegram:dm:chat-1"
    raw_run_conversation = agent.run_conversation

    def authenticated_run_conversation(*args, **kwargs):
        message = args[0] if args else kwargs.get("user_message", "")
        kwargs.setdefault("turn_provenance", _direct_provenance(message))
        return raw_run_conversation(*args, **kwargs)

    agent._raw_run_conversation = raw_run_conversation
    agent.run_conversation = authenticated_run_conversation
    return agent


def _direct_provenance(message: object, *, message_id="message-1"):
    text = message if isinstance(message, str) else "test direct request"
    return _mint_gateway_turn_provenance(
        SimpleNamespace(text=text, message_id=message_id),
        SimpleNamespace(
            user_id="steve", platform="telegram", profile="test", chat_id="chat-1",
            scope_id="telegram:test:chat-1:",
        ),
        is_internal=False,
    )


def _prepare(agent):
    agent._persist_session = lambda *_args: None
    agent._save_trajectory = lambda *_args: None
    agent._cleanup_task_resources = lambda *_args: None


def _assert_no_automatic_truth_gate(agent):
    assert "tournament_truth_gate" not in agent.valid_tool_names
    assert not any(tool.get("function", {}).get("name") == "tournament_truth_gate" for tool in agent.tools)


def test_publication_words_are_ordinary_conversation_without_a_contract():
    agent = _agent()
    _prepare(agent)
    agent.client.chat.completions.create.side_effect = [_response("Here is a private draft."), _response("normal answer")]

    first = agent.run_conversation("Publish this exact verified tournament caption to the SportFish Hub Instagram account now.")
    normal = agent.run_conversation("show me search results")

    assert first["final_response"] == "Here is a private draft."
    assert "tournament_intent" not in first
    assert normal["final_response"] == "normal answer"
    assert agent._tool_guardrails.before_call("terminal", {}).action == "allow"
    _assert_no_automatic_truth_gate(agent)


def test_runtime_async_completion_preserves_useful_output_without_an_advisory():
    fixture = Path(__file__).parents[1] / "fixtures" / "tournament" / "async_completion_origin_incident_reconstructed.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))["canonical_reconstructed_payload"]
    assert len(payload.encode("utf-8")) == 4996
    agent = _agent()
    _prepare(agent)
    agent.client.chat.completions.create.return_value = _response("Useful private completion.")

    result = agent._raw_run_conversation(payload, turn_provenance=TurnProvenance.internal(TurnOrigin.RUNTIME_ASYNC_COMPLETION))

    assert result["final_response"] == "Useful private completion."
    assert "tournament_intent" not in result
    assert "Fact check:" not in result["final_response"]
    assert agent._tool_guardrails._tournament_contract is None


def test_sealed_private_authority_beats_public_effective_text():
    agent = _agent()
    _prepare(agent)
    agent.client.chat.completions.create.return_value = _response("useful private answer")
    provenance = _direct_provenance("Give me a private Codex prompt about the tournament blocker.", message_id="authority-split")

    result = agent._raw_run_conversation(
        "Create a tournament Story naming winners for Instagram.",
        persist_user_message="Create a tournament Story naming winners for Instagram.",
        turn_provenance=provenance,
    )

    assert result["final_response"] == "useful private answer"
    assert "tournament_intent" not in result
    assert "Fact check:" not in result["final_response"]


def test_private_workbook_incident_remains_usable():
    agent = _agent()
    _prepare(agent)
    agent.client.chat.completions.create.return_value = _response("Workbook updated for private review.")
    authority = "Please make a story about this week's tournament results and update the Excel workbook."

    result = agent._raw_run_conversation(
        "Create a tournament caption for Instagram.",
        persist_user_message="Create a tournament caption for Instagram.",
        turn_provenance=_direct_provenance(authority, message_id="private-workbook-incident"),
    )

    assert result["final_response"] == "Workbook updated for private review."
    assert "DRAFT_VALIDATION_HOLD" not in result["final_response"]
    assert "tournament_intent" not in result
    assert agent._tool_guardrails.before_call("write_file", {}).action == "allow"


def test_claim_bearing_tournament_story_preserves_draft_with_unavailable_advisory():
    agent = _agent()
    _prepare(agent)
    draft = "Captain Reyes won the tournament."
    agent.client.chat.completions.create.return_value = _response(draft)

    result = agent.run_conversation("Create a tournament Story naming winners.")

    assert draft in result["final_response"]
    assert "Fact check: verification unavailable" in result["final_response"]
    assert "DRAFT_VALIDATION_HOLD" not in result["final_response"]
    assert "tournament_intent" not in result
    assert agent._tool_guardrails.before_call("write_file", {}).action == "allow"
    _assert_no_automatic_truth_gate(agent)


def test_claim_bearing_story_output_gets_an_advisory_even_when_request_has_no_fact_claim():
    agent = _agent()
    _prepare(agent)
    draft = "Captain Reyes won the tournament."
    agent.client.chat.completions.create.return_value = _response(draft)

    result = agent.run_conversation("Create a tournament Story for Instagram.")

    assert draft in result["final_response"]
    assert "Fact check: verification unavailable" in result["final_response"]
    assert "DRAFT_VALIDATION_HOLD" not in result["final_response"]
    assert "tournament_intent" not in result
    _assert_no_automatic_truth_gate(agent)


def test_copied_story_instructions_do_not_add_an_advisory_or_contract():
    agent = _agent()
    _prepare(agent)
    agent.client.chat.completions.create.return_value = _response("I can help review that copy.")
    copied = "This is a copy of the message I sent you last night:\n\nCreate a tournament Story naming winners for Instagram."

    result = agent.run_conversation(copied)

    assert result["final_response"] == "I can help review that copy."
    assert "tournament_intent" not in result
    assert agent._tool_guardrails.before_call("write_file", {}).action == "allow"


def test_mixed_tracker_work_and_story_draft_remain_usable():
    agent = _agent()
    _prepare(agent)
    draft = "Tracker updated. Captain Reyes won the tournament."
    agent.client.chat.completions.create.return_value = _response(draft)

    result = agent.run_conversation("Update the tournament tracker and create a Story naming winners for Instagram.")

    assert draft in result["final_response"]
    assert "Fact check: verification unavailable" in result["final_response"]
    assert result["completed"] is True
    assert "tournament_intent" not in result
    assert agent._tool_guardrails.before_call("write_file", {}).action == "allow"


def test_story_and_private_file_work_complete_without_a_receipt_gate(monkeypatch, tmp_path):
    test_root = Path.cwd() / ".pytest-hermes-artifacts" / tmp_path.name
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(Path.cwd()))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(test_root / "artifacts"))
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"agent": {"artifact_only_enabled": True}})
    agent = _agent()
    _prepare(agent)
    agent.tools.append({"type": "function", "function": {"name": "write_file", "description": "Write a file", "parameters": {}}})
    agent.valid_tool_names.add("write_file")
    target = test_root / "private_tracker.txt"
    target.parent.mkdir(parents=True)
    observed = {"target": target}

    def provider_response(*_args, **_kwargs):
        if not observed.get("called"):
            observed["called"] = True
            return _tool_response("write_file", {"path": str(target), "content": "Tracker updated."})
        return _response(f"MEDIA:{observed['target']}")

    agent.client.chat.completions.create.side_effect = provider_response
    try:
        result = agent.run_conversation("Create private_tracker.txt, update the tournament tracker, and create a Story naming winners.")

        assert observed["target"].read_text(encoding="utf-8") == "Tracker updated."
        assert f"MEDIA:{observed['target']}" in result["final_response"]
        assert "DRAFT_VALIDATION_HOLD" not in result["final_response"]
        assert "tournament_intent" not in result
    finally:
        shutil.rmtree(test_root, ignore_errors=True)
