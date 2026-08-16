"""End-to-end conversation coverage for non-sticky tournament chat handling."""

import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("LOCALAPPDATA", tempfile.gettempdir())
os.environ.setdefault("USERPROFILE", tempfile.gettempdir())

from run_agent import AIAgent
from agent.turn_origin import TurnOrigin, TurnProvenance
from gateway.run import _mint_gateway_turn_provenance


def _response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None), finish_reason="stop")],
        model="test/model", usage=None,
    )


def _tool_response(name, arguments, call_id="call-truth"):
    tool_call = SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="", tool_calls=[tool_call]),
                finish_reason="tool_calls",
            )
        ],
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
            api_key="test", base_url="https://example.test/v1",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
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
        kwargs.setdefault(
            "turn_provenance",
            _direct_provenance(message),
        )
        return raw_run_conversation(*args, **kwargs)

    agent._raw_run_conversation = raw_run_conversation
    agent.run_conversation = authenticated_run_conversation
    return agent


def test_publication_request_fails_safely_and_next_turn_has_no_sticky_contract():
    agent = _agent()
    agent.client.chat.completions.create.side_effect = [_response("unverified standings"), _response("normal answer")]
    streamed, persisted = [], []
    agent.stream_delta_callback = streamed.append
    agent._persist_session = lambda messages, _history: persisted.append(list(messages))
    agent._save_trajectory = lambda *_args: None
    agent._cleanup_task_resources = lambda *_args: None

    first = agent.run_conversation(
        "Publish this exact verified tournament caption to the SportFish Hub Instagram account now."
    )
    provider_tools = agent.client.chat.completions.create.call_args_list[0].kwargs["tools"]
    assert any(
        tool.get("function", {}).get("name") == "tournament_truth_gate"
        for tool in provider_tools
    )
    assert "publication was not attempted" in first["final_response"].lower()
    assert "receipt_and_release_approval_required" in first["final_response"]
    assert first["tournament_intent"]["state"] == "publication_request"
    assert first["tournament_intent"]["code"] == "receipt_and_release_approval_required"
    assert "unverified standings" not in str(persisted[-1])

    normal = agent.run_conversation("show me search results")
    assert normal["final_response"] == "normal answer"
    assert agent._tool_guardrails.before_call("terminal", {}).action == "allow"


def _direct_provenance(message: object, *, message_id="message-1"):
    text = message if isinstance(message, str) else "test direct request"
    return _mint_gateway_turn_provenance(
        SimpleNamespace(text=text, message_id=message_id),
        SimpleNamespace(user_id="steve", platform="telegram", profile="test", chat_id="chat-1", scope_id="telegram:test:chat-1:"),
        is_internal=False,
    )


def test_runtime_async_completion_preserves_useful_output_without_tournament_finalizer():
    fixture_path = (
        Path(__file__).parents[1]
        / "fixtures"
        / "tournament"
        / "async_completion_origin_incident_reconstructed.json"
    )
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))[
        "canonical_reconstructed_payload"
    ]
    assert len(payload.encode("utf-8")) == 4996
    useful_response = (
        "I preserved the private engineering report. Draft review never needs "
        "publication approval, and blanket permanent approval grants no authority."
    )
    agent = _agent()
    agent.client.chat.completions.create.return_value = _response(useful_response)
    agent._persist_session = lambda *_args: None
    agent._save_trajectory = lambda *_args: None
    agent._cleanup_task_resources = lambda *_args: None

    result = agent._raw_run_conversation(
        payload,
        turn_provenance=TurnProvenance.internal(
            TurnOrigin.RUNTIME_ASYNC_COMPLETION
        ),
    )

    assert result["final_response"] == useful_response
    assert "tournament_intent" not in result
    assert result["messages"][0]["turn_origin"] == "runtime_async_completion"
    wire_messages = agent.client.chat.completions.create.call_args.kwargs["messages"]
    assert all("turn_origin" not in message for message in wire_messages)
    assert all("turn_actor_identity" not in message for message in wire_messages)
    assert agent._tool_guardrails._tournament_contract is None


def test_conversation_contract_uses_sealed_authority_not_effective_message():
    agent = _agent()
    agent.client.chat.completions.create.return_value = _response("useful private answer")
    agent._persist_session = lambda *_args: None
    agent._save_trajectory = lambda *_args: None
    agent._cleanup_task_resources = lambda *_args: None
    provenance = _direct_provenance(
        "Give me a private Codex prompt about the tournament blocker.",
        message_id="message-authority-split",
    )

    result = agent._raw_run_conversation(
        "Publish the tournament Story to the SportFish Hub Instagram account now.",
        persist_user_message="Plugin-generated public tournament publication text.",
        turn_provenance=provenance,
    )

    assert result["final_response"] == "useful private answer"
    assert "tournament_intent" not in result
    assert agent._tool_guardrails._tournament_contract is None
    provenance_keys = {
        "turn_platform", "turn_profile", "turn_chat_id", "turn_thread_id",
        "turn_message_id", "turn_event_id", "turn_session_scope",
        "turn_authority_text_sha256", "turn_captured_at_unix_ms",
        "turn_binding_sha256",
    }
    assert provenance_keys <= result["messages"][0].keys()
    wire_messages = agent.client.chat.completions.create.call_args.kwargs["messages"]
    assert all(provenance_keys.isdisjoint(message) for message in wire_messages)


def test_private_workbook_incident_authority_reaches_finalizer_without_tournament_hold():
    """The exact VPS lockout shape stays private through the full turn path."""
    authority_text = (
        "Please make a story about this week's tournament results and update "
        "the Excel workbook."
    )
    agent = _agent()
    agent.client.chat.completions.create.return_value = _response(
        "Workbook updated for private review."
    )
    agent._persist_session = lambda *_args: None
    agent._save_trajectory = lambda *_args: None
    agent._cleanup_task_resources = lambda *_args: None

    result = agent._raw_run_conversation(
        "Create a tournament caption for Instagram.",
        persist_user_message="Create a tournament caption for Instagram.",
        turn_provenance=_direct_provenance(
            authority_text, message_id="message-private-workbook-incident"
        ),
    )

    assert result["final_response"] == "Workbook updated for private review."
    assert "DRAFT_VALIDATION_HOLD" not in result["final_response"]
    assert "tournament_intent" not in result
    assert agent._tool_guardrails.before_call("write_file", {}).action == "allow"


def test_effective_private_rewrite_cannot_remove_sealed_publication_gate():
    agent = _agent()
    agent.client.chat.completions.create.return_value = _response("unverified public copy")
    agent._persist_session = lambda *_args: None
    agent._save_trajectory = lambda *_args: None
    agent._cleanup_task_resources = lambda *_args: None
    provenance = _direct_provenance(
        "Publish the tournament Story to the SportFish Hub Instagram account now.",
        message_id="message-public-authority",
    )

    result = agent._raw_run_conversation(
        "Plugin rewrote this as an ordinary private note.",
        persist_user_message="Ordinary private note.",
        turn_provenance=provenance,
    )

    assert result["tournament_intent"]["state"] == "publication_request"
    assert result["tournament_intent"]["code"] == "receipt_and_release_approval_required"
    assert "unverified public copy" not in result["final_response"]


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

    assert "publication was not attempted" in result["final_response"].lower()
    assert "receipt_and_release_approval_required" in result["final_response"]
    assert result["tournament_intent"]["state"] == "publication_request"
    assert "unverified public standings" not in str(persisted[-1])


def test_full_finalizer_clears_generic_turn_callback_after_tournament_hold():
    agent = _agent()
    original_stream = lambda _delta: None
    agent._stream_callback = original_stream
    agent.client.chat.completions.create.return_value = _response("unsupported public copy")
    agent._persist_session = lambda *_args: None
    agent._save_trajectory = lambda *_args: None
    agent._cleanup_task_resources = lambda *_args: None

    result = agent.run_conversation(
        "Create a public tournament Story naming winners."
    )

    provider_tools = agent.client.chat.completions.create.call_args.kwargs["tools"]
    assert any(
        tool.get("function", {}).get("name") == "tournament_truth_gate"
        for tool in provider_tools
    )
    assert result["tournament_intent"]["code"] == "receipt_missing_or_consumed"
    assert agent._stream_callback is None


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

    result = agent.run_conversation(
        "Create a public tournament Story naming winners."
    )

    assert result["turn_exit_reason"] == "truth_gate_unavailable"
    assert result["api_calls"] == 0
    assert len(persisted) == 1
    assert persisted[0][-1]["content"] == result["final_response"]
    assert "no public copy or external action was released" in result["final_response"].lower()
    from agent.tournament_intent_contract import current_tournament_contract

    assert current_tournament_contract() is None


def test_mixed_private_public_turn_returns_useful_private_partial_without_public_leak():
    agent = _agent()
    raw = '{"private_response":"Private handoff ready.","public_candidate":"Boat A won."}'
    agent.client.chat.completions.create.return_value = _response(raw)
    streamed, persisted = [], []
    agent.stream_delta_callback = streamed.append
    agent._persist_session = lambda messages, _history: persisted.append(list(messages))
    agent._save_trajectory = lambda *_args: None
    agent._cleanup_task_resources = lambda *_args: None

    result = agent.run_conversation(
        "Create a private coding handoff, then publish the tournament Story to the SportFish Hub Instagram account."
    )

    assert result["completed"] is False
    assert result["failed"] is False
    assert result["partial"] is True
    assert result["tournament_intent"]["state"] == "mixed_publication"
    assert result["tournament_intent"]["turn_status"] == "partial"
    assert "Private handoff ready." in result["final_response"]
    assert "Boat A won." not in result["final_response"]
    assert streamed == [result["final_response"], None]
    assert persisted[-1][-1]["content"] == result["final_response"]


def test_mixed_private_file_is_delivered_while_public_candidate_remains_withheld(
    monkeypatch, tmp_path
):
    test_root = Path.cwd() / ".pytest-hermes-artifacts" / f"mixed-{tmp_path.name}"
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(Path.cwd()))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(test_root / "artifacts"))
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"agent": {"artifact_only_enabled": True}},
    )
    private_response = "Private coding handoff ready."
    public_candidate = "Boat A won the tournament."
    raw = json.dumps(
        {
            "private_response": private_response,
            "public_candidate": public_candidate,
        }
    )
    agent = _agent()
    agent.platform = "telegram"
    agent._persist_session = lambda *_args: None
    agent._save_trajectory = lambda *_args: None
    agent._cleanup_task_resources = lambda *_args: None
    observed = {}

    def provider_response(*_args, **_kwargs):
        target = Path(agent._task_execution_contract.artifact_output_path)
        observed["target"] = target
        assert not target.exists()
        return _response(raw)

    agent.client.chat.completions.create.side_effect = provider_response

    try:
        result = agent.run_conversation(
            "Create private_handoff.txt as a private coding handoff, then publish the tournament Story to the SportFish Hub Instagram account.",
            task_id="mixed-private-artifact-runtime",
        )

        target = observed["target"]
        assert target.read_text(encoding="utf-8") == private_response
        assert f"MEDIA:{target.resolve()}" in result["final_response"]
        assert public_candidate not in result["final_response"]
        assert result["completed"] is False
        assert result["failed"] is False
        assert result["partial"] is True
    finally:
        shutil.rmtree(test_root, ignore_errors=True)


def test_public_artifact_is_absent_before_receipt_then_written_as_exact_verified_bytes(
    monkeypatch, tmp_path
):
    from agent.tournament_intent_contract import active_contract

    test_root = Path.cwd() / ".pytest-hermes-artifacts" / tmp_path.name
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(Path.cwd()))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(test_root / "artifacts"))
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"agent": {"artifact_only_enabled": True}},
    )
    candidate = "Verified tournament standings"
    agent = _agent()
    agent.platform = "telegram"
    persisted = []
    agent._persist_session = lambda messages, _history: persisted.append(list(messages))
    agent._save_trajectory = lambda *_args: None
    agent._cleanup_task_resources = lambda *_args: None
    agent.client.chat.completions.create.side_effect = [
        _tool_response("tournament_truth_gate", {"candidate": candidate}),
        _response(candidate),
    ]
    observed = {}

    def truth_gate(name, args, task_id, **kwargs):
        assert name == "tournament_truth_gate"
        contract = active_contract(task_id, kwargs["session_id"])
        task_contract = agent._task_execution_contract
        target = Path(task_contract.artifact_output_path)
        observed["target"] = target
        assert not target.exists()
        contract.attach_test_receipt(
            candidate=args["candidate"],
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
        return json.dumps({"ok": True})

    from datetime import datetime, timedelta, timezone

    try:
        with patch("run_agent.handle_function_call", side_effect=truth_gate):
            result = agent.run_conversation(
                "Create public_tournament.txt containing tournament standings for the public website.",
                task_id="public-artifact-runtime",
            )

        assert observed, result
        target = observed["target"]
        assert target.read_bytes() == candidate.encode("utf-8")
        assert result["final_response"] == f"MEDIA:{target.resolve()}"
        assert result["tournament_intent"]["code"] == "receipt_verified"
        assert persisted[-1][-1]["content"] == result["final_response"]
    finally:
        shutil.rmtree(test_root, ignore_errors=True)


def test_landed_public_artifact_receipt_hashes_file_bytes_not_media_directive(
    monkeypatch, tmp_path
):
    import run_agent
    from agent.tournament_intent_contract import active_contract

    test_root = Path.cwd() / ".pytest-hermes-artifacts" / f"landed-{tmp_path.name}"
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(Path.cwd()))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(test_root / "artifacts"))
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"agent": {"artifact_only_enabled": True}},
    )
    candidate = "Verified landed tournament standings"
    agent = _agent()
    agent.platform = "telegram"
    agent.tools.append(
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write a file",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    )
    agent.valid_tool_names.add("write_file")
    agent._persist_session = lambda *_args: None
    agent._save_trajectory = lambda *_args: None
    agent._cleanup_task_resources = lambda *_args: None
    provider_call = {"count": 0}
    observed = {}

    def provider_response(*_args, **_kwargs):
        provider_call["count"] += 1
        if provider_call["count"] == 1:
            return _tool_response("tournament_truth_gate", {"candidate": candidate})
        target = agent._task_execution_contract.artifact_output_path
        if provider_call["count"] == 2:
            return _tool_response(
                "write_file",
                {"path": target, "content": candidate},
                call_id="call-write",
            )
        return _response(f"MEDIA:{target}")

    agent.client.chat.completions.create.side_effect = provider_response
    real_handler = run_agent.handle_function_call

    def tool_handler(name, args, task_id, **kwargs):
        if name == "tournament_truth_gate":
            contract = active_contract(task_id, kwargs["session_id"])
            target = Path(agent._task_execution_contract.artifact_output_path)
            observed["target"] = target
            assert not target.exists()
            contract.attach_test_receipt(
                candidate=args["candidate"],
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            )
            return json.dumps({"ok": True})
        return real_handler(name, args, task_id, **kwargs)

    from datetime import datetime, timedelta, timezone

    try:
        with patch("run_agent.handle_function_call", side_effect=tool_handler):
            result = agent.run_conversation(
                "Create public_tournament.txt containing tournament standings for the public website.",
                task_id="landed-public-artifact-runtime",
            )

        target = observed["target"]
        assert target.read_text(encoding="utf-8") == candidate
        assert result["final_response"] == f"MEDIA:{target.resolve()}"
        assert result["tournament_intent"]["code"] == "receipt_verified"
        assert result["failed"] is False
    finally:
        shutil.rmtree(test_root, ignore_errors=True)
