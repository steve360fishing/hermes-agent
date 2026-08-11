from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import agent.tournament_intent_contract as tournament_intent_contract
from agent.tool_guardrails import ToolCallGuardrailController
from agent.tournament_intent_contract import (
    TournamentIntentState,
    TournamentReleaseApproval,
    begin_tournament_intent_contract,
    classify_tournament_intent,
    clear_tournament_intent_contract,
    finalize_tournament_output,
)


PRIVATE_CASES = (
    (
        "What tournaments do we have coming up this week, starting today and ending Sunday?",
        TournamentIntentState.PRIVATE_INQUIRY,
    ),
    (
        "Create internal research notes about tournament dates.",
        TournamentIntentState.PRIVATE_ARTIFACT,
    ),
    (
        "Give me a Codex prompt to fix tournament truth-gate blocks.",
        TournamentIntentState.PRIVATE_ARTIFACT,
    ),
    (
        "Write a private Codex test using the phrase ‘Create a public Instagram Story naming tournament winners.’",
        TournamentIntentState.PRIVATE_ARTIFACT,
    ),
    (
        "Do not publish; analyze whether this public tournament post is supported.",
        TournamentIntentState.PRIVATE_INQUIRY,
    ),
)


@pytest.mark.parametrize(("message", "expected"), PRIVATE_CASES)
def test_private_and_embedded_language_never_activates_public_contract(message, expected):
    assert classify_tournament_intent(message) is expected
    agent = _agent()
    assert begin_tournament_intent_contract(agent, message=message, task_id="private") is None
    assert agent._tool_guardrails.before_call("terminal", {}).action == "allow"
    assert agent._tool_guardrails.before_call("write_file", {}).action == "allow"


def test_public_draft_and_publication_are_distinct_states():
    assert classify_tournament_intent(
        "Create a public Instagram Story naming tournament winners."
    ) is TournamentIntentState.PUBLIC_FACING_DRAFT
    assert classify_tournament_intent(
        "Publish that approved tournament Story now."
    ) is TournamentIntentState.PUBLICATION_REQUEST
    assert classify_tournament_intent(
        "Publish that approved Story now."
    ) is TournamentIntentState.PUBLICATION_REQUEST
    assert classify_tournament_intent(
        "Create a public tournament audit report for the website."
    ) is TournamentIntentState.PUBLIC_FACING_DRAFT
    assert classify_tournament_intent(
        "Validate the tournament truth receipt only; do not publish."
    ) is TournamentIntentState.RECEIPT_VALIDATION
    assert classify_tournament_intent(
        "Record the exact tournament release approval only; do not publish."
    ) is TournamentIntentState.RELEASE_APPROVAL


@pytest.mark.parametrize("platform", ("cron", "CRON", "CrOn"))
@pytest.mark.parametrize(
    "message",
    (
        "Create a public tournament Story and post the newsletter.",
        "Create private tournament research notes for Steve.",
        (
            "Tournament newsletter post Story image website Codex report: "
            "prepare an internal handoff and do not publish or send anything."
        ),
    ),
)
@pytest.mark.parametrize("with_stream_callback", (False, True))
def test_cron_platform_bypasses_before_classifier_and_leaves_agent_untouched(
    monkeypatch, platform, message, with_stream_callback
):
    classifier_calls = []

    def unexpected_classifier(value):
        classifier_calls.append(value)
        raise AssertionError("cron must bypass before tournament classification")

    monkeypatch.setattr(
        tournament_intent_contract,
        "classify_tournament_intent",
        unexpected_classifier,
    )
    agent = _agent(platform=platform)
    existing_tool = {"type": "function", "function": {"name": "existing_tool"}}
    agent.tools.append(existing_tool)
    agent.valid_tool_names.add("existing_tool")
    stream_delta_callback = lambda _value: None
    stream_callback = lambda _value: None
    persist_callback = lambda *_args: None
    supplied_stream_callback = (lambda _value: None) if with_stream_callback else None
    agent.stream_delta_callback = stream_delta_callback
    agent._stream_callback = stream_callback
    agent._persist_session = persist_callback
    original_guardrail_contract = agent._tool_guardrails._tournament_contract

    contract = begin_tournament_intent_contract(
        agent,
        message=message,
        task_id="cron-job",
        stream_callback=supplied_stream_callback,
    )

    assert contract is None
    assert classifier_calls == []
    assert agent.tools == [existing_tool]
    assert agent.valid_tool_names == {"existing_tool"}
    assert agent.stream_delta_callback is stream_delta_callback
    assert agent._stream_callback is stream_callback
    assert agent._persist_session is persist_callback
    assert agent._tool_guardrails._tournament_contract is original_guardrail_contract
    assert agent._tournament_intent_contract is None
    assert "tournament_truth_gate" not in agent.valid_tool_names
    assert agent._tool_guardrails.before_call("terminal", {}).action == "allow"


def test_public_contract_exposes_dispatchable_truth_gate_in_first_tool_surface():
    agent = _agent()
    contract = begin_tournament_intent_contract(
        agent,
        message="Create a public Instagram Story naming tournament winners.",
        task_id="public-draft",
    )
    assert contract is not None
    assert contract.preflight_error == ""
    assert any(
        tool.get("function", {}).get("name") == "tournament_truth_gate"
        for tool in agent.tools
    )
    assert "tournament_truth_gate" in agent.valid_tool_names
    from tools.registry import registry

    assert registry.get_entry("tournament_truth_gate") is not None


def test_public_contract_cleanup_removes_only_request_local_tool_wiring():
    agent = _agent()
    contract = begin_tournament_intent_contract(
        agent,
        message="Create a public Instagram Story naming tournament winners.",
        task_id="cleanup",
    )
    assert contract is not None
    clear_tournament_intent_contract(agent)
    assert agent._tournament_intent_contract is None
    assert agent._tool_guardrails._tournament_contract is None
    assert "tournament_truth_gate" not in agent.valid_tool_names
    assert not any(
        tool.get("function", {}).get("name") == "tournament_truth_gate"
        for tool in agent.tools
    )


def test_public_contract_allows_target_listing_but_not_sending_without_authority():
    agent = _agent()
    begin_tournament_intent_contract(
        agent,
        message="Publish that approved tournament Story now.",
        task_id="target-list",
    )
    assert agent._tool_guardrails.before_call("send_message", {"action": "list"}).action == "allow"
    denied = agent._tool_guardrails.before_call(
        "send_message",
        {"action": "send", "target": "telegram:channel-42", "message": "Winner copy"},
    )
    assert denied.action == "deny"
    assert denied.code == "receipt_and_release_approval_required"


def test_missing_registered_truth_gate_fails_before_provider_request(monkeypatch):
    from tools.registry import registry

    monkeypatch.setattr(registry, "get_entry", lambda _name: None)
    agent = _agent()
    contract = begin_tournament_intent_contract(
        agent,
        message="Create a public Instagram Story naming tournament winners.",
        task_id="missing-gate",
    )
    assert contract is not None
    assert contract.preflight_error == "truth_gate_unavailable"


def test_public_draft_missing_receipt_blocks_claim_bytes_without_opaque_tokens():
    agent = _agent()
    contract = begin_tournament_intent_contract(
        agent,
        message="Create a public Instagram Story naming tournament winners.",
        task_id="missing-receipt",
    )
    messages = [
        {"role": "user", "content": "Create a public tournament Story."},
        {"role": "assistant", "content": "Boat A won."},
    ]
    response, telemetry, failed = finalize_tournament_output(
        agent, candidate="Boat A won.", messages=messages
    )
    assert failed is True
    assert telemetry["code"] == "receipt_missing_or_consumed"
    assert "public tournament copy was not released" in response.lower()
    assert "ROUTE_HOLD" not in response
    assert "PUBLIC_ARTIFACT_BLOCKED" not in response
    assert "Boat A won" not in response
    assert messages[-1]["content"] == response
    assert agent._tournament_intent_contract is None


def test_public_draft_valid_receipt_releases_exact_candidate_once():
    agent = _agent()
    contract = begin_tournament_intent_contract(
        agent,
        message="Create a public Instagram Story naming tournament winners.",
        task_id="valid-draft",
    )
    _bind_test_receipt(contract, "Verified winner copy")
    messages = [{"role": "user", "content": "draft"}, {"role": "assistant", "content": "Verified winner copy"}]
    response, telemetry, failed = finalize_tournament_output(
        agent, candidate="Verified winner copy", messages=messages
    )
    assert failed is False
    assert response == "Verified winner copy"
    assert telemetry["code"] == "receipt_verified"
    assert contract.receipt_used is True
    assert agent._tournament_intent_contract is None


def test_public_draft_ignores_and_does_not_consume_irrelevant_release_approval():
    agent = _agent()
    contract = begin_tournament_intent_contract(
        agent,
        message="Create a public Instagram Story naming tournament winners.",
        task_id="draft-with-approval",
    )
    candidate = "Verified winner copy"
    _bind_test_receipt(contract, candidate)
    approval = TournamentReleaseApproval(
        destination="telegram:channel-42",
        candidate_sha256=contract.candidate_sha256(candidate),
        identity="steve",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        idempotency_key="irrelevant-draft-approval",
    )
    assert contract.attach_release_approval(approval)
    messages = [{"role": "user", "content": "draft"}, {"role": "assistant", "content": candidate}]
    response, _, failed = finalize_tournament_output(agent, candidate=candidate, messages=messages)
    assert failed is False
    assert response == candidate
    assert approval.state == "available"


@pytest.mark.parametrize(
    ("setup", "expected_code"),
    (
        ("expired", "receipt_expired"),
        ("mismatched", "candidate_bytes_mismatch"),
        ("consumed", "receipt_missing_or_consumed"),
    ),
)
def test_public_draft_receipt_failure_matrix(setup, expected_code):
    agent = _agent()
    contract = begin_tournament_intent_contract(
        agent,
        message="Create a public Instagram Story naming tournament winners.",
        task_id=f"draft-{setup}",
    )
    candidate = "Verified winner copy"
    if setup == "expired":
        contract.attach_test_receipt(
            candidate=candidate,
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        )
    else:
        _bind_test_receipt(contract, "different bytes" if setup == "mismatched" else candidate)
        if setup == "consumed":
            contract.receipt_used = True
    assert contract.verify_receipt(candidate).code == expected_code


def test_streaming_public_draft_buffers_until_exact_receipt_then_releases_once():
    streamed = []
    agent = _agent()
    contract = begin_tournament_intent_contract(
        agent,
        message="Create a public Instagram Story naming tournament winners.",
        task_id="streaming-draft",
        stream_callback=streamed.append,
    )
    contract.buffer("unverified leak")
    assert streamed == []
    _bind_test_receipt(contract, "Verified winner copy")
    messages = [{"role": "user", "content": "draft"}, {"role": "assistant", "content": "Verified winner copy"}]
    response, telemetry, failed = finalize_tournament_output(
        agent,
        candidate="Verified winner copy",
        messages=messages,
    )
    assert failed is False
    assert response == "Verified winner copy"
    assert telemetry["code"] == "receipt_verified"
    assert streamed == ["Verified winner copy", None]


def test_false_public_hold_does_not_poison_followup_private_file_handoff():
    agent = _agent()
    begin_tournament_intent_contract(
        agent,
        message="Create a public Instagram Story naming tournament winners.",
        task_id="prior-false-hold",
    )
    messages = [{"role": "user", "content": "draft"}, {"role": "assistant", "content": "Unverified copy"}]
    finalize_tournament_output(agent, candidate="Unverified copy", messages=messages)

    followup = "Give me a .txt Codex prompt to fix tournament truth-gate blocks."
    assert classify_tournament_intent(followup) is TournamentIntentState.PRIVATE_ARTIFACT
    assert begin_tournament_intent_contract(
        agent,
        message=followup,
        task_id="private-followup",
    ) is None
    assert agent._tool_guardrails.before_call("write_file", {}).action == "allow"


def test_private_artifact_can_be_created_read_back_and_delivered(tmp_path):
    agent = _agent()
    prompt = "Give me a Codex prompt to fix tournament truth-gate blocks."
    assert begin_tournament_intent_contract(agent, message=prompt, task_id="private-file") is None
    target = tmp_path / "tournament-gate-repair.txt"
    content = "Repair the private tournament artifact path without weakening public release checks."
    assert agent._tool_guardrails.before_call(
        "write_file", {"path": str(target), "content": content}
    ).action == "allow"
    target.write_text(content, encoding="utf-8")
    delivered = target.read_text(encoding="utf-8")
    assert delivered == content


def test_publication_requires_receipt_and_exact_release_approval():
    agent = _agent()
    contract = begin_tournament_intent_contract(
        agent,
        message="Publish that approved tournament Story now.",
        task_id="publication",
    )
    candidate = "Verified winner copy"
    destination = "telegram:channel-42"
    _bind_test_receipt(contract, candidate)

    missing = contract.authorize_external_action(
        tool_name="send_message",
        candidate=candidate,
        destination=destination,
        identity="steve",
        idempotency_key="release-1",
    )
    assert missing.allowed is False
    assert missing.code == "release_approval_required"

    contract.attach_release_approval(
        TournamentReleaseApproval(
            destination=destination,
            candidate_sha256=contract.candidate_sha256(candidate),
            identity="steve",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            idempotency_key="release-1",
        )
    )
    allowed = contract.authorize_external_action(
        tool_name="send_message",
        candidate=candidate,
        destination=destination,
        identity="steve",
        idempotency_key="release-1",
    )
    assert allowed.allowed is True
    assert contract.release_state == "in_flight"
    repeated_preflight = contract.authorize_external_action(
        tool_name="send_message",
        candidate=candidate,
        destination=destination,
        identity="steve",
        idempotency_key="release-1",
    )
    assert repeated_preflight.allowed is True
    contract.record_external_result(success=True, ambiguous=False)
    assert contract.release_state == "consumed"
    assert contract.receipt_used is True


def test_approval_never_bypasses_missing_or_mismatched_truth_receipt():
    agent = _agent()
    contract = begin_tournament_intent_contract(
        agent,
        message="Publish that approved tournament Story now.",
        task_id="approval-no-truth",
    )
    candidate = "Unverified copy"
    contract.attach_release_approval(
        TournamentReleaseApproval(
            destination="telegram:channel-42",
            candidate_sha256=contract.candidate_sha256(candidate),
            identity="steve",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            idempotency_key="release-2",
        )
    )
    decision = contract.authorize_external_action(
        tool_name="send_message",
        candidate=candidate,
        destination="telegram:channel-42",
        identity="steve",
        idempotency_key="release-2",
    )
    assert decision.allowed is False
    assert decision.code == "receipt_missing_or_consumed"
    assert contract.release_approval is not None
    assert contract.release_approval.state == "available"


def test_publication_with_neither_authority_reports_both_missing():
    agent = _agent()
    contract = begin_tournament_intent_contract(
        agent,
        message="Publish that approved tournament Story now.",
        task_id="publication-neither",
    )
    decision = contract.authorize_external_action(
        tool_name="send_message",
        candidate="Unverified copy",
        destination="telegram:channel-42",
        identity="steve",
        idempotency_key="release-none",
    )
    assert decision.code == "receipt_and_release_approval_required"


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        ("expired", "release_approval_expired"),
        ("destination", "release_approval_mismatch"),
        ("identity", "release_approval_mismatch"),
        ("idempotency", "release_approval_mismatch"),
        ("consumed", "release_approval_consumed"),
    ),
)
def test_publication_release_approval_failure_matrix(mutation, expected_code):
    agent = _agent()
    contract = begin_tournament_intent_contract(
        agent,
        message="Publish that approved tournament Story now.",
        task_id=f"publication-{mutation}",
    )
    candidate = "Verified winner copy"
    destination = "telegram:channel-42"
    _bind_test_receipt(contract, candidate)
    approval = TournamentReleaseApproval(
        destination=destination,
        candidate_sha256=contract.candidate_sha256(candidate),
        identity="steve",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        idempotency_key="release-matrix",
    )
    assert contract.attach_release_approval(approval)
    if mutation == "expired":
        approval.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    elif mutation == "consumed":
        approval.state = "consumed"
    decision = contract.authorize_external_action(
        tool_name="send_message",
        candidate=candidate,
        destination=("telegram:other" if mutation == "destination" else destination),
        identity=("other" if mutation == "identity" else "steve"),
        idempotency_key=("other" if mutation == "idempotency" else "release-matrix"),
    )
    assert decision.code == expected_code


def test_ambiguous_external_result_is_not_replayed():
    agent = _agent()
    contract = begin_tournament_intent_contract(
        agent,
        message="Publish that approved tournament Story now.",
        task_id="ambiguous-release",
    )
    candidate = "Verified winner copy"
    destination = "telegram:channel-42"
    _bind_test_receipt(contract, candidate)
    contract.attach_release_approval(
        TournamentReleaseApproval(
            destination=destination,
            candidate_sha256=contract.candidate_sha256(candidate),
            identity="steve",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            idempotency_key="release-3",
        )
    )
    assert contract.authorize_external_action(
        tool_name="send_message",
        candidate=candidate,
        destination=destination,
        identity="steve",
        idempotency_key="release-3",
    ).allowed
    contract.record_external_result(success=False, ambiguous=True)
    assert contract.release_state == "ambiguous"
    replay = contract.authorize_external_action(
        tool_name="send_message",
        candidate=candidate,
        destination=destination,
        identity="steve",
        idempotency_key="release-3",
    )
    assert replay.allowed is False
    assert replay.code == "release_outcome_ambiguous"


def test_successful_bound_publication_returns_confirmation_without_rechecking_consumed_receipt():
    agent = _agent()
    contract = begin_tournament_intent_contract(
        agent,
        message="Publish that approved tournament Story now.",
        task_id="publication-success",
    )
    candidate = "Verified winner copy"
    _bind_test_receipt(contract, candidate)
    contract.attach_release_approval(
        TournamentReleaseApproval(
            destination="telegram:channel-42",
            candidate_sha256=contract.candidate_sha256(candidate),
            identity=contract.actor_identity,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            idempotency_key="release-success",
        )
    )
    assert agent._tool_guardrails.before_call(
        "send_message",
        {"action": "send", "target": "telegram:channel-42", "message": candidate},
    ).action == "allow"
    agent._tool_guardrails.after_call(
        "send_message",
        {"action": "send", "target": "telegram:channel-42", "message": candidate},
        '{"ok":true}',
        failed=False,
    )
    messages = [{"role": "user", "content": "publish"}, {"role": "assistant", "content": "Sent."}]
    response, telemetry, failed = finalize_tournament_output(
        agent,
        candidate="Sent.",
        messages=messages,
    )
    assert failed is False
    assert response == "Publication completed through the exact receipt- and approval-bound action."
    assert telemetry["code"] == "release_consumed"
    assert messages[-1]["content"] == response


def _agent(*, platform: str = "telegram"):
    controller = ToolCallGuardrailController()
    return SimpleNamespace(
        session_id="session-1",
        platform=platform,
        tools=[],
        valid_tool_names=set(),
        stream_delta_callback=None,
        _stream_callback=None,
        _persist_session=None,
        _tool_guardrails=controller,
        _tournament_intent_contract=None,
        _response_was_previewed=False,
    )


def _bind_test_receipt(contract, candidate: str) -> None:
    contract.attach_test_receipt(
        candidate=candidate,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
