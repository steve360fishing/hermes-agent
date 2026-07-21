import hashlib
import json
from datetime import datetime, timedelta, timezone

from agent import tournament_research_contract as contract_module
from agent.tournament_research_contract import (
    TournamentIntent,
    begin_tournament_research_contract,
    canonical_json_sha256,
    clear_tournament_research_contract,
    classify_tournament_intent,
    finalize_tournament_output,
)


class Guardrails:
    def __init__(self): self.contract = None
    def set_tournament_contract(self, contract): self.contract = contract


class FakeAgent:
    def __init__(self):
        self.streamed, self.tts = [], []
        self.stream_delta_callback, self._stream_callback = self.streamed.append, self.tts.append
        self._response_was_previewed = False
        self.session_id, self._current_turn_id, self.platform = "session-1", "turn-1", "telegram"
        self._tool_guardrails = Guardrails()


def _roots(tmp_path, monkeypatch):
    roots = [tmp_path / name for name in ("receipts", "journal", "snapshots")]
    for root in roots: root.mkdir(parents=True)
    monkeypatch.setattr(contract_module, "configured_runtime_roots", lambda: contract_module.RuntimeRoots(*roots))
    return roots


def _attach(tmp_path, monkeypatch, agent, *, task_id, candidate, intent=TournamentIntent.PUBLIC):
    receipt_root, _journal, _snapshots = _roots(tmp_path, monkeypatch)
    state = begin_tournament_research_contract(agent, message="private tournament standings" if intent is TournamentIntent.PRIVATE else "publish tournament standings", task_id=task_id)
    metadata = {"factual_claims": [{"claim_id": "c1"}], "public_surfaces": [{"claim_ids": ["c1"]}]}
    now = datetime.now(timezone.utc)
    payload = contract_module.build_artifact_payload(candidate, state.destination, metadata)
    receipt = {
        "schema_version": "tournament_route_preflight.v2",
        "source_repository": contract_module.AUDIT_SOURCE_REPOSITORY,
        "contract_commit": contract_module.AUDIT_CONTRACT_COMMIT,
        "decision": "ALLOW_PRIVATE_ANSWER" if intent is TournamentIntent.PRIVATE else "ALLOW_PUBLIC_ARTIFACT",
        "issued_at_utc": now.isoformat(), "expires_at_utc": (now + timedelta(minutes=15)).isoformat(),
        "allowed_entrypoints": [state.entrypoint], "artifact_payload_hash": canonical_json_sha256(payload),
    }
    receipt["receipt_hash"] = canonical_json_sha256(receipt)
    path = receipt_root / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert state.attach_receipt(
        receipt_path=path, candidate=candidate, metadata=metadata, audit_request={},
        expires_at=now + timedelta(minutes=15),
    )
    monkeypatch.setattr(
        contract_module, "validate_audit_sink",
        lambda _contract, _candidate: contract_module.TournamentReceiptDecision(True, "receipt_sink_verified"),
    )
    return state


def test_classifier_requires_identity_or_sportfish_context_and_defaults_factual_questions_private():
    assert classify_tournament_intent("show me search results") is None
    assert classify_tournament_intent("show me results") is None
    assert classify_tournament_intent("marlin results") is TournamentIntent.PRIVATE
    assert classify_tournament_intent("tournament standings") is TournamentIntent.PRIVATE
    assert classify_tournament_intent("publish tournament standings") is TournamentIntent.PUBLIC
    assert classify_tournament_intent("make a tournament results carousel") is TournamentIntent.PUBLIC
    assert classify_tournament_intent("audit tournament scoring") is TournamentIntent.PRIVATE
    assert classify_tournament_intent("captain scored a photo") is None
    assert classify_tournament_intent("Which team won the Super Bowl?") is None
    assert classify_tournament_intent("make a tournament flyer") is TournamentIntent.PUBLIC
    assert classify_tournament_intent("add tournament standings to the website") is TournamentIntent.PUBLIC


def test_selected_current_journal_alias_is_protected_without_event_hardcoding(monkeypatch):
    monkeypatch.setattr(contract_module, "_trusted_journal_aliases", lambda: {"blue water classic"})
    assert classify_tournament_intent("Who won the Blue Water Classic?") is TournamentIntent.PRIVATE


def test_journal_alias_resolves_relative_to_pointer_parent_and_malformed_pointer_fails_safe(tmp_path, monkeypatch):
    _receipts, journal_root, _snapshots = _roots(tmp_path, monkeypatch)
    pointer_dir = journal_root / "pointers"
    pointer_dir.mkdir()
    pointer_path = journal_root / "LATEST-JOURNAL.json"
    pointer_path.write_text(json.dumps({"canonical_journal_path": "pointers/current.json"}), encoding="utf-8")
    (pointer_dir / "current.json").write_text(json.dumps({"selected_tournaments": [{"aliases": ["Blue Water Classic"]}]}), encoding="utf-8")
    assert contract_module._trusted_journal_aliases() == {"blue water classic"}
    pointer_path.write_text("{malformed", encoding="utf-8")
    assert contract_module._trusted_journal_aliases() is None


def test_canonical_json_preserves_unicode_bytes():
    assert canonical_json_sha256({"name": "Curaçao"}) != canonical_json_sha256({"name": "Cura\\u00e7ao"})


def test_audit_fixture_is_a_compatible_isolated_repair_head_placeholder():
    fixture = json.loads((contract_module.Path(__file__).parents[1] / "fixtures" / "tournament_route_preflight_v2_a23b3016.json").read_text(encoding="utf-8"))
    assert fixture["schema_version"] == contract_module.AUDIT_SCHEMA_VERSION
    assert fixture["source_repository"] == contract_module.AUDIT_SOURCE_REPOSITORY
    assert fixture["contract_commit"] == contract_module.AUDIT_CONTRACT_COMMIT
    assert fixture["audit_re_review"] == "passed"


def test_missing_receipt_blocks_and_no_stream_delta_escapes():
    agent = FakeAgent()
    begin_tournament_research_contract(agent, message="publish tournament results", task_id="task-1")
    agent.stream_delta_callback("unverified")
    output, telemetry, failed = finalize_tournament_output(agent, candidate="unverified", messages=[])
    assert output.startswith("PUBLIC_ARTIFACT_BLOCKED:")
    assert failed and agent.streamed == [] and telemetry["accepted"] is False


def test_rejection_redacts_tool_and_intermediate_content_before_persistence_shape():
    agent = FakeAgent()
    begin_tournament_research_contract(agent, message="publish tournament results", task_id="task-redact")
    messages = [
        {"role": "user", "content": "publish tournament results"},
        {"role": "assistant", "content": "intermediate", "tool_calls": [{"id": "x"}]},
        {"role": "tool", "content": "unverified tool result"},
        {"role": "assistant", "content": "unverified candidate"},
    ]
    output, _telemetry, failed = finalize_tournament_output(agent, candidate="unverified candidate", messages=messages)
    assert failed and messages == [{"role": "user", "content": "publish tournament results"}, {"role": "assistant", "content": output}]


def test_protected_turn_adds_gate_only_temporarily_and_stale_cleanup_is_duck_typed():
    agent = FakeAgent()
    agent.tools = [{"type": "function", "function": {"name": "terminal"}}]
    agent.valid_tool_names = {"terminal"}
    begin_tournament_research_contract(agent, message="publish tournament standings", task_id="task-schema")
    assert "tournament_truth_gate" in agent.valid_tool_names
    clear_tournament_research_contract(agent)
    assert [tool["function"]["name"] for tool in agent.tools] == ["terminal"]

    cleaned = []
    agent._tournament_research_contract = type("Stale", (), {"cleanup": lambda self, _agent: cleaned.append(True)})()
    clear_tournament_research_contract(agent)
    assert cleaned == [True]


def test_valid_receipt_is_exact_candidate_single_use_and_releases_after_finalization(tmp_path, monkeypatch):
    agent = FakeAgent()
    _attach(tmp_path, monkeypatch, agent, task_id="task-2", candidate="verified standings")
    agent.stream_delta_callback("verified standings")
    output, _telemetry, failed = finalize_tournament_output(agent, candidate="verified standings", messages=[])
    assert output == "verified standings" and not failed
    assert agent.streamed == ["verified standings", None]
    assert agent._tool_guardrails.contract is None


def test_candidate_change_or_private_receipt_on_public_turn_fails_closed(tmp_path, monkeypatch):
    agent = FakeAgent()
    _attach(tmp_path, monkeypatch, agent, task_id="task-3", candidate="safe")
    output, telemetry, failed = finalize_tournament_output(agent, candidate="changed", messages=[])
    assert failed and output.startswith("PUBLIC_ARTIFACT_BLOCKED:") and telemetry["code"] == "candidate_bytes_mismatch"

    agent = FakeAgent()
    state = _attach(tmp_path / "private", monkeypatch, agent, task_id="task-4", candidate="safe", intent=TournamentIntent.PRIVATE)
    state.intent = TournamentIntent.PUBLIC
    output, _telemetry, failed = finalize_tournament_output(agent, candidate="safe", messages=[])
    assert failed and output.startswith("PUBLIC_ARTIFACT_BLOCKED:")


def test_sink_time_audit_failure_blocks_rotated_or_tampered_truth(tmp_path, monkeypatch):
    agent = FakeAgent()
    _attach(tmp_path, monkeypatch, agent, task_id="task-sink", candidate="safe")
    monkeypatch.setattr(
        contract_module, "validate_audit_sink",
        lambda _contract, _candidate: contract_module.TournamentReceiptDecision(False, "audit_preflight_failed"),
    )
    output, telemetry, failed = finalize_tournament_output(agent, candidate="safe", messages=[])
    assert failed and output.startswith("PUBLIC_ARTIFACT_BLOCKED:")
    assert telemetry["code"] == "audit_preflight_failed"
