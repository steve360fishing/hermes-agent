import hashlib
import json
from datetime import datetime, timedelta, timezone

from agent import tournament_research_contract as contract_module
from agent.tournament_research_contract import (
    TournamentIntent,
    begin_tournament_research_contract,
    canonical_json_sha256,
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
        "decision": "ALLOW_PRIVATE_ANSWER" if intent is TournamentIntent.PRIVATE else "ALLOW_PUBLIC_ARTIFACT",
        "issued_at_utc": now.isoformat(), "expires_at_utc": (now + timedelta(minutes=15)).isoformat(),
        "allowed_entrypoints": [state.entrypoint], "artifact_payload_hash": canonical_json_sha256(payload),
    }
    receipt["receipt_hash"] = canonical_json_sha256(receipt)
    path = receipt_root / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert state.attach_receipt(receipt_path=path, candidate=candidate, metadata=metadata, expires_at=now + timedelta(minutes=15))
    return state


def test_classifier_has_no_event_specific_exception_and_delivery_ambiguity_is_public():
    assert classify_tournament_intent("Bermuda results") is TournamentIntent.PUBLIC
    assert classify_tournament_intent("audit tournament scoring") is TournamentIntent.PRIVATE
    assert classify_tournament_intent("captain scored a photo") is None


def test_missing_receipt_blocks_and_no_stream_delta_escapes():
    agent = FakeAgent()
    begin_tournament_research_contract(agent, message="publish tournament results", task_id="task-1")
    agent.stream_delta_callback("unverified")
    output, telemetry, failed = finalize_tournament_output(agent, candidate="unverified", messages=[])
    assert output.startswith("PUBLIC_ARTIFACT_BLOCKED:")
    assert failed and agent.streamed == [] and telemetry["accepted"] is False


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
