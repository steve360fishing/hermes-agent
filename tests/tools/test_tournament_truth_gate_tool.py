import json
import hashlib
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from agent.tool_guardrails import ToolCallGuardrailController
from agent.tournament_intent_contract import (
    begin_tournament_intent_contract,
    clear_tournament_intent_contract,
    finalize_tournament_output,
)
from gateway.run import _mint_gateway_turn_provenance
from agent.tournament_truth_support import canonical_json_sha256
from tools import tournament_truth_gate_tool as tool
from tools import tournament_source_capture_tool as capture_tool


def _bound_direct_provenance(message: str, session_id: str):
    """Use the gateway ingress mint and the matching agent request identity."""
    return _mint_gateway_turn_provenance(
        SimpleNamespace(text=message, message_id=f"message-{session_id}"),
        SimpleNamespace(
            user_id="steve",
            platform="telegram",
            profile="test",
            chat_id="chat-1",
            thread_id=None,
            scope_id=f"agent:test:telegram:chat-1:{session_id}",
        ),
        is_internal=False,
    )


def _bound_agent(session_id: str):
    return SimpleNamespace(
        session_id=session_id,
        platform="telegram",
        _chat_id="chat-1",
        _thread_id="",
        _gateway_session_key=f"agent:test:telegram:chat-1:{session_id}",
        tools=[],
        valid_tool_names=set(),
        stream_delta_callback=None,
        _stream_callback=None,
        _persist_session=None,
        _tool_guardrails=ToolCallGuardrailController(),
        _tournament_intent_contract=None,
    )


def _trusted_snapshot(path):
    evidence = {
        "schema_version": "tournament_trusted_snapshot.v1",
        "pulled_at_utc": datetime.now(timezone.utc).isoformat(),
        "verified_claims": [{"claim_type": "winner", "value": "Rascal"}],
        "verified_finality": {
            "displayed": True, "standings_final": False, "payout_final": False,
        },
    }
    raw = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
    path.write_text(raw, encoding="utf-8")
    return hashlib.sha256(raw.encode()).hexdigest(), evidence


def test_validator_requires_an_active_request_local_public_contract():
    result=json.loads(tool.run_tournament_truth_gate({"candidate":"draft","request":{},"artifact_metadata":{}},task_id="none",session_id="none"))
    assert result["code"] == "truth_gate_no_active_contract"


def test_validator_dispatch_binds_exact_receipt_to_active_contract(tmp_path, monkeypatch):
    roots = SimpleNamespace(
        receipt_root=tmp_path / "receipts",
        journal_root=tmp_path / "journals",
        source_snapshot_root=tmp_path / "snapshots",
    )
    for root in roots.__dict__.values():
        root.mkdir()
    snapshot = roots.source_snapshot_root / "official.json"
    snapshot_sha256, source_evidence = _trusted_snapshot(snapshot)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    receipt = {
        "schema_version": "tournament_route_preflight.v2",
        "decision": "ALLOW_PUBLIC_ARTIFACT",
        "issued_at_utc": (expires_at - timedelta(minutes=15)).isoformat(),
        "expires_at_utc": expires_at.isoformat(),
        "allowed_entrypoints": ["direct_public"],
        "artifact_payload_hash": canonical_json_sha256(
            {
                "surface": "story",
                "content": "Verified public copy",
                "destination": "platform:telegram:chat-1",
            }
        ),
    }
    receipt["receipt_hash"] = canonical_json_sha256(receipt)
    receipt_path = roots.receipt_root / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(tool, "configured_runtime_roots", lambda: roots)
    monkeypatch.setattr(
        tool,
        "_run_preflight",
        lambda *_args, **_kwargs: (receipt_path, receipt, "receipt_loaded"),
    )
    agent = _bound_agent("session-tool")
    contract = begin_tournament_intent_contract(
        agent,
        message="Create a public tournament Story for Instagram naming winners.",
        task_id="tool-bind",
        turn_provenance=_bound_direct_provenance(
            "Create a public tournament Story for Instagram naming winners.", "session-tool"
        ),
    )
    result = json.loads(
        tool.run_tournament_truth_gate(
            {
                "candidate": "Verified public copy",
                "request": {
                    "evidence_manifest": [
                        {
                            "source_snapshot_path": str(snapshot),
                            "source_snapshot_sha256": snapshot_sha256,
                        }
                    ]
                },
                "artifact_metadata": {"surface": "story"},
            },
            task_id="tool-bind",
            session_id="session-tool",
        )
    )
    assert result["accepted"] is True
    assert contract.receipt_candidate_sha256 == contract.candidate_sha256("Verified public copy")
    assert contract.receipt_path == receipt_path
    assert contract.audit_request["evidence_manifest"][0]["source_evidence"] == source_evidence
    assert contract.audit_request["evidence_manifest"][0]["displayed"] is True
    clear_tournament_intent_contract(agent)


def test_validator_rejects_a_hash_valid_blocked_receipt(tmp_path, monkeypatch):
    roots = SimpleNamespace(
        receipt_root=tmp_path / "receipts",
        journal_root=tmp_path / "journals",
        source_snapshot_root=tmp_path / "snapshots",
    )
    for root in roots.__dict__.values():
        root.mkdir()
    snapshot = roots.source_snapshot_root / "official.json"
    snapshot_sha256, _ = _trusted_snapshot(snapshot)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    receipt = {
        "schema_version": "tournament_route_preflight.v2",
        "decision": "BLOCKED",
        "issued_at_utc": (expires_at - timedelta(minutes=15)).isoformat(),
        "expires_at_utc": expires_at.isoformat(),
        "allowed_entrypoints": ["direct_public"],
        "artifact_payload_hash": canonical_json_sha256(
            {"content": "Blocked copy", "destination": "platform:telegram"}
        ),
    }
    receipt["receipt_hash"] = canonical_json_sha256(receipt)
    receipt_path = roots.receipt_root / "blocked.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(tool, "configured_runtime_roots", lambda: roots)
    monkeypatch.setattr(tool, "_run_preflight", lambda *_args, **_kwargs: (receipt_path, receipt, "receipt_loaded"))
    agent = _bound_agent("session-blocked")
    begin_tournament_intent_contract(
        agent,
        message="Create a public tournament Story for Instagram naming winners.",
        task_id="tool-blocked",
        turn_provenance=_bound_direct_provenance(
            "Create a public tournament Story for Instagram naming winners.", "session-blocked"
        ),
    )
    result = json.loads(
        tool.run_tournament_truth_gate(
            {
                "candidate": "Blocked copy",
                "request": {
                    "evidence_manifest": [
                        {
                            "source_snapshot_path": str(snapshot),
                            "source_snapshot_sha256": snapshot_sha256,
                        }
                    ]
                },
                "artifact_metadata": {},
            },
            task_id="tool-blocked",
            session_id="session-blocked",
        )
    )
    assert result["code"] == "audit_receipt_binding_failed"


def test_snapshot_hydration_rejects_model_hash_and_overwrites_model_evidence(tmp_path):
    snapshot = tmp_path / "official.json"
    digest, evidence = _trusted_snapshot(snapshot)
    request = {
        "evidence_manifest": [{
            "source_snapshot_path": str(snapshot),
            "source_snapshot_sha256": digest,
            "source_evidence": {"verified_claims": [{"claim_type": "winner", "value": "Fabricated"}]},
            "displayed": False,
        }]
    }
    hydrated, code = tool._hydrate_snapshot_ingestion(request, tmp_path)
    assert code is None
    assert hydrated["evidence_manifest"][0]["source_evidence"] == evidence
    assert hydrated["evidence_manifest"][0]["displayed"] is True

    request["evidence_manifest"][0]["source_snapshot_sha256"] = "0" * 64
    hydrated, code = tool._hydrate_snapshot_ingestion(request, tmp_path)
    assert hydrated is None
    assert code == "trusted_source_snapshot_hash_mismatch"


def test_capture_manifest_flows_to_truth_gate_without_model_authored_evidence(tmp_path, monkeypatch):
    roots = SimpleNamespace(
        receipt_root=tmp_path / "receipts", journal_root=tmp_path / "journals",
        source_snapshot_root=tmp_path / "snapshots",
    )
    for root in roots.__dict__.values():
        root.mkdir()
    source_map = {
        "provider": "CatchStat",
        "direct_url": "https://results.example.test/final",
        "provider_host": "results.example.test",
        "category_ids": [9312],
        "category_labels": ["Overall"],
        "identity": {"tournament_id": "event-1", "event_header": "Example Open", "year": 2026},
    }
    (roots.journal_root / "current.json").write_text(json.dumps({
        "selected_tournaments": [{
            "tournament_key": "registered-event", "tournament_name": "Example Open",
            "year": 2026, "source_map": source_map,
        }]
    }), encoding="utf-8")
    (roots.journal_root / "LATEST-JOURNAL.json").write_text(
        json.dumps({"canonical_journal_path": "current.json"}), encoding="utf-8"
    )
    monkeypatch.setattr(capture_tool, "configured_runtime_roots", lambda: roots)

    class TrustedCaptureError(ValueError):
        pass

    def capture_registered_source(**_kwargs):
        captured_at = datetime.now(timezone.utc).isoformat()
        snapshot_payload = {
            "schema_version": "tournament_trusted_snapshot.v1",
            "pulled_at_utc": captured_at,
            "verified_claims": [
                {"claim_type": "overall_winner", "value": "BAR South"},
                {"claim_type": "points", "value": "4300"},
            ],
            "verified_finality": {
                "displayed": True,
                "standings_final": False,
                "payout_final": False,
            },
        }
        snapshot_text = json.dumps(
            snapshot_payload, sort_keys=True, separators=(",", ":")
        )
        snapshot_path = roots.source_snapshot_root / "registered-event.capture.json"
        snapshot_path.write_text(snapshot_text, encoding="utf-8")
        return {
            "capture_kind": "registered_direct_source",
            "source_id": "registered-event",
            "source_snapshot_path": str(snapshot_path),
            "source_snapshot_sha256": hashlib.sha256(snapshot_text.encode()).hexdigest(),
            "pulled_at_utc": captured_at,
            "event_status": "displayed",
            "source_url": source_map["direct_url"],
            "provider_host": source_map["provider_host"],
            "status_code": 200,
            "content_type": "application/json",
            "byte_count": len(snapshot_text.encode()),
        }

    monkeypatch.setitem(
        sys.modules,
        "audit_agent.tournament_trusted_capture",
        SimpleNamespace(
            TrustedCaptureError=TrustedCaptureError,
            capture_registered_source=capture_registered_source,
        ),
    )
    monkeypatch.setattr(capture_tool, "_runtime", lambda: object())
    captured = json.loads(capture_tool.run_tournament_source_capture({"source_id": "registered-event"}))
    manifest = captured["evidence_manifest"][0]
    assert set(manifest) == {
        "capture_kind", "source_id", "source_snapshot_path", "source_snapshot_sha256",
        "pulled_at_utc", "event_status", "source_url", "provider_host", "status_code",
        "content_type", "byte_count",
    }

    candidate = "BAR South leads the displayed Overall board."
    claim = {
        "claim_id": "winner", "evidence_row_id": "winner-row",
        "tournament_key": "registered-event", "year": 2026,
        "claim_type": "overall_winner", "value": "BAR South", "unit": None,
        "finality_level": "displayed",
    }
    request = {
        "target_year": 2026, "request_classification": "current",
        "intended_tournaments": [{"tournament_key": "registered-event"}],
        "held_or_excluded": [],
        "routes": [{
            "tournament_key": "registered-event", "official_name": "Example Open", "year": 2026,
            "provider": "CatchStat", "direct_url": source_map["direct_url"],
            "provider_host": source_map["provider_host"], "source_role": "host_scoring",
            "event_header": "Example Open", "category_ids": [9312], "category_labels": ["Overall"],
            "freshness": "captured", "identity": source_map["identity"],
        }],
        "evidence_manifest": [{
            **manifest, "row_id": "winner-row", "tournament_key": "registered-event",
            "tournament_name": "Example Open", "year": 2026, "public_label": "Example Open",
            "source_url": source_map["direct_url"], "source_role": "host_scoring",
            "event_year_proof": 2026, "confidence_score": 97,
            "confidence_reason": "Registered CatchStat capture.", "why_not_higher": "Displayed board.",
            "stale_year_check": "pass", "direct_scoring_checked": True,
            "allowed_claims": [{key: value for key, value in claim.items() if key != "evidence_row_id"}],
        }],
    }
    metadata = {
        "content_domain": "tournament_results", "factual_claims": [claim],
        "public_surfaces": [{"surface": "private_telegram_draft", "text": candidate, "claim_ids": ["winner"]}],
    }
    monkeypatch.setattr(tool, "configured_runtime_roots", lambda: roots)
    monkeypatch.setattr(
        "agent.tournament_intent_contract.configured_runtime_roots", lambda: roots
    )

    def bound_preflight(_roots, request_payload, *, suffix):
        output_dir = roots.receipt_root / suffix
        output_dir.mkdir()
        now = datetime.now(timezone.utc)
        receipt = {
            "schema_version": "tournament_route_preflight.v2",
            "decision": "ALLOW_PUBLIC_ARTIFACT",
            "issued_at_utc": now.isoformat(),
            "expires_at_utc": (now + timedelta(minutes=15)).isoformat(),
            "allowed_entrypoints": request_payload["allowed_entrypoints"],
            "artifact_payload_hash": canonical_json_sha256(
                request_payload["artifact_payload"]
            ),
        }
        receipt["receipt_hash"] = canonical_json_sha256(receipt)
        receipt_path = output_dir / "receipt.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        return receipt_path, receipt, "receipt_loaded"

    monkeypatch.setattr(tool, "_run_preflight", bound_preflight)

    def require_public_entrypoint_receipt(
        *, entrypoint, artifact_payload, receipt_path, approved_receipt_root, **_kwargs
    ):
        assert receipt_path.resolve().is_relative_to(approved_receipt_root.resolve())
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert entrypoint in receipt["allowed_entrypoints"]
        assert receipt["artifact_payload_hash"] == canonical_json_sha256(
            artifact_payload
        )
        assert receipt["receipt_hash"] == canonical_json_sha256(
            {key: value for key, value in receipt.items() if key != "receipt_hash"}
        )

    monkeypatch.setitem(
        sys.modules,
        "audit_agent.tournament_artifact_gate",
        SimpleNamespace(
            require_public_entrypoint_receipt=require_public_entrypoint_receipt
        ),
    )
    agent = _bound_agent("capture-gate")
    contract = begin_tournament_intent_contract(
        agent,
        message="Create a public tournament Story for Instagram.",
        task_id="capture-gate",
        turn_provenance=_bound_direct_provenance(
            "Create a public tournament Story for Instagram.", "capture-gate"
        ),
    )
    result = json.loads(tool.run_tournament_truth_gate(
        {"candidate": candidate, "request": request, "artifact_metadata": metadata},
        task_id="capture-gate", session_id="capture-gate",
    ))
    assert result["accepted"] is True
    evidence = contract.audit_request["evidence_manifest"][0]["source_evidence"]
    assert {("overall_winner", "BAR South"), ("points", "4300")} <= {
        (item["claim_type"], item["value"]) for item in evidence["verified_claims"]
    }
    assert contract.receipt_candidate_sha256 == contract.candidate_sha256(candidate)
    messages = [{"role": "user", "content": "draft"}, {"role": "assistant", "content": candidate}]
    response, telemetry, failed = finalize_tournament_output(
        agent, candidate=candidate, messages=messages
    )
    assert failed is False, (response, telemetry)
    assert response == candidate
    assert telemetry["code"] == "receipt_verified"
    assert messages[-1]["content"] == candidate
    assert agent._tournament_intent_contract is None
