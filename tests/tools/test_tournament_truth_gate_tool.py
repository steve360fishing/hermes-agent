import json
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from agent.tool_guardrails import ToolCallGuardrailController
from agent.tournament_intent_contract import (
    begin_tournament_intent_contract,
    clear_tournament_intent_contract,
    finalize_tournament_output,
)
from agent.tournament_truth_support import canonical_json_sha256
from tools import tournament_truth_gate_tool as tool
from tools import tournament_source_capture_tool as capture_tool


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
                "destination": "platform:telegram",
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
    agent = SimpleNamespace(
        session_id="session-tool",
        platform="telegram",
        tools=[],
        valid_tool_names=set(),
        stream_delta_callback=None,
        _stream_callback=None,
        _persist_session=None,
        _tool_guardrails=ToolCallGuardrailController(),
        _tournament_intent_contract=None,
    )
    contract = begin_tournament_intent_contract(
        agent,
        message="Create a public tournament Story naming winners.",
        task_id="tool-bind",
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
    agent = SimpleNamespace(
        session_id="session-blocked",
        platform="telegram",
        tools=[],
        valid_tool_names=set(),
        stream_delta_callback=None,
        _stream_callback=None,
        _persist_session=None,
        _tool_guardrails=ToolCallGuardrailController(),
        _tournament_intent_contract=None,
    )
    begin_tournament_intent_contract(
        agent,
        message="Create a public tournament Story naming winners.",
        task_id="tool-blocked",
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
    from audit_agent.tournament_trusted_capture import TrustedCaptureRuntime

    body = json.dumps({
        "categoryId": 9312, "totalCount": 1,
        "catchLogData": [{"rank": 1, "team": "BAR South", "points": 4300}],
    }).encode()
    monkeypatch.setattr(capture_tool, "_runtime", lambda: TrustedCaptureRuntime(
        transport=lambda url, **_kwargs: (200, {"content-type": "application/json"}, body, url),
        resolver=lambda host: ("8.8.8.8",),
    ))
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

    def real_preflight(_roots, request_payload, *, suffix):
        from audit_agent.tournament_artifact_gate import run_tournament_artifact_preflight_command

        output_dir = roots.receipt_root / suffix
        output_dir.mkdir()
        request_path = output_dir / "request.json"
        request_path.write_text(json.dumps(request_payload), encoding="utf-8")
        result = run_tournament_artifact_preflight_command(
            request_json=request_path,
            journal_pointer=roots.journal_root / "LATEST-JOURNAL.json",
            approved_journal_root=roots.journal_root,
            approved_source_snapshot_root=roots.source_snapshot_root,
            approved_receipt_root=roots.receipt_root,
            output_dir=output_dir / "issued",
            now=datetime.now(timezone.utc),
        )
        receipt_path = Path(result["receipt_path"]) if result.get("receipt_path") else None
        receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path else None
        return receipt_path, receipt, "receipt_loaded" if receipt_path else "audit_preflight_failed"

    monkeypatch.setattr(tool, "_run_preflight", real_preflight)
    agent = SimpleNamespace(session_id="capture-gate", platform="telegram", tools=[], valid_tool_names=set(),
        stream_delta_callback=None, _stream_callback=None, _persist_session=None,
        _tool_guardrails=ToolCallGuardrailController(), _tournament_intent_contract=None)
    contract = begin_tournament_intent_contract(agent, message="Create a public tournament Story.", task_id="capture-gate")
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
