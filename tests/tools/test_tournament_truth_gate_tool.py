import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from agent.tool_guardrails import ToolCallGuardrailController
from agent.tournament_intent_contract import (
    begin_tournament_intent_contract,
    clear_tournament_intent_contract,
)
from agent.tournament_truth_support import canonical_json_sha256
from tools import tournament_truth_gate_tool as tool


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
    snapshot.write_text("{}", encoding="utf-8")
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
                            "source_snapshot_sha256": "a" * 64,
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
    snapshot.write_text("{}", encoding="utf-8")
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
                            "source_snapshot_sha256": "a" * 64,
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
