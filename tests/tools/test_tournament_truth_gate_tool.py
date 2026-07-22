import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from agent import tournament_research_contract as contract_module
from agent.tournament_research_contract import (
    RuntimeRoots,
    begin_tournament_research_contract,
    canonical_json_sha256,
)
from tools import tournament_truth_gate_tool as tool


class Agent:
    session_id = "session"
    _current_turn_id = "turn"
    platform = "local"
    stream_delta_callback = None
    _stream_callback = None
    _tool_guardrails = None


def _roots(tmp_path):
    roots = RuntimeRoots(tmp_path / "receipts", tmp_path / "journal", tmp_path / "snapshots")
    for path in (roots.receipt_root, roots.journal_root, roots.source_snapshot_root):
        path.mkdir(parents=True)
    return roots


def test_tool_requires_read_only_trusted_source_snapshot_before_running_provider_command(tmp_path, monkeypatch):
    roots = _roots(tmp_path)
    monkeypatch.setattr(tool, "configured_runtime_roots", lambda: roots)
    agent = Agent()
    begin_tournament_research_contract(agent, message="publish tournament standings", task_id="task", external_action=True)
    result = json.loads(tool.run_tournament_truth_gate({"candidate": "answer", "request": {}, "artifact_metadata": {}}, task_id="task", session_id="session"))
    assert result["code"] == "trusted_source_snapshot_required"


def test_tool_runs_argument_list_once_and_binds_only_trusted_v2_receipt(tmp_path, monkeypatch):
    roots = _roots(tmp_path)
    snapshot = roots.source_snapshot_root / "source.json"
    snapshot.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(tool, "configured_runtime_roots", lambda: roots)
    monkeypatch.setattr(contract_module, "configured_runtime_roots", lambda: roots)
    agent = Agent()
    contract = begin_tournament_research_contract(agent, message="publish tournament standings", task_id="task", external_action=True)
    request = {"evidence_manifest": [{"source_snapshot_path": str(snapshot), "source_snapshot_sha256": "0" * 64}]}
    metadata = {"factual_claims": [], "public_surfaces": []}

    def fake_run(command, **kwargs):
        output_dir = roots.receipt_root / "hermes-preflight" / contract.nonce / "preflight"
        payload = contract_module.build_artifact_payload("answer", contract.destination, metadata)
        now = datetime.now(timezone.utc)
        receipt = {
            "schema_version": "tournament_route_preflight.v2", "decision": "ALLOW_PUBLIC_ARTIFACT",
            "source_repository": contract_module.AUDIT_SOURCE_REPOSITORY,
            "contract_commit": contract_module.AUDIT_CONTRACT_COMMIT,
            "allowed_entrypoints": [contract.entrypoint], "artifact_payload_hash": canonical_json_sha256(payload),
            "issued_at_utc": now.isoformat(), "expires_at_utc": (now + timedelta(minutes=15)).isoformat(),
        }
        receipt["receipt_hash"] = canonical_json_sha256(receipt)
        path = output_dir / "receipt.json"
        path.write_text(json.dumps(receipt), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout=json.dumps({"receipt_path": str(path)}))

    monkeypatch.setattr(tool.subprocess, "run", fake_run)
    output = json.loads(tool.run_tournament_truth_gate({"candidate": "answer", "request": request, "artifact_metadata": metadata}, task_id="task", session_id="session"))
    assert output["accepted"] is True
    assert contract.has_valid_receipt() is True


def test_tool_rejects_a_receipt_path_other_than_its_nonce_output_dir(tmp_path, monkeypatch):
    roots = _roots(tmp_path)
    snapshot = roots.source_snapshot_root / "source.json"
    snapshot.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(tool, "configured_runtime_roots", lambda: roots)
    agent = Agent()
    begin_tournament_research_contract(agent, message="publish tournament standings", task_id="task", external_action=True)
    request = {"evidence_manifest": [{"source_snapshot_path": str(snapshot), "source_snapshot_sha256": "0" * 64}]}

    def fake_run(_command, **_kwargs):
        other = roots.receipt_root / "other-receipt.json"
        other.write_text("{}", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout=json.dumps({"receipt_path": str(other)}))

    monkeypatch.setattr(tool.subprocess, "run", fake_run)
    output = json.loads(tool.run_tournament_truth_gate({"candidate": "answer", "request": request, "artifact_metadata": {}}, task_id="task", session_id="session"))
    assert output["code"] == "audit_receipt_path_mismatch"
