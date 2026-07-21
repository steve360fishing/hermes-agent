"""Self-registering bridge to the installed SportFish tournament audit gate.

The model may propose evidence, but cannot choose roots, command arguments, or
the receipt path.  The audit command consumes only existing trusted snapshots;
there is intentionally no provider fetch or retry path here.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

from agent.tournament_research_contract import (
    AUDIT_SCHEMA_VERSION,
    active_contract,
    build_artifact_payload,
    canonical_json_sha256,
    configured_runtime_roots,
    contained_path,
)
from tools.registry import registry, tool_error, tool_result


TIMEOUT_SECONDS = 45


def _audit_command() -> list[str]:
    """Use the runtime's installed audit module without involving a shell."""
    return [sys.executable, "-m", "audit_agent", "tournament-artifact-preflight"]


def _read_only_snapshot_ingestion(request: Mapping[str, Any], snapshot_root: Path) -> str | None:
    """Require model references to resolve to already-captured trusted snapshots."""
    rows = request.get("evidence_manifest")
    if not isinstance(rows, list) or not rows:
        return "trusted_source_snapshot_required"
    for row in rows:
        if not isinstance(row, Mapping):
            return "trusted_source_snapshot_invalid"
        path = contained_path(snapshot_root, str(row.get("source_snapshot_path") or ""))
        digest = row.get("source_snapshot_sha256")
        if path is None or not path.is_file() or not isinstance(digest, str) or len(digest) != 64:
            return "trusted_source_snapshot_missing"
    return None


def run_tournament_truth_gate(
    args: Mapping[str, Any], *, task_id: str = "", session_id: str = "", **_kwargs: Any
) -> str:
    """Run one bounded preflight and bind its receipt to the current turn."""
    contract = active_contract(str(task_id or ""), str(session_id or ""))
    if contract is None:
        return tool_error("Tournament truth gate has no active request-local contract.")
    candidate = args.get("candidate")
    request = args.get("request")
    metadata = args.get("artifact_metadata", {})
    if not isinstance(candidate, str) or not candidate:
        return tool_error("candidate must be the exact non-empty proposed final answer.")
    if not isinstance(request, Mapping) or not isinstance(metadata, Mapping):
        return tool_error("request and artifact_metadata must be objects.")
    roots = configured_runtime_roots()
    if roots is None:
        return tool_error("Trusted tournament runtime roots are not configured.", code="trusted_runtime_roots_unavailable")
    snapshot_error = _read_only_snapshot_ingestion(request, roots.source_snapshot_root)
    if snapshot_error:
        return tool_error("No trusted direct-source snapshot is available; provider fetch was not attempted.", code=snapshot_error)

    artifact_payload = build_artifact_payload(candidate, contract.destination, metadata)
    request_payload = dict(request)
    request_payload["artifact_type"] = "private_answer" if contract.intent.value == "private" else "public_copy"
    request_payload["allowed_entrypoints"] = [contract.entrypoint]
    request_payload["artifact_payload"] = artifact_payload
    request_payload.pop("receipt_path", None)
    request_payload.pop("journal_pointer", None)

    output_dir = roots.receipt_root / "hermes-preflight" / contract.nonce
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", dir=output_dir, delete=False) as handle:
            json.dump(request_payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            request_path = Path(handle.name)
        command = _audit_command() + [
            "--request-json", str(request_path), "--journal-pointer", str(roots.journal_root / "LATEST-JOURNAL.json"),
            "--approved-journal-root", str(roots.journal_root), "--approved-source-snapshot-root", str(roots.source_snapshot_root),
            "--approved-receipt-root", str(roots.receipt_root), "--output-dir", str(output_dir),
        ]
        completed = subprocess.run(command, shell=False, capture_output=True, text=True, timeout=TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired:
        return tool_error("Tournament audit preflight timed out; no retry was attempted.", code="audit_preflight_timeout")
    except (OSError, ValueError) as exc:
        return tool_error("Tournament audit preflight could not start.", code="audit_preflight_unavailable")
    finally:
        try:
            request_path.unlink(missing_ok=True)
        except UnboundLocalError:
            pass

    if completed.returncode != 0:
        return tool_error("Tournament audit preflight rejected or failed.", code="audit_preflight_failed", exit_code=completed.returncode)
    try:
        result = json.loads(completed.stdout)
        receipt_path = contained_path(roots.receipt_root, str(result.get("receipt_path") or ""))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path else None
    except (OSError, ValueError, TypeError, UnicodeDecodeError):
        return tool_error("Tournament audit returned no readable trusted receipt.", code="audit_receipt_invalid")
    expires_at = _parse_expiry(receipt.get("expires_at_utc") if isinstance(receipt, Mapping) else None)
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema_version") != AUDIT_SCHEMA_VERSION
        or receipt.get("receipt_hash") != canonical_json_sha256({k: v for k, v in receipt.items() if k != "receipt_hash"})
        or expires_at is None
        or not contract.attach_receipt(receipt_path=receipt_path, candidate=candidate, metadata=metadata, expires_at=expires_at)
    ):
        return tool_error("Tournament audit receipt was not safely bound to this turn.", code="audit_receipt_binding_failed")
    return tool_result(
        accepted=True,
        schema_version=AUDIT_SCHEMA_VERSION,
        receipt_sha256=canonical_json_sha256(receipt),
        candidate_sha256=canonical_json_sha256({"candidate": candidate}),
        expires_at_utc=expires_at.isoformat(),
    )


def _parse_expiry(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else None


TOURNAMENT_TRUTH_GATE_SCHEMA = {
    "name": "tournament_truth_gate",
    "description": "Run the trusted, read-only tournament source preflight for the exact proposed final answer. No provider fetch is attempted.",
    "parameters": {
        "type": "object",
        "properties": {
            "candidate": {"type": "string", "description": "Exact final answer bytes to authorize."},
            "request": {"type": "object", "description": "Audit evidence request; all sources must reference trusted snapshots."},
            "artifact_metadata": {"type": "object", "description": "Claim/surface metadata bound with the candidate."},
        },
        "required": ["candidate", "request", "artifact_metadata"],
    },
}


registry.register(
    name="tournament_truth_gate", toolset="sportfish", schema=TOURNAMENT_TRUTH_GATE_SCHEMA,
    handler=lambda args, **kw: run_tournament_truth_gate(args, **kw), emoji="🧭", max_result_size_chars=8_000,
)
