"""Self-registering bridge to the installed SportFish tournament audit gate.

The model may propose evidence, but cannot choose roots, command arguments, or
the receipt path.  The audit command consumes only existing trusted snapshots;
there is intentionally no provider fetch or retry path here.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

from agent.tournament_truth_support import (
    AUDIT_SCHEMA_VERSION,
    build_artifact_payload,
    canonical_json_sha256,
    configured_runtime_roots,
    contained_path,
    secure_read_contained_text,
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


def _build_request_payload(candidate: str, request: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(request)
    payload.setdefault("artifact_type", "private_answer")
    payload.setdefault("allowed_entrypoints", ["advisory_validation"])
    payload["artifact_payload"] = build_artifact_payload(candidate, "tool:tournament_truth_gate", metadata)
    payload.pop("receipt_path", None)
    payload.pop("journal_pointer", None)
    return payload


def _run_preflight(roots, request_payload: Mapping[str, Any], *, suffix: str):
    import secrets
    output_dir = roots.receipt_root / "hermes-preflight" / secrets.token_urlsafe(18) / suffix
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", dir=output_dir, delete=False) as handle:
            json.dump(request_payload, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            request_path = Path(handle.name)
        command = _audit_command() + [
            "--request-json", str(request_path), "--journal-pointer", str(roots.journal_root / "LATEST-JOURNAL.json"),
            "--approved-journal-root", str(roots.journal_root), "--approved-source-snapshot-root", str(roots.source_snapshot_root),
            "--approved-receipt-root", str(roots.receipt_root), "--output-dir", str(output_dir),
        ]
        completed = subprocess.run(
            command, shell=False, stdin=subprocess.DEVNULL, capture_output=True,
            text=True, timeout=TIMEOUT_SECONDS, check=False,
            cwd=roots.source_snapshot_root,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONUTF8": "1", "PYTHONSAFEPATH": "1"},
        )
    except subprocess.TimeoutExpired:
        return None, None, "audit_preflight_timeout"
    except (OSError, ValueError):
        return None, None, "audit_preflight_unavailable"
    finally:
        try: request_path.unlink(missing_ok=True)
        except UnboundLocalError: pass
    if completed.returncode != 0:
        return None, None, "audit_preflight_failed"
    try:
        result = json.loads(completed.stdout)
        expected_receipt_path = output_dir / "receipt.json"
        receipt_path = contained_path(roots.receipt_root, expected_receipt_path)
        if str(result.get("receipt_path") or "") != str(expected_receipt_path):
            return None, None, "audit_receipt_path_mismatch"
        receipt_text = secure_read_contained_text(roots.receipt_root, receipt_path) if receipt_path else None
        receipt = json.loads(receipt_text) if receipt_text else None
    except (OSError, ValueError, TypeError, UnicodeDecodeError):
        return None, None, "audit_receipt_invalid"
    return receipt_path, receipt, "receipt_loaded"


def run_tournament_truth_gate(
    args: Mapping[str, Any], *, task_id: str = "", session_id: str = "", **_kwargs: Any
) -> str:
    """Run one bounded advisory preflight; it never affects turn delivery."""
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

    request_payload = _build_request_payload(candidate, request, metadata)
    receipt_path, receipt, code = _run_preflight(roots, request_payload, suffix="preflight")
    if receipt_path is None:
        return tool_error("Tournament audit preflight rejected or failed.", code=code)
    valid = isinstance(receipt, Mapping) and receipt.get("schema_version") == AUDIT_SCHEMA_VERSION and receipt.get("receipt_hash") == canonical_json_sha256({k: v for k, v in receipt.items() if k != "receipt_hash"})
    return tool_result(accepted=bool(valid), advisory=True, code=("verified" if valid else "receipt_invalid"), receipt_sha256=(canonical_json_sha256(receipt) if valid else ""), candidate_sha256=canonical_json_sha256({"candidate": candidate}))


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
