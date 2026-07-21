"""Request-local, fail-closed release contract for tournament truth.

The audit repository owns tournament truth.  This module only binds a receipt
created by its installed ``tournament-artifact-preflight`` command to one
Hermes turn.  Model-provided routes, evidence, and receipt paths are never
trusted inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import threading
from typing import Any, Callable, Mapping


AUDIT_SCHEMA_VERSION = "tournament_route_preflight.v2"
RECEIPT_LIFETIME = timedelta(minutes=15)


class TournamentIntent(str, Enum):
    PRIVATE = "private"
    PUBLIC = "public"


@dataclass(frozen=True)
class RuntimeRoots:
    receipt_root: Path
    journal_root: Path
    source_snapshot_root: Path


@dataclass(frozen=True)
class TournamentReceiptDecision:
    accepted: bool
    code: str


_CONTRACTS: dict[tuple[str, str], "TournamentResearchContract"] = {}
_CONTRACTS_LOCK = threading.RLock()
_TOURNAMENT_IDENTITY_CUE = re.compile(
    r"\b(?:tournaments?|catchstat|leaderboards?|standings?|weigh[- ]?ins?|"
    r"calcutta|release\s+points?)\b",
    re.IGNORECASE,
)
_SPORTFISH_CUE = re.compile(
    r"\b(?:boat|angler|captain|team|fleet|marlin|sailfish|wahoo|mahi|"
    r"tuna|kingfish|game\s+fish)\b",
    re.IGNORECASE,
)
_PRIVATE_CUE = re.compile(r"\b(?:private|internal|draft|research|audit|review|verify)\b", re.IGNORECASE)
_RESULT_CUE = re.compile(r"\b(?:results?|scor(?:e|ing)|winner|won|placing|points?)\b", re.IGNORECASE)
_PUBLIC_CUE = re.compile(r"\b(?:public|publish|post|send|announce|share|carousel|story|newsletter|image|caption)\b", re.IGNORECASE)


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _trusted_journal_aliases() -> set[str] | None:
    """Read selected event names only from the current trusted journal pointer."""
    roots = configured_runtime_roots()
    if roots is None:
        return None
    pointer_path = contained_path(roots.journal_root, roots.journal_root / "LATEST-JOURNAL.json")
    if pointer_path is None:
        return None
    try:
        pointer_text = secure_read_contained_text(roots.journal_root, pointer_path)
        pointer = json.loads(pointer_text) if pointer_text else None
        journal_path = contained_path(roots.journal_root, pointer.get("canonical_journal_path", ""))
        journal_text = secure_read_contained_text(roots.journal_root, journal_path) if journal_path else None
        journal = json.loads(journal_text) if journal_text else None
    except (OSError, ValueError, TypeError, UnicodeDecodeError):
        return None
    selected = journal.get("selected_tournaments") if isinstance(journal, Mapping) else None
    if not isinstance(selected, list):
        return None
    aliases: set[str] = set()
    for item in selected:
        if not isinstance(item, Mapping):
            continue
        for key in ("tournament_key", "tournament_name", "official_name", "name"):
            value = item.get(key)
            if isinstance(value, str) and len(value.strip()) >= 4:
                aliases.add(value.casefold())
        for value in item.get("aliases", []) if isinstance(item.get("aliases"), list) else []:
            if isinstance(value, str) and len(value.strip()) >= 4:
                aliases.add(value.casefold())
    return aliases


def classify_tournament_intent(message: object) -> TournamentIntent | None:
    """Classify tournament-result intent without event-specific hardcoding.

    Generic "results" is ordinary chat. Protected intent needs a tournament
    identity, sportfish entity plus results/scoring, or a selected current
    journal alias. Ordinary factual questions default to private.
    """
    if not isinstance(message, str):
        return None
    explicit_identity = bool(_TOURNAMENT_IDENTITY_CUE.search(message))
    sportfish_result = bool(_SPORTFISH_CUE.search(message) and _RESULT_CUE.search(message))
    aliases = _trusted_journal_aliases()
    alias_match = bool(aliases and any(alias in message.casefold() for alias in aliases))
    if not explicit_identity and not sportfish_result and not alias_match:
        return None
    return TournamentIntent.PUBLIC if _PUBLIC_CUE.search(message) else TournamentIntent.PRIVATE


def configured_runtime_roots() -> RuntimeRoots | None:
    """Read roots from process-owned configuration, never a tool argument."""
    try:
        from hermes_cli.config import load_config_readonly

        config = (load_config_readonly() or {}).get("tournament_truth_gate", {})
    except Exception:
        return None
    if not isinstance(config, Mapping):
        return None
    values = [config.get(key) for key in ("receipt_root", "journal_root", "source_snapshot_root")]
    if not all(isinstance(value, str) and value.strip() for value in values):
        return None
    try:
        roots = RuntimeRoots(*(Path(value).resolve(strict=True) for value in values))
    except OSError:
        return None
    if not all(path.is_dir() for path in (roots.receipt_root, roots.journal_root, roots.source_snapshot_root)):
        return None
    return roots


def contained_path(root: Path, candidate: str | Path, *, must_exist: bool = True) -> Path | None:
    try:
        resolved = Path(candidate).resolve(strict=must_exist)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        return None
    return resolved


def secure_read_contained_text(root: Path, candidate: str | Path) -> str | None:
    """Read a contained regular file and reject symlinks or identity races."""
    path = contained_path(root, candidate)
    if path is None:
        return None
    try:
        before = os.lstat(path)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            return None
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            after = os.fstat(descriptor)
            if not stat.S_ISREG(after.st_mode) or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                return None
            return os.read(descriptor, 16 * 1024 * 1024).decode("utf-8")
        finally:
            os.close(descriptor)
    except (OSError, UnicodeDecodeError):
        return None


def build_artifact_payload(candidate: str, destination: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the receipt's artifact hash independent of model-controlled routing."""
    payload = {key: value for key, value in metadata.items() if key not in {"content", "destination"}}
    payload["content"] = candidate
    payload["destination"] = destination
    return payload


@dataclass
class TournamentResearchContract:
    intent: TournamentIntent
    task_id: str
    session_id: str
    turn_id: str
    entrypoint: str
    destination: str
    nonce: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    callbacks: list[Callable[[str | None], None]] = field(default_factory=list)
    stream_deltas: list[str] = field(default_factory=list)
    original_stream_delta_callback: Callable[[str | None], None] | None = None
    original_stream_callback: Callable[[str | None], None] | None = None
    buffer_callback: Callable[[str | None], None] | None = None
    receipt_path: Path | None = None
    receipt_candidate_sha256: str | None = None
    receipt_metadata: dict[str, Any] | None = None
    audit_request: dict[str, Any] | None = None
    receipt_expires_at: datetime | None = None
    used: bool = False
    closed: bool = False

    def buffer(self, delta: str | None) -> None:
        if isinstance(delta, str) and delta:
            self.stream_deltas.append(delta)

    def attach_receipt(
        self, *, receipt_path: Path, candidate: str, metadata: Mapping[str, Any], audit_request: Mapping[str, Any], expires_at: datetime
    ) -> bool:
        if self.closed or self.used or expires_at <= datetime.now(timezone.utc):
            return False
        self.receipt_path = receipt_path
        self.receipt_candidate_sha256 = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
        self.receipt_metadata = dict(metadata)
        self.audit_request = dict(audit_request)
        self.receipt_expires_at = expires_at
        return True

    def has_valid_receipt(self) -> bool:
        return bool(
            not self.closed
            and not self.used
            and self.receipt_path
            and self.receipt_candidate_sha256
            and self.receipt_metadata is not None
            and self.audit_request is not None
            and self.receipt_expires_at
            and self.receipt_expires_at > datetime.now(timezone.utc)
        )

    def telemetry(self, *, accepted: bool, code: str, candidate: str) -> dict[str, object]:
        return {
            "intent": self.intent.value,
            "accepted": accepted,
            "code": code,
            "candidate_sha256_prefix": hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:12],
            "streamed_chars": sum(len(delta) for delta in self.stream_deltas),
        }

    def release(self) -> bool:
        delivered = False
        for callback in self.callbacks:
            try:
                for delta in self.stream_deltas:
                    callback(delta)
                callback(None)
                delivered = bool(self.stream_deltas) or delivered
            except Exception:
                pass
        self.stream_deltas.clear()
        return delivered

    def cleanup(self, agent) -> None:
        if self.closed:
            return
        self.closed = True
        if getattr(agent, "stream_delta_callback", None) is self.buffer_callback:
            agent.stream_delta_callback = self.original_stream_delta_callback
        if getattr(agent, "_stream_callback", None) is self.buffer_callback:
            agent._stream_callback = self.original_stream_callback
        guardrails = getattr(agent, "_tool_guardrails", None)
        if guardrails is not None:
            guardrails.set_tournament_contract(None)
        with _CONTRACTS_LOCK:
            _CONTRACTS.pop((self.task_id, self.session_id), None)
        self.stream_deltas.clear()


def begin_tournament_research_contract(agent, *, message: object, task_id: str, stream_callback=None) -> TournamentResearchContract | None:
    intent = classify_tournament_intent(message)
    if intent is None:
        return None
    session_id = str(getattr(agent, "session_id", "") or "")
    # This is created before build_turn_context assigns its provider request ID;
    # task IDs are per invocation and therefore the stable turn identity here.
    turn_id = f"task:{task_id}"
    entrypoint = "private_answer" if intent is TournamentIntent.PRIVATE else "direct_public"
    platform = str(getattr(agent, "platform", "") or "local")
    contract = TournamentResearchContract(
        intent=intent, task_id=task_id, session_id=session_id, turn_id=turn_id,
        entrypoint=entrypoint, destination=f"platform:{platform}",
    )
    contract.buffer_callback = contract.buffer
    contract.original_stream_delta_callback = getattr(agent, "stream_delta_callback", None)
    contract.original_stream_callback = getattr(agent, "_stream_callback", None)
    for callback in (contract.original_stream_delta_callback, contract.original_stream_callback, stream_callback):
        if callable(callback) and not any(_same_callback(callback, existing) for existing in contract.callbacks):
            contract.callbacks.append(callback)
    agent.stream_delta_callback = contract.buffer_callback
    agent._stream_callback = contract.buffer_callback
    agent._tournament_research_contract = contract
    guardrails = getattr(agent, "_tool_guardrails", None)
    if guardrails is not None:
        guardrails.set_tournament_contract(contract)
    with _CONTRACTS_LOCK:
        _CONTRACTS[(task_id, session_id)] = contract
    return contract


def active_contract(task_id: str, session_id: str) -> TournamentResearchContract | None:
    with _CONTRACTS_LOCK:
        return _CONTRACTS.get((task_id, session_id))


def _same_callback(left, right) -> bool:
    return left is right or (
        getattr(left, "__self__", None) is getattr(right, "__self__", None)
        and getattr(left, "__func__", None) is getattr(right, "__func__", None)
    )


def clear_tournament_research_contract(agent) -> None:
    contract = getattr(agent, "_tournament_research_contract", None)
    if isinstance(contract, TournamentResearchContract):
        contract.cleanup(agent)
    agent._tournament_research_contract = None


def _blocked_response(intent: TournamentIntent, reason: str) -> str:
    code = "ROUTE_HOLD" if intent is TournamentIntent.PRIVATE else "PUBLIC_ARTIFACT_BLOCKED"
    return f"{code}: Tournament output was held ({reason}). Re-run the SportFish audit with a current trusted receipt."


def _load_receipt(contract: TournamentResearchContract) -> tuple[Mapping[str, Any] | None, str]:
    roots = configured_runtime_roots()
    if roots is None or contract.receipt_path is None:
        return None, "trusted_runtime_roots_unavailable"
    receipt_path = contained_path(roots.receipt_root, contract.receipt_path)
    if receipt_path is None or not receipt_path.is_file():
        return None, "receipt_path_untrusted"
    try:
        receipt_text = secure_read_contained_text(roots.receipt_root, receipt_path)
        receipt = json.loads(receipt_text) if receipt_text else None
    except (OSError, ValueError, UnicodeDecodeError):
        return None, "receipt_unreadable"
    if not isinstance(receipt, Mapping):
        return None, "receipt_invalid"
    return receipt, "receipt_loaded"


def _parse_utc(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else None


def validate_audit_sink(contract: TournamentResearchContract, candidate: str) -> TournamentReceiptDecision:
    """Delegate current journal/snapshot validation to the audit command."""
    try:
        from tools.tournament_truth_gate_tool import validate_tournament_sink

        accepted, code = validate_tournament_sink(contract, candidate)
        return TournamentReceiptDecision(accepted, code)
    except Exception:
        return TournamentReceiptDecision(False, "audit_sink_validator_unavailable")


def _verify_receipt(contract: TournamentResearchContract, candidate: str) -> TournamentReceiptDecision:
    if not contract.has_valid_receipt() or contract.receipt_metadata is None:
        return TournamentReceiptDecision(False, "receipt_missing_or_consumed")
    if hashlib.sha256(candidate.encode("utf-8")).hexdigest() != contract.receipt_candidate_sha256:
        return TournamentReceiptDecision(False, "candidate_bytes_mismatch")
    receipt, code = _load_receipt(contract)
    if receipt is None:
        return TournamentReceiptDecision(False, code)
    if receipt.get("schema_version") != AUDIT_SCHEMA_VERSION:
        return TournamentReceiptDecision(False, "receipt_schema_mismatch")
    if receipt.get("receipt_hash") != canonical_json_sha256({k: v for k, v in receipt.items() if k != "receipt_hash"}):
        return TournamentReceiptDecision(False, "receipt_hash_mismatch")
    expected_decision = "ALLOW_PRIVATE_ANSWER" if contract.intent is TournamentIntent.PRIVATE else "ALLOW_PUBLIC_ARTIFACT"
    if receipt.get("decision") != expected_decision:
        return TournamentReceiptDecision(False, "receipt_visibility_mismatch")
    if contract.entrypoint not in receipt.get("allowed_entrypoints", []):
        return TournamentReceiptDecision(False, "receipt_entrypoint_mismatch")
    issued_at = _parse_utc(receipt.get("issued_at_utc"))
    expires_at = _parse_utc(receipt.get("expires_at_utc"))
    now = datetime.now(timezone.utc)
    if not issued_at or not expires_at or now < issued_at or now > expires_at or expires_at - issued_at != RECEIPT_LIFETIME:
        return TournamentReceiptDecision(False, "receipt_expired")
    payload = build_artifact_payload(candidate, contract.destination, contract.receipt_metadata)
    if receipt.get("artifact_payload_hash") != canonical_json_sha256(payload):
        return TournamentReceiptDecision(False, "receipt_payload_mismatch")
    # Re-run the audit-owned preflight at the sink. This detects pointer
    # rotation and snapshot changes after the tool call without reimplementing
    # the audit repository's truth rules in Hermes.
    return validate_audit_sink(contract, candidate)


def _redact_current_turn(messages: list[dict[str, Any]], replacement: str) -> None:
    for message in reversed(messages):
        if message.get("role") == "user":
            break
        if message.get("role") == "assistant" and not message.get("tool_calls"):
            message["content"] = replacement
            return


def finalize_tournament_output(agent, *, candidate: str | None, messages: list[dict[str, Any]]) -> tuple[str | None, dict[str, object] | None, bool]:
    contract = getattr(agent, "_tournament_research_contract", None)
    if not isinstance(contract, TournamentResearchContract):
        return candidate, None, False
    candidate_text = candidate or ""
    decision = _verify_receipt(contract, candidate_text)
    if decision.accepted:
        contract.used = True
        delivered = contract.release()
        agent._response_was_previewed = delivered
        telemetry = contract.telemetry(accepted=True, code=decision.code, candidate=candidate_text)
        contract.cleanup(agent)
        agent._tournament_research_contract = None
        return candidate, telemetry, False
    response = _blocked_response(contract.intent, decision.code)
    _redact_current_turn(messages, response)
    agent._response_was_previewed = False
    telemetry = contract.telemetry(accepted=False, code=decision.code, candidate=candidate_text)
    contract.cleanup(agent)
    agent._tournament_research_contract = None
    return response, telemetry, True
