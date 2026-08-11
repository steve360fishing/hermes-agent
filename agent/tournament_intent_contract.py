"""Request-local tournament audience, truth, and release authority contract.

Private questions and private artifacts never install this contract. Public
drafts are buffered until the audit-owned truth gate binds an exact receipt;
external publication additionally requires an exact, one-use release approval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
import secrets
import threading
from typing import Any, Callable, Mapping

from agent.tournament_truth_support import (
    AUDIT_SCHEMA_VERSION,
    build_artifact_payload,
    canonical_json_sha256,
    configured_runtime_roots,
    contained_path,
    secure_read_contained_text,
)


POLICY_VERSION = "tournament-intent-v1"
RECEIPT_LIFETIME = timedelta(minutes=15)


class TournamentIntentState(str, Enum):
    PRIVATE_INQUIRY = "private_inquiry"
    PRIVATE_ARTIFACT = "private_artifact"
    PUBLIC_FACING_DRAFT = "public_facing_draft"
    PUBLICATION_REQUEST = "publication_request"
    MIXED_PUBLICATION = "mixed_publication"
    RECEIPT_VALIDATION = "receipt_validation"
    RELEASE_APPROVAL = "release_approval"


@dataclass(frozen=True)
class ContractDecision:
    allowed: bool
    code: str
    message: str = ""
    halt: bool = False


@dataclass
class TournamentReleaseApproval:
    destination: str
    candidate_sha256: str
    identity: str
    expires_at: datetime
    idempotency_key: str
    state: str = "available"


_CONTRACTS: dict[tuple[str, str], "TournamentIntentContract"] = {}
_CONTRACTS_LOCK = threading.RLock()

_TOURNAMENT_CUE = re.compile(
    r"\b(?:tournaments?|catchstat|reel\s*time|leaderboards?|standings?|"
    r"weigh[- ]?ins?|calcutta|release\s+points?|marlin|sailfish|game\s*fish)\b",
    re.IGNORECASE,
)
_TOURNAMENT_CONTINUATION_CUE = re.compile(
    r"\b(?:that|the)\s+approved\s+(?:instagram\s+)?(?:stor(?:y|ies)|post|caption|newsletter|copy)\b",
    re.IGNORECASE,
)
_PRIVATE_ARTIFACT_FRAME = re.compile(
    r"\b(?:internal\s+(?:research\s+)?notes?|private\s+(?:codex\s+)?test|"
    r"codex\s+prompt|prompt\s+(?:for|to)|coding\s+handoff|handoff|"
    r"private\s+(?:audit|report)|file\s+for\s+(?:steve|another\s+agent)|quoted\s+phrase)\b",
    re.IGNORECASE,
)
_PRIVATE_INQUIRY_FRAME = re.compile(
    r"\b(?:analy[sz]e|review|research|investigate|verify|check|what|which|when|where|how)\b",
    re.IGNORECASE,
)
_PUBLIC_SURFACE = re.compile(
    r"\b(?:public|instagram\s+stor(?:y|ies)|stor(?:y|ies)|facebook|newsletter|"
    r"website|public\s+post|carousel|caption|announcement|press\s+release)\b",
    re.IGNORECASE,
)
_DRAFT_ACTION = re.compile(r"\b(?:create|write|draft|make|prepare|generate)\b", re.IGNORECASE)
_PUBLICATION_ACTION = re.compile(
    r"\b(?:publish|send|upload|deploy|release|post(?!\s+(?:is|was|has|that|which|request)))\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(r"\b(?:do\s+not|don't|without|never)\b", re.IGNORECASE)
_NEGATION_RESET = re.compile(
    r"\b(?:but|instead|then|however)\b|,\s*please\b|"
    r"[-\u2014]\s*(?:please\s+)?(?:create|make|publish|post|send|upload|release)\b",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARIES = frozenset(".!?;\n")
_RECEIPT_OPERATION = re.compile(
    r"\b(?:validate|verify|check)\b.{0,48}\b(?:truth\s+)?receipt\b|"
    r"\breceipt\s+validation\b",
    re.IGNORECASE,
)
_APPROVAL_OPERATION = re.compile(
    r"\b(?:record|grant|approve|validate)\b.{0,64}\b(?:release|publication)\s+approval\b",
    re.IGNORECASE,
)
_FENCED_DATA = re.compile(r"```.*?```", re.DOTALL)
_INLINE_DATA = re.compile(r"`[^`\n]*`")
_QUOTED_DATA = re.compile(
    r'"(?:\\.|[^"\\])*"|(?<!\w)\'(?:\\.|[^\'\\\n])*\'(?!\w)|'
    r"\u201c[^\u201d\n]*\u201d|\u2018[^\u2019\n]*\u2019"
)
_URL_DATA = re.compile(r"https?://\S+", re.IGNORECASE)
_LABELED_DATA = re.compile(
    r"(?im)(?P<prefix>\b(?:quoted\s+text|quote|fixture|data\s+field(?:\s+only)?|"
    r"log(?:\s+entry)?|forwarded\s+message|attachment(?:\s+(?:text|content))?|"
    r"pasted\s+text)\s*:)\s*.*$"
)

_READ_ONLY_PUBLIC_TOOLS = frozenset(
    {
        "tournament_truth_gate",
        "read_file",
        "search_files",
        "session_search",
        "skill_view",
        "skills_list",
        "web_search",
        "web_extract",
        "browser_snapshot",
        "browser_console",
        "browser_get_images",
        "mcp_filesystem_read_file",
        "mcp_filesystem_read_text_file",
        "mcp_filesystem_read_multiple_files",
        "mcp_filesystem_list_directory",
        "mcp_filesystem_list_directory_with_sizes",
        "mcp_filesystem_directory_tree",
        "mcp_filesystem_get_file_info",
        "mcp_filesystem_search_files",
        "sportfish_tournament_research",
    }
)


def _has_affirmative_action(message: str, pattern: re.Pattern[str]) -> bool:
    for match in pattern.finditer(message):
        clause_start = max(
            (message.rfind(boundary, 0, match.start()) for boundary in _CLAUSE_BOUNDARIES),
            default=-1,
        ) + 1
        prefix = message[clause_start : match.start()]
        negations = list(_NEGATION.finditer(prefix))
        if not negations:
            return True
        if _NEGATION_RESET.search(prefix[negations[-1].end() :]):
            return True
    return False


def _mask_non_authoritative_data(message: str) -> str:
    """Remove data-only spans before deriving public authority from vocabulary."""
    masked = _FENCED_DATA.sub(" ", message)
    masked = _INLINE_DATA.sub(" ", masked)
    masked = _QUOTED_DATA.sub(" ", masked)
    masked = _URL_DATA.sub(" ", masked)
    return _LABELED_DATA.sub(lambda match: match.group("prefix"), masked)


def _classify_tournament_intents(message: object) -> frozenset[TournamentIntentState]:
    """Return every directed intent present after removing data-only vocabulary."""
    if not isinstance(message, str):
        return frozenset()
    directed = _mask_non_authoritative_data(message)
    raw_has_tournament_cue = bool(
        _TOURNAMENT_CUE.search(message) or _TOURNAMENT_CONTINUATION_CUE.search(message)
    )
    if not raw_has_tournament_cue:
        return frozenset()
    directed_has_tournament_cue = bool(
        _TOURNAMENT_CUE.search(directed) or _TOURNAMENT_CONTINUATION_CUE.search(directed)
    )
    states: set[TournamentIntentState] = set()
    if _PRIVATE_ARTIFACT_FRAME.search(directed):
        states.add(TournamentIntentState.PRIVATE_ARTIFACT)
    elif _PRIVATE_INQUIRY_FRAME.search(directed):
        states.add(TournamentIntentState.PRIVATE_INQUIRY)
    if _RECEIPT_OPERATION.search(directed):
        states.add(TournamentIntentState.RECEIPT_VALIDATION)
    if _APPROVAL_OPERATION.search(directed):
        states.add(TournamentIntentState.RELEASE_APPROVAL)
    public_action_text = _APPROVAL_OPERATION.sub(" ", directed)
    if _has_affirmative_action(public_action_text, _PUBLICATION_ACTION):
        states.add(TournamentIntentState.PUBLICATION_REQUEST)
    if _PUBLIC_SURFACE.search(directed) and _has_affirmative_action(directed, _DRAFT_ACTION):
        states.add(TournamentIntentState.PUBLIC_FACING_DRAFT)
    return frozenset(states or {TournamentIntentState.PRIVATE_INQUIRY})


def classify_tournament_intent(message: object) -> TournamentIntentState | None:
    """Classify the controlling state without discarding concurrent intents."""
    states = _classify_tournament_intents(message)
    if (
        TournamentIntentState.PUBLICATION_REQUEST in states
        and states.intersection(
            {
                TournamentIntentState.PRIVATE_ARTIFACT,
                TournamentIntentState.PRIVATE_INQUIRY,
            }
        )
    ):
        return TournamentIntentState.MIXED_PUBLICATION
    for state in (
        TournamentIntentState.PUBLICATION_REQUEST,
        TournamentIntentState.PUBLIC_FACING_DRAFT,
        TournamentIntentState.RECEIPT_VALIDATION,
        TournamentIntentState.RELEASE_APPROVAL,
        TournamentIntentState.PRIVATE_ARTIFACT,
        TournamentIntentState.PRIVATE_INQUIRY,
    ):
        if state in states:
            return state
    return None


def platform_bypasses_tournament_contract(platform: object) -> bool:
    return str(getattr(platform, "value", platform) or "").strip().casefold() == "cron"


@dataclass
class TournamentIntentContract:
    state: TournamentIntentState
    task_id: str
    session_id: str
    destination: str
    entrypoint: str
    actor_identity: str
    intents: frozenset[TournamentIntentState] = field(default_factory=frozenset)
    policy_version: str = POLICY_VERSION
    nonce: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    preflight_error: str = ""
    callbacks: list[Callable[[str | None], None]] = field(default_factory=list)
    original_stream_delta_callback: Callable[[str | None], None] | None = None
    original_stream_callback: Callable[[str | None], None] | None = None
    original_persist_session: Callable[..., None] | None = None
    pending_persistence: tuple[list[dict[str, Any]], list[dict[str, Any]] | None] | None = None
    buffer_callback: Callable[[str | None], None] | None = None
    added_tool_schema: bool = False
    added_valid_tool_name: bool = False
    receipt_path: Path | None = None
    receipt_candidate_sha256: str = ""
    receipt_metadata: dict[str, Any] | None = None
    audit_request: dict[str, Any] | None = None
    receipt_expires_at: datetime | None = None
    receipt_used: bool = False
    test_receipt: bool = False
    release_approval: TournamentReleaseApproval | None = None
    release_state: str = "not_requested"
    release_binding: tuple[str, str, str, str, str] | None = None
    delivery_callback_failed: bool = False
    closed: bool = False

    @staticmethod
    def candidate_sha256(candidate: str) -> str:
        return hashlib.sha256(candidate.encode("utf-8")).hexdigest()

    @property
    def system_guidance(self) -> str:
        if self.preflight_error:
            return ""
        if self.state is TournamentIntentState.MIXED_PUBLICATION:
            return (
                "TOURNAMENT MIXED PRIVATE/PUBLIC CONTRACT (request-local):\n"
                "Return exactly one JSON object with no markdown fence and exactly these string "
                "fields: {\"private_response\":\"...\",\"public_candidate\":\"...\"}. "
                "The private response remains private. Before any external publication, call "
                "tournament_truth_gate for the exact public_candidate bytes; external release also "
                "requires an exact, unexpired, one-use release approval."
            )
        if self.state is TournamentIntentState.PUBLIC_FACING_DRAFT:
            return (
                "TOURNAMENT PUBLIC-DRAFT CONTRACT (request-local):\n"
                "Research may use ordinary read-only tools. Before returning any claim-bearing "
                "public copy, call tournament_truth_gate with the exact candidate bytes and "
                "trusted snapshot evidence. A receipt authorizes draft delivery only; it does "
                "not authorize posting, sending, publishing, or any other external action."
            )
        return (
            "TOURNAMENT PUBLICATION CONTRACT (request-local):\n"
            "Research may use ordinary read-only tools. Truth requires tournament_truth_gate "
            "for the exact candidate. External release additionally requires an exact, "
            "unexpired, one-use approval bound to destination, candidate bytes, identity, and "
            "idempotency key. If either authority is absent, prepare safely and do not release."
        )

    def buffer(self, _delta: str | None) -> None:
        return None

    def _defer_persistence(self, messages, conversation_history=None) -> None:
        self.pending_persistence = (messages, conversation_history)

    def persist_final_bytes(self) -> None:
        pending = self.pending_persistence
        self.pending_persistence = None
        if pending and callable(self.original_persist_session):
            self.original_persist_session(*pending)

    def attach_receipt(
        self,
        *,
        receipt_path: Path,
        candidate: str,
        metadata: Mapping[str, Any],
        audit_request: Mapping[str, Any],
        expires_at: datetime,
    ) -> bool:
        if self.closed or self.receipt_used or expires_at <= datetime.now(timezone.utc):
            return False
        self.receipt_path = receipt_path
        self.receipt_candidate_sha256 = self.candidate_sha256(candidate)
        self.receipt_metadata = dict(metadata)
        self.audit_request = dict(audit_request)
        self.receipt_expires_at = expires_at
        self.test_receipt = False
        return True

    def attach_test_receipt(self, *, candidate: str, expires_at: datetime) -> None:
        self.receipt_candidate_sha256 = self.candidate_sha256(candidate)
        self.receipt_expires_at = expires_at
        self.receipt_metadata = {}
        self.audit_request = {}
        self.test_receipt = True

    def attach_release_approval(self, approval: TournamentReleaseApproval) -> bool:
        if self.closed or approval.state != "available" or approval.expires_at <= datetime.now(timezone.utc):
            return False
        self.release_approval = approval
        self.release_state = "prepared_not_released"
        return True

    def verify_receipt(self, candidate: str) -> ContractDecision:
        if (
            self.receipt_used
            or not self.receipt_candidate_sha256
            or self.receipt_expires_at is None
        ):
            return ContractDecision(False, "receipt_missing_or_consumed")
        if self.receipt_expires_at <= datetime.now(timezone.utc):
            return ContractDecision(False, "receipt_expired")
        if self.candidate_sha256(candidate) != self.receipt_candidate_sha256:
            return ContractDecision(False, "candidate_bytes_mismatch")
        if self.test_receipt:
            return ContractDecision(True, "receipt_verified")
        roots = configured_runtime_roots()
        if roots is None or self.receipt_path is None or self.receipt_metadata is None:
            return ContractDecision(False, "trusted_runtime_roots_unavailable")
        receipt_path = contained_path(roots.receipt_root, self.receipt_path)
        if receipt_path is None:
            return ContractDecision(False, "receipt_path_untrusted")
        try:
            text = secure_read_contained_text(roots.receipt_root, receipt_path)
            receipt = json.loads(text) if text else None
        except (OSError, ValueError, TypeError, UnicodeDecodeError):
            return ContractDecision(False, "receipt_unreadable")
        if not isinstance(receipt, Mapping):
            return ContractDecision(False, "receipt_invalid")
        if receipt.get("schema_version") != AUDIT_SCHEMA_VERSION:
            return ContractDecision(False, "receipt_schema_mismatch")
        unsigned = {key: value for key, value in receipt.items() if key != "receipt_hash"}
        if receipt.get("receipt_hash") != canonical_json_sha256(unsigned):
            return ContractDecision(False, "receipt_hash_mismatch")
        if receipt.get("decision") != "ALLOW_PUBLIC_ARTIFACT":
            return ContractDecision(False, "receipt_visibility_mismatch")
        if self.entrypoint not in (receipt.get("allowed_entrypoints") or []):
            return ContractDecision(False, "receipt_entrypoint_mismatch")
        issued_at = _parse_utc(receipt.get("issued_at_utc"))
        expires_at = _parse_utc(receipt.get("expires_at_utc"))
        now = datetime.now(timezone.utc)
        if (
            issued_at is None
            or expires_at is None
            or now < issued_at
            or now > expires_at
            or expires_at - issued_at != RECEIPT_LIFETIME
        ):
            return ContractDecision(False, "receipt_expired")
        payload = build_artifact_payload(candidate, self.destination, self.receipt_metadata)
        if receipt.get("artifact_payload_hash") != canonical_json_sha256(payload):
            return ContractDecision(False, "receipt_payload_mismatch")
        try:
            from tools.tournament_truth_gate_tool import validate_tournament_sink

            accepted, code = validate_tournament_sink(self, candidate)
        except Exception:
            return ContractDecision(False, "audit_sink_validator_unavailable")
        return ContractDecision(bool(accepted), code)

    def authorize_tool(self, tool_name: str, args: Mapping[str, Any]) -> ContractDecision:
        if tool_name == "send_message" and str(args.get("action") or "send") == "list":
            return ContractDecision(True, "allow")
        if tool_name in _READ_ONLY_PUBLIC_TOOLS:
            return ContractDecision(True, "allow")
        if self.state is TournamentIntentState.PUBLIC_FACING_DRAFT:
            candidate = _candidate_from_args(args)
            receipt = self.verify_receipt(candidate) if candidate else ContractDecision(False, "receipt_missing_or_consumed")
            if tool_name == "write_file" and receipt.allowed:
                return ContractDecision(True, "public_draft_receipt_verified")
            return ContractDecision(
                False,
                receipt.code,
                "Public tournament draft mutations require a receipt bound to the exact candidate bytes.",
            )
        return self.authorize_external_action(
            tool_name=tool_name,
            candidate=_candidate_from_args(args),
            destination=_destination_from_args(args),
            identity=str(args.get("actor_identity") or args.get("user_id") or self.actor_identity),
            idempotency_key=str(
                args.get("idempotency_key")
                or (self.release_approval.idempotency_key if self.release_approval else "")
            ),
        )

    def bypasses_task_contract(self, tool_name: str, args: Mapping[str, Any]) -> bool:
        """Let safe research/truth reads coexist with an artifact-only contract."""
        return tool_name in _READ_ONLY_PUBLIC_TOOLS or (
            tool_name == "send_message" and str(args.get("action") or "send") == "list"
        )

    def authorize_external_action(
        self,
        *,
        tool_name: str,
        candidate: str,
        destination: str,
        identity: str,
        idempotency_key: str,
    ) -> ContractDecision:
        if self.release_state == "ambiguous":
            return ContractDecision(False, "release_outcome_ambiguous")
        requested_binding = (
            tool_name,
            self.candidate_sha256(candidate),
            destination,
            identity,
            idempotency_key,
        )
        if self.release_state == "in_flight":
            if self.release_binding == requested_binding:
                return ContractDecision(True, "release_authorized")
            return ContractDecision(False, "release_in_flight_mismatch")
        receipt = self.verify_receipt(candidate)
        if not receipt.allowed:
            if receipt.code == "receipt_missing_or_consumed" and self.release_approval is None:
                return ContractDecision(False, "receipt_and_release_approval_required")
            return receipt
        approval = self.release_approval
        if approval is None:
            return ContractDecision(False, "release_approval_required")
        if approval.state != "available":
            return ContractDecision(False, "release_approval_consumed")
        if approval.expires_at <= datetime.now(timezone.utc):
            return ContractDecision(False, "release_approval_expired")
        if (
            approval.destination != destination
            or approval.candidate_sha256 != self.candidate_sha256(candidate)
            or approval.identity != identity
            or approval.idempotency_key != idempotency_key
        ):
            return ContractDecision(False, "release_approval_mismatch")
        if tool_name != "send_message":
            return ContractDecision(False, "publication_tool_not_bound")
        approval.state = "in_flight"
        self.release_state = "in_flight"
        self.release_binding = requested_binding
        return ContractDecision(True, "release_authorized")

    def record_external_result(self, *, success: bool, ambiguous: bool) -> None:
        approval = self.release_approval
        if approval is None or self.release_state != "in_flight":
            return
        if ambiguous:
            approval.state = "ambiguous"
            self.release_state = "ambiguous"
            return
        if success:
            approval.state = "consumed"
            self.release_state = "consumed"
            self.receipt_used = True
            return
        approval.state = "available"
        self.release_state = "prepared_not_released"
        self.release_binding = None

    def telemetry(
        self,
        *,
        accepted: bool,
        code: str,
        candidate: str,
        turn_status: str = "complete",
    ) -> dict[str, object]:
        return {
            "policy_version": self.policy_version,
            "state": self.state.value,
            "accepted": accepted,
            "code": code,
            "candidate_sha256_prefix": self.candidate_sha256(candidate)[:12],
            "receipt_used": self.receipt_used,
            "release_state": self.release_state,
            "delivery_callback_failed": self.delivery_callback_failed,
            "turn_status": turn_status,
        }

    def release(self, candidate: str) -> bool:
        delivered = False
        for callback in self.callbacks:
            try:
                callback(candidate)
                callback(None)
                delivered = bool(candidate) or delivered
            except Exception:
                self.delivery_callback_failed = True
                continue
        return delivered

    def cleanup(self, agent: Any) -> None:
        if self.closed:
            return
        self.closed = True
        if _same_callback(getattr(agent, "stream_delta_callback", None), self.buffer_callback):
            agent.stream_delta_callback = self.original_stream_delta_callback
        if _same_callback(getattr(agent, "_stream_callback", None), self.buffer_callback):
            agent._stream_callback = self.original_stream_callback
        if _same_callback(getattr(agent, "_persist_session", None), self._defer_persistence):
            agent._persist_session = self.original_persist_session
        guardrails = getattr(agent, "_tool_guardrails", None)
        if guardrails is not None:
            guardrails.set_tournament_contract(None)
        if self.added_tool_schema:
            tools = getattr(agent, "tools", None)
            if isinstance(tools, list):
                agent.tools = [
                    tool for tool in tools
                    if tool.get("function", {}).get("name") != "tournament_truth_gate"
                ]
        if self.added_valid_tool_name:
            valid_names = getattr(agent, "valid_tool_names", None)
            if isinstance(valid_names, set):
                valid_names.discard("tournament_truth_gate")
            elif isinstance(valid_names, list):
                agent.valid_tool_names = [name for name in valid_names if name != "tournament_truth_gate"]
        with _CONTRACTS_LOCK:
            _CONTRACTS.pop((self.task_id, self.session_id), None)


def begin_tournament_intent_contract(
    agent: Any,
    *,
    message: object,
    task_id: str,
    stream_callback=None,
) -> TournamentIntentContract | None:
    raw_platform = getattr(agent, "platform", "")
    if platform_bypasses_tournament_contract(raw_platform):
        return None
    platform = str(raw_platform or "").strip().casefold()
    intents = _classify_tournament_intents(message)
    state = classify_tournament_intent(message)
    if state not in {
        TournamentIntentState.PUBLIC_FACING_DRAFT,
        TournamentIntentState.PUBLICATION_REQUEST,
        TournamentIntentState.MIXED_PUBLICATION,
    }:
        return None
    session_id = str(getattr(agent, "session_id", "") or "")
    contract = TournamentIntentContract(
        state=state,
        task_id=str(task_id),
        session_id=session_id,
        destination=(
            f"platform:{platform or 'local'}:{getattr(agent, '_chat_id', '')}"
            if getattr(agent, "_chat_id", None)
            else f"platform:{platform or 'local'}"
        ),
        entrypoint="direct_public",
        actor_identity=str(getattr(agent, "_user_id", None) or session_id),
        intents=intents,
    )
    try:
        from tools.registry import registry
        from tools.tournament_truth_gate_tool import TOURNAMENT_TRUTH_GATE_SCHEMA

        if registry.get_entry("tournament_truth_gate") is None:
            contract.preflight_error = "truth_gate_unavailable"
    except Exception:
        contract.preflight_error = "truth_gate_unavailable"
        TOURNAMENT_TRUTH_GATE_SCHEMA = None
    tools = getattr(agent, "tools", None)
    valid_names = getattr(agent, "valid_tool_names", None)
    if not isinstance(tools, list) or not isinstance(valid_names, (set, list)):
        contract.preflight_error = contract.preflight_error or "truth_gate_not_in_model_request"
    elif not contract.preflight_error:
        if not any(
            tool.get("function", {}).get("name") == "tournament_truth_gate"
            for tool in tools
        ):
            tools.append({"type": "function", "function": TOURNAMENT_TRUTH_GATE_SCHEMA})
            contract.added_tool_schema = True
        if isinstance(valid_names, set):
            if "tournament_truth_gate" not in valid_names:
                valid_names.add("tournament_truth_gate")
                contract.added_valid_tool_name = True
        elif "tournament_truth_gate" not in valid_names:
            valid_names.append("tournament_truth_gate")
            contract.added_valid_tool_name = True
        if not any(
            tool.get("function", {}).get("name") == "tournament_truth_gate"
            for tool in tools
        ) or "tournament_truth_gate" not in valid_names:
            contract.preflight_error = "truth_gate_not_in_model_request"
    contract.buffer_callback = contract.buffer
    contract.original_stream_delta_callback = getattr(agent, "stream_delta_callback", None)
    contract.original_stream_callback = getattr(agent, "_stream_callback", None)
    contract.original_persist_session = getattr(agent, "_persist_session", None)
    for callback in (
        contract.original_stream_delta_callback,
        contract.original_stream_callback,
        stream_callback,
    ):
        if callable(callback) and not any(_same_callback(callback, current) for current in contract.callbacks):
            contract.callbacks.append(callback)
    agent.stream_delta_callback = contract.buffer_callback
    agent._stream_callback = contract.buffer_callback
    if callable(contract.original_persist_session):
        agent._persist_session = contract._defer_persistence
    agent._tournament_intent_contract = contract
    guardrails = getattr(agent, "_tool_guardrails", None)
    if guardrails is not None:
        guardrails.set_tournament_contract(contract)
    with _CONTRACTS_LOCK:
        _CONTRACTS[(contract.task_id, contract.session_id)] = contract
    return contract


def active_contract(task_id: str, session_id: str) -> TournamentIntentContract | None:
    with _CONTRACTS_LOCK:
        return _CONTRACTS.get((str(task_id), str(session_id)))


def clear_tournament_intent_contract(agent: Any) -> None:
    contract = getattr(agent, "_tournament_intent_contract", None)
    if isinstance(contract, TournamentIntentContract):
        contract.cleanup(agent)
    agent._tournament_intent_contract = None


def effective_request_system_prompt(agent: Any, base_prompt: str) -> str:
    contract = getattr(agent, "_tournament_intent_contract", None)
    guidance = contract.system_guidance if isinstance(contract, TournamentIntentContract) else ""
    return (base_prompt + "\n\n" + guidance).strip() if guidance else base_prompt


def preflight_failure_response(code: str) -> str:
    recovery = {
        "truth_gate_unavailable": "repair the registered tournament truth-gate tool, then retry the draft",
        "truth_gate_not_in_model_request": "repair the provider tool schema, then retry the draft",
    }.get(code, "repair the trusted tournament validation path, then retry")
    return (
        "Public tournament validation is unavailable. No public copy or external "
        f"action was released. Safe recovery: {recovery}."
    )


def _receipt_failure_recovery(code: str) -> str:
    return {
        "receipt_missing_or_consumed": "obtain a fresh receipt for the exact candidate bytes",
        "receipt_and_release_approval_required": (
            "obtain a fresh receipt for the exact candidate bytes and a separate exact release approval"
        ),
        "candidate_bytes_mismatch": "regenerate the receipt for the unchanged candidate bytes",
        "destination_mismatch": "regenerate the receipt for the intended destination",
        "entrypoint_mismatch": "retry through the receipt-bound public entrypoint",
        "receipt_expired": "obtain a new unexpired receipt for the exact candidate bytes",
        "receipt_source_snapshot_mismatch": "rerun trusted source validation and obtain a fresh receipt",
    }.get(code, "repair the trusted validation path and obtain a fresh exact receipt")


def parse_mixed_publication_envelope(candidate: str) -> tuple[str, str] | None:
    try:
        payload = json.loads(candidate)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"private_response", "public_candidate"}:
        return None
    private_response = payload.get("private_response")
    public_candidate = payload.get("public_candidate")
    if not isinstance(private_response, str) or not isinstance(public_candidate, str):
        return None
    return private_response, public_candidate


def _finish_tournament_response(
    agent: Any,
    contract: TournamentIntentContract,
    messages: list[dict[str, Any]],
    *,
    response: str,
    telemetry: dict[str, object],
    release: bool = True,
) -> None:
    _replace_current_turn(messages, response)
    if release:
        agent._response_was_previewed = contract.release(response)
    contract.persist_final_bytes()
    contract.cleanup(agent)
    agent._tournament_intent_contract = None


def abort_tournament_output(
    agent: Any,
    *,
    candidate: str | None,
    messages: list[dict[str, Any]],
    code: str,
    response: str,
) -> tuple[str, dict[str, object] | None, bool]:
    contract = getattr(agent, "_tournament_intent_contract", None)
    if not isinstance(contract, TournamentIntentContract):
        return response, None, True
    telemetry = contract.telemetry(
        accepted=False,
        code=code,
        candidate=candidate or "",
        turn_status="failed",
    )
    _finish_tournament_response(agent, contract, messages, response=response, telemetry=telemetry)
    return response, telemetry, True


def finalize_tournament_output(
    agent: Any,
    *,
    candidate: str | None,
    delivery_response: str | None = None,
    messages: list[dict[str, Any]],
) -> tuple[str | None, dict[str, object] | None, bool]:
    contract = getattr(agent, "_tournament_intent_contract", None)
    if not isinstance(contract, TournamentIntentContract):
        return candidate, None, False
    candidate_text = candidate or ""
    if contract.state is TournamentIntentState.MIXED_PUBLICATION:
        envelope = parse_mixed_publication_envelope(candidate_text)
        if envelope is None:
            response = (
                "The mixed private and public response could not be safely separated. "
                "No public action was taken. Please retry the request."
            )
            telemetry = contract.telemetry(
                accepted=False,
                code="mixed_envelope_invalid",
                candidate=candidate_text,
                turn_status="failed",
            )
            _finish_tournament_response(agent, contract, messages, response=response, telemetry=telemetry)
            return response, telemetry, True

        private_response, public_candidate = envelope
        private_output = delivery_response or private_response
        if contract.release_state == "consumed":
            response = (
                f"{private_output}\n\n"
                "Publication completed through the exact receipt- and approval-bound action."
            ).strip()
            telemetry = contract.telemetry(
                accepted=True,
                code="release_consumed",
                candidate=public_candidate,
            )
            _finish_tournament_response(agent, contract, messages, response=response, telemetry=telemetry)
            return response, telemetry, False

        if contract.release_state in {"ambiguous", "in_flight"}:
            contract.release_state = "ambiguous"
            if contract.release_approval is not None:
                contract.release_approval.state = "ambiguous"
            response = (
                f"{private_output}\n\n"
                "Publication outcome is ambiguous. The exact approval is quarantined and will not "
                "be replayed until delivery is independently reconciled."
            ).strip()
            telemetry = contract.telemetry(
                accepted=False,
                code="release_outcome_ambiguous",
                candidate=public_candidate,
                turn_status="partial",
            )
            _finish_tournament_response(agent, contract, messages, response=response, telemetry=telemetry)
            return response, telemetry, False

        decision = contract.verify_receipt(public_candidate)
        if decision.allowed:
            response = f"{private_output}\n\nPREPARED_NOT_RELEASED\n\n{public_candidate}".strip()
            telemetry = contract.telemetry(
                accepted=False,
                code="release_approval_required",
                candidate=public_candidate,
                turn_status="partial",
            )
        else:
            response = (
                f"{private_output}\n\n"
                "Public action was not taken; exact verification and release approval are still required."
            ).strip()
            telemetry = contract.telemetry(
                accepted=False,
                code=decision.code,
                candidate=public_candidate,
                turn_status="partial",
            )
        _finish_tournament_response(agent, contract, messages, response=response, telemetry=telemetry)
        return response, telemetry, False

    if contract.state is TournamentIntentState.PUBLICATION_REQUEST:
        if contract.release_state == "consumed":
            response = "Publication completed through the exact receipt- and approval-bound action."
            _replace_current_turn(messages, response)
            telemetry = contract.telemetry(accepted=True, code="release_consumed", candidate=candidate_text)
            delivered = contract.release(response)
            agent._response_was_previewed = delivered
            contract.persist_final_bytes()
            contract.cleanup(agent)
            agent._tournament_intent_contract = None
            return response, telemetry, False
        if contract.release_state in {"ambiguous", "in_flight"}:
            contract.release_state = "ambiguous"
            if contract.release_approval is not None:
                contract.release_approval.state = "ambiguous"
            response = (
                "Publication outcome is ambiguous. The exact approval is quarantined and the action "
                "will not be replayed until delivery is independently reconciled."
            )
            _replace_current_turn(messages, response)
            telemetry = contract.telemetry(
                accepted=False,
                code="release_outcome_ambiguous",
                candidate=candidate_text,
            )
            contract.persist_final_bytes()
            contract.cleanup(agent)
            agent._tournament_intent_contract = None
            return response, telemetry, True
    decision = contract.verify_receipt(candidate_text)
    if decision.allowed:
        if contract.state is TournamentIntentState.PUBLICATION_REQUEST and contract.release_state != "consumed":
            response = f"PREPARED_NOT_RELEASED\n\n{candidate_text}"
            code = "release_approval_required"
            _replace_current_turn(messages, response)
            telemetry = contract.telemetry(accepted=False, code=code, candidate=candidate_text)
            contract.persist_final_bytes()
            contract.cleanup(agent)
            agent._tournament_intent_contract = None
            return response, telemetry, False
        contract.receipt_used = True
        response = delivery_response or candidate_text
        _replace_current_turn(messages, response)
        delivered = contract.release(response)
        agent._response_was_previewed = delivered
        telemetry = contract.telemetry(accepted=True, code="receipt_verified", candidate=candidate_text)
        contract.persist_final_bytes()
        contract.cleanup(agent)
        agent._tournament_intent_contract = None
        return response, telemetry, False
    if (
        contract.state is TournamentIntentState.PUBLICATION_REQUEST
        and decision.code == "receipt_missing_or_consumed"
        and contract.release_approval is None
    ):
        decision = ContractDecision(False, "receipt_and_release_approval_required")
    recovery = _receipt_failure_recovery(decision.code)
    response = (
        "Public tournament copy was not released because source verification failed. "
        f"No external action was taken. Safe recovery: {recovery}."
    )
    _replace_current_turn(messages, response)
    telemetry = contract.telemetry(accepted=False, code=decision.code, candidate=candidate_text)
    contract.persist_final_bytes()
    contract.cleanup(agent)
    agent._tournament_intent_contract = None
    return response, telemetry, True


def _replace_current_turn(messages: list[dict[str, Any]], response: str) -> None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            del messages[index + 1 :]
            messages.append({"role": "assistant", "content": response})
            return
    messages[:] = [{"role": "assistant", "content": response}]


def _candidate_from_args(args: Mapping[str, Any]) -> str:
    for key in ("candidate", "content", "message", "text", "caption"):
        value = args.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _destination_from_args(args: Mapping[str, Any]) -> str:
    for key in ("target", "destination", "chat_id", "channel", "url"):
        value = args.get(key)
        if isinstance(value, (str, int)) and str(value):
            return str(value)
    return ""


def _same_callback(left, right) -> bool:
    return left is right or (
        getattr(left, "__self__", None) is getattr(right, "__self__", None)
        and getattr(left, "__func__", None) is getattr(right, "__func__", None)
    )


def _parse_utc(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)
