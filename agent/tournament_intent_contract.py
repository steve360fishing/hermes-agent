"""Request-local tournament audience, truth, and release authority contract.

Private questions and private artifacts never install this contract. Public
drafts are buffered until the audit-owned truth gate binds an exact receipt;
external publication additionally requires an exact, one-use release approval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib
import json
import logging
from pathlib import Path
import re
import secrets
from typing import Any, Callable, Mapping


logger = logging.getLogger(__name__)

from agent.tournament_truth_support import (
    AUDIT_SCHEMA_VERSION,
    build_artifact_payload,
    canonical_json_sha256,
    configured_runtime_roots,
    contained_path,
    secure_read_contained_text,
)
from agent.tournament_release_state import (
    PendingPublicationPacket,
    ReleaseApprovalIntake,
    ReleaseState,
    TournamentReleaseStore,
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
    RELEASE_APPROVAL_DISCUSSION_OR_GRANT = "release_approval_discussion_or_grant"
    BOUND_RELEASE_APPROVAL_INTAKE = "bound_release_approval_intake"
    RELEASE_REVOCATION_OR_QUESTION = "release_revocation_or_question"
    # Compatibility alias: approval language is never an execution request.
    RELEASE_APPROVAL = "release_approval_discussion_or_grant"


class TournamentResponsePartKind(str, Enum):
    PRIVATE_EXPLANATION = "PRIVATE_EXPLANATION"
    PUBLIC_CANDIDATE = "PUBLIC_CANDIDATE"
    HOLD = "HOLD"


@dataclass(frozen=True)
class TournamentResponsePart:
    kind: TournamentResponsePartKind
    text: str


@dataclass(frozen=True)
class TournamentResponseEnvelope:
    private_explanation: TournamentResponsePart
    public_candidate: TournamentResponsePart


@dataclass(frozen=True)
class TournamentTruthAdvisory:
    """A non-blocking fact-check status attached only to the active turn."""

    code: str
    message: str


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


_PENDING_PUBLICATIONS = TournamentReleaseStore()


@dataclass(frozen=True)
class TournamentTurnExecutionContext:
    """Immutable request-local policy state propagated into tool workers."""

    contract: "TournamentIntentContract | None" = None


_ACTIVE_TOURNAMENT_CONTEXT: ContextVar[TournamentTurnExecutionContext] = (
    ContextVar(
        "active_tournament_execution_context",
        default=TournamentTurnExecutionContext(),
    )
)
_ACTIVE_TOURNAMENT_TRUTH_ADVISORY: ContextVar[TournamentTruthAdvisory | None] = (
    ContextVar("active_tournament_truth_advisory", default=None)
)


class _StableTournamentCallbackMux:
    """One stable callback that buffers only while a tournament turn is active."""

    __slots__ = ("fallback",)

    def __init__(self, fallback: Callable[[str | None], None] | None):
        self.fallback = fallback

    def __call__(self, value: str | None) -> None:
        contract = current_tournament_contract()
        if contract is not None:
            contract.buffer(value)
            return
        if callable(self.fallback):
            self.fallback(value)


class _StableTournamentPersistenceMux:
    """One stable persistence seam that defers only the active public turn."""

    __slots__ = ("fallback",)

    def __init__(self, fallback: Callable[..., None] | None):
        self.fallback = fallback

    def __call__(self, messages, conversation_history=None) -> None:
        contract = current_tournament_contract()
        if contract is not None:
            contract.pending_persistence = (messages, conversation_history)
            return
        if callable(self.fallback):
            self.fallback(messages, conversation_history)


def current_tournament_contract() -> "TournamentIntentContract | None":
    contract = _ACTIVE_TOURNAMENT_CONTEXT.get().contract
    if isinstance(contract, TournamentIntentContract) and not contract.closed:
        return contract
    return None


def bind_tournament_contract(contract: "TournamentIntentContract | None") -> None:
    """Bind request authority to the current execution context, never the agent."""
    _ACTIVE_TOURNAMENT_CONTEXT.set(TournamentTurnExecutionContext(contract=contract))


_CLAIM_BEARING_TOURNAMENT_DRAFT = re.compile(
    r"\b(?:winner|winners|won|result(?:s)?|standing(?:s)?|leaderboard|"
    r"place|placing|score|payout|weigh[- ]?in)\b",
    re.IGNORECASE,
)


def begin_tournament_truth_advisory(
    agent: Any, *, message: object, turn_provenance=None
) -> TournamentTruthAdvisory | None:
    """Mark a direct claim-bearing draft for a non-blocking fact-check note.

    This deliberately does not install a release contract, change tools, read
    sources, or ask the model to use a tool. The current runtime has no
    request-bound trusted snapshot seam, so the only honest automatic outcome
    is an unavailable advisory while preserving the model's useful draft.
    """
    _ACTIVE_TOURNAMENT_TRUTH_ADVISORY.set(None)
    from agent.turn_origin import coerce_turn_provenance

    provenance = coerce_turn_provenance(turn_provenance)
    if not provenance.is_authenticated_direct_user or not _matches_current_request_binding(
        agent, provenance
    ):
        return None
    directed = _authority_directed_text(provenance.authority_text)
    if not (
        _TOURNAMENT_CUE.search(directed)
        and _has_explicit_public_draft_operation(directed)
        and _CLAIM_BEARING_TOURNAMENT_DRAFT.search(directed)
    ):
        return None
    advisory = TournamentTruthAdvisory(
        code="trusted_snapshot_unavailable",
        message=(
            "Fact check: verification unavailable — no trusted tournament snapshot "
            "was bound to this request."
        ),
    )
    _ACTIVE_TOURNAMENT_TRUTH_ADVISORY.set(advisory)
    return advisory


def clear_tournament_truth_advisory() -> None:
    _ACTIVE_TOURNAMENT_TRUTH_ADVISORY.set(None)


def append_tournament_truth_advisory(response: str | None) -> str | None:
    """Keep useful output intact while attaching a concise current-turn note."""
    advisory = _ACTIVE_TOURNAMENT_TRUTH_ADVISORY.get()
    if advisory is None or not isinstance(response, str) or not response.strip():
        return response
    return f"{response.rstrip()}\n\n{advisory.message}"


def _install_stable_runtime_muxes(
    agent: Any,
) -> tuple[Callable[..., None] | None, Callable[..., None] | None]:
    callback = getattr(agent, "stream_delta_callback", None)
    if not isinstance(callback, _StableTournamentCallbackMux):
        callback = _StableTournamentCallbackMux(callback if callable(callback) else None)
        agent.stream_delta_callback = callback

    persistence = getattr(agent, "_persist_session", None)
    if not isinstance(persistence, _StableTournamentPersistenceMux):
        persistence = _StableTournamentPersistenceMux(
            persistence if callable(persistence) else None
        )
        agent._persist_session = persistence
    return callback.fallback, persistence.fallback


def _matches_current_request_binding(agent: Any, provenance: Any) -> bool:
    """Bind a sealed ingress envelope to this agent's immutable route identity."""
    raw_platform = getattr(agent, "platform", "")
    platform = str(getattr(raw_platform, "value", raw_platform) or "").strip()
    chat_id = str(getattr(agent, "_chat_id", "") or "")
    thread_id = str(getattr(agent, "_thread_id", "") or "")
    if not platform or not chat_id:
        return False
    if provenance.platform != platform or provenance.chat_id != chat_id:
        return False
    if provenance.thread_id != thread_id:
        return False

    gateway_session_key = str(
        getattr(agent, "_gateway_session_key", "") or ""
    )
    if not gateway_session_key:
        return False
    namespace = (
        "agent:main"
        if provenance.profile in ("", "default")
        else f"agent:{provenance.profile}"
    )
    if not gateway_session_key.startswith(f"{namespace}:{platform}:"):
        return False

    def _has_session_component(value: str) -> bool:
        return bool(value) and (
            f":{value}:" in gateway_session_key
            or gateway_session_key.endswith(f":{value}")
        )

    if not _has_session_component(chat_id):
        return False
    if thread_id and not _has_session_component(thread_id):
        return False

    return provenance.matches_bound_request(
        platform=platform,
        profile=provenance.profile,
        chat_id=chat_id,
        thread_id=thread_id,
        message_id=provenance.message_id,
        event_id=provenance.event_id,
        session_scope=provenance.session_scope,
        authority_text=provenance.authority_text,
    )

_TOURNAMENT_CUE = re.compile(
    r"\b(?:tournaments?|catchstat|reel\s*time|leaderboards?|standings?|"
    r"weigh[- ]?ins?|calcutta|release\s+points?|marlin|sailfish|game\s*fish)\b",
    re.IGNORECASE,
)
_TOURNAMENT_CONTINUATION_CUE = re.compile(
    r"\b(?:that|the)\s+(?:exact\s+)?approved\s+(?:instagram\s+)?(?:stor(?:y|ies)|post|caption|newsletter|copy)\b",
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
_PUBLIC_OUTPUT_ARTIFACT = (
    r"(?:stor(?:y|ies)|caption|post|carousel|announcement|(?:audit\s+)?report|"
    r"copy|page|press\s+release|[a-z0-9_-]+\.txt)"
)
_PUBLIC_PLATFORM_OUTPUT_ARTIFACT = (
    r"(?:stor(?:y|ies)|caption|post|carousel|announcement|copy|page|press\s+release)"
)
_PUBLIC_DESTINATION = r"(?:instagram|facebook|newsletter|website|cms)"
_EXPLICIT_PUBLIC_OUTPUT = re.compile(
    # A public output must bind an artifact to an actual destination. Bare
    # "public" and destination-first labels can describe private input data,
    # so neither one is authority without the artifact-to-destination relation.
    rf"\b{_PUBLIC_DESTINATION}\s+{_PUBLIC_PLATFORM_OUTPUT_ARTIFACT}\b|"
    rf"\b{_PUBLIC_OUTPUT_ARTIFACT}\b(?:\s+\w+){{0,5}}\s+"
    rf"(?:for|to|on|in|via)\s+(?:the\s+)?(?:public\s+)?{_PUBLIC_DESTINATION}\b",
    re.IGNORECASE,
)
# In Steve's Telegram workflow, capitalized "Story" is the established shorthand
# for an Instagram Story. Lowercase narrative "story" remains ordinary content.
_IMPLICIT_INSTAGRAM_STORY_OUTPUT = re.compile(r"\bStor(?:y|ies)\b")
_DRAFT_ACTION = re.compile(r"\b(?:create|write|draft|make|prepare|generate)\b", re.IGNORECASE)
_PUBLICATION_ACTION = re.compile(
    r"\b(?:publish|send|upload|deploy|release)\b|"
    r"\bpost(?=\s+(?:that\s+exact|this|the|it|now)\b|[.!?,;]|$)",
    re.IGNORECASE,
)
_NEGATION = re.compile(r"\b(?:do\s+not|don't|without|never)\b", re.IGNORECASE)
_NEGATION_RESET = re.compile(
    r"\b(?:but|instead|then|however)\b|,\s*please\b|"
    r"[-\u2014]\s*(?:please\s+)?(?:create|make|publish|post|send|upload|release)\b",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY = re.compile(r"[!?;\n]|\.(?=\s|$)")
_RECEIPT_OPERATION = re.compile(
    r"\b(?:validate|verify|check)\b.{0,48}\b(?:truth\s+)?receipt\b|"
    r"\breceipt\s+validation\b",
    re.IGNORECASE,
)
_APPROVAL_OPERATION = re.compile(
    r"\b(?:record|grant|approve|validate|have|give|supply)\b.{0,64}\b(?:release|publication)\s+approval\b",
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
# A plan/runbook pasted into a chat can contain imperative verbs, but it is
# data about a future action rather than an instruction to take one now.  Keep
# this structural boundary separate from lexical intent classification.
_META_DOCUMENT_START = re.compile(
    r"(?im)^\s*(?:#{1,6}\s+.*\b(?:plan|runbook|implementation|execution)\b|"
    r"(?:plan_ready|execution\s+guardrails|declared\s+(?:checks|skills))\b|"
    r"this\s+is\s+what\s+(?:codex|hermes)\s+is\s+going\s+to\s+do\s*:|"
    r"this\s+is\s+(?:a\s+)?copy\s+of\s+(?:"
    r"my\s+previous\s+message|"
    r"the\s+message\s+i\s+sent\s+you\s+(?:last\s+night|yesterday|earlier)"
    r")\s*:?)",
)
_EXPLICIT_BOUND_PUBLIC_SURFACE = re.compile(
    r"\b(?:instagram|facebook|newsletter|website|cms)\b.*\b(?:account|caption|story|post|page)\b|"
    r"\b(?:caption|story|post|page)\b.*\b(?:instagram|facebook|newsletter|website|cms)\b",
    re.IGNORECASE,
)
_BOUND_APPROVAL_INTAKE = re.compile(
    r"^APPROVE_TOURNAMENT_RELEASE\s+action_id=(?P<action>[a-f0-9]{32})\s+checksum=(?P<checksum>[a-f0-9]{64})$",
    re.IGNORECASE,
)
_REVOCATION_OR_QUESTION = re.compile(
    r"\b(?:revoke|withdraw|cancel|question|why|whether|explain|does|is|can)\b.{0,64}\b(?:release|publication)\s+approval\b",
    re.IGNORECASE,
)
_INTERNAL_PUBLIC_TOOLS = frozenset(
    {
        "memory",
        "tournament_source_capture",
        "tournament_truth_gate",
    }
)


def _has_affirmative_action(message: str, pattern: re.Pattern[str]) -> bool:
    for match in pattern.finditer(message):
        clause_start, _ = _clause_bounds(message, match.start())
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


def _authority_directed_text(message: object) -> str:
    """Return only speaker-attributable instructions, never a pasted plan body."""
    if not isinstance(message, str):
        return ""
    directed = _mask_non_authoritative_data(message)
    marker = _META_DOCUMENT_START.search(directed)
    return directed[: marker.start()] if marker is not None else directed


def _has_explicit_public_draft_operation(message: str) -> bool:
    """Require an affirmative draft action and its public output in one clause."""
    for action in _DRAFT_ACTION.finditer(message):
        clause_start, clause_end = _clause_bounds(message, action.start())
        clause = message[clause_start:clause_end]
        if (
            _has_affirmative_action(clause, _DRAFT_ACTION)
            and (
                _EXPLICIT_PUBLIC_OUTPUT.search(clause)
                or _IMPLICIT_INSTAGRAM_STORY_OUTPUT.search(clause)
            )
        ):
            return True
    return False


def _clause_bounds(message: str, offset: int) -> tuple[int, int]:
    """Return sentence-like bounds without splitting a filename extension."""
    boundaries = [match.start() for match in _CLAUSE_BOUNDARY.finditer(message)]
    start = max((position for position in boundaries if position < offset), default=-1) + 1
    end = min((position for position in boundaries if position >= offset), default=len(message))
    return start, end


def _installs_tournament_contract(
    state: TournamentIntentState | None,
    message: object,
    *,
    trusted_publication_context: bool,
) -> bool:
    """Authority installation is stricter than vocabulary classification."""
    directed = _authority_directed_text(message)
    if not directed.strip():
        return False
    if state is TournamentIntentState.PUBLIC_FACING_DRAFT:
        return _has_explicit_public_draft_operation(directed)
    if state in {
        TournamentIntentState.PUBLICATION_REQUEST,
        TournamentIntentState.MIXED_PUBLICATION,
    }:
        publication_text = re.sub(
            r"\b(?:release|publication)[\s-]+approval(?:[\s-]+blocker)?\b",
            " ",
            directed,
            flags=re.IGNORECASE,
        )
        publication_text = re.sub(
            r"\bpress\s+release\b", " ", publication_text, flags=re.IGNORECASE
        )
        has_target = bool(_EXPLICIT_BOUND_PUBLIC_SURFACE.search(directed)) or bool(
            trusted_publication_context
            and _TOURNAMENT_CONTINUATION_CUE.search(directed)
        )
        return has_target and _has_affirmative_action(publication_text, _PUBLICATION_ACTION)
    return False


def _classify_tournament_intents(
    message: object, *, trusted_publication_context: bool = False
) -> frozenset[TournamentIntentState]:
    """Return every directed intent present after removing data-only vocabulary."""
    if not isinstance(message, str):
        return frozenset()
    directed = _mask_non_authoritative_data(message)
    raw_has_tournament_cue = bool(
        _TOURNAMENT_CUE.search(message)
        or (trusted_publication_context and _TOURNAMENT_CONTINUATION_CUE.search(message))
        or _APPROVAL_OPERATION.search(message) or _REVOCATION_OR_QUESTION.search(message)
    )
    if not raw_has_tournament_cue and not trusted_publication_context:
        return (
            frozenset({TournamentIntentState.PRIVATE_INQUIRY})
            if _EXPLICIT_PUBLIC_OUTPUT.search(directed)
            else frozenset()
        )
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
    if _REVOCATION_OR_QUESTION.search(directed):
        states.add(TournamentIntentState.RELEASE_REVOCATION_OR_QUESTION)
    elif _APPROVAL_OPERATION.search(directed) and _has_affirmative_action(directed, _APPROVAL_OPERATION):
        states.add(TournamentIntentState.RELEASE_APPROVAL_DISCUSSION_OR_GRANT)
    # Approval is an authority noun, never a publication instruction itself.
    public_action_text = re.sub(r"\b(?:release|publication)[\s-]+approval(?:[\s-]+blocker)?\b", " ", directed, flags=re.IGNORECASE)
    public_action_text = re.sub(
        r"\bpress\s+release\b", " ", public_action_text, flags=re.IGNORECASE
    )
    if _has_affirmative_action(public_action_text, _PUBLICATION_ACTION):
        states.add(TournamentIntentState.PUBLICATION_REQUEST)
    if _has_explicit_public_draft_operation(directed):
        states.add(TournamentIntentState.PUBLIC_FACING_DRAFT)
    return frozenset(states or {TournamentIntentState.PRIVATE_INQUIRY})


def classify_tournament_intent(
    message: object, *, trusted_publication_context: bool = False
) -> TournamentIntentState | None:
    """Classify the controlling state without discarding concurrent intents."""
    states = _classify_tournament_intents(
        message, trusted_publication_context=trusted_publication_context
    )
    if (
        TournamentIntentState.PUBLICATION_REQUEST in states
        and states.intersection(
            {
                TournamentIntentState.PRIVATE_ARTIFACT,
                TournamentIntentState.PRIVATE_INQUIRY,
            }
        )
    ):
        # A private handoff that discusses approval vocabulary is not mixed;
        # only a separate affirmative publication clause may create mixed work.
        masked = _mask_non_authoritative_data(message) if isinstance(message, str) else ""
        masked = re.sub(r"\b(?:release|publication)[\s-]+approval(?:[\s-]+blocker)?\b", " ", masked, flags=re.IGNORECASE)
        if not _has_affirmative_action(masked, _PUBLICATION_ACTION):
            return TournamentIntentState.PRIVATE_ARTIFACT
        return TournamentIntentState.MIXED_PUBLICATION
    for state in (
        TournamentIntentState.PUBLICATION_REQUEST,
        TournamentIntentState.PUBLIC_FACING_DRAFT,
        TournamentIntentState.RECEIPT_VALIDATION,
        TournamentIntentState.BOUND_RELEASE_APPROVAL_INTAKE,
        TournamentIntentState.RELEASE_APPROVAL_DISCUSSION_OR_GRANT,
        TournamentIntentState.RELEASE_REVOCATION_OR_QUESTION,
        TournamentIntentState.PRIVATE_ARTIFACT,
        TournamentIntentState.PRIVATE_INQUIRY,
    ):
        if state in states:
            return state
    return None


def platform_bypasses_tournament_contract(platform: object) -> bool:
    return str(getattr(platform, "value", platform) or "").strip().casefold() == "cron"


def classify_bound_release_approval_intake(
    message: object, *, session_id: str, authenticated_identity: str, turn_provenance=None
) -> tuple[TournamentIntentState | None, PendingPublicationPacket | None]:
    """Resolve an authenticated exact packet response without granting execution."""
    if not isinstance(message, str):
        return None, None
    match = _BOUND_APPROVAL_INTAKE.fullmatch(message.strip())
    if match is None:
        return None, None
    packet = _PENDING_PUBLICATIONS.current_action(
        match.group("action"), session_id, provenance=turn_provenance
    )
    if (
        packet is None
        or packet.actor_identity != str(authenticated_identity)
        or packet.state is not ReleaseState.PREPARED
        or packet.checksum() != match.group("checksum").lower()
    ):
        return None, None
    return TournamentIntentState.BOUND_RELEASE_APPROVAL_INTAKE, packet


@dataclass
class TournamentIntentContract:
    state: TournamentIntentState
    task_id: str
    session_id: str
    destination: str
    entrypoint: str
    actor_identity: str
    turn_provenance: Any = None
    intents: frozenset[TournamentIntentState] = field(default_factory=frozenset)
    policy_version: str = POLICY_VERSION
    nonce: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    # Immutable owner identity: finalization and cleanup may affect only this
    # exact request, never whichever contract last touched the shared agent.
    turn_token: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    preflight_error: str = ""
    callbacks: list[Callable[[str | None], None]] = field(default_factory=list)
    original_stream_delta_callback: Callable[[str | None], None] | None = None
    original_stream_callback: Callable[[str | None], None] | None = None
    original_persist_session: Callable[..., None] | None = None
    pending_persistence: tuple[list[dict[str, Any]], list[dict[str, Any]] | None] | None = None
    buffer_callback: Callable[[str | None], None] | None = None
    added_tool_schema: bool = False
    added_valid_tool_name: bool = False
    added_tool_schemas: set[str] = field(default_factory=set)
    added_valid_tool_names: set[str] = field(default_factory=set)
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
    pending_publication: PendingPublicationPacket | None = None
    external_publication_sink: str = ""
    delivery_callback_failed: bool = False
    closed: bool = False

    def __setattr__(self, name: str, value: object) -> None:
        if name == "turn_token" and "turn_token" in self.__dict__:
            raise AttributeError("turn_token is immutable")
        super().__setattr__(name, value)

    @staticmethod
    def candidate_sha256(candidate: str) -> str:
        return hashlib.sha256(candidate.encode("utf-8")).hexdigest()

    def owns_turn(self, other: object) -> bool:
        own_token = getattr(self, "turn_token", None)
        other_token = getattr(other, "turn_token", None)
        return bool(
            isinstance(other, TournamentIntentContract)
            and isinstance(own_token, str)
            and isinstance(other_token, str)
            and own_token
            and other_token
            and secrets.compare_digest(own_token, other_token)
        )

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

    def attach_release_approval(
        self, approval: TournamentReleaseApproval, *, packet: PendingPublicationPacket
    ) -> bool:
        stored = _PENDING_PUBLICATIONS.current_action(
            packet.pending_action_id, self.session_id, provenance=self.turn_provenance
        )
        if (
            self.closed
            or stored is not packet
            or packet.state not in {ReleaseState.APPROVED, ReleaseState.FAILED_PRE_DISPATCH}
            or (
                packet.state is ReleaseState.FAILED_PRE_DISPATCH
                and not packet.retryable_pre_dispatch
            )
            or packet.action_tool != "send_message"
            or approval.state != "available"
            or approval.expires_at <= datetime.now(timezone.utc)
            or approval.destination != packet.external_publication_sink
            or approval.candidate_sha256 != packet.candidate_sha256
            or approval.identity != packet.actor_identity
            or approval.idempotency_key != packet.idempotency_key
        ):
            return False
        self.pending_publication = packet
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
        if tool_name in _INTERNAL_PUBLIC_TOOLS:
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
        return (
            tool_name in (_READ_ONLY_PUBLIC_TOOLS | _INTERNAL_PUBLIC_TOOLS)
            or (
            tool_name == "send_message" and str(args.get("action") or "send") == "list"
            )
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
            return ContractDecision(False, "release_already_in_flight")
        receipt = self.verify_receipt(candidate)
        if not receipt.allowed:
            if receipt.code == "receipt_missing_or_consumed" and self.release_approval is None:
                return ContractDecision(False, "receipt_and_release_approval_required")
            return receipt
        approval = self.release_approval
        if approval is None:
            packet = _PENDING_PUBLICATIONS.approved_for(
                session_id=self.session_id,
                destination=destination,
                candidate_sha256=self.candidate_sha256(candidate),
                identity=identity,
                idempotency_key=idempotency_key,
                provenance=self.turn_provenance,
            )
            if packet is not None:
                self.attach_release_approval(
                    TournamentReleaseApproval(
                        destination=packet.destination,
                        candidate_sha256=packet.candidate_sha256,
                        identity=packet.actor_identity,
                        expires_at=packet.expires_at,
                        idempotency_key=packet.idempotency_key,
                    ),
                    packet=packet,
                )
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
        pending_state = (
            self.pending_publication.state if self.pending_publication is not None else None
        )
        if (
            self.pending_publication is None
            or pending_state not in {ReleaseState.APPROVED, ReleaseState.FAILED_PRE_DISPATCH}
            or not _PENDING_PUBLICATIONS.transition(
                self.pending_publication,
                expected=pending_state,
                target=ReleaseState.IN_FLIGHT,
                provenance=self.turn_provenance,
            )
        ):
            approval.state = "available"
            self.release_state = "prepared_not_released"
            self.release_binding = None
            return ContractDecision(False, "release_state_transition_failed")
        return ContractDecision(True, "release_authorized")

    def record_external_result(self, *, success: bool, ambiguous: bool) -> None:
        approval = self.release_approval
        if approval is None or self.release_state != "in_flight":
            return
        if ambiguous:
            approval.state = "ambiguous"
            self.release_state = "ambiguous"
            if self.pending_publication is not None:
                _PENDING_PUBLICATIONS.transition(
                    self.pending_publication, expected=ReleaseState.IN_FLIGHT, target=ReleaseState.AMBIGUOUS,
                    provenance=self.turn_provenance,
                )
            return
        if success:
            approval.state = "consumed"
            self.release_state = "consumed"
            self.receipt_used = True
            if self.pending_publication is not None:
                _PENDING_PUBLICATIONS.transition(
                    self.pending_publication, expected=ReleaseState.IN_FLIGHT, target=ReleaseState.CONSUMED,
                    provenance=self.turn_provenance,
                )
            return
        approval.state = "available"
        self.release_state = "prepared_not_released"
        self.release_binding = None
        if self.pending_publication is not None:
            _PENDING_PUBLICATIONS.transition(
                self.pending_publication, expected=ReleaseState.IN_FLIGHT, target=ReleaseState.FAILED_PRE_DISPATCH,
                provenance=self.turn_provenance,
            )

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

    def cleanup(self, agent: Any) -> bool:
        """Close only this request's authority; stable runtime seams stay installed."""
        if self.closed:
            return True
        active = current_tournament_contract()
        if isinstance(active, TournamentIntentContract) and not self.owns_turn(active):
            return False
        owns_active = self.owns_turn(active)
        self.closed = True
        if owns_active:
            bind_tournament_contract(None)
        return True


def begin_tournament_intent_contract(
    agent: Any,
    *,
    message: object,
    task_id: str,
    stream_callback=None,
    turn_provenance=None,
) -> TournamentIntentContract | None:
    raw_platform = getattr(agent, "platform", "")
    if platform_bypasses_tournament_contract(raw_platform):
        return None
    from agent.turn_origin import coerce_turn_provenance

    provenance = coerce_turn_provenance(turn_provenance)
    if not provenance.is_authenticated_direct_user or not _matches_current_request_binding(
        agent, provenance
    ):
        return None
    # Only the adapter-sealed original command is authority-bearing.  The
    # caller's effective/persist text may contain plugin, replay, or model data.
    authority_message = provenance.authority_text
    authenticated_identity = provenance.actor_identity
    intake_state, packet = classify_bound_release_approval_intake(
        authority_message,
        session_id=str(getattr(agent, "session_id", "") or ""),
        authenticated_identity=authenticated_identity,
        turn_provenance=provenance,
    )
    if intake_state is TournamentIntentState.BOUND_RELEASE_APPROVAL_INTAKE and packet is not None:
        intake_authenticated_tournament_release_approval(
            task_id=task_id, session_id=packet.session_id, destination=packet.destination,
            candidate_sha256=packet.candidate_sha256,
            authenticated_identity=authenticated_identity,
            idempotency_key=packet.idempotency_key, pending_action_id=packet.pending_action_id,
            packet_checksum=packet.checksum(),
            turn_provenance=provenance,
        )
        return None
    platform = str(raw_platform or "").strip().casefold()
    session_id = str(getattr(agent, "session_id", "") or "")
    pending_context = _PENDING_PUBLICATIONS.current_for_session(
        session_id, provenance=provenance
    )
    trusted_context = pending_context is not None
    intents = _classify_tournament_intents(
        authority_message, trusted_publication_context=trusted_context
    )
    state = classify_tournament_intent(
        authority_message, trusted_publication_context=trusted_context
    )
    if state is TournamentIntentState.RELEASE_REVOCATION_OR_QUESTION:
        directed = _mask_non_authoritative_data(authority_message)
        if re.search(r"\b(?:revoke|withdraw|cancel)\b", directed, re.IGNORECASE):
            _PENDING_PUBLICATIONS.revoke_session(
                session_id=session_id,
                authenticated_identity=authenticated_identity,
                provenance=provenance,
            )
        return None
    if not _installs_tournament_contract(
        state, authority_message, trusted_publication_context=trusted_context
    ):
        return None
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
        actor_identity=authenticated_identity,
        turn_provenance=provenance,
        intents=intents,
        external_publication_sink=_publication_sink_from_message(authority_message),
    )
    contract.pending_publication = pending_context
    try:
        from tools.registry import registry
        from tools.tournament_truth_gate_tool import TOURNAMENT_TRUTH_GATE_SCHEMA
        from tools.tournament_source_capture_tool import TOURNAMENT_SOURCE_CAPTURE_SCHEMA

        if registry.get_entry("tournament_truth_gate") is None:
            contract.preflight_error = "truth_gate_unavailable"
    except Exception:
        contract.preflight_error = "truth_gate_unavailable"
        TOURNAMENT_TRUTH_GATE_SCHEMA = TOURNAMENT_SOURCE_CAPTURE_SCHEMA = None
    tools = getattr(agent, "tools", None)
    valid_names = getattr(agent, "valid_tool_names", None)
    if not isinstance(tools, list) or not isinstance(valid_names, (set, list)):
        contract.preflight_error = contract.preflight_error or "truth_gate_not_in_model_request"
    elif not contract.preflight_error:
        for schema in (TOURNAMENT_TRUTH_GATE_SCHEMA, TOURNAMENT_SOURCE_CAPTURE_SCHEMA):
            name = schema.get("name") if isinstance(schema, Mapping) else ""
            if name and not any(tool.get("function", {}).get("name") == name for tool in tools):
                tools.append({"type": "function", "function": schema})
        if isinstance(valid_names, set):
            for name in ("tournament_truth_gate", "tournament_source_capture"):
                if name not in valid_names:
                    valid_names.add(name)
        else:
            for name in ("tournament_truth_gate", "tournament_source_capture"):
                if name not in valid_names:
                    valid_names.append(name)
        if not any(
            tool.get("function", {}).get("name") == "tournament_truth_gate"
            for tool in tools
        ) or "tournament_truth_gate" not in valid_names:
            contract.preflight_error = "truth_gate_not_in_model_request"
    contract.buffer_callback = contract.buffer
    base_stream_delta, base_persistence = _install_stable_runtime_muxes(agent)
    contract.original_stream_delta_callback = base_stream_delta
    contract.original_stream_callback = getattr(agent, "_stream_callback", None)
    contract.original_persist_session = base_persistence
    for callback in (
        contract.original_stream_delta_callback,
        contract.original_stream_callback,
        stream_callback,
    ):
        if callable(callback) and not any(_same_callback(callback, current) for current in contract.callbacks):
            contract.callbacks.append(callback)
    bind_tournament_contract(contract)
    return contract


def active_contract(task_id: str, session_id: str) -> TournamentIntentContract | None:
    contract = current_tournament_contract()
    if (
        contract is not None
        and contract.task_id == str(task_id)
        and contract.session_id == str(session_id)
    ):
        return contract
    return None


def prepare_tournament_publication(
    contract: TournamentIntentContract,
    *,
    candidate: str,
    destination: str,
    idempotency_key: str,
) -> PendingPublicationPacket:
    """Freeze the exact, receipt-bound external action for separate approval.

    This creates no delivery side effect and deliberately cannot accept prose as
    approval.  The gateway must later supply authenticated identity and every
    bound field to :func:`intake_authenticated_tournament_release_approval`.
    """
    if contract.closed or contract.state not in {
        TournamentIntentState.PUBLICATION_REQUEST,
        TournamentIntentState.MIXED_PUBLICATION,
    }:
        raise ValueError("publication contract is not active")
    receipt = contract.verify_receipt(candidate)
    if not receipt.allowed:
        raise ValueError(f"cannot prepare publication without exact receipt: {receipt.code}")
    if not destination or destination == contract.destination:
        raise ValueError("external publication sink must be explicit and distinct from private delivery")
    if not contract.test_receipt:
        try:
            from tools.tournament_truth_gate_tool import validate_tournament_publication_sink

            accepted, code = validate_tournament_publication_sink(
                contract, candidate, destination
            )
        except Exception as exc:
            raise ValueError("publication sink truth validator unavailable") from exc
        if not accepted:
            raise ValueError(f"publication sink truth revalidation failed: {code}")
    packet = PendingPublicationPacket(
        task_id=contract.task_id,
        session_id=contract.session_id,
        destination=destination,
        candidate_sha256=contract.candidate_sha256(candidate),
        actor_identity=contract.actor_identity,
        idempotency_key=idempotency_key,
        expires_at=contract.receipt_expires_at or datetime.now(timezone.utc),
        action_tool="send_message",
        private_delivery_surface=contract.destination,
        external_publication_sink=destination,
    )
    contract.pending_publication = _PENDING_PUBLICATIONS.prepare(
        packet, provenance=contract.turn_provenance
    )
    return contract.pending_publication


def intake_authenticated_tournament_release_approval(
    *,
    task_id: str,
    session_id: str,
    destination: str,
    candidate_sha256: str,
    authenticated_identity: str,
    idempotency_key: str,
    pending_action_id: str,
    packet_checksum: str,
    turn_provenance=None,
) -> ContractDecision:
    """Record a gateway-authenticated exact approval without dispatching it."""
    result = _PENDING_PUBLICATIONS.approve_current(
        ReleaseApprovalIntake(
            task_id=task_id,
            session_id=session_id,
            destination=destination,
            candidate_sha256=candidate_sha256,
            authenticated_identity=authenticated_identity,
            idempotency_key=idempotency_key,
            pending_action_id=pending_action_id,
            packet_checksum=packet_checksum,
        ),
        provenance=turn_provenance,
    )
    if not result.accepted or result.packet is None:
        return ContractDecision(False, result.code)
    contract = active_contract(task_id, session_id)
    if contract is None or contract.closed:
        return ContractDecision(True, "release_approval_recorded")
    contract.pending_publication = result.packet
    attached = contract.attach_release_approval(
        TournamentReleaseApproval(
            destination=result.packet.destination,
            candidate_sha256=result.packet.candidate_sha256,
            identity=result.packet.actor_identity,
            expires_at=result.packet.expires_at,
            idempotency_key=result.packet.idempotency_key,
        ),
        packet=result.packet,
    )
    return ContractDecision(attached, "release_approval_recorded" if attached else "release_approval_attach_failed")


def clear_tournament_intent_contract(
    agent: Any, *, expected_contract: TournamentIntentContract | None = None
) -> bool:
    contract = current_tournament_contract()
    if expected_contract is not None and not expected_contract.owns_turn(contract):
        return False
    if not isinstance(contract, TournamentIntentContract):
        bind_tournament_contract(None)
        return True
    return bool(contract.cleanup(agent))


def effective_request_system_prompt(agent: Any, base_prompt: str) -> str:
    contract = current_tournament_contract()
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


def parse_mixed_publication_envelope(
    candidate: str,
) -> TournamentResponseEnvelope | None:
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
    return TournamentResponseEnvelope(
        private_explanation=TournamentResponsePart(
            TournamentResponsePartKind.PRIVATE_EXPLANATION,
            private_response,
        ),
        public_candidate=TournamentResponsePart(
            TournamentResponsePartKind.PUBLIC_CANDIDATE,
            public_candidate,
        ),
    )


def _render_response_parts(*parts: TournamentResponsePart) -> str:
    return "\n\n".join(part.text for part in parts if part.text).strip()


def _typed_private_parts(
    messages: list[dict[str, Any]],
) -> tuple[TournamentResponsePart, ...]:
    turn_start = 0
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            turn_start = index + 1
            break
    parts: list[TournamentResponsePart] = []
    for message in messages[turn_start:]:
        kind = message.get("tournament_response_part") or message.get(
            "response_part_kind"
        )
        content = message.get("content")
        if (
            kind == TournamentResponsePartKind.PRIVATE_EXPLANATION.value
            and isinstance(content, str)
            and content
        ):
            parts.append(
                TournamentResponsePart(
                    TournamentResponsePartKind.PRIVATE_EXPLANATION, content
                )
            )
    return tuple(parts)


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


def abort_tournament_output(
    agent: Any,
    *,
    candidate: str | None,
    messages: list[dict[str, Any]],
    code: str,
    response: str,
    contract: TournamentIntentContract | None = None,
) -> tuple[str, dict[str, object] | None, bool]:
    contract = contract or current_tournament_contract()
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
    contract: TournamentIntentContract | None = None,
) -> tuple[str | None, dict[str, object] | None, bool]:
    contract = contract or current_tournament_contract()
    if not isinstance(contract, TournamentIntentContract):
        return append_tournament_truth_advisory(candidate), None, False
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

        private_response = envelope.private_explanation.text
        public_candidate = envelope.public_candidate.text
        private_part = TournamentResponsePart(
            TournamentResponsePartKind.PRIVATE_EXPLANATION,
            delivery_response or private_response,
        )
        private_output = private_part.text
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
            response = _render_response_parts(
                private_part,
                TournamentResponsePart(
                    TournamentResponsePartKind.HOLD, "PREPARED_NOT_RELEASED"
                ),
                envelope.public_candidate,
            )
            telemetry = contract.telemetry(
                accepted=False,
                code="release_approval_required",
                candidate=public_candidate,
                turn_status="partial",
            )
        else:
            response = _render_response_parts(
                private_part,
                TournamentResponsePart(
                    TournamentResponsePartKind.HOLD,
                    "Public action was not taken; exact verification and release approval are still required.",
                ),
            )
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
            clear_tournament_intent_contract(agent, expected_contract=contract)
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
            clear_tournament_intent_contract(agent, expected_contract=contract)
            return response, telemetry, True
    decision = contract.verify_receipt(candidate_text)
    if decision.allowed:
        if contract.state is TournamentIntentState.PUBLICATION_REQUEST and contract.release_state != "consumed":
            try:
                packet = contract.pending_publication or prepare_tournament_publication(
                    contract,
                    candidate=candidate_text,
                    destination=contract.external_publication_sink,
                    idempotency_key=f"tournament-{contract.nonce}",
                )
                response = (
                    "PREPARED_NOT_RELEASED\n\n"
                    f"{candidate_text}\n\n"
                    f"Pending action: {packet.pending_action_id}\n"
                    f"Checksum: {packet.checksum()}\n"
                    f"Destination: {packet.external_publication_sink}\n"
                    "Approve only this exact unexpired packet through the authenticated approval response."
                )
            except ValueError as exc:
                response = (
                    "PUBLICATION_PREPARATION_HOLD\n\n"
                    f"Code: publication_packet_not_prepared\nSafe recovery: {exc}"
                )
            code = "release_approval_required"
            _replace_current_turn(messages, response)
            telemetry = contract.telemetry(accepted=False, code=code, candidate=candidate_text)
            contract.persist_final_bytes()
            contract.cleanup(agent)
            clear_tournament_intent_contract(agent, expected_contract=contract)
            return response, telemetry, False
        if (
            delivery_response is not None
            and not delivery_response.startswith("MEDIA:")
            and contract.candidate_sha256(delivery_response) != contract.candidate_sha256(candidate_text)
        ):
            if contract.state is TournamentIntentState.PUBLIC_FACING_DRAFT:
                response = (
                    "DRAFT_VALIDATION_HOLD\n\n"
                    "Code: candidate_bytes_mismatch\n"
                    "Only the changed claim-bearing draft was withheld. Safe recovery: "
                    "freeze and verify the final delivery bytes before private delivery."
                )
            else:
                response = (
                    "Publication was not attempted because final candidate bytes changed "
                    "after verification. Code: candidate_bytes_mismatch."
                )
            _replace_current_turn(messages, response)
            telemetry = contract.telemetry(accepted=False, code="candidate_bytes_mismatch", candidate=candidate_text)
            contract.persist_final_bytes()
            contract.cleanup(agent)
            clear_tournament_intent_contract(agent, expected_contract=contract)
            return response, telemetry, True
        contract.receipt_used = True
        response = delivery_response or candidate_text
        _replace_current_turn(messages, response)
        delivered = contract.release(response)
        agent._response_was_previewed = delivered
        telemetry = contract.telemetry(accepted=True, code="receipt_verified", candidate=candidate_text)
        contract.persist_final_bytes()
        contract.cleanup(agent)
        clear_tournament_intent_contract(agent, expected_contract=contract)
        return response, telemetry, False
    if (
        contract.state is TournamentIntentState.PUBLICATION_REQUEST
        and decision.code == "receipt_missing_or_consumed"
        and contract.release_approval is None
    ):
        decision = ContractDecision(False, "receipt_and_release_approval_required")
    recovery = _receipt_failure_recovery(decision.code)
    if contract.state is TournamentIntentState.PUBLIC_FACING_DRAFT:
        hold_part = TournamentResponsePart(
            TournamentResponsePartKind.HOLD,
            (
                "DRAFT_VALIDATION_HOLD\n\n"
                f"Code: {decision.code}\n"
                "Only the unsupported claim-bearing draft was withheld. "
                f"Safe recovery: {recovery}."
            ),
        )
        response = _render_response_parts(*_typed_private_parts(messages), hold_part)
    else:
        response = (
            "Publication was not attempted because its required truth authority is invalid. "
            f"Code: {decision.code}. Safe recovery: {recovery}."
        )
    _replace_current_turn(messages, response)
    telemetry = contract.telemetry(accepted=False, code=decision.code, candidate=candidate_text)
    contract.persist_final_bytes()
    contract.cleanup(agent)
    clear_tournament_intent_contract(agent, expected_contract=contract)
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


def _publication_sink_from_message(message: object) -> str:
    if not isinstance(message, str):
        return ""
    lowered = message.casefold()
    if "instagram" in lowered:
        return "instagram:sportfish-hub" if "sportfish hub" in lowered else "instagram:unspecified"
    if "facebook" in lowered:
        return "facebook:unspecified"
    if "newsletter" in lowered:
        return "newsletter:unspecified"
    if "website" in lowered or "cms" in lowered:
        return "cms:unspecified"
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
