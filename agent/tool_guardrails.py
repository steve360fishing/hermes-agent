"""Pure tool-call loop guardrail primitives.

The controller in this module is intentionally side-effect free: it tracks
per-turn tool-call observations and returns decisions. Runtime code owns whether
those decisions become warning guidance, synthetic tool results, or controlled
turn halts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from utils import safe_json_loads
from agent.tool_result_classification import file_mutation_result_landed

if TYPE_CHECKING:
    from agent.task_execution_contract import TaskExecutionContract


IDEMPOTENT_TOOL_NAMES = frozenset(
    {
        "read_file",
        "search_files",
        "web_search",
        "web_extract",
        "session_search",
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
    }
)

MUTATING_TOOL_NAMES = frozenset(
    {
        "terminal",
        "execute_code",
        "write_file",
        "patch",
        "todo",
        "memory",
        "skill_manage",
        "browser_click",
        "browser_type",
        "browser_press",
        "browser_scroll",
        "browser_navigate",
        "send_message",
        "tournament_source_capture",
        "cronjob",
        "delegate_task",
        "process",
    }
)


class TournamentToolEffect(str, Enum):
    READ_RESEARCH = "read_research"
    TRUSTED_CAPTURE = "trusted_capture"
    PRIVATE_MEMORY = "private_memory"
    INTERNAL_DIAGNOSTIC = "internal_diagnostic"
    PRIVATE_HANDOFF = "private_handoff"
    PRIVATE_DELIVERY = "private_delivery"
    PUBLIC_CANDIDATE_WRITE = "public_candidate_write"
    EXTERNAL_PUBLICATION = "external_publication"
    TRUSTED_SNAPSHOT_WRITE = "trusted_snapshot_write"
    UNKNOWN_MUTATION = "unknown_mutation"


class TournamentDestinationKind(str, Enum):
    NONE = "none"
    LOCAL_PRIVATE = "local_private"
    TRUSTED_SNAPSHOT_ROOT = "trusted_snapshot_root"
    PRIVATE_SURFACE = "private_surface"
    PUBLIC_SINK = "public_sink"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TournamentToolAction:
    effect: TournamentToolEffect
    destination_kind: TournamentDestinationKind
    destination: str = ""


_PUBLIC_DESTINATION_PREFIXES = (
    "instagram",
    "facebook",
    "cms",
    "website",
    "newsletter",
    "email",
    "twitter",
    "x:",
)


def _declared_destination(args: Mapping[str, Any]) -> str:
    for key in (
        "external_publication_sink",
        "publication_sink",
        "target",
        "destination",
        "chat_id",
        "channel",
        "url",
        "path",
        "file_path",
    ):
        value = args.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value).strip()
    return ""


def _is_public_destination(destination: str, args: Mapping[str, Any]) -> bool:
    lowered = destination.strip().casefold()
    visibility = str(args.get("visibility") or "").strip().casefold()
    declared_effect = str(args.get("effect") or "").strip().casefold()
    if visibility in {"public", "external_publication"}:
        return True
    if declared_effect in {"publish", "publication", "external_publication"}:
        return True
    if lowered.startswith(_PUBLIC_DESTINATION_PREFIXES):
        return True
    if "@" in lowered and not lowered.startswith(("matrix:", "telegram:")):
        return True
    return (
        ":channel" in lowered
        or ":#" in lowered
        or lowered.startswith("telegram:-")
    )


def _declares_external_publication(args: Mapping[str, Any]) -> bool:
    if any(
        isinstance(args.get(key), (str, int)) and str(args.get(key)).strip()
        for key in ("external_publication_sink", "publication_sink")
    ):
        return True
    visibility = str(args.get("visibility") or "").strip().casefold()
    declared_effect = str(args.get("effect") or "").strip().casefold()
    return visibility in {"public", "external_publication"} or declared_effect in {
        "publish", "publication", "external_publication"
    }


def _is_private_messaging_destination(destination: str) -> bool:
    lowered = destination.strip().casefold()
    if not lowered:
        return False
    if ":private" in lowered or ":dm" in lowered:
        return True
    if lowered in {
        "telegram", "discord", "slack", "signal", "matrix", "whatsapp",
        "imessage", "photon", "feishu", "weixin", "yuanbao",
    }:
        return True
    if lowered.startswith("telegram:"):
        target = lowered.split(":", 1)[1].split(":", 1)[0]
        return target.isdigit()
    return False


def _is_trusted_snapshot_target(destination: str) -> bool:
    if not destination:
        return False
    try:
        from agent.tournament_truth_support import configured_runtime_roots

        roots = configured_runtime_roots()
        if roots is None:
            return False
        candidate = Path(destination).resolve(strict=False)
        candidate.relative_to(roots.source_snapshot_root.resolve(strict=True))
        return True
    except (OSError, ValueError, TypeError):
        return False


def classify_tournament_tool_action(
    tool_name: str,
    args: Mapping[str, Any] | None,
    *,
    execution_contract: Any = None,
    tournament_contract: Any = None,
) -> TournamentToolAction:
    """Classify observable effect and sink without deriving authority from text."""
    args = _coerce_args(args)
    destination = _declared_destination(args)
    if tool_name in IDEMPOTENT_TOOL_NAMES or tool_name in {
        "sportfish_tournament_research", "tournament_truth_gate",
    }:
        return TournamentToolAction(
            TournamentToolEffect.READ_RESEARCH, TournamentDestinationKind.NONE
        )
    if tool_name == "tournament_source_capture":
        return TournamentToolAction(
            TournamentToolEffect.TRUSTED_CAPTURE,
            TournamentDestinationKind.TRUSTED_SNAPSHOT_ROOT,
        )
    if tool_name == "memory":
        return TournamentToolAction(
            TournamentToolEffect.PRIVATE_MEMORY,
            TournamentDestinationKind.LOCAL_PRIVATE,
        )

    declared_effect = str(args.get("effect") or args.get("purpose") or "").casefold()
    if declared_effect in {"internal_diagnostic", "diagnostic", "private_diagnostic"}:
        return TournamentToolAction(
            TournamentToolEffect.INTERNAL_DIAGNOSTIC,
            TournamentDestinationKind.LOCAL_PRIVATE,
            destination,
        )
    if declared_effect in {"private_handoff", "codex_handoff"}:
        return TournamentToolAction(
            TournamentToolEffect.PRIVATE_HANDOFF,
            TournamentDestinationKind.LOCAL_PRIVATE,
            destination,
        )

    if tool_name in {"write_file", "patch"}:
        if _is_trusted_snapshot_target(destination):
            return TournamentToolAction(
                TournamentToolEffect.TRUSTED_SNAPSHOT_WRITE,
                TournamentDestinationKind.TRUSTED_SNAPSHOT_ROOT,
                destination,
            )
        expected_path = str(
            getattr(execution_contract, "artifact_output_path", "") or ""
        )
        if expected_path and destination == expected_path:
            state = str(getattr(getattr(tournament_contract, "state", None), "value", ""))
            if state == "mixed_publication":
                return TournamentToolAction(
                    TournamentToolEffect.PRIVATE_HANDOFF,
                    TournamentDestinationKind.LOCAL_PRIVATE,
                    destination,
                )
            if tournament_contract is not None:
                return TournamentToolAction(
                    TournamentToolEffect.PUBLIC_CANDIDATE_WRITE,
                    TournamentDestinationKind.LOCAL_PRIVATE,
                    destination,
                )
        if str(args.get("visibility") or "").casefold() == "public" or declared_effect in {
            "public_candidate", "claim_bearing_public_candidate",
        }:
            return TournamentToolAction(
                TournamentToolEffect.PUBLIC_CANDIDATE_WRITE,
                TournamentDestinationKind.LOCAL_PRIVATE,
                destination,
            )

    if tool_name == "send_message":
        action = str(args.get("action") or "send").casefold()
        if action in {"list", "react", "unreact"}:
            return TournamentToolAction(
                TournamentToolEffect.READ_RESEARCH,
                TournamentDestinationKind.NONE,
                destination,
            )
        if _is_public_destination(destination, args):
            return TournamentToolAction(
                TournamentToolEffect.EXTERNAL_PUBLICATION,
                TournamentDestinationKind.PUBLIC_SINK,
                destination,
            )
        if _is_private_messaging_destination(destination):
            return TournamentToolAction(
                TournamentToolEffect.PRIVATE_DELIVERY,
                TournamentDestinationKind.PRIVATE_SURFACE,
                destination,
            )

    if _declares_external_publication(args):
        return TournamentToolAction(
            TournamentToolEffect.EXTERNAL_PUBLICATION,
            TournamentDestinationKind.PUBLIC_SINK,
            destination,
        )
    return TournamentToolAction(
        TournamentToolEffect.UNKNOWN_MUTATION,
        TournamentDestinationKind.UNKNOWN,
        destination,
    )


@dataclass(frozen=True)
class ToolCallGuardrailConfig:
    """Thresholds for per-turn tool-call loop detection.

    Warnings are enabled by default and never prevent tool execution. Hard stops
    are explicit opt-in so interactive CLI/TUI sessions get a gentle nudge unless
    the user enables circuit-breaker behavior in config.yaml.
    """

    warnings_enabled: bool = True
    hard_stop_enabled: bool = False
    exact_failure_warn_after: int = 2
    exact_failure_block_after: int = 5
    same_tool_failure_warn_after: int = 3
    same_tool_failure_halt_after: int = 8
    no_progress_warn_after: int = 2
    no_progress_block_after: int = 5
    idempotent_tools: frozenset[str] = field(default_factory=lambda: IDEMPOTENT_TOOL_NAMES)
    mutating_tools: frozenset[str] = field(default_factory=lambda: MUTATING_TOOL_NAMES)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ToolCallGuardrailConfig":
        """Build config from the `tool_loop_guardrails` config.yaml section."""
        if not isinstance(data, Mapping):
            return cls()

        warn_after = data.get("warn_after")
        if not isinstance(warn_after, Mapping):
            warn_after = {}
        hard_stop_after = data.get("hard_stop_after")
        if not isinstance(hard_stop_after, Mapping):
            hard_stop_after = {}

        defaults = cls()
        return cls(
            warnings_enabled=_as_bool(data.get("warnings_enabled"), defaults.warnings_enabled),
            hard_stop_enabled=_as_bool(data.get("hard_stop_enabled"), defaults.hard_stop_enabled),
            exact_failure_warn_after=_positive_int(
                warn_after.get("exact_failure", data.get("exact_failure_warn_after")),
                defaults.exact_failure_warn_after,
            ),
            same_tool_failure_warn_after=_positive_int(
                warn_after.get("same_tool_failure", data.get("same_tool_failure_warn_after")),
                defaults.same_tool_failure_warn_after,
            ),
            no_progress_warn_after=_positive_int(
                warn_after.get("idempotent_no_progress", data.get("no_progress_warn_after")),
                defaults.no_progress_warn_after,
            ),
            exact_failure_block_after=_positive_int(
                hard_stop_after.get("exact_failure", data.get("exact_failure_block_after")),
                defaults.exact_failure_block_after,
            ),
            same_tool_failure_halt_after=_positive_int(
                hard_stop_after.get("same_tool_failure", data.get("same_tool_failure_halt_after")),
                defaults.same_tool_failure_halt_after,
            ),
            no_progress_block_after=_positive_int(
                hard_stop_after.get("idempotent_no_progress", data.get("no_progress_block_after")),
                defaults.no_progress_block_after,
            ),
        )


@dataclass(frozen=True)
class ToolCallSignature:
    """Stable, non-reversible identity for a tool name plus canonical args."""

    tool_name: str
    args_hash: str

    @classmethod
    def from_call(cls, tool_name: str, args: Mapping[str, Any] | None) -> "ToolCallSignature":
        canonical = canonical_tool_args(args or {})
        return cls(tool_name=tool_name, args_hash=_sha256(canonical))

    def to_metadata(self) -> dict[str, str]:
        """Return public metadata without raw argument values."""
        return {"tool_name": self.tool_name, "args_hash": self.args_hash}


@dataclass(frozen=True)
class ToolGuardrailDecision:
    """Decision returned by the tool-call guardrail controller."""

    action: str = "allow"  # allow | warn | deny | block | halt
    code: str = "allow"
    message: str = ""
    tool_name: str = ""
    count: int = 0
    signature: ToolCallSignature | None = None

    @property
    def allows_execution(self) -> bool:
        return self.action in {"allow", "warn"}

    @property
    def should_halt(self) -> bool:
        return self.action in {"block", "halt"}

    def to_metadata(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "action": self.action,
            "code": self.code,
            "message": self.message,
            "tool_name": self.tool_name,
            "count": self.count,
        }
        if self.signature is not None:
            data["signature"] = self.signature.to_metadata()
        return data


def canonical_tool_args(args: Mapping[str, Any]) -> str:
    """Return sorted compact JSON for parsed tool arguments."""
    if not isinstance(args, Mapping):
        raise TypeError(f"tool args must be a mapping, got {type(args).__name__}")
    return json.dumps(
        args,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def classify_tool_failure(tool_name: str, result: str | None) -> tuple[bool, str]:
    """Safety-fallback classifier used only when callers don't pass ``failed``.

    Mirrors ``agent.display._detect_tool_failure`` exactly so the guardrail
    never disagrees with the CLI's user-visible ``[error]`` tag. Production
    callers in ``run_agent.py`` always pass an explicit ``failed=`` derived
    from ``_detect_tool_failure``; this function exists so standalone callers
    (tests, tooling) still get consistent behavior.
    """
    if result is None:
        return False, ""
    if file_mutation_result_landed(tool_name, result):
        return False, ""

    if tool_name == "terminal":
        data = safe_json_loads(result)
        if isinstance(data, dict):
            exit_code = data.get("exit_code")
            if exit_code is not None and exit_code != 0:
                return True, f" [exit {exit_code}]"
        return False, ""

    if tool_name == "memory":
        data = safe_json_loads(result)
        if isinstance(data, dict):
            if data.get("success") is False and "exceed the limit" in data.get("error", ""):
                return True, " [full]"

    lower = result[:500].lower()
    if '"error"' in lower or '"failed"' in lower or result.startswith("Error"):
        return True, " [error]"

    return False, ""


class ToolCallGuardrailController:
    """Per-turn controller for repeated failed/non-progressing tool calls."""

    def __init__(self, config: ToolCallGuardrailConfig | None = None):
        self.config = config or ToolCallGuardrailConfig()
        self._execution_contract: TaskExecutionContract | None = None
        self.reset_for_turn()

    @property
    def _tournament_contract(self) -> Any | None:
        """Resolve tournament authority from the propagated request context."""
        from agent.tournament_intent_contract import current_tournament_contract

        return current_tournament_contract()

    def set_execution_contract(self, contract: TaskExecutionContract | None) -> None:
        """Bind the request-local policy evaluated before loop guardrails."""
        self._execution_contract = contract

    def set_tournament_contract(self, contract: Any | None) -> None:
        """Compatibility seam backed by request-local context, not controller state."""
        from agent.tournament_intent_contract import bind_tournament_contract

        bind_tournament_contract(contract)

    def _tournament_preflight_args(
        self,
        tool_name: str,
        args: Mapping[str, Any] | None,
        signature: ToolCallSignature,
    ) -> ToolGuardrailDecision | None:
        contract = self._tournament_contract
        action_model = classify_tournament_tool_action(
            tool_name,
            args,
            execution_contract=self._execution_contract,
            tournament_contract=contract,
        )
        if action_model.effect is TournamentToolEffect.TRUSTED_SNAPSHOT_WRITE:
            return ToolGuardrailDecision(
                action="deny",
                code="trusted_snapshot_write_requires_capture_tool",
                message=(
                    "Trusted tournament snapshots may be created only by the "
                    "runtime-owned tournament_source_capture capability."
                ),
                tool_name=tool_name,
                signature=signature,
            )
        if contract is None:
            if action_model.effect is TournamentToolEffect.EXTERNAL_PUBLICATION:
                return ToolGuardrailDecision(
                    action="deny",
                    code="external_publication_contract_required",
                    message=(
                        "External publication requires an active exact tournament truth "
                        "and one-use destination approval contract."
                    ),
                    tool_name=tool_name,
                    signature=signature,
                )
            if action_model.effect is TournamentToolEffect.PUBLIC_CANDIDATE_WRITE:
                return ToolGuardrailDecision(
                    action="deny",
                    code="public_candidate_contract_required",
                    message=(
                        "Claim-bearing public candidate writes require an active exact "
                        "tournament truth contract."
                    ),
                    tool_name=tool_name,
                    signature=signature,
                )
            return None
        if action_model.effect in {
            TournamentToolEffect.READ_RESEARCH,
            TournamentToolEffect.TRUSTED_CAPTURE,
            TournamentToolEffect.PRIVATE_MEMORY,
            TournamentToolEffect.INTERNAL_DIAGNOSTIC,
            TournamentToolEffect.PRIVATE_HANDOFF,
        }:
            return None
        if action_model.effect is TournamentToolEffect.PRIVATE_DELIVERY:
            expected_surface = str(getattr(contract, "destination", "") or "")
            expected_parts = expected_surface.split(":", 2)
            expected_platform = expected_parts[1] if len(expected_parts) == 3 else ""
            expected_chat = expected_parts[2] if len(expected_parts) == 3 else ""
            normalized_private_targets = {
                f"{expected_platform}:{expected_chat}",
                f"{expected_platform}:private:{expected_chat}",
                f"{expected_platform}:dm:{expected_chat}",
            }
            if (
                not expected_platform
                or not expected_chat
                or action_model.destination not in normalized_private_targets
            ):
                return ToolGuardrailDecision(
                    action="deny",
                    code="private_delivery_destination_mismatch",
                    message=(
                        "Truth authority for private draft delivery is bound to the exact "
                        "authenticated private conversation."
                    ),
                    tool_name=tool_name,
                    signature=signature,
                )
            candidate = str(
                (args or {}).get("message")
                or (args or {}).get("content")
                or (args or {}).get("candidate")
                or ""
            )
            try:
                authorization = contract.verify_receipt(candidate)
            except Exception:
                authorization = None
            if authorization is not None and authorization.allowed:
                return None
            return ToolGuardrailDecision(
                action="deny",
                code=getattr(authorization, "code", "receipt_missing_or_consumed"),
                message=(
                    "Private delivery of the claim-bearing tournament candidate requires "
                    "truth authority for those exact bytes; release approval is not consumed."
                ),
                tool_name=tool_name,
                signature=signature,
            )
        try:
            authorization = contract.authorize_tool(tool_name, _coerce_args(args))
        except Exception:
            authorization = None
        if authorization is not None and authorization.allowed:
            return None
        action = "block" if getattr(authorization, "halt", False) else "deny"
        return ToolGuardrailDecision(
            action=action,
            code=getattr(authorization, "code", "tournament_contract_unavailable"),
            message=getattr(
                authorization,
                "message",
                "The request-local tournament authority contract could not authorize this tool.",
            ),
            tool_name=tool_name,
            signature=signature,
        )

    def _tournament_bypasses_task_contract(
        self,
        tool_name: str,
        args: Mapping[str, Any] | None,
    ) -> bool:
        contract = self._tournament_contract
        if contract is None:
            return False
        action_model = classify_tournament_tool_action(
            tool_name,
            args,
            execution_contract=self._execution_contract,
            tournament_contract=contract,
        )
        if action_model.effect in {
            TournamentToolEffect.READ_RESEARCH,
            TournamentToolEffect.TRUSTED_CAPTURE,
            TournamentToolEffect.PRIVATE_MEMORY,
            TournamentToolEffect.INTERNAL_DIAGNOSTIC,
        }:
            return True
        try:
            return bool(contract.bypasses_task_contract(tool_name, _coerce_args(args)))
        except Exception:
            return False

    def bound_result(self, result: str | None) -> str:
        if self._execution_contract is None:
            return "" if result is None else str(result)
        return self._execution_contract.bound_tool_result(result)

    def preflight_request_contract(
        self,
        tool_name: str,
        args: Mapping[str, Any] | None,
    ) -> ToolGuardrailDecision:
        """Apply request-local shape checks before tool middleware."""
        signature = ToolCallSignature.from_call(tool_name, _coerce_args(args))
        tournament_decision = self._tournament_preflight_args(tool_name, args, signature)
        if tournament_decision is not None:
            return tournament_decision
        if self._tournament_bypasses_task_contract(tool_name, args):
            return ToolGuardrailDecision(tool_name=tool_name, signature=signature)
        if self._execution_contract is None:
            return ToolGuardrailDecision(tool_name=tool_name, signature=signature)
        authorization = self._execution_contract.preflight_tool(tool_name, _coerce_args(args))
        if authorization.allowed:
            return ToolGuardrailDecision(tool_name=tool_name, signature=signature)
        action = "block" if authorization.halt else "deny"
        decision = ToolGuardrailDecision(
            action=action,
            code=authorization.code,
            message=authorization.message,
            tool_name=tool_name,
            count=getattr(self._execution_contract, "_tool_calls", 0),
            signature=signature,
        )
        if decision.should_halt:
            self._halt_decision = decision
        return decision

    def reset_for_turn(self) -> None:
        self._exact_failure_counts: dict[ToolCallSignature, int] = {}
        self._same_tool_failure_counts: dict[str, int] = {}
        self._no_progress: dict[ToolCallSignature, tuple[str, int]] = {}
        self._halt_decision: ToolGuardrailDecision | None = None

    @property
    def halt_decision(self) -> ToolGuardrailDecision | None:
        return self._halt_decision

    def before_call(self, tool_name: str, args: Mapping[str, Any] | None) -> ToolGuardrailDecision:
        signature = ToolCallSignature.from_call(tool_name, _coerce_args(args))
        tournament_decision = self._tournament_preflight_args(tool_name, args, signature)
        if tournament_decision is not None:
            return tournament_decision
        bypasses_task_contract = self._tournament_bypasses_task_contract(tool_name, args)
        if self._execution_contract is not None and not bypasses_task_contract:
            authorization = self._execution_contract.before_tool(tool_name, _coerce_args(args))
            if not authorization.allowed:
                action = "block" if authorization.halt else "deny"
                decision = ToolGuardrailDecision(
                    action=action,
                    code=authorization.code,
                    message=authorization.message,
                    tool_name=tool_name,
                    count=getattr(self._execution_contract, "_tool_calls", 0),
                    signature=signature,
                )
                if decision.should_halt:
                    self._halt_decision = decision
                return decision
        if not self.config.hard_stop_enabled:
            return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

        exact_count = self._exact_failure_counts.get(signature, 0)
        if exact_count >= self.config.exact_failure_block_after:
            decision = ToolGuardrailDecision(
                action="block",
                code="repeated_exact_failure_block",
                message=(
                    f"Blocked {tool_name}: the same tool call failed {exact_count} "
                    "times with identical arguments. Stop retrying it unchanged; "
                    "change strategy or explain the blocker."
                ),
                tool_name=tool_name,
                count=exact_count,
                signature=signature,
            )
            self._halt_decision = decision
            return decision

        if self._is_idempotent(tool_name):
            record = self._no_progress.get(signature)
            if record is not None:
                _result_hash, repeat_count = record
                if repeat_count >= self.config.no_progress_block_after:
                    decision = ToolGuardrailDecision(
                        action="block",
                        code="idempotent_no_progress_block",
                        message=(
                            f"Blocked {tool_name}: this read-only call returned the same "
                            f"result {repeat_count} times. Stop repeating it unchanged; "
                            "use the result already provided or try a different query."
                        ),
                        tool_name=tool_name,
                        count=repeat_count,
                        signature=signature,
                    )
                    self._halt_decision = decision
                    return decision

        return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

    def after_call(
        self,
        tool_name: str,
        args: Mapping[str, Any] | None,
        result: str | None,
        *,
        failed: bool | None = None,
        no_dispatch_proven: bool = False,
    ) -> ToolGuardrailDecision:
        args = _coerce_args(args)
        signature = ToolCallSignature.from_call(tool_name, args)
        if failed is None:
            failed, _ = classify_tool_failure(tool_name, result)

        contract = self._tournament_contract
        if contract is not None and getattr(contract, "release_state", "") == "in_flight":
            # Once a provider-facing call entered the in-flight state, any
            # reported failure is ambiguous unless a caller proves no dispatch.
            ambiguous = bool(failed and not no_dispatch_proven)
            try:
                contract.record_external_result(success=not failed, ambiguous=ambiguous)
            except Exception:
                contract.release_state = "ambiguous"
                approval = getattr(contract, "release_approval", None)
                if approval is not None:
                    approval.state = "ambiguous"

        if failed:
            exact_count = self._exact_failure_counts.get(signature, 0) + 1
            self._exact_failure_counts[signature] = exact_count
            self._no_progress.pop(signature, None)

            same_count = self._same_tool_failure_counts.get(tool_name, 0) + 1
            self._same_tool_failure_counts[tool_name] = same_count

            if self.config.hard_stop_enabled and same_count >= self.config.same_tool_failure_halt_after:
                decision = ToolGuardrailDecision(
                    action="halt",
                    code="same_tool_failure_halt",
                    message=(
                        f"Stopped {tool_name}: it failed {same_count} times this turn. "
                        "Stop retrying the same failing tool path and choose a different approach."
                    ),
                    tool_name=tool_name,
                    count=same_count,
                    signature=signature,
                )
                self._halt_decision = decision
                return decision

            if self.config.warnings_enabled and exact_count >= self.config.exact_failure_warn_after:
                return ToolGuardrailDecision(
                    action="warn",
                    code="repeated_exact_failure_warning",
                    message=(
                        f"{tool_name} has failed {exact_count} times with identical arguments. "
                        "This looks like a loop; inspect the error and change strategy "
                        "instead of retrying it unchanged."
                    ),
                    tool_name=tool_name,
                    count=exact_count,
                    signature=signature,
                )

            if self.config.warnings_enabled and same_count >= self.config.same_tool_failure_warn_after:
                return ToolGuardrailDecision(
                    action="warn",
                    code="same_tool_failure_warning",
                    message=_tool_failure_recovery_hint(tool_name, same_count),
                    tool_name=tool_name,
                    count=same_count,
                    signature=signature,
                )

            return ToolGuardrailDecision(tool_name=tool_name, count=exact_count, signature=signature)

        self._exact_failure_counts.pop(signature, None)
        self._same_tool_failure_counts.pop(tool_name, None)

        if not self._is_idempotent(tool_name):
            self._no_progress.pop(signature, None)
            return ToolGuardrailDecision(tool_name=tool_name, signature=signature)

        result_hash = _result_hash(result)
        previous = self._no_progress.get(signature)
        repeat_count = 1
        if previous is not None and previous[0] == result_hash:
            repeat_count = previous[1] + 1
        self._no_progress[signature] = (result_hash, repeat_count)

        if self.config.warnings_enabled and repeat_count >= self.config.no_progress_warn_after:
            return ToolGuardrailDecision(
                action="warn",
                code="idempotent_no_progress_warning",
                message=(
                    f"{tool_name} returned the same result {repeat_count} times. "
                    "Use the result already provided or change the query instead of "
                    "repeating it unchanged."
                ),
                tool_name=tool_name,
                count=repeat_count,
                signature=signature,
            )

        return ToolGuardrailDecision(tool_name=tool_name, count=repeat_count, signature=signature)

    def _is_idempotent(self, tool_name: str) -> bool:
        if tool_name in self.config.mutating_tools:
            return False
        return tool_name in self.config.idempotent_tools


def toolguard_synthetic_result(decision: ToolGuardrailDecision) -> str:
    """Build a synthetic role=tool content string for a blocked tool call."""
    return json.dumps(
        {
            "error": decision.message,
            "guardrail": decision.to_metadata(),
        },
        ensure_ascii=False,
    )


def append_toolguard_guidance(result: str, decision: ToolGuardrailDecision) -> str:
    """Append runtime guidance to the current tool result content."""
    if decision.action not in {"warn", "halt"} or not decision.message:
        return result
    label = "Tool loop hard stop" if decision.action == "halt" else "Tool loop warning"
    suffix = (
        f"\n\n[{label}: "
        f"{decision.code}; count={decision.count}; {decision.message}]"
    )
    return (result or "") + suffix


def _tool_failure_recovery_hint(tool_name: str, count: int) -> str:
    """Action-oriented guidance for recovering from repeated tool failures."""
    common = (
        f"{tool_name} has failed {count} times this turn. This looks like a loop. "
        "Do not switch to text-only replies; keep using tools, but diagnose before retrying. "
        "First inspect the latest error/output and verify your assumptions. "
    )
    if tool_name == "terminal":
        return common + (
            "For terminal failures, run a small diagnostic such as `pwd && ls -la` "
            "in the same tool, then try an absolute path, a simpler command, a different "
            "working directory, or a different tool such as read_file/write_file/patch."
        )
    return common + (
        "Try different arguments, a narrower query/path, an absolute path when relevant, "
        "or a different tool that can make progress. If the blocker is external, report "
        "the blocker after one diagnostic attempt instead of repeating the same failing path."
    )


def _coerce_args(args: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return args if isinstance(args, Mapping) else {}


def _result_hash(result: str | None) -> str:
    parsed = safe_json_loads(result or "")
    if parsed is not None:
        try:
            canonical = json.dumps(
                parsed,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        except TypeError:
            canonical = str(parsed)
    else:
        canonical = result or ""
    return _sha256(canonical)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "enabled"}:
            return True
        if lowered in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


def _positive_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 1 else default


def _sha256(value: str) -> str:
    # surrogatepass: tool results scraped from the web can carry unpaired
    # UTF-16 surrogates (e.g. half of a mathematical-bold pair); a strict
    # encode raises and takes down the whole conversation loop. The hash only
    # needs deterministic bytes, not valid UTF-8.
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()
