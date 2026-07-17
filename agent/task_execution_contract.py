"""Request-local execution policy for simple text-artifact turns.

The policy is deliberately conservative. It recognizes only explicit requests
for a textual artifact and falls back to the normal agent for anything that
also asks Hermes to research, mutate state, render, execute, or publish.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.parse import urlparse, urlunparse

from agent.file_safety import get_safe_write_roots, is_write_denied


NORMAL = "normal"
ARTIFACT_ONLY = "artifact_only"
POLICY_VERSION = "artifact-only-v2"
MAX_ARTIFACT_BYTES = 49 * 1024 * 1024
_ARTIFACT_RECEIPT_LOCK = threading.RLock()
_ARTIFACT_RECEIPTS: dict[str, "TaskExecutionContract"] = {}
_TERMINAL_RECEIPT_STATES = frozenset({"delivered", "failed_preflight", "ambiguous"})
_ALLOWED_RECEIPT_TRANSITIONS = {
    "": frozenset({"allocated"}),
    "allocated": frozenset({"allocated", "written", "failed_preflight", "ambiguous"}),
    "written": frozenset({"written", "dispatching", "failed_preflight", "ambiguous"}),
    "dispatching": frozenset({"dispatching", "delivered", "ambiguous"}),
}

_SUPPORTED_ARTIFACT_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}
_QUOTED_FILENAME = re.compile(
    r"[\"'`](?P<name>[^\"'`\r\n]{1,160}\.(?:txt|md|markdown))[\"'`]",
    re.IGNORECASE,
)
_BARE_FILENAME = re.compile(
    r"(?<![\w.-])(?P<name>[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.(?:txt|md|markdown))(?![\w.])",
    re.IGNORECASE,
)
_TEXT_FILE_REQUEST = re.compile(
    r"(?:\b(?:plain[- ]?text|text|txt|markdown|md)\s+(?:document|file)\b|"
    r"(?<!\w)\.(?:txt|md|markdown)\b)",
    re.IGNORECASE,
)
_FILE_DELIVERY_VERB = re.compile(
    r"\b(?:return|give|write|draft|create|produce|generate|make|deliver|send|attach)\b",
    re.IGNORECASE,
)
_EXTERNAL_DELIVERY_TARGET = re.compile(
    r"\b(?:email|gmail|outlook|sms|text\s+message|customer|client|recipient|"
    r"webhook|upload\s+to)\b",
    re.IGNORECASE,
)

_ARTIFACT_NOUN = re.compile(
    r"\b(?:gpt\s+image\s+prompt|image\s+prompt|prompt|caption|copy|creative\s+brief|"
    r"brief\b(?!\s+(?:update|status|summary)))",
    re.IGNORECASE,
)
_ARTIFACT_VERB = re.compile(
    r"\b(?:return|give|write|draft|create|produce|generate|make)\b",
    re.IGNORECASE,
)
_EXPLICIT_ONLY = re.compile(
    r"\b(?:prompt[- ]only|return\s+only|give\s+me\s+only|paste[- ]ready)\b",
    re.IGNORECASE,
)
_OPERATIONAL_REQUEST = re.compile(
    r"\b(?:research|search|look\s*up|browse|render|export|build|implement|"
    r"execute|run|reconcile|sync|publish|post|send|deploy|restart|install|"
    r"edit\s+(?:the\s+)?(?:file|ledger|manifest|record)|update\s+(?:the\s+)?"
    r"(?:file|ledger|manifest|record)|png|jpe?g|image\s+file)\b",
    re.IGNORECASE,
)
_URL = re.compile(r"https://[^\s<>'\"]+", re.IGNORECASE)
_NEGATED_CONSTRAINT = re.compile(
    r"\b(?:do\s+not|don't|without)\b(?:(?:\.(?:txt|md|markdown))|[^.;,\n])*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ToolAuthorization:
    allowed: bool
    code: str = "allow"
    message: str = ""
    halt: bool = False


@dataclass
class TaskExecutionContract:
    lane: str
    decision_reason: str
    correlation_id: str
    artifact_output_path: str
    artifact_id: str = ""
    artifact_root: str = ""
    artifact_receipt_path: str = ""
    artifact_filename: str = ""
    artifact_extension: str = ""
    artifact_mime_type: str = ""
    artifact_route: str = "none"
    artifact_file_requested: bool = False
    preflight_error: str = ""
    explicit_urls: frozenset[str] = field(default_factory=frozenset)
    policy_version: str = POLICY_VERSION
    max_tool_calls: int = 8
    max_tool_result_chars: int = 50_000
    max_network_lookups: int = 1
    max_artifact_writes: int = 1
    max_history_chars: int = 12_000
    max_history_messages: int = 4
    _tool_calls: int = 0
    _allowed_calls: int = 0
    _denied_calls: int = 0
    _network_lookups: int = 0
    _artifact_writes: int = 0
    _tool_result_chars: int = 0
    _truncated_chars: int = 0
    _started_at: float = field(default_factory=time.monotonic, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _artifact_identity: tuple[int, int] | None = field(default=None, repr=False)
    active: bool = True

    @property
    def system_guidance(self) -> str:
        if self.lane != ARTIFACT_ONLY:
            return ""
        if self.preflight_error:
            return ""
        if self.artifact_file_requested:
            return (
                "REQUEST EXECUTION CONTRACT (artifact_only, fail closed):\n"
                "Create only the requested textual file. Do not search prior sessions, "
                "reconcile records, render/export media, delegate, consult a council, call "
                "image generation, run code or shell commands, or modify unrelated state. "
                "Use no tools except one explicit HTTPS URL lookup when supplied by the user "
                "and one write_file call. The only permitted destination is "
                f"{self.artifact_output_path}. Preserve filename {self.artifact_filename} "
                f"and MIME type {self.artifact_mime_type}. After the write succeeds, return "
                f"MEDIA:{self.artifact_output_path} on its own line so the gateway sends a "
                "Telegram document. Stop after the attachment reference."
            )
        return (
            "REQUEST EXECUTION CONTRACT (artifact_only, fail closed):\n"
            "Return the requested textual artifact directly. Do not search prior "
            "sessions, reconcile records, render/export media, delegate, consult a "
            "council, call image generation, run code or shell commands, or modify "
            "state. Use no tools unless the user supplied an explicit HTTPS URL that "
            "must be opened once. Return the artifact in chat and stop as soon as it "
            "is complete."
        )

    def before_tool(self, tool_name: str, args: Mapping[str, Any] | None) -> ToolAuthorization:
        if self.lane != ARTIFACT_ONLY:
            return ToolAuthorization(True)
        if not self.active:
            return ToolAuthorization(
                False,
                "execution_contract_expired",
                "The request-local execution contract has expired; start a new turn.",
                halt=True,
            )

        tool_name = str(tool_name or "")
        args = args if isinstance(args, Mapping) else {}
        with self._lock:
            self._tool_calls += 1
            if self._tool_calls > self.max_tool_calls:
                self._denied_calls += 1
                return ToolAuthorization(
                    False,
                    "artifact_tool_call_budget_exhausted",
                    "Artifact-only tool-call budget exhausted; return the artifact or explain the blocker.",
                    halt=True,
                )

            if tool_name == "web_extract":
                urls = _tool_urls(args)
                if (
                    len(urls) != 1
                    or any(_normalized_url(url) not in self.explicit_urls for url in urls)
                ):
                    self._denied_calls += 1
                    return ToolAuthorization(
                        False,
                        "artifact_lookup_not_explicit",
                        "Artifact-only requests may open only an HTTPS URL explicitly supplied by the user.",
                    )
                if self._network_lookups >= self.max_network_lookups:
                    self._denied_calls += 1
                    return ToolAuthorization(
                        False,
                        "artifact_lookup_limit",
                        "Artifact-only requests allow at most one explicit-URL lookup.",
                    )
                self._network_lookups += 1
                self._allowed_calls += 1
                return ToolAuthorization(True)

            if tool_name == "write_file":
                if not self.artifact_file_requested or self.preflight_error:
                    self._denied_calls += 1
                    return ToolAuthorization(
                        False,
                        "artifact_write_not_requested",
                        "This artifact-only request did not request a file attachment.",
                    )
                requested = os.path.normcase(os.path.realpath(str(args.get("path") or "")))
                allowed = os.path.normcase(os.path.realpath(self.artifact_output_path))
                if requested != allowed:
                    self._denied_calls += 1
                    return ToolAuthorization(
                        False,
                        "artifact_write_path_denied",
                        "Artifact-only writes are restricted to the generated task artifact path.",
                    )
                if self._artifact_writes >= self.max_artifact_writes:
                    self._denied_calls += 1
                    return ToolAuthorization(
                        False,
                        "artifact_write_limit",
                        "Artifact-only requests allow at most one generated artifact write.",
                    )
                validation_error = validate_artifact_output_path(
                    requested, self.artifact_root
                )
                if validation_error is not None:
                    self._denied_calls += 1
                    return ToolAuthorization(
                        False,
                        "artifact_write_preflight_denied",
                        "The generated artifact destination no longer passes the writer policy.",
                    )
                if not isinstance(args.get("content"), str) or not artifact_content_fits(args["content"]):
                    self._denied_calls += 1
                    return ToolAuthorization(
                        False,
                        "artifact_write_too_large",
                        "Artifact attachments are limited to 49 MB.",
                    )
                self._artifact_writes += 1
                self._allowed_calls += 1
                return ToolAuthorization(True)

            self._denied_calls += 1
            return ToolAuthorization(
                False,
                "artifact_tool_not_allowlisted",
                f"{tool_name or 'Unknown tool'} is not permitted for an artifact-only request.",
            )

    def preflight_tool(self, tool_name: str, args: Mapping[str, Any] | None) -> ToolAuthorization:
        """Reject disallowed shapes before request middleware sees arguments.

        Allowed shapes are consumed by ``before_tool`` after middleware has
        run, so transformed arguments and request quotas are still enforced.
        Denied shapes are consumed here to retain the finite attempt budget.
        """
        if self.lane != ARTIFACT_ONLY:
            return ToolAuthorization(True)
        args = args if isinstance(args, Mapping) else {}
        with self._lock:
            tool_budget_exhausted = self._tool_calls >= self.max_tool_calls
            lookup_budget_exhausted = self._network_lookups >= self.max_network_lookups
            write_budget_exhausted = self._artifact_writes >= self.max_artifact_writes
        if tool_budget_exhausted:
            return self.before_tool(tool_name, args)
        if tool_name == "web_extract":
            urls = _tool_urls(args)
            if (
                not lookup_budget_exhausted
                and len(urls) == 1
                and all(_normalized_url(url) in self.explicit_urls for url in urls)
            ):
                return ToolAuthorization(True)
        elif tool_name == "write_file":
            requested = os.path.normcase(os.path.realpath(str(args.get("path") or "")))
            allowed = os.path.normcase(os.path.realpath(self.artifact_output_path))
            if (
                self.active
                and self.artifact_file_requested
                and not self.preflight_error
                and not write_budget_exhausted
                and requested == allowed
                and validate_artifact_output_path(requested, self.artifact_root) is None
            ):
                return ToolAuthorization(True)
        return self.before_tool(tool_name, args)

    def bound_tool_result(self, result: str | None) -> str:
        text = "" if result is None else str(result)
        if self.lane != ARTIFACT_ONLY:
            return text
        with self._lock:
            remaining = self.max_tool_result_chars - self._tool_result_chars
            if remaining <= 0:
                self._truncated_chars += len(text)
                return "[Tool result omitted: artifact-only request budget exhausted.]"
            if len(text) <= remaining:
                self._tool_result_chars += len(text)
                return text
            kept = text[:remaining]
            self._tool_result_chars += len(kept)
            self._truncated_chars += len(text) - len(kept)
            return kept + "\n[Tool result truncated by artifact-only policy.]"

    def bound_conversation_history(self, history: Any) -> list[dict[str, str]]:
        if self.lane != ARTIFACT_ONLY:
            return list(history) if history else []
        candidates: list[dict[str, str]] = []
        for message in list(history or []):
            if not isinstance(message, Mapping):
                continue
            role = message.get("role")
            content = message.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                continue
            if message.get("tool_calls"):
                continue
            item = {"role": role, "content": content}
            if candidates and candidates[-1]["role"] == role:
                candidates[-1] = item
            else:
                candidates.append(item)
        candidates = candidates[-self.max_history_messages :]
        remaining = self.max_history_chars
        bounded: list[dict[str, str]] = []
        for message in reversed(candidates):
            if remaining <= 0:
                break
            content = message["content"]
            if len(content) > remaining:
                content = content[-remaining:]
            bounded.append({"role": message["role"], "content": content})
            remaining -= len(content)
        bounded.reverse()
        return bounded

    def first_event_latency_ms(self, event_timestamp: Any) -> int | None:
        if not isinstance(event_timestamp, (int, float)):
            return None
        return max(0, int((float(event_timestamp) - self._started_at) * 1000))

    def telemetry(
        self,
        *,
        first_event_ms: int | None = None,
        decision_status: str = "unknown",
    ) -> dict[str, Any]:
        with self._lock:
            data: dict[str, Any] = {
                "schema_version": 1,
                "correlation_id": self.correlation_id,
                "lane": self.lane,
                "policy_version": self.policy_version,
                "decision_reason": self.decision_reason,
                "decision_status": decision_status,
                "artifact_file_requested": self.artifact_file_requested,
                "artifact_extension": self.artifact_extension,
                "artifact_route": self.artifact_route,
                "artifact_preflight_ok": not bool(self.preflight_error),
                "tool_calls": self._tool_calls,
                "allowed_tool_calls": self._allowed_calls,
                "denied_tool_calls": self._denied_calls,
                "network_lookups": self._network_lookups,
                "artifact_writes": self._artifact_writes,
                "tool_result_chars": self._tool_result_chars,
                "truncated_chars": self._truncated_chars,
                "elapsed_ms": max(0, int((time.monotonic() - self._started_at) * 1000)),
            }
        if first_event_ms is not None:
            data["first_event_ms"] = max(0, int(first_event_ms))
        return data

    def deactivate(self) -> None:
        """Expire this request-local policy so stale references fail closed."""
        with self._lock:
            self.active = False


def build_task_execution_contract(
    message: Any, *, task_id: str, platform: Any = None
) -> TaskExecutionContract:
    text = message if isinstance(message, str) else str(message or "")
    correlation_id = hashlib.sha256(str(task_id).encode("utf-8")).hexdigest()[:16]
    trusted_text = _trusted_request_text(text)
    affirmative_text = _NEGATED_CONSTRAINT.sub("", trusted_text)
    file_requested, requested_filename, requested_extension = _requested_artifact_file(
        affirmative_text
    )
    lane, reason = _classify(trusted_text, file_requested=file_requested)
    output_path = ""
    artifact_root = ""
    artifact_route = "none"
    artifact_filename = ""
    artifact_mime_type = ""
    preflight_error = ""
    artifact_id = uuid.uuid4().hex
    artifact_receipt_path = ""
    if lane == ARTIFACT_ONLY and file_requested:
        artifact_filename = _safe_artifact_filename(
            requested_filename, requested_extension, correlation_id
        )
        requested_extension = os.path.splitext(artifact_filename)[1].lower()
        artifact_mime_type = _SUPPORTED_ARTIFACT_TYPES[requested_extension]
        if platform is not None and not platform_supports_document_delivery(platform):
            preflight_error = "artifact_delivery_unavailable"
        prepared = None if preflight_error else _prepare_artifact_output(artifact_filename, artifact_id)
        if prepared is None:
            preflight_error = preflight_error or "artifact_output_unavailable"
        else:
            artifact_root, output_path, artifact_route = prepared
            receipts_root = os.path.join(os.path.dirname(artifact_root), ".receipts")
            try:
                os.makedirs(receipts_root, mode=0o700, exist_ok=True)
            except OSError:
                preflight_error = "artifact_output_unavailable"
            else:
                artifact_receipt_path = os.path.join(receipts_root, f"{artifact_id}.json")
    explicit_urls = frozenset(
        filter(None, (_normalized_url(url.rstrip(".,);]")) for url in _URL.findall(text)))
    )
    contract = TaskExecutionContract(
        lane=lane,
        decision_reason=reason,
        correlation_id=correlation_id,
        artifact_output_path=output_path,
        artifact_id=artifact_id if file_requested else "",
        artifact_root=artifact_root,
        artifact_receipt_path=artifact_receipt_path,
        artifact_filename=artifact_filename,
        artifact_extension=requested_extension if file_requested else "",
        artifact_mime_type=artifact_mime_type,
        artifact_route=artifact_route,
        artifact_file_requested=file_requested,
        preflight_error=preflight_error,
        explicit_urls=explicit_urls,
    )
    if contract.artifact_file_requested and not contract.preflight_error:
        with _ARTIFACT_RECEIPT_LOCK:
            _ARTIFACT_RECEIPTS[_artifact_registry_key(contract.artifact_output_path)] = contract
            _write_artifact_receipt_locked(contract, state="allocated")
    return contract


def platform_supports_document_delivery(platform: Any) -> bool:
    """Static capability gate used before an attachment-only turn is activated."""
    return str(getattr(platform, "value", platform) or "").lower() == "telegram"


def artifact_content_fits(content: str) -> bool:
    try:
        return len(content.encode("utf-8")) <= MAX_ARTIFACT_BYTES
    except UnicodeError:
        return False


def _artifact_registry_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(os.path.expanduser(str(path or ""))))


def _stat_identity(value: os.stat_result) -> tuple[int, int] | None:
    device = int(getattr(value, "st_dev", 0) or 0)
    inode = int(getattr(value, "st_ino", 0) or 0)
    if inode == 0:
        return None
    return device, inode


def _artifact_error_code(exc: OSError, default: str) -> str:
    if isinstance(exc, FileExistsError):
        return "artifact_destination_exists"
    for item in getattr(exc, "args", ()):
        if isinstance(item, str) and re.fullmatch(r"artifact_[a-z_]+", item):
            return item
    return default


def _verified_artifact_bytes(contract: TaskExecutionContract) -> tuple[bytes, tuple[int, int]]:
    path = contract.artifact_output_path
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise OSError("artifact_not_regular")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        identity = _stat_identity(opened)
        if identity is None or _stat_identity(before) != identity:
            raise OSError("artifact_path_changed")
        after_open = os.lstat(path)
        if stat.S_ISLNK(after_open.st_mode) or _stat_identity(after_open) != identity:
            raise OSError("artifact_path_changed")
        chunks: list[bytes] = []
        remaining = MAX_ARTIFACT_BYTES + 1
        while remaining > 0:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        final_fd = os.fstat(fd)
        final_path = os.lstat(path)
        if _stat_identity(final_fd) != identity or _stat_identity(final_path) != identity:
            raise OSError("artifact_path_changed")
        payload = b"".join(chunks)
        if len(payload) > MAX_ARTIFACT_BYTES or len(payload) != final_fd.st_size:
            raise OSError("artifact_oversize_or_changed")
        return payload, identity
    finally:
        os.close(fd)


def write_registered_artifact(path: str, content: str) -> tuple[bool, str, int]:
    """Create one registered artifact through an exclusive, descriptor-pinned write."""
    key = _artifact_registry_key(path)
    with _ARTIFACT_RECEIPT_LOCK:
        contract = _ARTIFACT_RECEIPTS.get(key)
    if contract is None or not contract.active:
        return False, "artifact_contract_unavailable", 0
    if key != _artifact_registry_key(contract.artifact_output_path):
        return False, "artifact_write_path_denied", 0
    if not artifact_content_fits(content):
        return False, "artifact_write_too_large", 0
    payload = content.encode("utf-8")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fd: int | None = None
    created_identity: tuple[int, int] | None = None
    try:
        fd = os.open(contract.artifact_output_path, flags, 0o600)
        opened = os.fstat(fd)
        created_identity = _stat_identity(opened)
        if created_identity is None or not stat.S_ISREG(opened.st_mode):
            raise OSError("artifact_identity_unavailable")
        visible = os.lstat(contract.artifact_output_path)
        if stat.S_ISLNK(visible.st_mode) or _stat_identity(visible) != created_identity:
            raise OSError("artifact_path_changed")
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
        final_fd = os.fstat(fd)
        final_path = os.lstat(contract.artifact_output_path)
        if (
            _stat_identity(final_fd) != created_identity
            or _stat_identity(final_path) != created_identity
            or final_fd.st_size != len(payload)
        ):
            raise OSError("artifact_path_changed")
        contract._artifact_identity = created_identity
        return True, "", len(payload)
    except OSError as exc:
        return False, _artifact_error_code(exc, "artifact_write_failed"), 0
    finally:
        if fd is not None:
            os.close(fd)
        if created_identity is not None:
            try:
                current = os.lstat(contract.artifact_output_path)
                if _stat_identity(current) == created_identity and contract._artifact_identity != created_identity:
                    os.unlink(contract.artifact_output_path)
            except OSError:
                pass


def is_registered_artifact_path(path: str) -> bool:
    with _ARTIFACT_RECEIPT_LOCK:
        return _artifact_registry_key(path) in _ARTIFACT_RECEIPTS


def registered_artifact_identity_matches(path: str, opened: os.stat_result) -> bool:
    """Bind provider delivery to the inode verified by the canonical writer/finalizer."""
    with _ARTIFACT_RECEIPT_LOCK:
        contract = _ARTIFACT_RECEIPTS.get(_artifact_registry_key(path))
        if contract is None:
            return True
        return (
            contract._artifact_identity is not None
            and contract._artifact_identity == _stat_identity(opened)
        )


def record_artifact_written(contract: TaskExecutionContract) -> bool:
    """Atomically record the bytes actually present after the canonical writer ran."""
    try:
        path = os.path.realpath(contract.artifact_output_path)
        if validate_artifact_output_path(path, contract.artifact_root) is not None:
            finalize_artifact_contract(
                contract,
                state="failed_preflight",
                error_code="invalid_output_path",
            )
            return False
        payload, identity = _verified_artifact_bytes(contract)
        contract._artifact_identity = identity
        _write_artifact_receipt(
            contract,
            state="written",
            sha256=hashlib.sha256(payload).hexdigest(),
            size=len(payload),
        )
        return True
    except OSError as exc:
        finalize_artifact_contract(
            contract,
            state="failed_preflight",
            error_code=_artifact_error_code(exc, "artifact_unreadable"),
        )
        return False


def record_artifact_dispatch(path: str, *, state: str, message_id: Any = None, error_code: str = "") -> str | None:
    """Advance a registered artifact receipt without persisting user content."""
    cleanup_contract = None
    with _ARTIFACT_RECEIPT_LOCK:
        key = _artifact_registry_key(path)
        contract = _ARTIFACT_RECEIPTS.get(key)
        if contract is None:
            return None
        receipt = _write_artifact_receipt_locked(
            contract,
            state=state,
            message_id=message_id,
            error_code=error_code,
        )
        if receipt.get("state") in _TERMINAL_RECEIPT_STATES:
            _ARTIFACT_RECEIPTS.pop(key, None)
            cleanup_contract = contract
    if cleanup_contract is not None:
        _cleanup_task_owned_artifact(cleanup_contract)
    return contract.correlation_id


def finalize_artifact_contract(
    contract: TaskExecutionContract,
    *,
    state: str,
    error_code: str,
) -> None:
    """Terminalize a turn that cannot reach provider dispatch."""
    if state not in _TERMINAL_RECEIPT_STATES:
        raise ValueError("artifact terminal state required")
    key = _artifact_registry_key(contract.artifact_output_path)
    with _ARTIFACT_RECEIPT_LOCK:
        _write_artifact_receipt_locked(contract, state=state, error_code=error_code)
        _ARTIFACT_RECEIPTS.pop(key, None)
    _cleanup_task_owned_artifact(contract)


def _write_artifact_receipt(
    contract: TaskExecutionContract,
    *,
    state: str,
    sha256: str = "",
    size: int | None = None,
    message_id: Any = None,
    error_code: str = "",
) -> dict[str, Any]:
    with _ARTIFACT_RECEIPT_LOCK:
        return _write_artifact_receipt_locked(
            contract,
            state=state,
            sha256=sha256,
            size=size,
            message_id=message_id,
            error_code=error_code,
        )


def _write_artifact_receipt_locked(
    contract: TaskExecutionContract,
    *,
    state: str,
    sha256: str = "",
    size: int | None = None,
    message_id: Any = None,
    error_code: str = "",
) -> dict[str, Any]:
    if not contract.artifact_receipt_path:
        return {}
    prior: dict[str, Any] = {}
    try:
        with open(contract.artifact_receipt_path, "r", encoding="utf-8") as source:
            prior = json.load(source)
    except (OSError, ValueError):
        pass
    prior_state = str(prior.get("state", ""))
    if prior_state in _TERMINAL_RECEIPT_STATES:
        return prior
    if state not in _ALLOWED_RECEIPT_TRANSITIONS.get(prior_state, frozenset()):
        return prior
    receipt = {
        "id": contract.artifact_id,
        "turn": contract.correlation_id,
        "path": contract.artifact_output_path,
        "sha256": sha256 or prior.get("sha256", ""),
        "bytes": size if size is not None else prior.get("bytes", 0),
        "mime": contract.artifact_mime_type,
        "state": state,
        "attempt_count": int(prior.get("attempt_count", 0))
        + (1 if state == "dispatching" and prior_state != "dispatching" else 0),
        "platform_message_id": str(message_id) if message_id is not None else prior.get("platform_message_id", ""),
        "error_code": error_code,
    }
    try:
        fd, temporary = tempfile.mkstemp(prefix=".receipt-", dir=contract.artifact_root)
        with os.fdopen(fd, "w", encoding="utf-8") as destination:
            json.dump(receipt, destination, sort_keys=True, separators=(",", ":"))
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, contract.artifact_receipt_path)
    except OSError:
        try:
            os.unlink(temporary)
        except (OSError, UnboundLocalError):
            pass
    return receipt


def _cleanup_task_owned_artifact(contract: TaskExecutionContract) -> None:
    """Remove only the verified output and its dedicated, now-empty task directory."""
    root = os.path.abspath(contract.artifact_root)
    if not contract.artifact_id or os.path.basename(root) != contract.artifact_id:
        return
    output = os.path.abspath(contract.artifact_output_path)
    if os.path.dirname(output) != root:
        return
    if contract._artifact_identity is not None:
        try:
            current = os.lstat(output)
            if _stat_identity(current) == contract._artifact_identity:
                os.unlink(output)
        except OSError:
            pass
    try:
        os.rmdir(root)
    except OSError:
        pass


def _trusted_request_text(text: str) -> str:
    """Discard fenced, quoted, and inline-code examples before classification."""
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"(?m)^\s*>.*$", "", text)
    return re.sub(r"`[^`\r\n]*`", "", text)


def clear_task_execution_contract(agent: Any) -> None:
    """Clear and expire any policy left by the previous request."""
    contract = getattr(agent, "_task_execution_contract", None)
    if contract is not None:
        key = _artifact_registry_key(getattr(contract, "artifact_output_path", ""))
        with _ARTIFACT_RECEIPT_LOCK:
            registered = _ARTIFACT_RECEIPTS.get(key) is contract
            receipt = {}
            try:
                with open(contract.artifact_receipt_path, "r", encoding="utf-8") as source:
                    receipt = json.load(source)
            except (OSError, ValueError, TypeError):
                pass
        if registered and receipt.get("state") == "allocated":
            finalize_artifact_contract(
                contract,
                state="failed_preflight",
                error_code="artifact_turn_ended_before_write",
            )
        deactivate = getattr(contract, "deactivate", None)
        if callable(deactivate):
            deactivate()
    agent._task_execution_contract = None
    guardrails = getattr(agent, "_tool_guardrails", None)
    setter = getattr(guardrails, "set_execution_contract", None)
    if callable(setter):
        setter(None)


def validate_artifact_output_path(path: str, artifact_root: str) -> str | None:
    """Apply the same canonical writer policy before contract injection and use."""
    if not path or not artifact_root:
        return "missing_artifact_path"
    try:
        resolved_root = os.path.realpath(os.path.expanduser(artifact_root))
        resolved_path = os.path.realpath(os.path.expanduser(path))
        if os.path.commonpath([resolved_root, resolved_path]) != resolved_root:
            return "outside_artifact_root"
        if os.path.dirname(resolved_path) != resolved_root:
            return "nested_artifact_path"
        if os.path.splitext(resolved_path)[1].lower() not in _SUPPORTED_ARTIFACT_TYPES:
            return "unsupported_artifact_type"
        if is_write_denied(resolved_path):
            return "writer_policy_denied"
    except (OSError, ValueError):
        return "invalid_artifact_path"
    return None


def _prepare_artifact_output(
    filename: str, correlation_id: str
) -> tuple[str, str, str] | None:
    for base_root, route in _artifact_root_candidates():
        try:
            expanded = os.path.abspath(os.path.expanduser(base_root))
            if _path_has_symlink_component(expanded):
                continue
            probe_target = os.path.join(expanded, ".artifact-write-probe")
            if is_write_denied(probe_target):
                continue
            if os.path.lexists(expanded) and not os.path.isdir(expanded):
                continue
            os.makedirs(expanded, mode=0o700, exist_ok=True)
            if _path_has_symlink_component(expanded):
                continue
            resolved_base = os.path.realpath(expanded)
            try:
                os.chmod(resolved_base, 0o700)
            except OSError:
                pass
            task_root = os.path.join(resolved_base, correlation_id)
            if _path_has_symlink_component(task_root):
                continue
            if is_write_denied(os.path.join(task_root, ".artifact-write-probe")):
                continue
            os.makedirs(task_root, mode=0o700, exist_ok=True)
            if _path_has_symlink_component(task_root):
                continue
            resolved_root = os.path.realpath(task_root)
            try:
                os.chmod(resolved_root, 0o700)
            except OSError:
                pass
            output_path = os.path.join(resolved_root, filename)
            if validate_artifact_output_path(output_path, resolved_root) is not None:
                continue
            fd, probe_path = tempfile.mkstemp(prefix=".artifact-write-probe-", dir=resolved_root)
            try:
                os.close(fd)
            finally:
                try:
                    os.unlink(probe_path)
                except OSError:
                    pass
            return resolved_root, output_path, route
        except (OSError, ValueError):
            continue
    return None


def _path_has_symlink_component(path: str) -> bool:
    """Reject a candidate when any existing path component is a symlink."""
    current = os.path.abspath(os.path.expanduser(path))
    while True:
        try:
            if os.path.lexists(current) and os.path.islink(current):
                return True
        except (OSError, ValueError):
            return True
        parent = os.path.dirname(current)
        if parent == current:
            return False
        current = parent


def _artifact_root_candidates() -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    preferred = os.getenv("HERMES_ARTIFACT_ROOT") or os.path.join(
        tempfile.gettempdir(), "hermes-artifacts"
    )
    candidates.append((preferred, "primary"))
    configured_fallback = os.getenv("HERMES_ARTIFACT_FALLBACK_ROOT")
    if configured_fallback:
        candidates.append((configured_fallback, "configured_fallback"))
    for safe_root in sorted(get_safe_write_roots()):
        candidates.append((os.path.join(safe_root, "hermes-artifacts"), "safe_root_fallback"))

    unique: list[tuple[str, str]] = []
    seen: set[str] = set()
    for root, route in candidates:
        resolved = os.path.normcase(os.path.realpath(os.path.expanduser(root)))
        if resolved not in seen:
            seen.add(resolved)
            unique.append((root, route))
    return unique


def _requested_artifact_file(text: str) -> tuple[bool, str, str]:
    match = _QUOTED_FILENAME.search(text) or _BARE_FILENAME.search(text)
    if match:
        filename = match.group("name")
        extension = os.path.splitext(filename)[1].lower()
        return True, filename, extension
    type_match = _TEXT_FILE_REQUEST.search(text)
    if not type_match:
        return False, "", ""
    matched = type_match.group(0).lower()
    extension = ".md" if "markdown" in matched or re.search(r"(?:^|\W)\.md\b", matched) else ".txt"
    return True, "", extension


def _safe_artifact_filename(raw_name: str, extension: str, correlation_id: str) -> str:
    extension = extension.lower() if extension.lower() in _SUPPORTED_ARTIFACT_TYPES else ".txt"
    if raw_name:
        basename = raw_name.replace("\\", "/").rsplit("/", 1)[-1]
        stem = os.path.splitext(basename)[0]
        stem = re.sub(r"[^A-Za-z0-9._ -]+", "-", stem)
        stem = re.sub(r"[\s_]+", "-", stem).strip(" .-")
    else:
        stem = ""
    if not stem:
        stem = f"artifact-{correlation_id}"
    max_stem = 128 - len(extension)
    return f"{stem[:max_stem]}{extension}"


def effective_request_system_prompt(agent: Any, base_prompt: str) -> str:
    """Append existing volatile guidance, then the request-local policy."""
    effective = base_prompt or ""
    ephemeral = getattr(agent, "ephemeral_system_prompt", None)
    if ephemeral:
        effective = (effective + "\n\n" + str(ephemeral)).strip()
    contract = getattr(agent, "_task_execution_contract", None)
    guidance = getattr(contract, "system_guidance", "") if contract is not None else ""
    if guidance:
        effective = (effective + "\n\n" + guidance).strip()
    return effective


def _classify(text: str, *, file_requested: bool = False) -> tuple[str, str]:
    if not text.strip():
        return NORMAL, "empty_request"
    affirmative_text = _NEGATED_CONSTRAINT.sub("", text)
    delivery_context = _BARE_FILENAME.sub(
        "", _QUOTED_FILENAME.sub("", affirmative_text)
    )
    if (
        file_requested
        and _FILE_DELIVERY_VERB.search(affirmative_text)
        and not _EXTERNAL_DELIVERY_TARGET.search(delivery_context)
    ):
        operational_terms = [
            match.group(0).lower()
            for match in _OPERATIONAL_REQUEST.finditer(affirmative_text)
        ]
        if not operational_terms or all(term == "send" for term in operational_terms):
            return ARTIFACT_ONLY, "explicit_text_file_artifact"
    if _OPERATIONAL_REQUEST.search(affirmative_text):
        return NORMAL, "operational_or_research_request"
    if _ARTIFACT_NOUN.search(text) and (_ARTIFACT_VERB.search(text) or _EXPLICIT_ONLY.search(text)):
        return ARTIFACT_ONLY, "explicit_text_artifact"
    return NORMAL, "ambiguous_request"


def _tool_urls(args: Mapping[str, Any]) -> list[str]:
    raw = args.get("urls", args.get("url"))
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, (list, tuple)):
        return [item for item in raw if isinstance(item, str)]
    return []


def _normalized_url(url: str) -> str:
    try:
        parsed = urlparse(url)
    except Exception:
        return ""
    host = (parsed.hostname or "").lower()
    if parsed.scheme.lower() != "https" or not host:
        return ""
    port = f":{parsed.port}" if parsed.port is not None else ""
    netloc = host + port
    path = parsed.path or ""
    return urlunparse(("https", netloc, path, "", parsed.query, ""))
