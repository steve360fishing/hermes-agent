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
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping
from urllib.parse import urlparse, urlunparse

from agent.file_safety import get_safe_write_roots, is_write_denied


NORMAL = "normal"
ARTIFACT_ONLY = "artifact_only"
POLICY_VERSION = "artifact-only-v2"
MAX_ARTIFACT_BYTES = 49 * 1024 * 1024
MAX_PENDING_ARTIFACTS = 16
MAX_PENDING_ARTIFACT_BYTES = 128 * 1024 * 1024
ARTIFACT_WRITTEN_TTL_SECONDS = 60 * 60
_ARTIFACT_RECEIPT_LOCK = threading.RLock()
_ARTIFACT_RECEIPTS: dict[str, "TaskExecutionContract"] = {}
_TERMINAL_RECEIPT_STATES = frozenset({"delivered", "failed_preflight", "ambiguous"})
_ALLOWED_RECEIPT_TRANSITIONS = {
    "": frozenset({"allocated"}),
    "allocated": frozenset({"allocated", "written", "failed_preflight", "ambiguous"}),
    "written": frozenset({"written", "dispatching", "failed_preflight", "ambiguous"}),
    "dispatching": frozenset({"dispatching", "delivered", "ambiguous"}),
}
_WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


class ArtifactReceiptPersistenceError(OSError):
    """Durable receipt state could not be read or committed."""


class ArtifactPathSecurityError(OSError):
    """A path could not be opened without traversing mutable aliases."""

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
    _artifact_parent_chain: tuple[tuple[str, tuple[int, int]], ...] = field(
        default_factory=tuple,
        repr=False,
    )
    _receipt_parent_chain: tuple[tuple[str, tuple[int, int]], ...] = field(
        default_factory=tuple,
        repr=False,
    )
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
    artifact_parent_chain: tuple[tuple[str, tuple[int, int]], ...] = ()
    receipt_parent_chain: tuple[tuple[str, tuple[int, int]], ...] = ()
    if lane == ARTIFACT_ONLY and file_requested:
        artifact_filename = _safe_artifact_filename(
            requested_filename, requested_extension, correlation_id
        )
        requested_extension = os.path.splitext(artifact_filename)[1].lower()
        artifact_mime_type = _SUPPORTED_ARTIFACT_TYPES[requested_extension]
        if not platform_supports_document_delivery(platform):
            preflight_error = "artifact_delivery_unavailable"
            lane = NORMAL
            reason = "artifact_delivery_unavailable"
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
                try:
                    artifact_parent_chain = _capture_parent_chain(artifact_root)
                    receipt_parent_chain = _capture_parent_chain(receipts_root)
                except OSError:
                    preflight_error = "artifact_output_unavailable"
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
        _artifact_parent_chain=artifact_parent_chain,
        _receipt_parent_chain=receipt_parent_chain,
    )
    if contract.artifact_file_requested and not contract.preflight_error:
        with _ARTIFACT_RECEIPT_LOCK:
            key = _artifact_registry_key(contract.artifact_output_path)
            _ARTIFACT_RECEIPTS[key] = contract
            try:
                _write_artifact_receipt_locked(contract, state="allocated")
            except ArtifactReceiptPersistenceError:
                _ARTIFACT_RECEIPTS.pop(key, None)
                contract.preflight_error = "artifact_receipt_unavailable"
                contract.lane = NORMAL
    if contract.preflight_error and contract.artifact_root:
        _cleanup_task_owned_artifact(contract)
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


def _is_reparse_or_symlink(value: os.stat_result) -> bool:
    return stat.S_ISLNK(value.st_mode) or bool(
        int(getattr(value, "st_file_attributes", 0) or 0)
        & _WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT
    )


def _absolute_path_components(path: str) -> list[str]:
    absolute = os.path.abspath(os.path.expanduser(path))
    drive, tail = os.path.splitdrive(absolute)
    anchor = drive + os.sep if drive else os.sep
    components = [anchor]
    current = anchor
    for part in tail.replace("/", os.sep).split(os.sep):
        if not part:
            continue
        current = os.path.join(current, part)
        components.append(current)
    return components


def _capture_parent_chain(path: str) -> tuple[tuple[str, tuple[int, int]], ...]:
    chain: list[tuple[str, tuple[int, int]]] = []
    for component in _absolute_path_components(path):
        value = os.lstat(component)
        identity = _stat_identity(value)
        if (
            identity is None
            or _is_reparse_or_symlink(value)
            or not stat.S_ISDIR(value.st_mode)
        ):
            raise ArtifactPathSecurityError("artifact_parent_unsafe")
        chain.append((component, identity))
    return tuple(chain)


def _verify_parent_chain(
    chain: tuple[tuple[str, tuple[int, int]], ...],
) -> None:
    if not chain:
        raise ArtifactPathSecurityError("artifact_parent_unavailable")
    for component, expected in chain:
        value = os.lstat(component)
        if (
            _is_reparse_or_symlink(value)
            or not stat.S_ISDIR(value.st_mode)
            or _stat_identity(value) != expected
        ):
            raise ArtifactPathSecurityError("artifact_parent_changed")


@contextmanager
def _hold_windows_parent_chain(
    chain: tuple[tuple[str, tuple[int, int]], ...],
) -> Iterator[None]:
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        create_file.restype = wintypes.HANDLE
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (wintypes.HANDLE,)
        close_handle.restype = wintypes.BOOL
    except Exception as exc:  # pragma: no cover - platform import boundary
        raise ArtifactPathSecurityError("artifact_secure_path_unavailable") from exc

    handles: list[Any] = []
    invalid = ctypes.c_void_p(-1).value
    try:
        for component, _expected in chain:
            handle = create_file(
                component,
                0x80,  # FILE_READ_ATTRIBUTES
                0x1 | 0x2,  # FILE_SHARE_READ | FILE_SHARE_WRITE; deliberately no DELETE
                None,
                3,  # OPEN_EXISTING
                0x02000000 | 0x00200000,  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
                None,
            )
            if handle in (None, invalid):
                raise ArtifactPathSecurityError("artifact_secure_path_unavailable")
            handles.append(handle)
        _verify_parent_chain(chain)
        yield None
    finally:
        for handle in reversed(handles):
            close_handle(handle)


@contextmanager
def _hold_posix_parent_chain(
    chain: tuple[tuple[str, tuple[int, int]], ...],
) -> Iterator[int]:
    if (
        os.open not in getattr(os, "supports_dir_fd", set())
        or not getattr(os, "O_NOFOLLOW", 0)
        or not getattr(os, "O_DIRECTORY", 0)
    ):
        raise ArtifactPathSecurityError("artifact_secure_path_unavailable")
    descriptors: list[int] = []
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        try:
            first_path, first_identity = chain[0]
            current = os.open(first_path, flags)
            descriptors.append(current)
            if _stat_identity(os.fstat(current)) != first_identity:
                raise ArtifactPathSecurityError("artifact_parent_changed")
            for component, expected in chain[1:]:
                name = os.path.basename(component.rstrip(os.sep))
                current = os.open(name, flags, dir_fd=current)
                descriptors.append(current)
                opened = os.fstat(current)
                if (
                    _stat_identity(opened) != expected
                    or not stat.S_ISDIR(opened.st_mode)
                ):
                    raise ArtifactPathSecurityError("artifact_parent_changed")
        except ArtifactPathSecurityError:
            raise
        except OSError as exc:
            raise ArtifactPathSecurityError("artifact_parent_changed") from exc
        yield current
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


@contextmanager
def _hold_parent_chain(
    chain: tuple[tuple[str, tuple[int, int]], ...],
) -> Iterator[int | None]:
    if os.name == "nt":
        with _hold_windows_parent_chain(chain):
            yield None
    else:
        with _hold_posix_parent_chain(chain) as descriptor:
            yield descriptor


def _leaf_lstat(path: str, parent_fd: int | None) -> os.stat_result:
    if parent_fd is None:
        return os.lstat(path)
    return os.stat(
        os.path.basename(path),
        dir_fd=parent_fd,
        follow_symlinks=False,
    )


def _open_leaf_descriptor(
    path: str,
    parent_fd: int | None,
    *,
    flags: int,
    mode: int = 0o600,
    must_exist: bool,
) -> tuple[int, tuple[int, int]]:
    before = None
    if must_exist:
        before = _leaf_lstat(path, parent_fd)
        if _is_reparse_or_symlink(before) or not stat.S_ISREG(before.st_mode):
            raise ArtifactPathSecurityError("artifact_not_regular")
    open_flags = flags | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    target = path if parent_fd is None else os.path.basename(path)
    fd = os.open(target, open_flags, mode, **({"dir_fd": parent_fd} if parent_fd is not None else {}))
    try:
        opened = os.fstat(fd)
        identity = _stat_identity(opened)
        after = _leaf_lstat(path, parent_fd)
        if (
            identity is None
            or not stat.S_ISREG(opened.st_mode)
            or _is_reparse_or_symlink(after)
            or _stat_identity(after) != identity
            or (before is not None and _stat_identity(before) != identity)
        ):
            raise ArtifactPathSecurityError("artifact_path_changed")
        return fd, identity
    except BaseException:
        os.close(fd)
        raise


def open_path_descriptor_no_reparse(path: str) -> int:
    """Open an arbitrary document through a no-follow parent chain."""
    absolute = os.path.abspath(os.path.expanduser(path))
    chain = _capture_parent_chain(os.path.dirname(absolute))
    with _hold_parent_chain(chain) as parent_fd:
        fd, _identity = _open_leaf_descriptor(
            absolute,
            parent_fd,
            flags=os.O_RDONLY,
            must_exist=True,
        )
        return fd


def _artifact_error_code(exc: OSError, default: str) -> str:
    if isinstance(exc, FileExistsError):
        return "artifact_destination_exists"
    for item in getattr(exc, "args", ()):
        if isinstance(item, str) and re.fullmatch(r"artifact_[a-z_]+", item):
            return item
    return default


def _descriptor_bytes(fd: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    remaining = MAX_ARTIFACT_BYTES + 1
    while remaining > 0:
        chunk = os.read(fd, min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    payload = b"".join(chunks)
    if len(payload) > MAX_ARTIFACT_BYTES:
        raise ArtifactPathSecurityError("artifact_oversize_or_changed")
    return payload


def _verified_artifact_bytes(contract: TaskExecutionContract) -> tuple[bytes, tuple[int, int]]:
    path = contract.artifact_output_path
    with _hold_parent_chain(contract._artifact_parent_chain) as parent_fd:
        fd, identity = _open_leaf_descriptor(
            path,
            parent_fd,
            flags=os.O_RDONLY,
            must_exist=True,
        )
    try:
        payload = _descriptor_bytes(fd)
        final_fd = os.fstat(fd)
        if _stat_identity(final_fd) != identity or len(payload) != final_fd.st_size:
            raise ArtifactPathSecurityError("artifact_oversize_or_changed")
        return payload, identity
    finally:
        os.close(fd)


def write_registered_artifact(
    path: str,
    content: str,
    writer: Callable[[str, str], Any] | None = None,
) -> tuple[bool, str, int]:
    """Write through the terminal backend while pinning the delivery parent."""
    key = _artifact_registry_key(path)
    with _ARTIFACT_RECEIPT_LOCK:
        contract = _ARTIFACT_RECEIPTS.get(key)
    if contract is None or not contract.active:
        return False, "artifact_contract_unavailable", 0
    if key != _artifact_registry_key(contract.artifact_output_path):
        return False, "artifact_write_path_denied", 0
    if not artifact_content_fits(content):
        return False, "artifact_write_too_large", 0
    if writer is None:
        return False, "artifact_backend_unavailable", 0
    try:
        with _hold_parent_chain(contract._artifact_parent_chain) as parent_fd:
            created_identity = None
            try:
                _leaf_lstat(contract.artifact_output_path, parent_fd)
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError("artifact_destination_exists")
            backend_path = contract.artifact_output_path
            if os.name != "nt":
                proc_parent = f"/proc/{os.getpid()}/fd/{parent_fd}"
                if not os.path.isdir(proc_parent):
                    raise ArtifactPathSecurityError("artifact_secure_backend_unavailable")
                backend_path = os.path.join(proc_parent, contract.artifact_filename)
            try:
                try:
                    result = writer(backend_path, content)
                except Exception as exc:
                    raise OSError("artifact_backend_write_failed") from exc
                result_dict = result.to_dict() if hasattr(result, "to_dict") else {}
                if result_dict.get("error"):
                    raise OSError("artifact_backend_write_failed")
                fd, created_identity = _open_leaf_descriptor(
                    contract.artifact_output_path,
                    parent_fd,
                    flags=os.O_RDONLY,
                    must_exist=True,
                )
                try:
                    payload = _descriptor_bytes(fd)
                    if os.fstat(fd).st_size != len(payload):
                        raise ArtifactPathSecurityError("artifact_oversize_or_changed")
                finally:
                    os.close(fd)
                _verify_parent_chain(contract._artifact_parent_chain)
                contract._artifact_identity = created_identity
                return True, "", len(payload)
            except OSError:
                if created_identity is not None:
                    try:
                        current = _leaf_lstat(contract.artifact_output_path, parent_fd)
                        if _stat_identity(current) == created_identity:
                            if parent_fd is None:
                                os.unlink(contract.artifact_output_path)
                            else:
                                os.unlink(contract.artifact_filename, dir_fd=parent_fd)
                    except OSError:
                        pass
                raise
    except OSError as exc:
        return False, _artifact_error_code(exc, "artifact_write_failed"), 0


def is_registered_artifact_path(path: str) -> bool:
    with _ARTIFACT_RECEIPT_LOCK:
        return _artifact_registry_key(path) in _ARTIFACT_RECEIPTS


def registered_artifact_correlation_id(path: str) -> str | None:
    with _ARTIFACT_RECEIPT_LOCK:
        contract = _ARTIFACT_RECEIPTS.get(_artifact_registry_key(path))
        return contract.correlation_id if contract is not None else None


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


def open_registered_artifact_descriptor(path: str) -> int | None:
    """Open a registered artifact and bind held bytes to its durable receipt."""
    key = _artifact_registry_key(path)
    with _ARTIFACT_RECEIPT_LOCK:
        contract = _ARTIFACT_RECEIPTS.get(key)
        if contract is None:
            return None
        receipt = _load_artifact_receipt_locked(contract, required=True)
        if (
            receipt.get("id") != contract.artifact_id
            or receipt.get("path") != contract.artifact_output_path
            or receipt.get("mime") != contract.artifact_mime_type
            or receipt.get("state") not in {"written", "dispatching"}
        ):
            raise ArtifactPathSecurityError("artifact_receipt_mismatch")
        with _hold_parent_chain(contract._artifact_parent_chain) as parent_fd:
            fd, identity = _open_leaf_descriptor(
                contract.artifact_output_path,
                parent_fd,
                flags=os.O_RDONLY,
                must_exist=True,
            )
            try:
                if contract._artifact_identity != identity:
                    raise ArtifactPathSecurityError("artifact_path_changed")
                payload = _descriptor_bytes(fd)
                digest = hashlib.sha256(payload).hexdigest()
                if (
                    int(receipt.get("bytes", -1)) != len(payload)
                    or receipt.get("sha256") != digest
                ):
                    raise ArtifactPathSecurityError("artifact_receipt_mismatch")
                os.lseek(fd, 0, os.SEEK_SET)
                return fd
            except BaseException:
                os.close(fd)
                raise


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
        _reconcile_artifact_store(os.path.dirname(contract.artifact_root))
        return True
    except ArtifactReceiptPersistenceError:
        raise
    except OSError as exc:
        finalize_artifact_contract(
            contract,
            state="failed_preflight",
            error_code=_artifact_error_code(exc, "artifact_unreadable"),
        )
        return False


def record_artifact_dispatch(
    path: str,
    *,
    state: str,
    message_id: Any = None,
    error_code: str = "",
    transport_attempt: bool = False,
) -> str | None:
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
            transport_attempt=transport_attempt,
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
    transport_attempt: bool = False,
) -> dict[str, Any]:
    with _ARTIFACT_RECEIPT_LOCK:
        return _write_artifact_receipt_locked(
            contract,
            state=state,
            sha256=sha256,
            size=size,
            message_id=message_id,
            error_code=error_code,
            transport_attempt=transport_attempt,
        )


def _load_artifact_receipt_locked(
    contract: TaskExecutionContract,
    *,
    required: bool,
) -> dict[str, Any]:
    if not contract.artifact_receipt_path:
        if required:
            raise ArtifactReceiptPersistenceError("artifact_receipt_missing")
        return {}
    try:
        with _hold_parent_chain(contract._receipt_parent_chain) as parent_fd:
            fd, _identity = _open_leaf_descriptor(
                contract.artifact_receipt_path,
                parent_fd,
                flags=os.O_RDONLY,
                must_exist=True,
            )
            try:
                raw = bytearray()
                while len(raw) <= 64 * 1024:
                    chunk = os.read(fd, min(8192, 64 * 1024 + 1 - len(raw)))
                    if not chunk:
                        break
                    raw.extend(chunk)
                if len(raw) > 64 * 1024:
                    raise ArtifactReceiptPersistenceError("artifact_receipt_unreadable")
                receipt = json.loads(raw.decode("utf-8"))
            finally:
                os.close(fd)
    except FileNotFoundError:
        if required:
            raise ArtifactReceiptPersistenceError("artifact_receipt_missing")
        return {}
    except (OSError, ValueError, TypeError) as exc:
        raise ArtifactReceiptPersistenceError("artifact_receipt_unreadable") from exc
    if not isinstance(receipt, dict):
        raise ArtifactReceiptPersistenceError("artifact_receipt_unreadable")
    return receipt


def _write_artifact_receipt_locked(
    contract: TaskExecutionContract,
    *,
    state: str,
    sha256: str = "",
    size: int | None = None,
    message_id: Any = None,
    error_code: str = "",
    transport_attempt: bool = False,
) -> dict[str, Any]:
    if not contract.artifact_receipt_path:
        raise ArtifactReceiptPersistenceError("artifact_receipt_missing")
    prior = _load_artifact_receipt_locked(
        contract,
        required=state != "allocated",
    )
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
        + (1 if state == "dispatching" and transport_attempt else 0),
        "platform_message_id": str(message_id) if message_id is not None else prior.get("platform_message_id", ""),
        "error_code": error_code,
        "updated_at": int(time.time()),
    }
    temporary_name = f".receipt-{uuid.uuid4().hex}.tmp"
    receipt_parent = os.path.dirname(contract.artifact_receipt_path)
    receipt_name = os.path.basename(contract.artifact_receipt_path)
    with _hold_parent_chain(contract._receipt_parent_chain) as parent_fd:
        temporary_path = os.path.join(receipt_parent, temporary_name)
        target = temporary_path if parent_fd is None else temporary_name
        open_kwargs = {"dir_fd": parent_fd} if parent_fd is not None else {}
        fd = -1
        try:
            fd = os.open(
                target,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                **open_kwargs,
            )
            payload = json.dumps(
                receipt,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            offset = 0
            while offset < len(payload):
                offset += os.write(fd, payload[offset:])
            os.fsync(fd)
            os.close(fd)
            fd = -1
            if parent_fd is None:
                os.replace(temporary_path, contract.artifact_receipt_path)
            else:
                if os.rename not in getattr(os, "supports_dir_fd", set()):
                    raise ArtifactPathSecurityError("artifact_secure_path_unavailable")
                os.rename(
                    temporary_name,
                    receipt_name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
            temporary_name = ""
            if parent_fd is not None:
                os.fsync(parent_fd)
        except OSError as exc:
            try:
                if fd >= 0:
                    os.close(fd)
            except OSError:
                pass
            try:
                if temporary_name:
                    if parent_fd is None:
                        os.unlink(temporary_path)
                    else:
                        os.unlink(temporary_name, dir_fd=parent_fd)
            except OSError:
                pass
            raise ArtifactReceiptPersistenceError("artifact_receipt_persistence_failed") from exc
    return receipt


def _orphan_contract_from_receipt(
    artifact_base: str,
    receipt_path: str,
    receipt: Mapping[str, Any],
) -> TaskExecutionContract | None:
    artifact_id = str(receipt.get("id", ""))
    output = os.path.abspath(str(receipt.get("path", "")))
    if not re.fullmatch(r"[0-9a-f]{32}", artifact_id):
        return None
    root = os.path.abspath(os.path.join(artifact_base, artifact_id))
    if os.path.dirname(output) != root or os.path.basename(root) != artifact_id:
        return None
    extension = os.path.splitext(output)[1].lower()
    if extension not in _SUPPORTED_ARTIFACT_TYPES:
        return None
    try:
        chain = _capture_parent_chain(root)
        receipt_chain = _capture_parent_chain(os.path.dirname(receipt_path))
    except OSError:
        return None
    contract = TaskExecutionContract(
        lane=NORMAL,
        decision_reason="artifact_reconciliation",
        correlation_id=str(receipt.get("turn", "")),
        artifact_output_path=output,
        artifact_id=artifact_id,
        artifact_root=root,
        artifact_receipt_path=receipt_path,
        artifact_filename=os.path.basename(output),
        artifact_extension=extension,
        artifact_mime_type=str(receipt.get("mime", "")),
        artifact_route="reconciled",
        artifact_file_requested=True,
        _artifact_parent_chain=chain,
        _receipt_parent_chain=receipt_chain,
    )
    try:
        payload, identity = _verified_artifact_bytes(contract)
    except OSError:
        return None
    if (
        len(payload) != int(receipt.get("bytes", -1))
        or hashlib.sha256(payload).hexdigest() != receipt.get("sha256")
    ):
        return None
    contract._artifact_identity = identity
    return contract


def _reconcile_artifact_store(artifact_base: str) -> None:
    """Bound abandoned written artifacts without retrying provider delivery."""
    artifact_base = os.path.abspath(artifact_base)
    receipts_root = os.path.join(artifact_base, ".receipts")
    try:
        receipt_paths = list(Path(receipts_root).glob("*.json"))
    except OSError:
        return
    records: list[tuple[int, str, dict[str, Any], Path]] = []
    now = time.time()
    for receipt_path in receipt_paths:
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            modified = receipt_path.stat().st_mtime_ns
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(receipt, dict) or receipt.get("state") not in {"written", "dispatching"}:
            continue
        records.append((modified, str(receipt.get("id", "")), receipt, receipt_path))
    records.sort(key=lambda item: (item[0], item[1]))

    selected: dict[str, tuple[dict[str, Any], Path, str]] = {}
    for modified, artifact_id, receipt, receipt_path in records:
        age = max(0.0, now - (modified / 1_000_000_000))
        if age > ARTIFACT_WRITTEN_TTL_SECONDS:
            terminal_state = "ambiguous" if receipt.get("state") == "dispatching" else "failed_preflight"
            selected[artifact_id] = (receipt, receipt_path, terminal_state)

    pending = [item for item in records if item[1] not in selected and item[2].get("state") == "written"]
    pending_bytes = sum(max(0, int(item[2].get("bytes", 0) or 0)) for item in pending)
    while len(pending) > MAX_PENDING_ARTIFACTS or pending_bytes > MAX_PENDING_ARTIFACT_BYTES:
        _modified, artifact_id, receipt, receipt_path = pending.pop(0)
        pending_bytes -= max(0, int(receipt.get("bytes", 0) or 0))
        selected[artifact_id] = (receipt, receipt_path, "failed_preflight")

    cleanup: list[TaskExecutionContract] = []
    with _ARTIFACT_RECEIPT_LOCK:
        for _artifact_id, (receipt, receipt_path, terminal_state) in selected.items():
            path = str(receipt.get("path", ""))
            key = _artifact_registry_key(path)
            contract = _ARTIFACT_RECEIPTS.get(key)
            if contract is None:
                contract = _orphan_contract_from_receipt(
                    artifact_base,
                    str(receipt_path),
                    receipt,
                )
            if contract is None:
                continue
            _write_artifact_receipt_locked(
                contract,
                state=terminal_state,
                error_code="artifact_dispatch_abandoned",
            )
            _ARTIFACT_RECEIPTS.pop(key, None)
            cleanup.append(contract)
    for contract in cleanup:
        _cleanup_task_owned_artifact(contract)


def _cleanup_task_owned_artifact(contract: TaskExecutionContract) -> None:
    """Remove only the verified output and its dedicated, now-empty task directory."""
    root = os.path.abspath(contract.artifact_root)
    if not contract.artifact_id or os.path.basename(root) != contract.artifact_id:
        return
    output = os.path.abspath(contract.artifact_output_path)
    if os.path.dirname(output) != root:
        return
    try:
        _verify_parent_chain(contract._artifact_parent_chain)
    except OSError:
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
    persistence_error: ArtifactReceiptPersistenceError | None = None
    if contract is not None:
        key = _artifact_registry_key(getattr(contract, "artifact_output_path", ""))
        with _ARTIFACT_RECEIPT_LOCK:
            registered = _ARTIFACT_RECEIPTS.get(key) is contract
            try:
                receipt = _load_artifact_receipt_locked(contract, required=registered)
            except ArtifactReceiptPersistenceError as exc:
                receipt = {}
                persistence_error = exc
        if registered and receipt.get("state") == "allocated":
            try:
                finalize_artifact_contract(
                    contract,
                    state="failed_preflight",
                    error_code="artifact_turn_ended_before_write",
                )
            except ArtifactReceiptPersistenceError as exc:
                persistence_error = exc
        deactivate = getattr(contract, "deactivate", None)
        if callable(deactivate):
            deactivate()
    agent._task_execution_contract = None
    guardrails = getattr(agent, "_tool_guardrails", None)
    setter = getattr(guardrails, "set_execution_contract", None)
    if callable(setter):
        setter(None)
    if persistence_error is not None:
        raise persistence_error


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
            _reconcile_artifact_store(resolved_base)
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
    """Reject a candidate when any existing component is a symlink/reparse point."""
    current = os.path.abspath(os.path.expanduser(path))
    while True:
        try:
            if os.path.lexists(current):
                value = os.lstat(current)
                if _is_reparse_or_symlink(value):
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
