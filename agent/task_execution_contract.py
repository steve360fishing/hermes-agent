"""Request-local execution policy for simple text-artifact turns.

The policy is deliberately conservative. It recognizes only explicit requests
for a textual artifact and falls back to the normal agent for anything that
also asks Hermes to research, mutate state, render, execute, or publish.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.parse import urlparse, urlunparse

from agent.file_safety import get_safe_write_roots, is_write_denied


NORMAL = "normal"
ARTIFACT_ONLY = "artifact_only"
POLICY_VERSION = "artifact-only-v2"

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
    artifact_root: str = ""
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


def build_task_execution_contract(message: Any, *, task_id: str) -> TaskExecutionContract:
    text = message if isinstance(message, str) else str(message or "")
    correlation_id = hashlib.sha256(str(task_id).encode("utf-8")).hexdigest()[:16]
    affirmative_text = _NEGATED_CONSTRAINT.sub("", text)
    file_requested, requested_filename, requested_extension = _requested_artifact_file(
        affirmative_text
    )
    lane, reason = _classify(text, file_requested=file_requested)
    output_path = ""
    artifact_root = ""
    artifact_route = "none"
    artifact_filename = ""
    artifact_mime_type = ""
    preflight_error = ""
    if lane == ARTIFACT_ONLY and file_requested:
        artifact_filename = _safe_artifact_filename(
            requested_filename, requested_extension, correlation_id
        )
        requested_extension = os.path.splitext(artifact_filename)[1].lower()
        artifact_mime_type = _SUPPORTED_ARTIFACT_TYPES[requested_extension]
        prepared = _prepare_artifact_output(artifact_filename, correlation_id)
        if prepared is None:
            preflight_error = "artifact_output_unavailable"
        else:
            artifact_root, output_path, artifact_route = prepared
    explicit_urls = frozenset(
        filter(None, (_normalized_url(url.rstrip(".,);]")) for url in _URL.findall(text)))
    )
    return TaskExecutionContract(
        lane=lane,
        decision_reason=reason,
        correlation_id=correlation_id,
        artifact_output_path=output_path,
        artifact_root=artifact_root,
        artifact_filename=artifact_filename,
        artifact_extension=requested_extension if file_requested else "",
        artifact_mime_type=artifact_mime_type,
        artifact_route=artifact_route,
        artifact_file_requested=file_requested,
        preflight_error=preflight_error,
        explicit_urls=explicit_urls,
    )


def clear_task_execution_contract(agent: Any) -> None:
    """Clear and expire any policy left by the previous request."""
    contract = getattr(agent, "_task_execution_contract", None)
    if contract is not None:
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
