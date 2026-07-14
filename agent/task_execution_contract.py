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


NORMAL = "normal"
ARTIFACT_ONLY = "artifact_only"
POLICY_VERSION = "artifact-only-v1"

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
    r"\b(?:do\s+not|don't|without)\b[^.;,\n]*",
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

    @property
    def system_guidance(self) -> str:
        if self.lane != ARTIFACT_ONLY:
            return ""
        return (
            "REQUEST EXECUTION CONTRACT (artifact_only, fail closed):\n"
            "Return the requested textual artifact directly. Do not search prior "
            "sessions, reconcile records, render/export media, delegate, consult a "
            "council, call image generation, run code or shell commands, or modify "
            "state. Use no tools unless the user supplied an explicit HTTPS URL that "
            "must be opened once. If a file artifact is explicitly necessary, the "
            f"only permitted destination is {self.artifact_output_path}. Prefer the "
            "chat response and stop as soon as the artifact is complete."
        )

    def before_tool(self, tool_name: str, args: Mapping[str, Any] | None) -> ToolAuthorization:
        if self.lane != ARTIFACT_ONLY:
            return ToolAuthorization(True)

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
                requested = os.path.normcase(os.path.abspath(str(args.get("path") or "")))
                allowed = os.path.normcase(os.path.abspath(self.artifact_output_path))
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
            requested = os.path.normcase(os.path.abspath(str(args.get("path") or "")))
            allowed = os.path.normcase(os.path.abspath(self.artifact_output_path))
            if not write_budget_exhausted and requested == allowed:
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


def build_task_execution_contract(message: Any, *, task_id: str) -> TaskExecutionContract:
    text = message if isinstance(message, str) else str(message or "")
    lane, reason = _classify(text)
    correlation_id = hashlib.sha256(str(task_id).encode("utf-8")).hexdigest()[:16]
    output_path = os.path.join(tempfile.gettempdir(), "hermes-artifacts", f"{correlation_id}.md")
    explicit_urls = frozenset(
        filter(None, (_normalized_url(url.rstrip(".,);]")) for url in _URL.findall(text)))
    )
    return TaskExecutionContract(
        lane=lane,
        decision_reason=reason,
        correlation_id=correlation_id,
        artifact_output_path=output_path,
        explicit_urls=explicit_urls,
    )


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


def _classify(text: str) -> tuple[str, str]:
    if not text.strip():
        return NORMAL, "empty_request"
    affirmative_text = _NEGATED_CONSTRAINT.sub("", text)
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
