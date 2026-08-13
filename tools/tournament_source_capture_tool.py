"""Narrow model-facing bridge for registered trusted tournament capture."""

from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
from typing import Any, Mapping
from urllib.parse import urlsplit

from agent.tournament_truth_support import configured_runtime_roots
from tools.registry import registry, tool_error, tool_result


_RUNTIME: object | None = None
_TIMEOUT_SECONDS = 15.0

_SAFE_TCAP_CODES = frozenset({
    "TCAP_POINTER_NOT_CURRENT",
    "TCAP_JOURNAL_INVALID",
    "TCAP_SOURCE_NOT_REGISTERED",
    "TCAP_SOURCE_MAP_INVALID",
    "TCAP_DEDUP_CONFLICT",
    "TCAP_SIZE_LIMIT",
    "TCAP_REDIRECT_LIMIT",
    "TCAP_REDIRECT_INVALID",
    "TCAP_REDIRECT_DOWNGRADE",
    "TCAP_HTTP_STATUS",
    "TCAP_CONTENT_TYPE",
    "TCAP_SNAPSHOT_VERIFY_FAILED",
    "TCAP_SNAPSHOT_ROOT_SYMLINK",
    "TCAP_URL_CREDENTIALS_FORBIDDEN",
    "TCAP_HTTPS_REQUIRED",
    "TCAP_HOST_AUTHORITY",
    "TCAP_DNS_EMPTY",
    "TCAP_DNS_INVALID",
    "TCAP_SSRF_PRIVATE_ADDRESS",
    "TCAP_SYMLINK_ESCAPE",
    "TCAP_PATH_ESCAPE",
    "TCAP_DNS_RESOLUTION_FAILED",
    "TCAP_TRANSPORT_FAILED",
    "TCAP_CAPTURE_INVALID",
})


def _capture_rejection_code(exc: BaseException) -> str:
    """Return one approved, target-free capture-rejection code."""
    detail = str(exc)
    code = detail.split(":", 1)[0]
    if code in _SAFE_TCAP_CODES:
        return code
    if code.startswith("TCAP_"):
        return "TCAP_CAPTURE_INVALID"
    lowered = detail.casefold()
    if "symlink" in lowered:
        return "TCAP_SYMLINK_ESCAPE"
    if "escape" in lowered or "outside" in lowered or "contain" in lowered:
        return "TCAP_PATH_ESCAPE"
    if "too large" in lowered or "exceeds" in lowered or "size limit" in lowered:
        return "TCAP_SIZE_LIMIT"
    if "credential" in lowered or "userinfo" in lowered:
        return "TCAP_URL_CREDENTIALS_FORBIDDEN"
    if "https" in lowered or "file scheme" in lowered:
        return "TCAP_HTTPS_REQUIRED"
    if "redirect" in lowered:
        return "TCAP_REDIRECT_INVALID"
    if "content" in lowered or "media type" in lowered:
        return "TCAP_CONTENT_TYPE"
    if "private" in lowered or "metadata" in lowered or "link-local" in lowered or "rfc1918" in lowered:
        return "TCAP_SSRF_PRIVATE_ADDRESS"
    if "invalid address" in lowered:
        return "TCAP_DNS_INVALID"
    if "resolve" in lowered or "dns" in lowered or "address" in lowered:
        return "TCAP_DNS_RESOLUTION_FAILED"
    if isinstance(exc, OSError):
        return "TCAP_TRANSPORT_FAILED"
    return "TCAP_CAPTURE_INVALID"


def install_tournament_capture_runtime(runtime: object) -> None:
    """Install the runtime-owned audit transport capability during bootstrap."""
    global _RUNTIME
    _RUNTIME = runtime


def _resolve_public(host: str) -> tuple[str, ...]:
    """Resolve a registered host and reject non-public destinations."""
    try:
        rows = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError("TCAP_DNS_RESOLUTION_FAILED") from exc
    values = tuple(sorted({row[4][0] for row in rows}))
    if not values:
        raise ValueError("TCAP_DNS_EMPTY")
    if any(not ipaddress.ip_address(value).is_global for value in values):
        raise ValueError("TCAP_SSRF_PRIVATE_ADDRESS")
    return values


def _transport(url: str, *, allowed_host: str, resolved_ips: tuple[str, ...], max_bytes: int):
    """Fetch exactly one TLS-pinned hop; audit code alone decides redirects."""
    parsed = urlsplit(url)
    if parsed.username or parsed.password:
        raise ValueError("TCAP_URL_CREDENTIALS_FORBIDDEN")
    if parsed.scheme != "https":
        raise ValueError("TCAP_HTTPS_REQUIRED")
    if not parsed.hostname or parsed.hostname.casefold() != allowed_host.casefold():
        raise ValueError("TCAP_HOST_AUTHORITY")
    if not resolved_ips:
        raise ValueError("TCAP_DNS_EMPTY")
    try:
        ip = str(ipaddress.ip_address(resolved_ips[0]))
    except ValueError as exc:
        raise ValueError("TCAP_DNS_INVALID") from exc
    target = (parsed.path or "/") + (("?" + parsed.query) if parsed.query else "")
    raw = socket.create_connection((ip, parsed.port or 443), timeout=_TIMEOUT_SECONDS)
    try:
        context = ssl.create_default_context()
        with context.wrap_socket(raw, server_hostname=parsed.hostname) as secure:
            secure.settimeout(_TIMEOUT_SECONDS)
            request = (
                f"GET {target} HTTP/1.1\r\nHost: {parsed.hostname}\r\n"
                "User-Agent: HermesTournamentCapture/1\r\nAccept-Encoding: identity\r\nConnection: close\r\n\r\n"
            ).encode("ascii")
            secure.sendall(request)
            response = http.client.HTTPResponse(secure)
            response.begin()
            body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise ValueError("TCAP_SIZE_LIMIT")
            return response.status, dict(response.getheaders()), body, url
    except Exception:
        raw.close()
        raise


def _runtime() -> object:
    global _RUNTIME
    if _RUNTIME is None:
        from audit_agent.tournament_trusted_capture import TrustedCaptureRuntime

        _RUNTIME = TrustedCaptureRuntime(transport=_transport, resolver=_resolve_public)
    return _RUNTIME


def run_tournament_source_capture(args: Mapping[str, Any], **_kwargs: Any) -> str:
    """Capture only an audit-journal registered source ID; never accepts URLs."""
    source_id = args.get("source_id")
    if not isinstance(source_id, str) or not source_id.strip():
        return tool_error("source_id must name a registered tournament source.", code="capture_source_id_required")
    roots = configured_runtime_roots()
    if roots is None:
        return tool_error("Trusted capture runtime is unavailable.", code="trusted_capture_unavailable")
    try:
        from audit_agent.tournament_trusted_capture import (
            TrustedCaptureError,
            capture_registered_source,
        )
    except ImportError:
        return tool_error("Trusted capture runtime is unavailable.", code="trusted_capture_unavailable")
    try:
        manifest = capture_registered_source(
            source_id=source_id.strip(),
            journal_pointer=roots.journal_root / "LATEST-JOURNAL.json",
            approved_journal_root=roots.journal_root,
            approved_snapshot_root=roots.source_snapshot_root,
            runtime=_runtime(),
        )
    except TrustedCaptureError as exc:
        return tool_error("Trusted tournament capture was rejected.", code=_capture_rejection_code(exc))
    except (OSError, TypeError, ValueError) as exc:
        return tool_error("Trusted tournament capture was rejected.", code=_capture_rejection_code(exc))
    return tool_result(accepted=True, advisory=False, code="captured", evidence_manifest=[manifest])


TOURNAMENT_SOURCE_CAPTURE_SCHEMA = {
    "name": "tournament_source_capture",
    "description": "Capture one registered tournament source into trusted evidence. This mutates private evidence only; source_id is the only accepted selector.",
    "parameters": {"type": "object", "properties": {"source_id": {"type": "string"}}, "required": ["source_id"], "additionalProperties": False},
}


registry.register(
    name="tournament_source_capture", toolset="sportfish", schema=TOURNAMENT_SOURCE_CAPTURE_SCHEMA,
    handler=lambda args, **kw: run_tournament_source_capture(args, **kw), emoji="🧭", max_result_size_chars=8_000,
)
