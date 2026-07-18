"""Fail-closed, source-only primitives for Hermes Rescue Plane V1.

This module intentionally has no provider, gateway, or rescue-action imports.
It is safe to import from the request path and from the independent reporter.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
import threading
import time
import uuid
from typing import Any, Iterator, Mapping


SAFE_MODE_OVERLAY_PATH = Path("/var/run/hermes-rescue/safe-mode-v1.json")
TURN_TELEMETRY_PATH = Path("/run/hermes-rescue/turn-telemetry-v1.json")
QUIESCENCE_SCHEMA_VERSION = "hermes-quiescence-snapshot-v1"
SAFE_MODE_SCHEMA_VERSION = "hermes-safe-mode-v1"
FIXTURE_ID = "HERMES-ARTIFACT-STICKY-20260717-v1"
FIXTURE_SHA256 = "d8d190a659e77d714b555dfbccf49f15222171db8bccaa37d38d31373c81f420"
_OVERLAY_KEYS = frozenset({"schema_version", "enabled", "incident_id", "issued_at", "disables"})


@dataclass(frozen=True)
class SafeModeOverlay:
    valid: bool
    disables: frozenset[str]
    reason: str

    @property
    def disables_artifact_only(self) -> bool:
        return not self.valid or "artifact_only" in self.disables


def _invalid_overlay(reason: str) -> SafeModeOverlay:
    return SafeModeOverlay(False, frozenset({"artifact_only"}), f"invalid_{reason}")


def read_safe_mode_overlay(path: Path | None = None) -> SafeModeOverlay:
    """Read one exact, regular, non-writable-by-others rescue overlay.

    A present but untrustworthy file disables artifact-only.  An absent file is
    deliberately neutral so safe mode never becomes sticky across turns.
    """
    path = SAFE_MODE_OVERLAY_PATH if path is None else path
    try:
        parent = path.parent
        if parent.is_symlink() or path.is_symlink():
            return _invalid_overlay("symlink")
        info = path.lstat()
    except FileNotFoundError:
        return SafeModeOverlay(True, frozenset(), "absent")
    except OSError:
        return _invalid_overlay("unreadable")
    if not stat.S_ISREG(info.st_mode):
        return _invalid_overlay("not_regular")
    # NTFS permission bits are not POSIX modes. The deployed Linux container
    # requires the exact mode; Windows remains testable without weakening it.
    if os.name != "nt" and stat.S_IMODE(info.st_mode) != 0o640:
        return _invalid_overlay("mode")
    if info.st_size > 4096:
        return _invalid_overlay("too_large")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return _invalid_overlay("json")
    if not isinstance(payload, dict) or set(payload) != _OVERLAY_KEYS:
        return _invalid_overlay("schema")
    if payload.get("schema_version") != SAFE_MODE_SCHEMA_VERSION or payload.get("enabled") is not True:
        return _invalid_overlay("schema")
    try:
        parsed = uuid.UUID(str(payload.get("incident_id", "")))
    except (ValueError, AttributeError):
        return _invalid_overlay("incident_id")
    if str(parsed) != str(payload["incident_id"]).lower():
        return _invalid_overlay("incident_id")
    issued_at = payload.get("issued_at")
    if not isinstance(issued_at, str) or not issued_at.endswith("Z"):
        return _invalid_overlay("issued_at")
    try:
        from datetime import datetime
        datetime.fromisoformat(issued_at[:-1] + "+00:00")
    except ValueError:
        return _invalid_overlay("issued_at")
    if payload.get("disables") != ["artifact_only"]:
        return _invalid_overlay("disables")
    return SafeModeOverlay(True, frozenset({"artifact_only"}), "enabled")


def rescue_overlay_disables_artifact_only() -> bool:
    return read_safe_mode_overlay().disables_artifact_only


def load_sticky_artifact_fixture(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FIXTURE_SHA256:
        raise ValueError("rescue fixture hash mismatch")
    payload = json.loads(raw)
    if payload.get("fixture_id") != FIXTURE_ID:
        raise ValueError("rescue fixture id mismatch")
    return payload


def _canonical_snapshot(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


@dataclass(frozen=True)
class QuiescenceValidation:
    valid: bool
    reason: str


def validate_quiescence_snapshot(
    snapshot: Mapping[str, Any] | None,
    *,
    keys: Mapping[str, bytes],
    now: float | None = None,
    seen_sequences: dict[str, int] | None = None,
) -> QuiescenceValidation:
    if not isinstance(snapshot, Mapping):
        return QuiescenceValidation(False, "missing")
    required = {
        "schema_version", "key_id", "sequence", "timestamp", "gateway_pid", "gateway_state",
        "active_turn_count", "active_tool_count", "active_provider_action_count", "source_sha", "image_id", "signature",
    }
    if set(snapshot) != required or snapshot.get("schema_version") != QUIESCENCE_SCHEMA_VERSION:
        return QuiescenceValidation(False, "malformed")
    key_id = snapshot.get("key_id")
    key = keys.get(key_id) if isinstance(key_id, str) else None
    if not isinstance(key, bytes) or len(key) != 32:
        return QuiescenceValidation(False, "unknown_key")
    signature = snapshot.get("signature")
    unsigned = {name: value for name, value in snapshot.items() if name != "signature"}
    expected = hmac.new(key, _canonical_snapshot(unsigned), hashlib.sha256).hexdigest()
    if not isinstance(signature, str) or not hmac.compare_digest(signature, expected):
        return QuiescenceValidation(False, "invalid_signature")
    try:
        sequence = int(snapshot["sequence"])
        timestamp = float(snapshot["timestamp"])
        counts = [int(snapshot[name]) for name in ("active_turn_count", "active_tool_count", "active_provider_action_count")]
    except (TypeError, ValueError):
        return QuiescenceValidation(False, "malformed")
    if sequence < 1 or any(count < 0 for count in counts):
        return QuiescenceValidation(False, "malformed")
    current = time.time() if now is None else now
    if timestamp > current + 1 or current - timestamp >= 15:
        return QuiescenceValidation(False, "stale")
    if seen_sequences is not None:
        prior = seen_sequences.get(key_id, 0)
        if sequence <= prior:
            return QuiescenceValidation(False, "replayed")
        seen_sequences[key_id] = sequence
    return QuiescenceValidation(True, "valid")


class RescueExecutionTelemetry:
    """Thread-safe active-work accounting shared by turn, tool, and provider paths."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._turns: dict[str, dict[str, Any]] = {}
        self._active_turns: set[str] = set()
        self._active_tools = 0
        self._active_provider_actions = 0
        self._sequence = 0

    def start_turn(self, turn_id: str, *, lane: str, artifact_requested: bool, now: float | None = None) -> None:
        with self._lock:
            self._turns[turn_id] = {"turn_id": turn_id, "lane": lane, "artifact_requested": artifact_requested, "started_at": time.time() if now is None else now}
            self._active_turns.add(turn_id)

    def finish_turn(self, turn_id: str, *, now: float | None = None) -> None:
        with self._lock:
            if turn_id in self._turns:
                self._turns[turn_id]["completed_at"] = time.time() if now is None else now
            self._active_turns.discard(turn_id)

    @contextmanager
    def active_tool(self) -> Iterator[None]:
        with self._lock:
            self._active_tools += 1
        try:
            yield
        finally:
            with self._lock:
                self._active_tools = max(0, self._active_tools - 1)

    @contextmanager
    def active_provider_action(self) -> Iterator[None]:
        with self._lock:
            self._active_provider_actions += 1
        try:
            yield
        finally:
            with self._lock:
                self._active_provider_actions = max(0, self._active_provider_actions - 1)

    def active_counts(self) -> tuple[int, int, int]:
        with self._lock:
            return len(self._active_turns), self._active_tools, self._active_provider_actions

    def turn_record(self, turn_id: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._turns[turn_id])

    def _state(self) -> dict[str, Any]:
        turns = sorted((dict(record) for record in self._turns.values()), key=lambda item: item["turn_id"])
        active_turns, active_tools, active_providers = self.active_counts()
        return {"schema_version": "hermes-rescue-turn-telemetry-v1", "turns": turns, "active_turn_count": active_turns, "active_tool_count": active_tools, "active_provider_action_count": active_providers}

    def write(self, path: Path = TURN_TELEMETRY_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(_canonical_snapshot(self._state()))
        os.replace(temporary, path)

    def quiescence_snapshot(self, *, key: bytes, key_id: str, gateway_pid: int | None, gateway_state: str, source_sha: str, image_id: str, now: float | None = None) -> dict[str, Any]:
        if len(key) != 32:
            raise ValueError("quiescence key must be 32 bytes")
        with self._lock:
            self._sequence += 1
            active_turns, active_tools, active_providers = self.active_counts()
            unsigned: dict[str, Any] = {"schema_version": QUIESCENCE_SCHEMA_VERSION, "key_id": key_id, "sequence": self._sequence, "timestamp": time.time() if now is None else now, "gateway_pid": gateway_pid, "gateway_state": gateway_state, "active_turn_count": active_turns, "active_tool_count": active_tools, "active_provider_action_count": active_providers, "source_sha": source_sha, "image_id": image_id}
        unsigned["signature"] = hmac.new(key, _canonical_snapshot(unsigned), hashlib.sha256).hexdigest()
        return unsigned


GLOBAL_RESCUE_TELEMETRY = RescueExecutionTelemetry()
