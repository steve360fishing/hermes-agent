"""Independent, reporter-owned rescue telemetry aggregation and signing."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import socket
import stat
import struct
import threading
import time
from typing import Any, Callable, Mapping

import psutil

from agent.rescue_plane_core import (
    EVENT_SCHEMA_VERSION,
    KeyRing,
    KeySlot,
    QUIESCENCE_SCHEMA_VERSION,
    atomic_write_secure,
    _secure_read,
    read_hmac_key,
    recover_snapshot_sequence,
    sign_quiescence_snapshot,
)


_EVENT_BASE = frozenset({"schema_version", "event", "event_id", "turn_id"})
_EVENT_WORK = _EVENT_BASE | {"work_id"}
_EVENT_TURN_START = _EVENT_BASE | {"lane", "artifact_requested"}
_POLICY_RETENTION_SECONDS = 120.0
_CONTINUITY_INITIALIZED = b"hermes-rescue-continuity-initialized-v1"
_RECOVERY_STABLE_EMISSIONS = 3
_RECOVERY_MAX_TTL_SECONDS = 300.0
_RECOVERY_ID_CAPACITY = 256
_RECOVERABLE_DEGRADATION_REASONS = frozenset({"continuity_gap", "legacy_unattributed"})
_DEGRADATION_REASONS = frozenset({
    "active_work_bound",
    "background_unknown",
    "capacity_exhausted",
    "continuity_gap",
    "gateway_unknown",
    "legacy_unattributed",
    "recovery_ledger_exhausted",
    "sequence_exhausted",
    "worker_crash",
})


def _posix_uid() -> int:
    """Return the current POSIX uid or reject an unsupported identity API."""
    getuid = getattr(os, "getuid", None)
    if not callable(getuid):
        raise PermissionError("rescue reporter requires POSIX uid support")
    uid = getuid()
    if type(uid) is not int or uid < 0:
        raise PermissionError("invalid POSIX uid")
    return uid


class ReporterCapacityExhausted(ValueError):
    """Reporter cannot retain another turn without violating policy history."""

    def __init__(self, message: str, *, reason: str = "capacity_exhausted") -> None:
        super().__init__(message)
        self.reason = reason


def _bounded_text(value: Any, *, limit: int = 256) -> bool:
    return type(value) is str and 1 <= len(value) <= limit


def _strict_json_object(raw: bytes | str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    parsed = json.loads(raw, object_pairs_hook=reject_duplicates)
    if type(parsed) is not dict:
        raise ValueError("JSON payload must be an object")
    return parsed


class ReporterState:
    """Reporter-owned aggregate built from authenticated process events."""

    def __init__(self, *, max_turns: int = 256, max_work: int = 4096) -> None:
        if type(max_turns) is not int or not 1 <= max_turns <= 256:
            raise ValueError("invalid turn bound")
        if type(max_work) is not int or not 1 <= max_work <= 100_000:
            raise ValueError("invalid work bound")
        self.max_turns = max_turns
        self.max_work = max_work
        self.turns: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.tools: dict[str, dict[str, Any]] = {}
        self.providers: dict[str, dict[str, Any]] = {}
        self.backgrounds: dict[str, dict[str, Any]] = {}
        self.seen_events: OrderedDict[str, None] = OrderedDict()
        self.telemetry_health = "healthy"
        self.reconciled_crashes = 0
        self.event_generation = 0
        self.degradation_reasons: set[str] = set()
        self.recovery_idle_emissions = 0
        self.consumed_recovery_ids: list[str] = []

    def degrade(self, reason: str) -> None:
        if reason not in _DEGRADATION_REASONS:
            raise ValueError("unknown degradation reason")
        self.telemetry_health = "degraded"
        self.degradation_reasons.add(reason)
        self.recovery_idle_emissions = 0

    def note_activity(self) -> None:
        self.recovery_idle_emissions = 0

    def recover(self, authorization_id: str | None = None) -> None:
        if self.telemetry_health != "degraded":
            raise ValueError("reporter is not degraded")
        if not self.degradation_reasons or not self.degradation_reasons.issubset(
            _RECOVERABLE_DEGRADATION_REASONS
        ):
            raise ValueError("reporter degradation is not recoverable")
        if authorization_id is not None:
            if authorization_id in self.consumed_recovery_ids:
                raise ValueError("recovery authorization already consumed")
            if len(self.consumed_recovery_ids) >= _RECOVERY_ID_CAPACITY:
                self.degrade("recovery_ledger_exhausted")
                raise ReporterCapacityExhausted(
                    "recovery authorization ledger exhausted"
                )
            self.consumed_recovery_ids.append(authorization_id)
        self.telemetry_health = "healthy"
        self.degradation_reasons.clear()
        self.recovery_idle_emissions = 0

    def _validate_event(self, event: Mapping[str, Any]) -> str:
        if type(event) is not dict:
            raise ValueError("event must be an object")
        kind = event.get("event")
        if kind == "turn_start":
            expected = _EVENT_TURN_START
        elif kind in {"turn_end"}:
            expected = _EVENT_BASE
        elif kind in {
            "tool_start",
            "tool_end",
            "provider_start",
            "provider_end",
            "background_start",
            "background_end",
            "background_unknown",
        }:
            expected = _EVENT_WORK
        else:
            raise ValueError("unknown event")
        if (
            set(event) != expected
            or event.get("schema_version") != EVENT_SCHEMA_VERSION
        ):
            raise ValueError("malformed event")
        if not _bounded_text(event.get("event_id"), limit=64):
            raise ValueError("invalid event id")
        if not _bounded_text(event.get("turn_id")):
            raise ValueError("invalid turn id")
        if "work_id" in event and not _bounded_text(event["work_id"], limit=64):
            raise ValueError("invalid work id")
        if kind == "turn_start":
            if event.get("lane") not in {"normal", "artifact_only"}:
                raise ValueError("invalid lane")
            if type(event.get("artifact_requested")) is not bool:
                raise ValueError("invalid artifact flag")
        return str(kind)

    def apply_event(
        self,
        event: Mapping[str, Any],
        *,
        peer_pid: int,
        peer_uid: int,
        now: float,
    ) -> bool:
        kind = self._validate_event(event)
        if (
            type(peer_pid) is not int
            or peer_pid <= 0
            or type(peer_uid) is not int
            or peer_uid < 0
        ):
            raise ValueError("invalid peer credentials")
        if type(now) not in {int, float} or not math.isfinite(float(now)):
            raise ValueError("invalid event time")
        event_id = str(event["event_id"])
        if event_id in self.seen_events:
            return False
        if self.event_generation >= (2**63 - 1):
            self.degrade("sequence_exhausted")
            raise ReporterCapacityExhausted(
                "event generation exhausted",
                reason="sequence_exhausted",
            )

        turn_id = str(event["turn_id"])
        if kind == "turn_start":
            self._prune(float(now))
            existing = self.turns.get(turn_id)
            if existing and existing["completed_at"] is None:
                raise ValueError("duplicate active turn")
            if existing:
                raise ValueError("retained turn id cannot be reused")
            if len(self.turns) >= self.max_turns:
                self.degrade("capacity_exhausted")
                raise ReporterCapacityExhausted("policy evidence capacity exhausted")
            self.turns[turn_id] = {
                "turn_id": turn_id,
                "lane": event["lane"],
                "artifact_requested": event["artifact_requested"],
                "started_at": float(now),
                "completed_at": None,
                "peer_pid": peer_pid,
                "peer_uid": peer_uid,
            }
            self.turns.move_to_end(turn_id)
        elif kind == "turn_end":
            turn = self.turns.get(turn_id)
            if (
                not turn
                or turn["completed_at"] is not None
                or turn["peer_pid"] != peer_pid
            ):
                raise ValueError("turn end does not match active owner")
            turn["completed_at"] = float(now)
        else:
            work_id = str(event["work_id"])
            if kind.startswith("tool_"):
                collection = self.tools
            elif kind.startswith("provider_"):
                collection = self.providers
            else:
                collection = self.backgrounds
            if kind.endswith("_start"):
                if work_id in collection:
                    raise ValueError("duplicate active work")
                if (
                    len(self.tools) + len(self.providers) + len(self.backgrounds)
                    >= self.max_work
                ):
                    self.degrade("active_work_bound")
                    raise ReporterCapacityExhausted(
                        "active work bound exceeded",
                        reason="active_work_bound",
                    )
                collection[work_id] = {
                    "turn_id": turn_id,
                    "peer_pid": peer_pid,
                    "peer_uid": peer_uid,
                    "started_at": float(now),
                }
            elif kind.endswith("_end"):
                active = collection.get(work_id)
                if (
                    not active
                    or active["peer_pid"] != peer_pid
                    or active["turn_id"] != turn_id
                ):
                    raise ValueError("work end does not match active owner")
                del collection[work_id]
            else:
                active = collection.get(work_id)
                if (
                    not active
                    or active["peer_pid"] != peer_pid
                    or active["turn_id"] != turn_id
                ):
                    raise ValueError("work uncertainty does not match active owner")
                self.degrade("background_unknown")
        self.seen_events[event_id] = None
        self.event_generation += 1
        while len(self.seen_events) > max(1024, self.max_work * 2):
            self.seen_events.popitem(last=False)
        self._prune(float(now))
        return True

    def _prune(self, now: float) -> None:
        expired = [
            turn_id
            for turn_id, turn in self.turns.items()
            if turn["completed_at"] is not None
            and now - float(turn["completed_at"]) >= _POLICY_RETENTION_SECONDS
        ]
        for turn_id in expired:
            self.turns.pop(turn_id, None)

    def reconcile(self, is_pid_alive: Callable[[int], bool], *, now: float) -> None:
        active_pids = {
            int(turn["peer_pid"])
            for turn in self.turns.values()
            if turn["completed_at"] is None
        }
        active_pids.update(int(item["peer_pid"]) for item in self.tools.values())
        active_pids.update(int(item["peer_pid"]) for item in self.providers.values())
        active_pids.update(int(item["peer_pid"]) for item in self.backgrounds.values())
        dead = {pid for pid in active_pids if not is_pid_alive(pid)}
        if dead:
            newly_degraded = self.telemetry_health != "degraded"
            self.degrade("worker_crash")
            if newly_degraded:
                self.reconciled_crashes = min(
                    100_000, self.reconciled_crashes + len(dead)
                )
        # Deliberately retain dead-worker records: absence of an authenticated
        # end event cannot become an idle authorization after a crash.

    def active_counts(self) -> tuple[int, int, int]:
        return (
            sum(turn["completed_at"] is None for turn in self.turns.values()),
            len(self.tools) + len(self.backgrounds),
            len(self.providers),
        )

    def policy_evidence(self) -> list[dict[str, Any]]:
        return [
            {
                "turn_id": turn["turn_id"],
                "lane": turn["lane"],
                "artifact_requested": turn["artifact_requested"],
                "started_at": turn["started_at"],
                "completed_at": turn["completed_at"],
            }
            for turn in self.turns.values()
        ]

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "hermes-rescue-reporter-state-v2",
            "max_turns": self.max_turns,
            "max_work": self.max_work,
            "turns": list(self.turns.values()),
            "tools": self.tools,
            "providers": self.providers,
            "backgrounds": self.backgrounds,
            "seen_events": list(self.seen_events),
            "telemetry_health": self.telemetry_health,
            "reconciled_crashes": self.reconciled_crashes,
            "event_generation": self.event_generation,
            "degradation_reasons": sorted(self.degradation_reasons),
            "recovery_idle_emissions": self.recovery_idle_emissions,
            "consumed_recovery_ids": self.consumed_recovery_ids,
        }

    def recovery_subject_payload(self) -> dict[str, Any]:
        return {
            "max_turns": self.max_turns,
            "max_work": self.max_work,
            "turns": list(self.turns.values()),
            "tools": self.tools,
            "providers": self.providers,
            "backgrounds": self.backgrounds,
            "seen_events": list(self.seen_events),
            "reconciled_crashes": self.reconciled_crashes,
            "event_generation": self.event_generation,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ReporterState":
        expected_v1 = {
            "schema_version",
            "max_turns",
            "max_work",
            "turns",
            "tools",
            "providers",
            "seen_events",
            "telemetry_health",
            "reconciled_crashes",
        }
        expected_v2 = expected_v1 | {
            "backgrounds",
            "degradation_reasons",
            "recovery_idle_emissions",
            "consumed_recovery_ids",
            "event_generation",
        }
        if type(payload) is not dict:
            raise ValueError("corrupt reporter state")
        is_v1 = payload.get("schema_version") == "hermes-rescue-reporter-state-v1"
        if is_v1:
            if frozenset(payload) not in {
                frozenset(expected_v1),
                frozenset(expected_v1 | {"backgrounds"}),
            }:
                raise ValueError("corrupt reporter state")
        elif payload.get(
            "schema_version"
        ) != "hermes-rescue-reporter-state-v2" or frozenset(payload) != frozenset(
            expected_v2
        ):
            raise ValueError("corrupt reporter state")
        state = cls(max_turns=payload["max_turns"], max_work=payload["max_work"])
        if payload["telemetry_health"] not in {"healthy", "degraded"}:
            raise ValueError("corrupt reporter health")
        if (
            type(payload["reconciled_crashes"]) is not int
            or not 0 <= payload["reconciled_crashes"] <= 100_000
        ):
            raise ValueError("corrupt crash count")
        turns = payload["turns"]
        if type(turns) is not list or len(turns) > state.max_turns:
            raise ValueError("corrupt turn state")
        for turn in turns:
            if type(turn) is not dict or set(turn) != {
                "turn_id",
                "lane",
                "artifact_requested",
                "started_at",
                "completed_at",
                "peer_pid",
                "peer_uid",
            }:
                raise ValueError("corrupt turn state")
            if (
                not _bounded_text(turn["turn_id"])
                or turn["lane"] not in {"normal", "artifact_only"}
                or type(turn["artifact_requested"]) is not bool
                or type(turn["peer_pid"]) is not int
                or turn["peer_pid"] <= 0
                or type(turn["peer_uid"]) is not int
                or turn["peer_uid"] < 0
                or type(turn["started_at"]) not in {int, float}
                or not math.isfinite(float(turn["started_at"]))
                or (
                    turn["completed_at"] is not None
                    and (
                        type(turn["completed_at"]) not in {int, float}
                        or not math.isfinite(float(turn["completed_at"]))
                    )
                )
            ):
                raise ValueError("corrupt turn state")
            state.turns[turn["turn_id"]] = dict(turn)
        for name, destination in (
            ("tools", state.tools),
            ("providers", state.providers),
            ("backgrounds", state.backgrounds),
        ):
            values = payload.get(name, {})
            if type(values) is not dict or len(values) > state.max_work:
                raise ValueError("corrupt work state")
            for work_id, item in values.items():
                if (
                    not _bounded_text(work_id, limit=64)
                    or type(item) is not dict
                    or set(item) != {"turn_id", "peer_pid", "peer_uid", "started_at"}
                    or not _bounded_text(item["turn_id"])
                    or type(item["peer_pid"]) is not int
                    or item["peer_pid"] <= 0
                    or type(item["peer_uid"]) is not int
                    or item["peer_uid"] < 0
                    or type(item["started_at"]) not in {int, float}
                    or not math.isfinite(float(item["started_at"]))
                ):
                    raise ValueError("corrupt work state")
                destination[work_id] = dict(item)
        if (
            len(state.tools) + len(state.providers) + len(state.backgrounds)
            > state.max_work
        ):
            raise ValueError("corrupt aggregate work state")
        seen = payload["seen_events"]
        if type(seen) is not list or len(seen) > max(1024, state.max_work * 2):
            raise ValueError("corrupt event state")
        for event_id in seen:
            if not _bounded_text(event_id, limit=64):
                raise ValueError("corrupt event state")
            state.seen_events[event_id] = None
        state.telemetry_health = payload["telemetry_health"]
        state.reconciled_crashes = payload["reconciled_crashes"]
        if is_v1:
            state.event_generation = len(state.seen_events)
            if state.telemetry_health == "degraded":
                state.degradation_reasons = {"legacy_unattributed"}
        else:
            reasons = payload["degradation_reasons"]
            idle_emissions = payload["recovery_idle_emissions"]
            consumed = payload["consumed_recovery_ids"]
            if (
                type(reasons) is not list
                or len(reasons) > len(_DEGRADATION_REASONS)
                or any(
                    type(reason) is not str or reason not in _DEGRADATION_REASONS
                    for reason in reasons
                )
                or len(set(reasons)) != len(reasons)
                or type(idle_emissions) is not int
                or not 0 <= idle_emissions < _RECOVERY_STABLE_EMISSIONS
                or type(consumed) is not list
                or len(consumed) > _RECOVERY_ID_CAPACITY
                or any(
                    type(value) is not str
                    or not __import__("re").fullmatch(
                        r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
                        value,
                    )
                    for value in consumed
                )
            ):
                raise ValueError("corrupt reporter recovery state")
            if type(payload["event_generation"]) is not int or not len(
                state.seen_events
            ) <= payload["event_generation"] <= (2**63 - 1):
                raise ValueError("corrupt reporter event generation")
            state.event_generation = payload["event_generation"]
            state.degradation_reasons = set(reasons)
            state.recovery_idle_emissions = idle_emissions
            state.consumed_recovery_ids = list(consumed)
            if (state.telemetry_health == "healthy") != (not reasons):
                raise ValueError("corrupt reporter recovery health")
        return state


def recovery_subject_hash(
    *,
    producer_epoch: str,
    state: ReporterState,
    count_floor: Mapping[str, int],
    policy_evidence_floor: list[dict[str, Any]],
) -> str:
    payload = {
        "schema_version": "hermes-rescue-recovery-subject-v1",
        "producer_epoch": producer_epoch,
        "aggregate": state.recovery_subject_payload(),
        "active_count_floor": dict(count_floor),
        "policy_evidence_floor": policy_evidence_floor,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _read_proc_uid(status: Path) -> int | None:
    try:
        for line in status.read_text(encoding="ascii").splitlines():
            if line.startswith("Uid:"):
                values = line.split()
                return int(values[1])
    except FileNotFoundError:
        return None
    except OSError:
        raise
    except (ValueError, IndexError) as exc:
        raise ValueError("malformed proc status") from exc
    raise ValueError("missing proc uid")


def discover_gateway_state(
    *,
    proc_root: Path = Path("/proc"),
    expected_uid: int,
) -> tuple[list[int], str]:
    pids: list[int] = []
    try:
        children = list(proc_root.iterdir())
    except OSError:
        return [], "unknown"
    for child in children:
        if not child.name.isdigit():
            continue
        try:
            process_uid = _read_proc_uid(child / "status")
        except (OSError, ValueError):
            return [], "unknown"
        if process_uid is None:
            continue
        if process_uid != expected_uid:
            continue
        try:
            raw = (child / "cmdline").read_bytes()
        except FileNotFoundError:
            continue
        except OSError:
            return [], "unknown"
        args = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
        if "gateway" not in args or "run" not in args:
            continue
        if not any("hermes" in arg for arg in args):
            continue
        pids.append(int(child.name))
    pids.sort()
    return pids, "active" if pids else "dead"


def _pid_alive(pid: int) -> bool:
    return type(pid) is int and pid > 0 and psutil.pid_exists(pid)


def acquire_reporter_lock(runtime_dir: Path) -> int:
    if os.name != "posix":
        raise RuntimeError("reporter lock requires POSIX")
    import fcntl

    path = runtime_dir / "reporter.lock"
    fd = os.open(
        path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(fd, 0o600)
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != _posix_uid()
            or stat.S_IMODE(info.st_mode) != 0o600
        ):
            raise PermissionError("insecure reporter lock")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("rescue reporter already running") from exc
        return fd
    except Exception:
        os.close(fd)
        raise


def require_effective_uid(expected_uid: int, *, actual_uid: int | None = None) -> None:
    actual = _posix_uid() if actual_uid is None else actual_uid
    if type(expected_uid) is not int or expected_uid <= 0 or actual != expected_uid:
        raise PermissionError("unexpected rescue reporter effective uid")


class QuiescenceReporter:
    """Owns the Unix socket, durable aggregate, and HMAC snapshot output."""

    def __init__(
        self,
        *,
        runtime_dir: Path,
        continuity_dir: Path,
        keyring: KeyRing,
        source_sha: str,
        image_id: str,
        expected_hermes_uid: int,
        max_turns: int = 256,
        recovery_authorization_path: Path | None = Path(
            "/run/hermes-rescue-secrets/telemetry-recovery-v1.json"
        ),
    ) -> None:
        self.runtime_dir = runtime_dir
        self.continuity_dir = continuity_dir
        self.socket_path = runtime_dir / "events.sock"
        self.output_path = runtime_dir / "quiescence-v1.json"
        self.continuity_state_path = continuity_dir / "continuity-state-v1.json"
        self.continuity_initialized_path = continuity_dir / "continuity-initialized-v1"
        self.state_path = self.continuity_state_path
        self.producer_state_path = self.continuity_state_path
        self.keyring = keyring
        self.source_sha = source_sha
        self.image_id = image_id
        self.expected_hermes_uid = expected_hermes_uid
        self.recovery_authorization_path = recovery_authorization_path
        runtime_info = runtime_dir.lstat()
        if (
            runtime_dir.is_symlink()
            or not stat.S_ISDIR(runtime_info.st_mode)
            or runtime_info.st_uid != _posix_uid()
            or stat.S_IMODE(runtime_info.st_mode) != 0o750
        ):
            raise PermissionError("insecure reporter runtime directory")
        self.runtime_gid = runtime_info.st_gid
        continuity_info = continuity_dir.lstat()
        if (
            continuity_dir.is_symlink()
            or not stat.S_ISDIR(continuity_info.st_mode)
            or continuity_info.st_uid != _posix_uid()
            or continuity_info.st_gid != self.runtime_gid
            or stat.S_IMODE(continuity_info.st_mode) != 0o750
        ):
            raise PermissionError("insecure reporter continuity directory")
        self.continuity_gid = continuity_info.st_gid
        self.count_floor = {
            "turn": 0,
            "tool": 0,
            "provider": 0,
        }
        self.policy_evidence_floor: list[dict[str, Any]] = []
        self.state = ReporterState(max_turns=max_turns)
        self.producer_epoch = ""
        self.sequence = 0
        self._load_or_create_continuity()
        self.recovery_authorization = self._load_recovery_authorization()
        self._lock = threading.RLock()

    def _recovery_subject_hash(self) -> str:
        return recovery_subject_hash(
            producer_epoch=self.producer_epoch,
            state=self.state,
            count_floor=self.count_floor,
            policy_evidence_floor=self.policy_evidence_floor,
        )

    def _load_recovery_authorization(self) -> dict[str, Any] | None:
        if self.recovery_authorization_path is None:
            return None
        try:
            raw = _secure_read(
                self.recovery_authorization_path,
                expected_uid=_posix_uid(),
                expected_gid=self.runtime_gid,
                file_mode=0o400,
                parent_mode=0o750,
                max_bytes=4096,
            )
        except FileNotFoundError:
            return None
        payload = _strict_json_object(raw)
        if (
            set(payload)
            != {
                "schema_version",
                "authorization_id",
                "expected_reason",
                "expected_producer_epoch",
                "expected_state_hash",
                "expected_prior_source_sha",
                "expected_prior_image_id",
                "expected_target_source_sha",
                "expected_target_image_id",
                "issued_at",
                "expires_at",
            }
            or payload["schema_version"] != "hermes-rescue-recovery-authorization-v1"
            or not __import__("re").fullmatch(
                r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
                str(payload["authorization_id"]),
            )
            or payload["expected_reason"] != "legacy_unattributed"
            or not __import__("re").fullmatch(
                r"[0-9a-f]{32}", str(payload["expected_producer_epoch"])
            )
            or not __import__("re").fullmatch(
                r"[0-9a-f]{64}", str(payload["expected_state_hash"])
            )
            or not __import__("re").fullmatch(
                r"[0-9a-f]{40}|[0-9a-f]{64}",
                str(payload["expected_prior_source_sha"]),
            )
            or not __import__("re").fullmatch(
                r"sha256:[0-9a-f]{64}", str(payload["expected_prior_image_id"])
            )
            or not __import__("re").fullmatch(
                r"[0-9a-f]{40}|[0-9a-f]{64}",
                str(payload["expected_target_source_sha"]),
            )
            or not __import__("re").fullmatch(
                r"sha256:[0-9a-f]{64}", str(payload["expected_target_image_id"])
            )
            or type(payload["issued_at"]) not in {int, float}
            or not math.isfinite(float(payload["issued_at"]))
            or type(payload["expires_at"]) not in {int, float}
            or not math.isfinite(float(payload["expires_at"]))
            or not 0
            < float(payload["expires_at"]) - float(payload["issued_at"])
            <= _RECOVERY_MAX_TTL_SECONDS
        ):
            raise ValueError("invalid telemetry recovery authorization")
        if (
            payload["expected_target_source_sha"] != self.source_sha
            or payload["expected_target_image_id"] != self.image_id
            or payload["expected_producer_epoch"] != self.producer_epoch
        ):
            return None
        recover_snapshot_sequence(
            self.output_path,
            keyring=self.keyring,
            expected_uid=_posix_uid(),
            expected_gid=self.runtime_gid,
        )
        snapshot = _strict_json_object(
            _secure_read(
                self.output_path,
                expected_uid=_posix_uid(),
                expected_gid=self.runtime_gid,
                file_mode=0o600,
                parent_mode=0o750,
                max_bytes=2_000_000,
            )
        )
        if (
            snapshot.get("producer_epoch") != self.producer_epoch
            or snapshot.get("source_sha") != payload["expected_prior_source_sha"]
            or snapshot.get("image_id") != payload["expected_prior_image_id"]
            or snapshot.get("telemetry_health") != "degraded"
        ):
            return None
        return payload

    def _load_or_create_continuity(self) -> None:
        try:
            raw = _secure_read(
                self.continuity_state_path,
                expected_uid=_posix_uid(),
                expected_gid=self.continuity_gid,
                file_mode=0o600,
                parent_mode=0o750,
                max_bytes=2_000_000,
            )
        except FileNotFoundError:
            self._recover_or_initialize_continuity()
            return
        payload = _strict_json_object(raw)
        if (
            set(payload)
            != {
                "schema_version",
                "producer_epoch",
                "sequence",
                "aggregate",
                "active_count_floor",
                "policy_evidence_floor",
            }
            or payload["schema_version"] != "hermes-rescue-continuity-state-v1"
            or type(payload["producer_epoch"]) is not str
            or not __import__("re").fullmatch(
                r"[0-9a-f]{32}", payload["producer_epoch"]
            )
            or type(payload["sequence"]) is not int
            or not 0 <= payload["sequence"] <= (2**63 - 1)
        ):
            raise ValueError("corrupt continuity state")
        floor = payload["active_count_floor"]
        if (
            type(floor) is not dict
            or set(floor) != {"turn", "tool", "provider"}
            or any(
                type(value) is not int or not 0 <= value <= 100_000
                for value in floor.values()
            )
            or type(payload["policy_evidence_floor"]) is not list
            or len(payload["policy_evidence_floor"]) > 256
        ):
            raise ValueError("corrupt continuity state")
        self.state = ReporterState.from_payload(payload["aggregate"])
        self.producer_epoch = str(payload["producer_epoch"])
        self.sequence = int(payload["sequence"])
        self.count_floor = dict(floor)
        self.policy_evidence_floor = [
            dict(item) for item in payload["policy_evidence_floor"]
        ]
        self._validate_policy_evidence_floor()
        self._persist_initialized_marker()

    def _recover_or_initialize_continuity(self) -> None:
        initialized = self._continuity_was_initialized()
        try:
            raw = _secure_read(
                self.output_path,
                expected_uid=_posix_uid(),
                expected_gid=self.runtime_gid,
                file_mode=0o600,
                parent_mode=0o750,
                max_bytes=2_000_000,
            )
        except FileNotFoundError:
            if initialized:
                raise ValueError("continuity state missing after initialization")
            self.producer_epoch = secrets.token_hex(16)
            self.sequence = 0
            self._persist_continuity_state()
            self._persist_initialized_marker()
            return
        snapshot = _strict_json_object(raw)
        sequence = recover_snapshot_sequence(
            self.output_path,
            keyring=self.keyring,
            expected_uid=_posix_uid(),
            expected_gid=self.runtime_gid,
        )
        self.producer_epoch = str(snapshot["producer_epoch"])
        self.sequence = sequence
        if (
            snapshot.get("telemetry_health") == "healthy"
            and snapshot.get("active_turn_count") == 0
            and snapshot.get("active_tool_count") == 0
            and snapshot.get("active_provider_action_count") == 0
            and not any(
                item.get("completed_at") is None
                for item in snapshot.get("policy_evidence", [])
                if type(item) is dict
            )
        ):
            self.state.degrade("continuity_gap")
        else:
            self.state.degrade("legacy_unattributed")
        self.count_floor = {
            "turn": int(snapshot["active_turn_count"]),
            "tool": int(snapshot["active_tool_count"]),
            "provider": int(snapshot["active_provider_action_count"]),
        }
        self.policy_evidence_floor = [
            dict(item) for item in snapshot["policy_evidence"]
        ]
        self._validate_policy_evidence_floor()
        self._persist_continuity_state()
        self._persist_initialized_marker()

    def _continuity_was_initialized(self) -> bool:
        try:
            raw = _secure_read(
                self.continuity_initialized_path,
                expected_uid=_posix_uid(),
                expected_gid=self.continuity_gid,
                file_mode=0o400,
                parent_mode=0o750,
                max_bytes=128,
            )
        except FileNotFoundError:
            return False
        if raw != _CONTINUITY_INITIALIZED:
            raise ValueError("corrupt continuity initialization marker")
        return True

    def _persist_initialized_marker(self) -> None:
        if self._continuity_was_initialized():
            return
        atomic_write_secure(
            self.continuity_initialized_path,
            _CONTINUITY_INITIALIZED,
            mode=0o400,
            expected_parent_uid=_posix_uid(),
            expected_parent_gid=self.continuity_gid,
        )

    def _validate_policy_evidence_floor(self) -> None:
        for item in self.policy_evidence_floor:
            if (
                type(item) is not dict
                or set(item)
                != {
                    "turn_id",
                    "lane",
                    "artifact_requested",
                    "started_at",
                    "completed_at",
                }
                or not _bounded_text(item["turn_id"])
                or item["lane"] not in {"normal", "artifact_only"}
                or type(item["artifact_requested"]) is not bool
                or type(item["started_at"]) not in {int, float}
                or not math.isfinite(float(item["started_at"]))
                or (
                    item["completed_at"] is not None
                    and (
                        type(item["completed_at"]) not in {int, float}
                        or not math.isfinite(float(item["completed_at"]))
                    )
                )
            ):
                raise ValueError("corrupt continuity policy evidence")

    def _persist_continuity_state(self) -> None:
        atomic_write_secure(
            self.continuity_state_path,
            json.dumps(
                {
                    "schema_version": "hermes-rescue-continuity-state-v1",
                    "producer_epoch": self.producer_epoch,
                    "sequence": self.sequence,
                    "aggregate": self.state.to_payload(),
                    "active_count_floor": self.count_floor,
                    "policy_evidence_floor": self.policy_evidence_floor,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii"),
            mode=0o600,
            expected_parent_uid=_posix_uid(),
            expected_parent_gid=self.continuity_gid,
        )

    def _persist_producer_state(self) -> None:
        self._persist_continuity_state()

    def _invalidate_snapshot(self) -> None:
        if os.name != "posix":
            self.output_path.unlink(missing_ok=True)
            return
        parent_fd = os.open(
            self.runtime_dir,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            try:
                output_fd = os.open(
                    self.output_path.name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_fd,
                )
            except FileNotFoundError:
                return
            try:
                info = os.fstat(output_fd)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_nlink != 1
                    or info.st_uid != _posix_uid()
                    or info.st_gid != self.runtime_gid
                ):
                    raise PermissionError("unsafe snapshot invalidation target")
            finally:
                os.close(output_fd)
            os.unlink(self.output_path.name, dir_fd=parent_fd)
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)

    def _persist_state(self) -> None:
        self._persist_continuity_state()

    def _maybe_recover(
        self,
        *,
        current: float,
        gateway_state: str,
        active_counts: tuple[int, int, int],
        policy_evidence: list[dict[str, Any]],
    ) -> bool:
        if self.state.telemetry_health != "degraded":
            if self.state.recovery_idle_emissions:
                self.state.recovery_idle_emissions = 0
                return True
            return False
        reasons = self.state.degradation_reasons
        if reasons == {"continuity_gap"}:
            authorization_id: str | None = None
        elif reasons == {"legacy_unattributed"}:
            authorization = self.recovery_authorization
            if (
                authorization is None
                or current < float(authorization["issued_at"])
                or current > float(authorization["expires_at"])
                or authorization["authorization_id"] in self.state.consumed_recovery_ids
                or authorization["expected_state_hash"] != self._recovery_subject_hash()
            ):
                if self.state.recovery_idle_emissions:
                    self.state.recovery_idle_emissions = 0
                    return True
                return False
            authorization_id = str(authorization["authorization_id"])
            if len(self.state.consumed_recovery_ids) >= _RECOVERY_ID_CAPACITY:
                self.state.degrade("recovery_ledger_exhausted")
                return True
        else:
            if self.state.recovery_idle_emissions:
                self.state.recovery_idle_emissions = 0
                return True
            return False
        safe_idle = (
            gateway_state == "active"
            and active_counts == (0, 0, 0)
            and all(value == 0 for value in self.count_floor.values())
            and self.state.reconciled_crashes == 0
            and not any(item["completed_at"] is None for item in policy_evidence)
        )
        if not safe_idle:
            if self.state.recovery_idle_emissions:
                self.state.recovery_idle_emissions = 0
                return True
            return False
        self.state.recovery_idle_emissions += 1
        if self.state.recovery_idle_emissions >= _RECOVERY_STABLE_EMISSIONS:
            self.state.recover(authorization_id)
        return True

    def process_event(
        self,
        payload: bytes,
        *,
        peer_pid: int,
        peer_uid: int,
        now: float | None = None,
    ) -> None:
        if peer_uid != self.expected_hermes_uid:
            raise PermissionError("unauthorized telemetry peer")
        if len(payload) > 8192:
            raise ValueError("event too large")
        event = _strict_json_object(payload)
        event_now = time.time() if now is None else now
        is_start = str(event.get("event", "")).endswith("_start")
        with self._lock:
            prior = ReporterState.from_payload(self.state.to_payload())
            durably_persisted = False
            try:
                changed = self.state.apply_event(
                    event,
                    peer_pid=peer_pid,
                    peer_uid=peer_uid,
                    now=event_now,
                )
                if changed:
                    self.state.note_activity()
                    if is_start:
                        self._invalidate_snapshot()
                    self._persist_state()
                    durably_persisted = True
                    if is_start:
                        self.emit_snapshot(now=float(event_now))
            except ReporterCapacityExhausted as exc:
                self.state = prior
                self.state.degrade(exc.reason)
                self._invalidate_snapshot()
                self._persist_state()
                self.emit_snapshot(now=float(event_now))
                raise
            except Exception:
                if not durably_persisted:
                    self.state = prior
                raise

    def emit_snapshot(self, *, now: float | None = None) -> dict[str, Any]:
        current = time.time() if now is None else now
        with self._lock:
            self.state.reconcile(_pid_alive, now=current)
            self._persist_state()
            gateway_pids, gateway_state = discover_gateway_state(
                expected_uid=self.expected_hermes_uid
            )
            if gateway_state == "unknown":
                self.state.degrade("gateway_unknown")
                self._persist_state()
            if self.sequence >= (2**63 - 1):
                self.state.degrade("sequence_exhausted")
                self._persist_state()
                self._invalidate_snapshot()
                raise OverflowError("producer sequence exhausted")
            active_turns, active_tools, active_providers = self.state.active_counts()
            active_turns = max(active_turns, self.count_floor["turn"])
            active_tools = max(active_tools, self.count_floor["tool"])
            active_providers = max(active_providers, self.count_floor["provider"])
            policy_evidence = list(self.policy_evidence_floor)
            retained_turn_ids = {item["turn_id"] for item in policy_evidence}
            policy_evidence.extend(
                item
                for item in self.state.policy_evidence()
                if item["turn_id"] not in retained_turn_ids
            )
            self.sequence += 1
            self._persist_producer_state()
            unsigned = {
                "schema_version": QUIESCENCE_SCHEMA_VERSION,
                "producer_epoch": self.producer_epoch,
                "sequence": self.sequence,
                "timestamp": current,
                "gateway_pids": gateway_pids,
                "gateway_state": gateway_state,
                "active_turn_count": active_turns,
                "active_tool_count": active_tools,
                "active_provider_action_count": active_providers,
                "source_sha": self.source_sha,
                "image_id": self.image_id,
                "telemetry_health": self.state.telemetry_health,
                "policy_evidence": policy_evidence,
            }
            snapshot = sign_quiescence_snapshot(unsigned, self.keyring, now=current)
            atomic_write_secure(
                self.output_path,
                json.dumps(
                    snapshot,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("ascii"),
                mode=0o600,
                expected_parent_uid=_posix_uid(),
                expected_parent_gid=self.runtime_gid,
            )
            prior_recovery_state = ReporterState.from_payload(self.state.to_payload())
            try:
                if self._maybe_recover(
                    current=current,
                    gateway_state=gateway_state,
                    active_counts=(active_turns, active_tools, active_providers),
                    policy_evidence=policy_evidence,
                ):
                    self._persist_state()
            except Exception:
                self.state = prior_recovery_state
                raise
            return snapshot

    def serve(
        self,
        *,
        interval: float = 10.0,
        stop_event: threading.Event | None = None,
    ) -> None:
        if os.name != "posix" or not hasattr(socket, "SO_PEERCRED"):
            raise RuntimeError("rescue reporter requires Linux peer credentials")
        lock_fd = acquire_reporter_lock(self.runtime_dir)
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o620)
            os.chown(self.socket_path, -1, self.runtime_gid)
            server.listen(128)
            server.settimeout(0.5)
            next_snapshot = time.monotonic()
            while stop_event is None or not stop_event.is_set():
                timeout = max(0.05, min(0.5, next_snapshot - time.monotonic()))
                server.settimeout(timeout)
                try:
                    connection, _ = server.accept()
                except socket.timeout:
                    connection = None
                if connection is not None:
                    with connection:
                        credentials = connection.getsockopt(
                            socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
                        )
                        peer_pid, peer_uid, _peer_gid = struct.unpack("3i", credentials)
                        payload = bytearray()
                        while len(payload) <= 8192:
                            chunk = connection.recv(4096)
                            if not chunk:
                                break
                            payload.extend(chunk)
                        try:
                            self.process_event(
                                bytes(payload),
                                peer_pid=peer_pid,
                                peer_uid=peer_uid,
                            )
                        except (
                            ValueError,
                            PermissionError,
                            json.JSONDecodeError,
                            OSError,
                        ):
                            connection.sendall(b"REJECT\n")
                        else:
                            connection.sendall(b"OK\n")
                if time.monotonic() >= next_snapshot:
                    self.emit_snapshot()
                    next_snapshot = time.monotonic() + interval
        finally:
            server.close()
            os.close(lock_fd)
            try:
                self.socket_path.unlink()
            except FileNotFoundError:
                pass


def _read_identity(
    path: Path,
    pattern: str,
    *,
    expected_uid: int,
    expected_gid: int,
) -> str:
    value = (
        _secure_read(
            path,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
            file_mode=0o400,
            parent_mode=0o750,
            max_bytes=256,
        )
        .decode("ascii")
        .strip()
    )
    if not __import__("re").fullmatch(pattern, value):
        raise ValueError(f"invalid identity file: {path.name}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["serve"])
    parser.add_argument(
        "--runtime-dir", type=Path, default=Path("/run/hermes-rescue-reporter")
    )
    parser.add_argument(
        "--continuity-dir", type=Path, default=Path("/var/lib/hermes-rescue")
    )
    parser.add_argument(
        "--recovery-authorization",
        type=Path,
        default=Path("/run/hermes-rescue-secrets/telemetry-recovery-v1.json"),
    )
    parser.add_argument(
        "--current-key",
        type=Path,
        default=Path("/run/hermes-rescue-secrets/hmac-current"),
    )
    parser.add_argument(
        "--current-key-id-file",
        type=Path,
        default=Path("/run/hermes-rescue-secrets/key-id-current"),
    )
    parser.add_argument(
        "--next-key",
        type=Path,
        default=Path("/run/hermes-rescue-secrets/hmac-next"),
    )
    parser.add_argument(
        "--next-key-id-file",
        type=Path,
        default=Path("/run/hermes-rescue-secrets/key-id-next"),
    )
    parser.add_argument(
        "--cutover-file",
        type=Path,
        default=Path("/run/hermes-rescue-secrets/key-cutover"),
    )
    parser.add_argument(
        "--source-sha-file",
        type=Path,
        default=Path("/run/hermes-rescue-secrets/source-sha"),
    )
    parser.add_argument(
        "--image-id-file",
        type=Path,
        default=Path("/run/hermes-rescue-secrets/image-id"),
    )
    parser.add_argument("--expected-hermes-uid", type=int, required=True)
    parser.add_argument("--expected-reporter-uid", type=int, required=True)
    args = parser.parse_args()

    reporter_uid = _posix_uid()
    require_effective_uid(args.expected_reporter_uid, actual_uid=reporter_uid)
    runtime_info = args.runtime_dir.lstat()
    if (
        args.runtime_dir.is_symlink()
        or not stat.S_ISDIR(runtime_info.st_mode)
        or runtime_info.st_uid != reporter_uid
        or stat.S_IMODE(runtime_info.st_mode) != 0o750
    ):
        raise PermissionError("insecure reporter runtime directory")
    runtime_gid = runtime_info.st_gid
    continuity_info = args.continuity_dir.lstat()
    if (
        args.continuity_dir.is_symlink()
        or not stat.S_ISDIR(continuity_info.st_mode)
        or continuity_info.st_uid != reporter_uid
        or continuity_info.st_gid != runtime_gid
        or stat.S_IMODE(continuity_info.st_mode) != 0o750
    ):
        raise PermissionError("insecure reporter continuity directory")
    current = KeySlot(
        _read_identity(
            args.current_key_id_file,
            r"[A-Za-z0-9_.-]{1,64}",
            expected_uid=reporter_uid,
            expected_gid=runtime_gid,
        ),
        read_hmac_key(
            args.current_key, expected_uid=reporter_uid, expected_gid=runtime_gid
        ),
    )
    next_slot = None
    cutover = None
    if (
        args.next_key.exists()
        or args.next_key_id_file.exists()
        or args.cutover_file.exists()
    ):
        if not (
            args.next_key.exists()
            and args.next_key_id_file.exists()
            and args.cutover_file.exists()
        ):
            raise ValueError("incomplete next-key rotation slot")
        next_slot = KeySlot(
            _read_identity(
                args.next_key_id_file,
                r"[A-Za-z0-9_.-]{1,64}",
                expected_uid=reporter_uid,
                expected_gid=runtime_gid,
            ),
            read_hmac_key(
                args.next_key, expected_uid=reporter_uid, expected_gid=runtime_gid
            ),
        )
        cutover = float(
            _read_identity(
                args.cutover_file,
                r"[0-9]+(?:\.[0-9]+)?",
                expected_uid=reporter_uid,
                expected_gid=runtime_gid,
            )
        )
    keyring = KeyRing(current=current, next=next_slot, cutover_at=cutover)
    source_sha = _read_identity(
        args.source_sha_file,
        r"[0-9a-f]{40}|[0-9a-f]{64}",
        expected_uid=reporter_uid,
        expected_gid=runtime_gid,
    )
    image_id = _read_identity(
        args.image_id_file,
        r"sha256:[0-9a-f]{64}",
        expected_uid=reporter_uid,
        expected_gid=runtime_gid,
    )
    QuiescenceReporter(
        runtime_dir=args.runtime_dir,
        continuity_dir=args.continuity_dir,
        keyring=keyring,
        source_sha=source_sha,
        image_id=image_id,
        expected_hermes_uid=args.expected_hermes_uid,
        recovery_authorization_path=args.recovery_authorization,
    ).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
