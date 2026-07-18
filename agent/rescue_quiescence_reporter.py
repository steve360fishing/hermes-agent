"""Independent, reporter-owned rescue telemetry aggregation and signing."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import math
import os
from pathlib import Path
import socket
import stat
import struct
import threading
import time
from typing import Any, Callable, Mapping

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
        if type(max_turns) is not int or not 1 <= max_turns <= 4096:
            raise ValueError("invalid turn bound")
        if type(max_work) is not int or not 1 <= max_work <= 100_000:
            raise ValueError("invalid work bound")
        self.max_turns = max_turns
        self.max_work = max_work
        self.turns: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.tools: dict[str, dict[str, Any]] = {}
        self.providers: dict[str, dict[str, Any]] = {}
        self.seen_events: OrderedDict[str, None] = OrderedDict()
        self.telemetry_health = "healthy"
        self.reconciled_crashes = 0

    def _validate_event(self, event: Mapping[str, Any]) -> str:
        if type(event) is not dict:
            raise ValueError("event must be an object")
        kind = event.get("event")
        if kind == "turn_start":
            expected = _EVENT_TURN_START
        elif kind in {"turn_end"}:
            expected = _EVENT_BASE
        elif kind in {"tool_start", "tool_end", "provider_start", "provider_end"}:
            expected = _EVENT_WORK
        else:
            raise ValueError("unknown event")
        if set(event) != expected or event.get("schema_version") != EVENT_SCHEMA_VERSION:
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
        if type(peer_pid) is not int or peer_pid <= 0 or type(peer_uid) is not int or peer_uid < 0:
            raise ValueError("invalid peer credentials")
        event_id = str(event["event_id"])
        if event_id in self.seen_events:
            return False
        self.seen_events[event_id] = None
        while len(self.seen_events) > max(1024, self.max_work * 2):
            self.seen_events.popitem(last=False)

        turn_id = str(event["turn_id"])
        if kind == "turn_start":
            existing = self.turns.get(turn_id)
            if existing and existing["completed_at"] is None:
                raise ValueError("duplicate active turn")
            if len([turn for turn in self.turns.values() if turn["completed_at"] is None]) >= self.max_turns:
                self.telemetry_health = "degraded"
                raise ValueError("active turn bound exceeded")
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
            if not turn or turn["completed_at"] is not None or turn["peer_pid"] != peer_pid:
                raise ValueError("turn end does not match active owner")
            turn["completed_at"] = float(now)
        else:
            work_id = str(event["work_id"])
            collection = self.tools if kind.startswith("tool_") else self.providers
            if kind.endswith("_start"):
                if work_id in collection:
                    raise ValueError("duplicate active work")
                if len(self.tools) + len(self.providers) >= self.max_work:
                    self.telemetry_health = "degraded"
                    raise ValueError("active work bound exceeded")
                collection[work_id] = {
                    "turn_id": turn_id,
                    "peer_pid": peer_pid,
                    "peer_uid": peer_uid,
                    "started_at": float(now),
                }
            else:
                active = collection.get(work_id)
                if not active or active["peer_pid"] != peer_pid or active["turn_id"] != turn_id:
                    raise ValueError("work end does not match active owner")
                del collection[work_id]
        self._prune()
        return True

    def _prune(self) -> None:
        completed = [
            turn_id
            for turn_id, turn in self.turns.items()
            if turn["completed_at"] is not None
        ]
        while len(self.turns) > self.max_turns and completed:
            self.turns.pop(completed.pop(0), None)

    def reconcile(self, is_pid_alive: Callable[[int], bool], *, now: float) -> None:
        active_pids = {
            int(turn["peer_pid"])
            for turn in self.turns.values()
            if turn["completed_at"] is None
        }
        active_pids.update(int(item["peer_pid"]) for item in self.tools.values())
        active_pids.update(int(item["peer_pid"]) for item in self.providers.values())
        dead = {pid for pid in active_pids if not is_pid_alive(pid)}
        if dead:
            newly_degraded = self.telemetry_health != "degraded"
            self.telemetry_health = "degraded"
            if newly_degraded:
                self.reconciled_crashes = min(
                    100_000, self.reconciled_crashes + len(dead)
                )
        # Deliberately retain dead-worker records: absence of an authenticated
        # end event cannot become an idle authorization after a crash.

    def active_counts(self) -> tuple[int, int, int]:
        return (
            sum(turn["completed_at"] is None for turn in self.turns.values()),
            len(self.tools),
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
            "schema_version": "hermes-rescue-reporter-state-v1",
            "max_turns": self.max_turns,
            "max_work": self.max_work,
            "turns": list(self.turns.values()),
            "tools": self.tools,
            "providers": self.providers,
            "seen_events": list(self.seen_events),
            "telemetry_health": self.telemetry_health,
            "reconciled_crashes": self.reconciled_crashes,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ReporterState":
        expected = {
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
        if type(payload) is not dict or set(payload) != expected:
            raise ValueError("corrupt reporter state")
        state = cls(max_turns=payload["max_turns"], max_work=payload["max_work"])
        if payload["schema_version"] != "hermes-rescue-reporter-state-v1":
            raise ValueError("corrupt reporter state")
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
        for name, destination in (("tools", state.tools), ("providers", state.providers)):
            values = payload[name]
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
        if len(state.tools) + len(state.providers) > state.max_work:
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
        return state


def _read_proc_uid(status: Path) -> int | None:
    try:
        for line in status.read_text(encoding="ascii").splitlines():
            if line.startswith("Uid:"):
                values = line.split()
                return int(values[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def discover_gateway_state(
    *,
    proc_root: Path = Path("/proc"),
    expected_uid: int,
) -> tuple[list[int], str]:
    pids: list[int] = []
    try:
        children = list(proc_root.iterdir())
    except OSError:
        return [], "dead"
    for child in children:
        if not child.name.isdigit():
            continue
        try:
            raw = (child / "cmdline").read_bytes()
        except OSError:
            continue
        args = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
        if "gateway" not in args or "run" not in args:
            continue
        if not any("hermes" in arg for arg in args):
            continue
        if _read_proc_uid(child / "status") != expected_uid:
            continue
        pids.append(int(child.name))
    pids.sort()
    return pids, "active" if pids else "dead"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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
            or info.st_uid != os.getuid()
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
    actual = os.getuid() if actual_uid is None else actual_uid
    if type(expected_uid) is not int or expected_uid <= 0 or actual != expected_uid:
        raise PermissionError("unexpected rescue reporter effective uid")


class QuiescenceReporter:
    """Owns the Unix socket, durable aggregate, and HMAC snapshot output."""

    def __init__(
        self,
        *,
        runtime_dir: Path,
        keyring: KeyRing,
        source_sha: str,
        image_id: str,
        expected_hermes_uid: int,
        sequence: int = 0,
        max_turns: int = 256,
    ) -> None:
        self.runtime_dir = runtime_dir
        self.socket_path = runtime_dir / "events.sock"
        self.state_path = runtime_dir / "reporter-state-v1.json"
        self.output_path = runtime_dir / "quiescence-v1.json"
        self.keyring = keyring
        self.source_sha = source_sha
        self.image_id = image_id
        self.expected_hermes_uid = expected_hermes_uid
        self.sequence = sequence
        runtime_info = runtime_dir.lstat()
        if (
            runtime_dir.is_symlink()
            or not stat.S_ISDIR(runtime_info.st_mode)
            or runtime_info.st_uid != os.getuid()
            or stat.S_IMODE(runtime_info.st_mode) != 0o750
        ):
            raise PermissionError("insecure reporter runtime directory")
        self.runtime_gid = runtime_info.st_gid
        try:
            persisted = _secure_read(
                self.state_path,
                expected_uid=os.getuid(),
                expected_gid=self.runtime_gid,
                file_mode=0o600,
                parent_mode=0o750,
                max_bytes=2_000_000,
            )
        except FileNotFoundError:
            self.state = ReporterState(max_turns=max_turns)
        else:
            self.state = ReporterState.from_payload(_strict_json_object(persisted))
        self._lock = threading.RLock()

    def _persist_state(self) -> None:
        atomic_write_secure(
            self.state_path,
            json.dumps(
                self.state.to_payload(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii"),
            mode=0o600,
            expected_parent_uid=os.getuid(),
            expected_parent_gid=self.runtime_gid,
        )

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
        with self._lock:
            prior = ReporterState.from_payload(self.state.to_payload())
            try:
                changed = self.state.apply_event(
                    event,
                    peer_pid=peer_pid,
                    peer_uid=peer_uid,
                    now=time.time() if now is None else now,
                )
                if changed:
                    self._persist_state()
            except Exception:
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
            self.sequence += 1
            active_turns, active_tools, active_providers = self.state.active_counts()
            unsigned = {
                "schema_version": QUIESCENCE_SCHEMA_VERSION,
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
                "policy_evidence": self.state.policy_evidence(),
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
                expected_parent_uid=os.getuid(),
                expected_parent_gid=self.runtime_gid,
            )
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
                        except (ValueError, PermissionError, json.JSONDecodeError, OSError):
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
    value = _secure_read(
        path,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
        file_mode=0o400,
        parent_mode=0o750,
        max_bytes=256,
    ).decode("ascii").strip()
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

    reporter_uid = os.getuid()
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
    current = KeySlot(
        _read_identity(
            args.current_key_id_file,
            r"[A-Za-z0-9_.-]{1,64}",
            expected_uid=reporter_uid,
            expected_gid=runtime_gid,
        ),
        read_hmac_key(args.current_key, expected_uid=reporter_uid, expected_gid=runtime_gid),
    )
    next_slot = None
    cutover = None
    if args.next_key.exists() or args.next_key_id_file.exists() or args.cutover_file.exists():
        if not (args.next_key.exists() and args.next_key_id_file.exists() and args.cutover_file.exists()):
            raise ValueError("incomplete next-key rotation slot")
        next_slot = KeySlot(
            _read_identity(
                args.next_key_id_file,
                r"[A-Za-z0-9_.-]{1,64}",
                expected_uid=reporter_uid,
                expected_gid=runtime_gid,
            ),
            read_hmac_key(args.next_key, expected_uid=reporter_uid, expected_gid=runtime_gid),
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
    sequence = recover_snapshot_sequence(
        args.runtime_dir / "quiescence-v1.json",
        keyring=keyring,
        expected_uid=reporter_uid,
        expected_gid=runtime_gid,
    )
    QuiescenceReporter(
        runtime_dir=args.runtime_dir,
        keyring=keyring,
        source_sha=source_sha,
        image_id=image_id,
        expected_hermes_uid=args.expected_hermes_uid,
        sequence=sequence,
    ).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
