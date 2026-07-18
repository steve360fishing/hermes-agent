"""Security boundary primitives for the Hermes Rescue Plane V1 core slice."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
import stat
import tempfile
import threading
import time
import uuid
from typing import Any, Iterator, Mapping


SAFE_MODE_OVERLAY_PATH = Path("/var/run/hermes-rescue/safe-mode-v1.json")
RESCUE_EVENT_SOCKET_PATH = Path("/run/hermes-rescue-reporter/events.sock")
RESCUE_TELEMETRY_REQUIRED_PATH = Path(
    "/var/lib/hermes-rescue/telemetry-required-v1.json"
)
RESCUE_REPORTER_UID = 10001
TELEMETRY_REQUIRED_MARKER = (
    b'{"required":true,"schema_version":'
    b'"hermes-rescue-telemetry-required-v1"}'
)
QUIESCENCE_SCHEMA_VERSION = "hermes-quiescence-snapshot-v1"
EVENT_SCHEMA_VERSION = "hermes-rescue-event-v1"
SAFE_MODE_SCHEMA_VERSION = "hermes-safe-mode-v1"
FIXTURE_ID = "HERMES-ARTIFACT-STICKY-20260717-v1"
FIXTURE_SHA256 = "d8d190a659e77d714b555dfbccf49f15222171db8bccaa37d38d31373c81f420"
_OVERLAY_KEYS = frozenset(
    {"schema_version", "enabled", "incident_id", "issued_at", "disables"}
)
_SNAPSHOT_KEYS = frozenset(
    {
        "schema_version",
        "producer_epoch",
        "key_id",
        "sequence",
        "timestamp",
        "gateway_pids",
        "gateway_state",
        "active_turn_count",
        "active_tool_count",
        "active_provider_action_count",
        "source_sha",
        "image_id",
        "telemetry_health",
        "policy_evidence",
        "signature",
    }
)
_POLICY_KEYS = frozenset(
    {"turn_id", "lane", "artifact_requested", "started_at", "completed_at"}
)
_HEX_RE = re.compile(r"^[0-9a-f]+$")
_REPLAY_LOCKS_GUARD = threading.Lock()
_REPLAY_LOCKS: dict[str, threading.RLock] = {}


@dataclass(frozen=True)
class SafeModeOverlay:
    valid: bool
    disables: frozenset[str]
    reason: str

    @property
    def disables_artifact_only(self) -> bool:
        return not self.valid or "artifact_only" in self.disables


@dataclass(frozen=True)
class KeySlot:
    key_id: str
    key: bytes

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", self.key_id):
            raise ValueError("invalid key id")
        if type(self.key) is not bytes or len(self.key) != 32:
            raise ValueError("HMAC key must be exactly 32 bytes")


@dataclass(frozen=True)
class KeyRing:
    current: KeySlot
    next: KeySlot | None = None
    cutover_at: float | None = None

    def __post_init__(self) -> None:
        if (self.next is None) != (self.cutover_at is None):
            raise ValueError("next key and cutover must be configured together")
        if self.next and self.next.key_id == self.current.key_id:
            raise ValueError("rotation key ids must differ")
        if self.cutover_at is not None and not _finite_number(self.cutover_at):
            raise ValueError("cutover must be finite")

    def slot(self, key_id: str) -> KeySlot | None:
        if key_id == self.current.key_id:
            return self.current
        if self.next and key_id == self.next.key_id:
            return self.next
        return None


@dataclass(frozen=True)
class QuiescenceValidation:
    valid: bool
    reason: str


def _invalid_overlay(reason: str) -> SafeModeOverlay:
    code = reason if reason.startswith("invalid_") else f"invalid_{reason}"
    return SafeModeOverlay(False, frozenset({"artifact_only"}), code)


def _finite_number(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _posix_secure_read(
    path: Path,
    *,
    expected_uid: int,
    expected_gid: int | None,
    file_mode: int,
    parent_mode: int | None,
    max_bytes: int,
) -> bytes:
    parent = path.parent
    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_fd = os.open(parent, parent_flags)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise PermissionError("invalid parent") from exc
    try:
        parent_stat = os.fstat(parent_fd)
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise PermissionError("invalid parent")
        if parent_stat.st_uid != expected_uid:
            raise PermissionError("invalid parent owner")
        if expected_gid is not None and parent_stat.st_gid != expected_gid:
            raise PermissionError("invalid parent gid")
        if parent_mode is not None and stat.S_IMODE(parent_stat.st_mode) != parent_mode:
            raise PermissionError("invalid parent mode")
        if parent_mode is None and stat.S_IMODE(parent_stat.st_mode) & 0o022:
            raise PermissionError("invalid parent mode")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path.name, flags, dir_fd=parent_fd)
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise PermissionError("invalid file") from exc
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise PermissionError("not regular")
            if info.st_nlink != 1:
                raise PermissionError("hardlink")
            if info.st_uid != expected_uid:
                raise PermissionError("owner")
            required_gid = parent_stat.st_gid if expected_gid is None else expected_gid
            if info.st_gid != required_gid:
                raise PermissionError("gid")
            if stat.S_IMODE(info.st_mode) != file_mode:
                raise PermissionError("mode")
            if info.st_size > max_bytes:
                raise PermissionError("too large")
            chunks: list[bytes] = []
            remaining = max_bytes + 1
            while remaining:
                chunk = os.read(fd, min(remaining, 65536))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            if len(raw) > max_bytes:
                raise PermissionError("too large")
            after = os.fstat(fd)
            if (after.st_dev, after.st_ino, after.st_size) != (
                info.st_dev,
                info.st_ino,
                info.st_size,
            ):
                raise PermissionError("file changed")
            return raw
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)


def _secure_read(
    path: Path,
    *,
    expected_uid: int,
    expected_gid: int | None,
    file_mode: int,
    parent_mode: int | None,
    max_bytes: int,
) -> bytes:
    if os.name == "posix":
        return _posix_secure_read(
            path,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
            file_mode=file_mode,
            parent_mode=parent_mode,
            max_bytes=max_bytes,
        )
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
        raise PermissionError("invalid file")
    return path.read_bytes()


def atomic_write_secure(
    path: Path,
    data: bytes,
    *,
    mode: int = 0o600,
    expected_parent_uid: int | None = None,
    expected_parent_gid: int | None = None,
) -> None:
    """Atomically replace one regular file using a random, exclusive temp."""
    if type(data) is not bytes:
        raise TypeError("atomic data must be bytes")
    if not path.parent.exists():
        if expected_parent_uid is not None or expected_parent_gid is not None:
            raise PermissionError("secure output parent is absent")
        path.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "posix":
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb", closefd=True) as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        return

    parent_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(path.parent, parent_flags)
    temporary_name = f".{path.name}.{secrets.token_hex(16)}.tmp"
    try:
        parent_stat = os.fstat(parent_fd)
        if not stat.S_ISDIR(parent_stat.st_mode) or stat.S_IMODE(parent_stat.st_mode) & 0o022:
            raise PermissionError("insecure output parent")
        if expected_parent_uid is not None and parent_stat.st_uid != expected_parent_uid:
            raise PermissionError("output parent owner")
        if expected_parent_gid is not None and parent_stat.st_gid != expected_parent_gid:
            raise PermissionError("output parent gid")
        try:
            current_fd = os.open(
                path.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            current_fd = None
        if current_fd is not None:
            try:
                current = os.fstat(current_fd)
                if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
                    raise PermissionError("unsafe output target")
                if current.st_uid != parent_stat.st_uid or current.st_gid != parent_stat.st_gid:
                    raise PermissionError("output target ownership")
            finally:
                os.close(current_fd)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(temporary_name, flags, mode, dir_fd=parent_fd)
        try:
            os.fchmod(fd, mode)
            os.fchown(fd, -1, parent_stat.st_gid)
            temporary_stat = os.fstat(fd)
            if (
                temporary_stat.st_uid != parent_stat.st_uid
                or temporary_stat.st_gid != parent_stat.st_gid
            ):
                raise PermissionError("temporary output ownership")
            offset = 0
            while offset < len(data):
                offset += os.write(fd, data[offset:])
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.fsync(parent_fd)
    finally:
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)


def path_is_on_readonly_mount(
    path: Path,
    *,
    mountinfo_path: Path = Path("/proc/self/mountinfo"),
) -> bool:
    """Require a non-root mountpoint whose effective mount options include ro."""
    target = os.path.realpath(os.path.abspath(path))
    best_mount = ""
    best_readonly = False
    try:
        lines = mountinfo_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for line in lines:
        left, separator, right = line.partition(" - ")
        if not separator:
            return False
        fields = left.split()
        right_fields = right.split()
        if len(fields) < 6 or len(right_fields) < 3:
            return False
        mountpoint = fields[4].replace("\\040", " ")
        normalized = os.path.realpath(os.path.abspath(mountpoint))
        if target != normalized and not target.startswith(normalized.rstrip(os.sep) + os.sep):
            continue
        if len(normalized) < len(best_mount):
            continue
        mount_options = set(fields[5].split(","))
        super_options = set(right_fields[2].split(","))
        best_mount = normalized
        best_readonly = "ro" in mount_options or "ro" in super_options
    return bool(best_mount and best_mount != os.path.abspath(os.sep) and best_readonly)


def read_safe_mode_overlay(
    path: Path | None = None,
    *,
    expected_uid: int = 0,
    expected_gid: int | None = None,
) -> SafeModeOverlay:
    path = SAFE_MODE_OVERLAY_PATH if path is None else path
    try:
        raw = _secure_read(
            path,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
            file_mode=0o640,
            parent_mode=0o750,
            max_bytes=4096,
        )
        if path == SAFE_MODE_OVERLAY_PATH and os.name == "posix":
            if not path_is_on_readonly_mount(path):
                return _invalid_overlay("mount")
    except FileNotFoundError:
        return SafeModeOverlay(True, frozenset(), "absent")
    except PermissionError as exc:
        reason = str(exc).replace(" ", "_")
        if reason in {"invalid_file", "file_changed"}:
            reason = "symlink"
        return _invalid_overlay(reason)
    except OSError:
        return _invalid_overlay("unreadable")
    try:
        payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError):
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
    if type(issued_at) is not str or not issued_at.endswith("Z"):
        return _invalid_overlay("issued_at")
    try:
        datetime.fromisoformat(issued_at[:-1] + "+00:00")
    except ValueError:
        return _invalid_overlay("issued_at")
    if payload.get("disables") != ["artifact_only"]:
        return _invalid_overlay("disables")
    return SafeModeOverlay(True, frozenset({"artifact_only"}), "enabled")


def rescue_overlay_disables_artifact_only() -> bool:
    return read_safe_mode_overlay().disables_artifact_only


def read_hmac_key(path: Path, *, expected_uid: int, expected_gid: int) -> bytes:
    raw = _secure_read(
        path,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
        file_mode=0o400,
        parent_mode=None,
        max_bytes=32,
    )
    if len(raw) != 32:
        raise PermissionError("key length")
    return raw


def load_sticky_artifact_fixture(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FIXTURE_SHA256:
        raise ValueError("rescue fixture hash mismatch")
    payload = json.loads(raw)
    if payload.get("fixture_id") != FIXTURE_ID:
        raise ValueError("rescue fixture id mismatch")
    return payload


def _validate_policy_evidence(value: Any) -> bool:
    if type(value) is not list or len(value) > 256:
        return False
    for item in value:
        if type(item) is not dict or set(item) != _POLICY_KEYS:
            return False
        if type(item["turn_id"]) is not str or not 1 <= len(item["turn_id"]) <= 256:
            return False
        if item["lane"] not in {"normal", "artifact_only"}:
            return False
        if type(item["artifact_requested"]) is not bool:
            return False
        if not _finite_number(item["started_at"]):
            return False
        completed = item["completed_at"]
        if completed is not None and not _finite_number(completed):
            return False
    return True


def _strict_snapshot(snapshot: Any) -> tuple[bool, str]:
    if type(snapshot) is not dict or set(snapshot) != _SNAPSHOT_KEYS:
        return False, "malformed"
    if snapshot["schema_version"] != QUIESCENCE_SCHEMA_VERSION:
        return False, "malformed"
    if type(snapshot["producer_epoch"]) is not str or not re.fullmatch(
        r"[0-9a-f]{32}", snapshot["producer_epoch"]
    ):
        return False, "malformed"
    if type(snapshot["key_id"]) is not str or not re.fullmatch(
        r"[A-Za-z0-9_.-]{1,64}", snapshot["key_id"]
    ):
        return False, "malformed"
    if (
        type(snapshot["sequence"]) is not int
        or not 1 <= snapshot["sequence"] <= (2**63 - 1)
    ):
        return False, "malformed"
    if not _finite_number(snapshot["timestamp"]):
        return False, "malformed"
    pids = snapshot["gateway_pids"]
    if type(pids) is not list or len(pids) > 64:
        return False, "malformed"
    if any(type(pid) is not int or pid <= 0 for pid in pids) or len(set(pids)) != len(pids):
        return False, "malformed"
    if snapshot["gateway_state"] not in {"active", "dead", "unknown"}:
        return False, "malformed"
    for name in (
        "active_turn_count",
        "active_tool_count",
        "active_provider_action_count",
    ):
        if type(snapshot[name]) is not int or not 0 <= snapshot[name] <= 100_000:
            return False, "malformed"
    source_sha = snapshot["source_sha"]
    if type(source_sha) is not str or len(source_sha) not in {40, 64} or not _HEX_RE.fullmatch(source_sha):
        return False, "malformed"
    image_id = snapshot["image_id"]
    if type(image_id) is not str or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        return False, "malformed"
    if snapshot["telemetry_health"] not in {"healthy", "degraded"}:
        return False, "malformed"
    if not _validate_policy_evidence(snapshot["policy_evidence"]):
        return False, "malformed"
    signature = snapshot["signature"]
    if type(signature) is not str or not re.fullmatch(r"[0-9a-f]{64}", signature):
        return False, "malformed"
    return True, "valid"


def _slot_allowed(key_id: str, timestamp: float, now: float, keyring: KeyRing) -> tuple[KeySlot | None, str]:
    slot = keyring.slot(key_id)
    if slot is None:
        return None, "unknown_key"
    if keyring.next is None:
        return (slot, "valid") if slot is keyring.current else (None, "unknown_key")
    assert keyring.cutover_at is not None
    if slot is keyring.current:
        if timestamp >= keyring.cutover_at:
            return None, "post_cutover_old_key"
        if now >= keyring.cutover_at and now - timestamp > 30:
            return None, "old_key_expired"
        return slot, "valid"
    if timestamp < keyring.cutover_at:
        return None, "pre_cutover_next_key"
    return slot, "valid"


def sign_quiescence_snapshot(
    unsigned: Mapping[str, Any],
    keyring: KeyRing,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    payload = dict(unsigned)
    signing_time = time.time() if now is None else now
    if "key_id" not in payload:
        payload["key_id"] = (
            keyring.next.key_id
            if keyring.next and keyring.cutover_at is not None and signing_time >= keyring.cutover_at
            else keyring.current.key_id
        )
    slot = keyring.slot(payload["key_id"])
    if slot is None:
        raise ValueError("unknown signing key")
    payload.pop("signature", None)
    payload["signature"] = hmac.new(slot.key, _canonical_json(payload), hashlib.sha256).hexdigest()
    valid, reason = _strict_snapshot(payload)
    if not valid:
        raise ValueError(reason)
    allowed, reason = _slot_allowed(
        payload["key_id"], float(payload["timestamp"]), signing_time, keyring
    )
    if allowed is None:
        raise ValueError(reason)
    return payload


def _verify_snapshot_signature(snapshot: Mapping[str, Any], keyring: KeyRing) -> bool:
    slot = keyring.slot(str(snapshot.get("key_id", "")))
    if slot is None:
        return False
    unsigned = {name: value for name, value in snapshot.items() if name != "signature"}
    expected = hmac.new(slot.key, _canonical_json(unsigned), hashlib.sha256).hexdigest()
    return hmac.compare_digest(str(snapshot.get("signature", "")), expected)


class ProducerEpochChanged(ValueError):
    """The verifier is pinned to a different durable producer identity."""


class DurableReplayState:
    """Mandatory durable monotonic sequence state for host-side verification."""

    def __init__(
        self,
        path: Path,
        *,
        expected_uid: int | None = None,
        expected_gid: int | None = None,
    ) -> None:
        self.path = path
        self.expected_uid = (
            os.getuid() if expected_uid is None and os.name == "posix" else expected_uid
        )
        self.expected_gid = (
            os.getgid() if expected_gid is None and os.name == "posix" else expected_gid
        )

    def _load(self) -> dict[str, Any]:
        if os.name == "posix":
            assert self.expected_uid is not None
            raw = _secure_read(
                self.path,
                expected_uid=self.expected_uid,
                expected_gid=self.expected_gid,
                file_mode=0o600,
                parent_mode=None,
                max_bytes=65_536,
            )
        else:
            raw = self.path.read_bytes()
        payload = json.loads(raw)
        if (
            type(payload) is not dict
            or set(payload)
            != {
                "schema_version",
                "producer_epoch",
                "last_sequence",
                "last_key_id",
            }
            or payload["schema_version"] != "hermes-rescue-replay-state-v1"
            or (
                payload["producer_epoch"] is not None
                and (
                    type(payload["producer_epoch"]) is not str
                    or not re.fullmatch(
                        r"[0-9a-f]{32}", payload["producer_epoch"]
                    )
                )
            )
            or type(payload["last_sequence"]) is not int
            or not 0 <= payload["last_sequence"] <= (2**63 - 1)
            or (
                payload["last_key_id"] is not None
                and (
                    type(payload["last_key_id"]) is not str
                    or not re.fullmatch(
                        r"[A-Za-z0-9_.-]{1,64}", payload["last_key_id"]
                    )
                )
            )
        ):
            raise ValueError("corrupt replay state")
        return payload

    def initialize(self) -> None:
        """Explicitly provision replay state once; never recreate missing state."""
        initial = _canonical_json(
            {
                "schema_version": "hermes-rescue-replay-state-v1",
                "producer_epoch": None,
                "last_sequence": 0,
                "last_key_id": None,
            }
        )
        try:
            self._load()
            return
        except FileNotFoundError:
            pass
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        parent_fd = None
        if os.name == "posix":
            parent_fd = os.open(
                self.path.parent,
                os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
            )
            parent_info = os.fstat(parent_fd)
            if (
                not stat.S_ISDIR(parent_info.st_mode)
                or stat.S_IMODE(parent_info.st_mode) & 0o022
                or (
                    self.expected_uid is not None
                    and parent_info.st_uid != self.expected_uid
                )
                or (
                    self.expected_gid is not None
                    and parent_info.st_gid != self.expected_gid
                )
            ):
                os.close(parent_fd)
                raise PermissionError("insecure replay parent")
            flags |= getattr(os, "O_NOFOLLOW", 0)
            try:
                fd = os.open(self.path.name, flags, 0o600, dir_fd=parent_fd)
            except Exception:
                os.close(parent_fd)
                raise
        else:
            fd = os.open(self.path, flags, 0o600)
        try:
            if os.name == "posix":
                os.fchmod(fd, 0o600)
                if self.expected_gid is not None:
                    os.fchown(fd, -1, self.expected_gid)
            offset = 0
            while offset < len(initial):
                offset += os.write(fd, initial[offset:])
            os.fsync(fd)
        finally:
            os.close(fd)
            if parent_fd is not None:
                try:
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)

    def accept(self, producer_epoch: str, key_id: str, sequence: int) -> bool:
        if (
            type(producer_epoch) is not str
            or not re.fullmatch(r"[0-9a-f]{32}", producer_epoch)
            or type(key_id) is not str
            or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", key_id)
            or type(sequence) is not int
            or not 1 <= sequence <= (2**63 - 1)
        ):
            raise ValueError("invalid replay candidate")
        lock_key = os.path.abspath(self.path)
        with _REPLAY_LOCKS_GUARD:
            process_lock = _REPLAY_LOCKS.setdefault(lock_key, threading.RLock())
        with process_lock:
            lock_fd = None
            parent_fd = None
            if os.name == "posix":
                import fcntl

                parent_fd = os.open(
                    self.path.parent,
                    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                )
                parent_info = os.fstat(parent_fd)
                if (
                    not stat.S_ISDIR(parent_info.st_mode)
                    or stat.S_IMODE(parent_info.st_mode) & 0o022
                    or (
                        self.expected_uid is not None
                        and parent_info.st_uid != self.expected_uid
                    )
                    or (
                        self.expected_gid is not None
                        and parent_info.st_gid != self.expected_gid
                    )
                ):
                    os.close(parent_fd)
                    parent_fd = None
                    raise PermissionError("insecure replay parent")
                try:
                    lock_name = f".{self.path.name}.lock"
                    lock_fd = os.open(
                        lock_name,
                        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                        0o600,
                        dir_fd=parent_fd,
                    )
                    os.fchmod(lock_fd, 0o600)
                    lock_info = os.fstat(lock_fd)
                    if (
                        not stat.S_ISREG(lock_info.st_mode)
                        or lock_info.st_nlink != 1
                        or lock_info.st_uid != self.expected_uid
                        or lock_info.st_gid != self.expected_gid
                    ):
                        raise PermissionError("insecure replay lock")
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
                except Exception:
                    if lock_fd is not None:
                        os.close(lock_fd)
                        lock_fd = None
                    os.close(parent_fd)
                    parent_fd = None
                    raise
            try:
                state = self._load()
                if (
                    state["producer_epoch"] is not None
                    and state["producer_epoch"] != producer_epoch
                ):
                    raise ProducerEpochChanged
                if sequence <= state["last_sequence"]:
                    return False
                state["producer_epoch"] = producer_epoch
                state["last_sequence"] = sequence
                state["last_key_id"] = key_id
                atomic_write_secure(
                    self.path,
                    _canonical_json(state),
                    expected_parent_uid=self.expected_uid,
                    expected_parent_gid=self.expected_gid,
                )
                return True
            finally:
                if lock_fd is not None:
                    import fcntl

                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    os.close(lock_fd)
                if parent_fd is not None:
                    os.close(parent_fd)


def validate_quiescence_snapshot(
    snapshot: Mapping[str, Any] | None,
    *,
    keyring: KeyRing,
    now: float,
    replay_state: DurableReplayState,
) -> QuiescenceValidation:
    if not _finite_number(now):
        return QuiescenceValidation(False, "invalid_verifier_time")
    valid, reason = _strict_snapshot(snapshot)
    if not valid:
        return QuiescenceValidation(False, reason)
    assert snapshot is not None
    if not _verify_snapshot_signature(snapshot, keyring):
        return QuiescenceValidation(False, "invalid_signature")
    timestamp = float(snapshot["timestamp"])
    slot, reason = _slot_allowed(str(snapshot["key_id"]), timestamp, now, keyring)
    if slot is None:
        return QuiescenceValidation(False, reason)
    if timestamp > now + 1 or now - timestamp >= 15:
        return QuiescenceValidation(False, "stale")
    if snapshot["gateway_state"] == "unknown":
        return QuiescenceValidation(False, "gateway_unknown")
    if snapshot["telemetry_health"] != "healthy":
        return QuiescenceValidation(False, "telemetry_degraded")
    try:
        accepted = replay_state.accept(
            str(snapshot["producer_epoch"]),
            str(snapshot["key_id"]),
            int(snapshot["sequence"]),
        )
    except ProducerEpochChanged:
        return QuiescenceValidation(False, "producer_epoch_changed")
    except (OSError, ValueError, json.JSONDecodeError):
        return QuiescenceValidation(False, "replay_state_unavailable")
    if not accepted:
        return QuiescenceValidation(False, "replayed")
    return QuiescenceValidation(True, "valid")


def recover_snapshot_sequence(
    path: Path,
    *,
    keyring: KeyRing,
    expected_uid: int,
    expected_gid: int,
) -> int:
    try:
        raw = _secure_read(
            path,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
            file_mode=0o600,
            parent_mode=None,
            max_bytes=1_000_000,
        )
    except FileNotFoundError:
        return 0
    snapshot = json.loads(raw)
    valid, reason = _strict_snapshot(snapshot)
    if not valid:
        raise ValueError(reason)
    if not _verify_snapshot_signature(snapshot, keyring):
        raise ValueError("invalid previous signature")
    return int(snapshot["sequence"])


class RescueTelemetryUnavailable(RuntimeError):
    pass


class RescueTelemetryClient:
    """One-event-per-connection client; the reporter derives all aggregate state."""

    def __init__(self, socket_path: Path = RESCUE_EVENT_SOCKET_PATH) -> None:
        self.socket_path = socket_path

    def emit(self, event: Mapping[str, Any]) -> None:
        payload = {"schema_version": EVENT_SCHEMA_VERSION, **dict(event)}
        raw = _canonical_json(payload)
        if len(raw) > 8192:
            raise RescueTelemetryUnavailable("rescue telemetry event too large")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(1.0)
                client.connect(str(self.socket_path))
                client.sendall(raw)
                client.shutdown(socket.SHUT_WR)
                acknowledgement = client.recv(16)
        except OSError as exc:
            raise RescueTelemetryUnavailable("rescue telemetry unavailable") from exc
        if acknowledgement != b"OK\n":
            raise RescueTelemetryUnavailable("rescue telemetry rejected")

    @contextmanager
    def turn(self, turn_id: str, *, lane: str, artifact_requested: bool) -> Iterator[None]:
        self.emit(
            {
                "event": "turn_start",
                "event_id": secrets.token_hex(16),
                "turn_id": turn_id,
                "lane": lane,
                "artifact_requested": artifact_requested,
            }
        )
        try:
            yield
        finally:
            self.emit(
                {
                    "event": "turn_end",
                    "event_id": secrets.token_hex(16),
                    "turn_id": turn_id,
                }
            )

    @contextmanager
    def active_work(self, kind: str, *, turn_id: str) -> Iterator[None]:
        if kind not in {"tool", "provider"}:
            raise ValueError("invalid rescue work kind")
        work_id = secrets.token_hex(16)
        self.emit(
            {
                "event": f"{kind}_start",
                "event_id": secrets.token_hex(16),
                "turn_id": turn_id,
                "work_id": work_id,
            }
        )
        try:
            yield
        finally:
            self.emit(
                {
                    "event": f"{kind}_end",
                    "event_id": secrets.token_hex(16),
                    "turn_id": turn_id,
                    "work_id": work_id,
                }
            )


class RescueBackgroundWork:
    """One reporter-owned active record that outlives its launching tool call."""

    def __init__(
        self,
        client: RescueTelemetryClient,
        *,
        turn_id: str,
        work_id: str,
    ) -> None:
        self.client = client
        self.turn_id = turn_id
        self.work_id = work_id
        self._finished = False
        self._unknown_reported = False
        self._lock = threading.Lock()

    def mark_unknown(self) -> None:
        """Degrade telemetry without removing this active reporter record."""
        with self._lock:
            if self._finished or self._unknown_reported:
                return
            self.client.emit(
                {
                    "event": "background_unknown",
                    "event_id": secrets.token_hex(16),
                    "turn_id": self.turn_id,
                    "work_id": self.work_id,
                }
            )
            self._unknown_reported = True

    def finish(self) -> None:
        with self._lock:
            if self._finished:
                return
            self.client.emit(
                {
                    "event": "background_end",
                    "event_id": secrets.token_hex(16),
                    "turn_id": self.turn_id,
                    "work_id": self.work_id,
                }
            )
            self._finished = True


_RESCUE_TOOL_CONTEXT: contextvars.ContextVar[
    tuple[RescueTelemetryClient | None, str] | None
] = contextvars.ContextVar("rescue_tool_context", default=None)
_CLIENT_UNSET = object()


@contextmanager
def rescue_tool_execution_scope(
    turn_id: str,
    *,
    client: RescueTelemetryClient | None | object = _CLIENT_UNSET,
) -> Iterator[None]:
    """Single accounting and required-telemetry boundary for every tool path."""
    existing = _RESCUE_TOOL_CONTEXT.get()
    if existing is not None:
        # Tool middleware and the registry boundary can legitimately nest.
        # The outermost scope owns both the required-telemetry check and the
        # active-work record; nested paths must not double count.
        yield
        return
    resolved = get_rescue_telemetry_client() if client is _CLIENT_UNSET else client
    normalized_turn_id = str(turn_id or "unscoped")
    token = _RESCUE_TOOL_CONTEXT.set((resolved, normalized_turn_id))
    scope = (
        resolved.active_work("tool", turn_id=normalized_turn_id)
        if resolved is not None
        else nullcontext()
    )
    try:
        with scope:
            yield
    finally:
        _RESCUE_TOOL_CONTEXT.reset(token)


def begin_rescue_background_work(
    kind: str,
    work_id: str,
) -> RescueBackgroundWork | None:
    """Persist background work before a registry launches its worker/process."""
    if kind not in {"terminal", "delegation"}:
        raise ValueError("invalid rescue background kind")
    normalized = f"{kind}:{work_id}"
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", normalized):
        raise ValueError("invalid rescue background work id")
    context = _RESCUE_TOOL_CONTEXT.get()
    if context is None:
        client = get_rescue_telemetry_client()
        turn_id = "unscoped"
    else:
        client, turn_id = context
    if client is None:
        return None
    client.emit(
        {
            "event": "background_start",
            "event_id": secrets.token_hex(16),
            "turn_id": turn_id,
            "work_id": normalized,
        }
    )
    return RescueBackgroundWork(client, turn_id=turn_id, work_id=normalized)


def _rescue_telemetry_is_required() -> bool:
    try:
        parent_info = RESCUE_TELEMETRY_REQUIRED_PATH.parent.lstat()
        marker_info = RESCUE_TELEMETRY_REQUIRED_PATH.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise RescueTelemetryUnavailable("rescue telemetry marker unavailable") from exc
    if (
        RESCUE_TELEMETRY_REQUIRED_PATH.parent.is_symlink()
        or not stat.S_ISDIR(parent_info.st_mode)
        or parent_info.st_uid != RESCUE_REPORTER_UID
        or parent_info.st_gid not in os.getgroups()
        or stat.S_IMODE(parent_info.st_mode) != 0o750
        or RESCUE_TELEMETRY_REQUIRED_PATH.is_symlink()
        or not stat.S_ISREG(marker_info.st_mode)
    ):
        raise RescueTelemetryUnavailable("insecure rescue telemetry marker")
    try:
        raw = _secure_read(
            RESCUE_TELEMETRY_REQUIRED_PATH,
            expected_uid=RESCUE_REPORTER_UID,
            expected_gid=parent_info.st_gid,
            file_mode=0o440,
            parent_mode=0o750,
            max_bytes=256,
        )
    except (OSError, PermissionError) as exc:
        raise RescueTelemetryUnavailable("insecure rescue telemetry marker") from exc
    if raw != TELEMETRY_REQUIRED_MARKER:
        raise RescueTelemetryUnavailable("invalid rescue telemetry marker")
    return True


def get_rescue_telemetry_client() -> RescueTelemetryClient | None:
    """Return configured telemetry or fail closed across a required restart gap."""
    if os.name != "posix" or not hasattr(socket, "AF_UNIX"):
        return None
    required = _rescue_telemetry_is_required()
    try:
        info = RESCUE_EVENT_SOCKET_PATH.lstat()
    except OSError:
        if required:
            raise RescueTelemetryUnavailable("required rescue telemetry unavailable")
        return None
    if (
        not stat.S_ISSOCK(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o620
        or info.st_uid == os.getuid()
        or info.st_gid not in os.getgroups()
    ):
        if required:
            raise RescueTelemetryUnavailable("required rescue telemetry socket is insecure")
        return None
    return RescueTelemetryClient(RESCUE_EVENT_SOCKET_PATH)
