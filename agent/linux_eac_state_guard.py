"""Linux host-only durable checkpoint store for Execution Authorization Contract V1.

The service that imports this module must run as root outside Hermes containers.
Its signing key is supplied through a systemd encrypted credential; this module
never creates, logs, or exposes key material.
"""
from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

try:  # Imported on Windows by cross-platform test discovery, never instantiated there.
    import fcntl
except ImportError:  # pragma: no cover - Linux-only implementation.
    fcntl = None  # type: ignore[assignment]


class LinuxEACStateGuardError(RuntimeError):
    pass


_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class LinuxEACStateGuard:
    """Atomic file-backed checkpoint adapter, confined to a root-owned directory."""

    def __init__(self, state_root: str | os.PathLike[str]) -> None:
        if fcntl is None:
            raise LinuxEACStateGuardError("Linux file locking is unavailable on this platform")
        self.root = Path(state_root)
        self._validate_root()

    def _validate_root(self) -> None:
        try:
            info = self.root.lstat()
        except OSError as error:
            raise LinuxEACStateGuardError("state root is unavailable") from error
        if (self.root.is_symlink() or not stat.S_ISDIR(info.st_mode)
                or stat.S_IMODE(info.st_mode) & 0o077):
            raise LinuxEACStateGuardError("state root must be a private regular directory")
        if os.name != "nt" and info.st_uid != 0:
            raise LinuxEACStateGuardError("state root must be root-owned")

    @staticmethod
    def _name(contract_id: str) -> str:
        if not isinstance(contract_id, str) or not _ID.fullmatch(contract_id):
            raise LinuxEACStateGuardError("contract id is invalid")
        return f"{contract_id}.checkpoint.json"

    def _path(self, contract_id: str) -> Path:
        return self.root / self._name(contract_id)

    @contextmanager
    def _lock(self, contract_id: str) -> Iterator[None]:
        path = self.root / f".{self._name(contract_id)}.lock"
        fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def read(self, contract_id: str) -> dict[str, Any] | None:
        path = self._path(contract_id)
        try:
            info = path.lstat()
        except FileNotFoundError:
            return None
        except OSError as error:
            raise LinuxEACStateGuardError("checkpoint unavailable") from error
        if path.is_symlink() or not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            raise LinuxEACStateGuardError("checkpoint path is unsafe")
        try:
            with path.open("rb") as handle:
                value = json.loads(handle.read().decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LinuxEACStateGuardError("checkpoint is invalid") from error
        if not isinstance(value, dict) or not isinstance(value.get("checkpoint_mac"), str):
            raise LinuxEACStateGuardError("checkpoint is invalid")
        return value

    def compare_and_set(self, contract_id: str, expected_mac: str | None, value: dict[str, Any]) -> bool:
        if not isinstance(value, dict) or not isinstance(value.get("checkpoint_mac"), str):
            raise LinuxEACStateGuardError("checkpoint value is invalid")
        with self._lock(contract_id):
            current = self.read(contract_id)
            if (current.get("checkpoint_mac") if current else None) != expected_mac:
                return False
            encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
            fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=self.root)
            try:
                os.fchmod(fd, 0o600)
                os.write(fd, encoded)
                os.fsync(fd)
                os.close(fd)
                fd = -1
                os.replace(temporary, self._path(contract_id))
                directory = os.open(self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                if fd >= 0:
                    os.close(fd)
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return True
