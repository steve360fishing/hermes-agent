"""Fail-closed Telegram polling liveness marker for the host watchdog."""

from __future__ import annotations

import json
import os
import re
import stat
import time
from pathlib import Path
from typing import Final


MARKER_NAME: Final = "telegram-liveness.json"
SCHEMA_VERSION: Final = 1
_SOURCE_SHA_PATTERN: Final = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


def _producer_source_sha() -> str | None:
    """Return the reviewed build/source revision, never a fabricated value."""
    candidates = (os.environ.get("HERMES_REVISION"), _read_baked_build_sha())
    for candidate in candidates:
        value = candidate.strip() if isinstance(candidate, str) else ""
        if _SOURCE_SHA_PATTERN.fullmatch(value):
            return value
    return None


def _read_baked_build_sha() -> str | None:
    try:
        return (Path(__file__).resolve().parents[3] / ".hermes_build_sha").read_text(
            encoding="utf-8"
        ).strip()
    except OSError:
        return None


def _marker_path() -> Path:
    return Path(os.environ.get("HERMES_HOME", "/opt/data")) / MARKER_NAME


def write_polling_liveness_marker(*, path: Path | None = None, source_sha: str | None = None) -> bool:
    """Atomically record successful polling evidence, returning False on any unsafe failure.

    The host reader requires a regular, uid-owned mode-0600 file.  This writer
    deliberately refuses symlinked directories or targets and never creates a
    marker without a full, reviewed producer source SHA.
    """
    target = path or _marker_path()
    sha = source_sha if source_sha is not None else _producer_source_sha()
    if not isinstance(sha, str) or not _SOURCE_SHA_PATTERN.fullmatch(sha):
        return False
    try:
        parent_info = os.lstat(target.parent)
        if not stat.S_ISDIR(parent_info.st_mode) or stat.S_ISLNK(parent_info.st_mode):
            return False
        try:
            existing = os.lstat(target)
        except FileNotFoundError:
            existing = None
        if existing is not None and (not stat.S_ISREG(existing.st_mode) or stat.S_ISLNK(existing.st_mode)):
            return False

        generated_unix = time.time()
        generated_monotonic = time.monotonic()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "producer_source_sha": sha,
            "generated_at_unix": generated_unix,
            "generated_at_monotonic": generated_monotonic,
            "telegram_last_success_unix": generated_unix,
            "telegram_last_success_monotonic": generated_monotonic,
        }
        encoded = (json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
        if os.name == "nt":  # pragma: no cover - deployment runs on Linux; exercised in Windows CI.
            return _write_windows_marker(target, encoded)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        directory_fd = os.open(target.parent, directory_flags)
    except OSError:
        return False

    temporary_name = f".{MARKER_NAME}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    temporary_fd: int | None = None
    try:
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        _write_all(temporary_fd, encoded)
        os.fsync(temporary_fd)
        os.fchmod(temporary_fd, 0o600)
        os.close(temporary_fd)
        temporary_fd = None
        # Re-check immediately before replacement so a concurrent symlink swap
        # cannot turn a best-effort marker into an unsafe write.
        try:
            existing = os.lstat(target.name, dir_fd=directory_fd)
        except FileNotFoundError:
            existing = None
        if existing is not None and (not stat.S_ISREG(existing.st_mode) or stat.S_ISLNK(existing.st_mode)):
            return False
        os.replace(temporary_name, target.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
        return True
    except OSError:
        return False
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except OSError:
            pass
        os.close(directory_fd)


def _write_windows_marker(target: Path, encoded: bytes) -> bool:
    """Portable test-host fallback; production uses the dir-fd no-follow path."""
    temporary = target.parent / f".{MARKER_NAME}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            existing = os.lstat(target)
        except FileNotFoundError:
            existing = None
        if existing is not None and (not stat.S_ISREG(existing.st_mode) or stat.S_ISLNK(existing.st_mode)):
            return False
        os.replace(temporary, target)
        os.chmod(target, 0o600)
        return True
    except OSError:
        return False
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except OSError:
            pass


def _write_all(descriptor: int, content: bytes) -> None:
    offset = 0
    while offset < len(content):
        written = os.write(descriptor, content[offset:])
        if written <= 0:
            raise OSError("short liveness marker write")
        offset += written
