"""Atomic, one-use approvals for typed request hashes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from contextlib import contextmanager
import json
import os
from pathlib import Path
from threading import Lock
from typing import Callable, Iterator
from uuid import uuid4

from .schemas import OperationRequest


class ApprovalError(ValueError):
    pass


@dataclass(frozen=True)
class ApprovalToken:
    approval_id: str
    request_hash: str
    caller_id: str
    expires_at: datetime
    nonce: str
    idempotency_key: str


class ApprovalStore:
    """One-use approvals, optionally durable across process restarts.

    A supplied ``state_path`` is the production-safe mode.  It uses an
    advisory process lock plus atomic replace so a second process cannot race
    a consume after reloading stale state.
    """

    def __init__(self, state_path: Path | None = None, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._state_path = state_path
        self._tokens: dict[str, ApprovalToken] = {}
        self._used: set[str] = set()
        self._nonces: set[str] = set()
        self._idempotency_keys: set[str] = set()
        self._lock = Lock()

    @contextmanager
    def _state_lock(self) -> Iterator[None]:
        if self._state_path is None:
            yield
            return
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self._state_path.with_suffix(self._state_path.suffix + ".lock")
        with lock_path.open("a+", encoding="utf-8") as handle:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                handle.write("0")
                handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _load(self) -> None:
        if self._state_path is None or not self._state_path.exists():
            return
        raw = json.loads(self._state_path.read_text(encoding="utf-8"))
        tokens = raw.get("tokens")
        used = raw.get("used")
        if not isinstance(tokens, dict) or not isinstance(used, list):
            raise ApprovalError("approval state is invalid")
        self._tokens = {
            approval_id: ApprovalToken(
                approval_id, entry["request_hash"], entry["caller_id"],
                datetime.fromisoformat(entry["expires_at"]), entry["nonce"], entry["idempotency_key"],
            )
            for approval_id, entry in tokens.items()
        }
        self._used = set(used)
        self._nonces = {token.nonce for token in self._tokens.values()}
        self._idempotency_keys = {token.idempotency_key for token in self._tokens.values()}

    def _save(self) -> None:
        if self._state_path is None:
            return
        payload = {
            "tokens": {
                approval_id: {
                    "request_hash": token.request_hash, "caller_id": token.caller_id,
                    "expires_at": token.expires_at.isoformat(), "nonce": token.nonce,
                    "idempotency_key": token.idempotency_key,
                }
                for approval_id, token in self._tokens.items()
            },
            "used": sorted(self._used),
        }
        temporary = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, self._state_path)

    def issue(self, request: OperationRequest, caller_id: str, expires_at: datetime, *, nonce: str | None = None, idempotency_key: str | None = None) -> ApprovalToken:
        if expires_at.tzinfo is None or expires_at <= self._clock():
            raise ApprovalError("expiry must be in the future and timezone-aware")
        if not caller_id:
            raise ApprovalError("caller_id is required")
        with self._lock:
            with self._state_lock():
                self._load()
                selected_nonce, selected_idempotency = nonce or str(uuid4()), idempotency_key or str(uuid4())
                if selected_nonce in self._nonces or selected_idempotency in self._idempotency_keys:
                    raise ApprovalError("nonce and idempotency_key must each be unique")
                token = ApprovalToken(str(uuid4()), request.request_hash(), caller_id, expires_at, selected_nonce, selected_idempotency)
                self._tokens[token.approval_id] = token
                self._nonces.add(token.nonce)
                self._idempotency_keys.add(token.idempotency_key)
                self._save()
        return token

    def consume(self, token: ApprovalToken, request: OperationRequest, caller_id: str) -> None:
        with self._lock:
            with self._state_lock():
                self._load()
                stored = self._tokens.get(token.approval_id)
                if stored != token:
                    raise ApprovalError("approval token is unknown or tampered")
                if token.approval_id in self._used:
                    raise ApprovalError("approval has already been consumed")
                if token.expires_at <= self._clock():
                    raise ApprovalError("approval has expired")
                if token.caller_id != caller_id or token.request_hash != request.request_hash():
                    raise ApprovalError("approval does not bind this caller and request")
                self._used.add(token.approval_id)
                self._save()
