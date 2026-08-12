"""Append-only JSONL audit records without credentials or raw secret values."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from typing import Any, Mapping
from uuid import uuid4
from uuid import UUID

from .schemas import SecretValue


class JsonlAuditJournal:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = Lock()

    @staticmethod
    def correlation_id() -> str:
        return str(uuid4())

    @staticmethod
    def validate_correlation_id(correlation_id: str) -> str:
        if not isinstance(correlation_id, str):
            raise ValueError("correlation_id must be a UUID string")
        try:
            return str(UUID(correlation_id))
        except (ValueError, AttributeError) as exc:
            raise ValueError("correlation_id must be a UUID string") from exc

    def append(self, event: str, correlation_id: str, fields: Mapping[str, Any]) -> None:
        self.validate_correlation_id(correlation_id)
        allowed = {"operation", "request_hash", "caller_id", "outcome", "retryable", "target_alias", "approval_id"}
        def scrub(value: Any) -> Any:
            if isinstance(value, SecretValue):
                return "[REDACTED]"
            if isinstance(value, Mapping):
                return {str(k): scrub(v) for k, v in value.items() if "secret" not in str(k).lower() and "credential" not in str(k).lower()}
            if isinstance(value, (list, tuple)):
                return [scrub(v) for v in value]
            return value
        safe_fields = {key: scrub(value) for key, value in fields.items() if key in allowed}
        record = {"event": event, "correlation_id": correlation_id, "at": datetime.now(timezone.utc).isoformat(), "fields": safe_fields}
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
