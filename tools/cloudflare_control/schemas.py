"""Closed operation schemas for an unverified Cloudflare provider boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
import re
from typing import Any, ClassVar, Mapping


class SchemaError(ValueError):
    """Input did not match an approved closed schema."""


class SecretValue:
    """Opaque secret holder whose normal display channels never reveal its value."""

    __slots__ = ("__value",)

    def __init__(self, value: str) -> None:
        if not isinstance(value, str) or not value:
            raise SchemaError("secret must be a non-empty string")
        self.__value = value

    def __repr__(self) -> str:
        return "SecretValue([REDACTED])"

    def __str__(self) -> str:
        return "[REDACTED]"


_ALIAS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_DIGEST = re.compile(r"^[a-f0-9]{64}$")
_INJECTION = re.compile(r"[\\/\r\n\x00]|\.\.|://|[$][(]|[;&|`]|--|(?i:body=|path=|url=|search=|execute=)")


def _text(value: Any, field: str, *, digest: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{field} must be a non-empty string")
    if _INJECTION.search(value):
        raise SchemaError(f"{field} contains a disallowed control shape")
    pattern = _DIGEST if digest else _ALIAS
    if not pattern.fullmatch(value):
        raise SchemaError(f"{field} has an invalid shape")
    return value


def _closed(raw: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = set(raw) - allowed
    missing = allowed - set(raw)
    if unknown or missing:
        raise SchemaError(f"closed schema mismatch: unknown={sorted(unknown)} missing={sorted(missing)}")


@dataclass(frozen=True)
class OperationRequest:
    """Base type. Provider/account field meanings remain UNVERIFIED."""

    operation: ClassVar[str]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "OperationRequest":
        raise NotImplementedError

    def arguments(self) -> dict[str, str]:
        raise NotImplementedError

    def request_hash(self) -> str:
        canonical = json.dumps(
            {"operation": self.operation, "arguments": self.arguments()},
            sort_keys=True, separators=(",", ":"),
        )
        return sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CloudflareCapabilities(OperationRequest):
    operation: ClassVar[str] = "cloudflare_capabilities"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CloudflareCapabilities":
        _closed(raw, set())
        return cls()

    def arguments(self) -> dict[str, str]:
        return {}


@dataclass(frozen=True)
class CloudflareStatus(OperationRequest):
    operation_id: str
    operation: ClassVar[str] = "cloudflare_status"

    def __post_init__(self) -> None:
        _text(self.operation_id, "operation_id")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CloudflareStatus":
        _closed(raw, {"operation_id"})
        return cls(_text(raw["operation_id"], "operation_id"))

    def arguments(self) -> dict[str, str]:
        return {"operation_id": self.operation_id}


@dataclass(frozen=True)
class CloudflareRead(OperationRequest):
    operation_id: str
    target_alias: str
    cursor: str | None = None
    limit: int | None = None
    operation: ClassVar[str] = "cloudflare_read"

    def __post_init__(self) -> None:
        _text(self.operation_id, "operation_id")
        _text(self.target_alias, "target_alias")
        if self.cursor is not None:
            _text(self.cursor, "cursor")
        if self.limit is not None and (type(self.limit) is not int or not 1 <= self.limit <= 100):
            raise SchemaError("limit must be an integer from 1 to 100")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CloudflareRead":
        allowed = {"operation_id", "target_alias", "cursor", "limit"}
        unknown, missing = set(raw) - allowed, {"operation_id", "target_alias"} - set(raw)
        if unknown or missing:
            raise SchemaError(f"closed schema mismatch: unknown={sorted(unknown)} missing={sorted(missing)}")
        return cls(raw["operation_id"], raw["target_alias"], raw.get("cursor"), raw.get("limit"))

    def arguments(self) -> dict[str, str]:
        result = {"operation_id": self.operation_id, "target_alias": self.target_alias}
        if self.cursor is not None:
            result["cursor"] = self.cursor
        if self.limit is not None:
            result["limit"] = str(self.limit)
        return result


@dataclass(frozen=True)
class CloudflareStage(OperationRequest):
    operation_id: str
    target_alias: str
    artifact_id: str
    expected_state_hash: str
    idempotency_key: str
    operation: ClassVar[str] = "cloudflare_stage"

    def __post_init__(self) -> None:
        _text(self.operation_id, "operation_id")
        _text(self.target_alias, "target_alias")
        _text(self.artifact_id, "artifact_id")
        _text(self.expected_state_hash, "expected_state_hash", digest=True)
        _text(self.idempotency_key, "idempotency_key")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CloudflareStage":
        _closed(raw, {"operation_id", "target_alias", "artifact_id", "expected_state_hash", "idempotency_key"})
        return cls(raw["operation_id"], raw["target_alias"], raw["artifact_id"], raw["expected_state_hash"], raw["idempotency_key"])

    def arguments(self) -> dict[str, str]:
        return {"operation_id": self.operation_id, "target_alias": self.target_alias, "artifact_id": self.artifact_id, "expected_state_hash": self.expected_state_hash, "idempotency_key": self.idempotency_key}


@dataclass(frozen=True)
class CloudflarePropose(OperationRequest):
    operation_id: str
    target_alias: str
    desired_state_hash: str
    expected_state_hash: str
    rollback_hash: str
    expires_at: str
    operation: ClassVar[str] = "cloudflare_propose"

    def __post_init__(self) -> None:
        _text(self.operation_id, "operation_id")
        _text(self.target_alias, "target_alias")
        _text(self.desired_state_hash, "desired_state_hash", digest=True)
        _text(self.expected_state_hash, "expected_state_hash", digest=True)
        _text(self.rollback_hash, "rollback_hash", digest=True)
        _text(self.expires_at, "expires_at")
        try:
            parsed = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SchemaError("expires_at must be ISO-8601") from exc
        if parsed.tzinfo is None:
            raise SchemaError("expires_at must include a timezone")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CloudflarePropose":
        _closed(raw, {"operation_id", "target_alias", "desired_state_hash", "expected_state_hash", "rollback_hash", "expires_at"})
        return cls(raw["operation_id"], raw["target_alias"], raw["desired_state_hash"], raw["expected_state_hash"], raw["rollback_hash"], raw["expires_at"])

    def arguments(self) -> dict[str, str]:
        return {"operation_id": self.operation_id, "target_alias": self.target_alias, "desired_state_hash": self.desired_state_hash, "expected_state_hash": self.expected_state_hash, "rollback_hash": self.rollback_hash, "expires_at": self.expires_at}


_REQUEST_TYPES: dict[str, type[OperationRequest]] = {
    item.operation: item
    for item in (CloudflareCapabilities, CloudflareStatus, CloudflareRead, CloudflareStage, CloudflarePropose)
}


def parse_operation(raw: Mapping[str, Any]) -> OperationRequest:
    _closed(raw, {"operation", "arguments"})
    operation = raw["operation"]
    if not isinstance(operation, str) or operation not in _REQUEST_TYPES:
        raise SchemaError("operation is not in the closed capability manifest")
    arguments = raw["arguments"]
    if not isinstance(arguments, Mapping):
        raise SchemaError("arguments must be an object")
    return _REQUEST_TYPES[operation].from_dict(arguments)


CAPABILITY_MANIFEST = {
    "provider_shapes": "UNVERIFIED",
    "operations": tuple(sorted(_REQUEST_TYPES)),
    "forbidden_surfaces": ("url", "path", "body", "search", "execute", "credentials"),
}
