"""Credential-free typed serializer for the VPS control boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .broker import BrokerResult, Outcome, TypedEnvelope
from .audit import JsonlAuditJournal
from .schemas import parse_operation


@dataclass(frozen=True)
class VpsControlRequest:
    operation: str
    arguments: Mapping[str, str]
    request_hash: str
    correlation_id: str

    @classmethod
    def from_envelope(cls, envelope: TypedEnvelope) -> "VpsControlRequest":
        return cls(envelope.request.operation, envelope.request.arguments(), envelope.request.request_hash(), envelope.correlation_id)

    def to_dict(self) -> dict[str, Any]:
        return {"operation": self.operation, "arguments": dict(self.arguments), "request_hash": self.request_hash, "correlation_id": self.correlation_id}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "VpsControlRequest":
        if set(raw) != {"operation", "arguments", "request_hash", "correlation_id"}:
            raise ValueError("VPS request schema is closed")
        request = parse_operation({"operation": raw["operation"], "arguments": raw["arguments"]})
        if raw["request_hash"] != request.request_hash():
            raise ValueError("VPS request binding is invalid")
        JsonlAuditJournal.validate_correlation_id(raw["correlation_id"])
        return cls(request.operation, request.arguments(), request.request_hash(), raw["correlation_id"])


@dataclass(frozen=True)
class VpsControlStatus:
    outcome: Outcome
    correlation_id: str
    detail: str
    retryable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"outcome": self.outcome.value, "correlation_id": self.correlation_id, "detail": self.detail, "retryable": False}


class VpsControlAdapter:
    """Serialization only; it has no credential inputs, environment access, or network calls."""

    @staticmethod
    def serialize(envelope: TypedEnvelope) -> dict[str, Any]:
        return VpsControlRequest.from_envelope(envelope).to_dict()

    @staticmethod
    def deserialize_status(raw: Mapping[str, Any]) -> BrokerResult:
        if set(raw) != {"outcome", "correlation_id", "detail", "retryable"}:
            raise ValueError("VPS status schema is closed")
        if not isinstance(raw["outcome"], str) or not isinstance(raw["detail"], str) or type(raw["retryable"]) is not bool:
            raise ValueError("VPS status field types are invalid")
        JsonlAuditJournal.validate_correlation_id(raw["correlation_id"])
        outcome = Outcome(raw["outcome"])
        if outcome is Outcome.UNKNOWN:
            return BrokerResult(outcome, raw["correlation_id"], raw["detail"], False)
        return BrokerResult(outcome, raw["correlation_id"], raw["detail"], False)
