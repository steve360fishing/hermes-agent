"""Process boundary broker with injected backend and terminal UNKNOWN failures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from .approvals import ApprovalStore, ApprovalToken
from .audit import JsonlAuditJournal
from .schemas import OperationRequest, SchemaError


class Outcome(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class ProductionState(str, Enum):
    PROPOSED = "PROPOSED"
    APPROVED_ONCE = "APPROVED_ONCE"
    EXECUTING = "EXECUTING"
    APPLIED = "APPLIED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"
    ROLLED_BACK = "ROLLED_BACK"


PRODUCTION_STATES = tuple(state.value for state in ProductionState)


@dataclass(frozen=True)
class TypedEnvelope:
    request: OperationRequest
    approval: ApprovalToken
    caller_id: str
    correlation_id: str

    def __post_init__(self) -> None:
        JsonlAuditJournal.validate_correlation_id(self.correlation_id)


@dataclass(frozen=True)
class BrokerResult:
    outcome: Outcome
    correlation_id: str
    detail: str
    retryable: bool = False


class TypedBackend(Protocol):
    def execute_typed(self, request: OperationRequest, correlation_id: str) -> BrokerResult: ...


class BrokerExecutor:
    """No URL/path/body/search/execute public API is intentionally provided."""

    def __init__(
        self,
        approvals: ApprovalStore,
        journal: JsonlAuditJournal,
        backend: TypedBackend,
        *,
        allowed_targets: frozenset[str],
    ) -> None:
        self._approvals, self._journal, self._backend = approvals, journal, backend
        self._allowed_targets = allowed_targets

    def dispatch(self, envelope: TypedEnvelope) -> BrokerResult:
        if type(envelope) is not TypedEnvelope or not isinstance(envelope.request, OperationRequest):
            raise TypeError("broker accepts only a TypedEnvelope with an OperationRequest")
        target_alias = getattr(envelope.request, "target_alias", None)
        if target_alias is not None and target_alias not in self._allowed_targets:
            raise SchemaError("target_alias is not in the executor allowlist")
        self._approvals.consume(envelope.approval, envelope.request, envelope.caller_id)
        self._journal.append("dispatch", envelope.correlation_id, {"operation": envelope.request.operation, "request_hash": envelope.request.request_hash(), "caller_id": envelope.caller_id})
        try:
            result = self._backend.execute_typed(envelope.request, envelope.correlation_id)
            if not isinstance(result, BrokerResult) or result.correlation_id != envelope.correlation_id:
                raise RuntimeError("backend returned an invalid typed result")
            if result.outcome is Outcome.UNKNOWN:
                result = BrokerResult(Outcome.UNKNOWN, envelope.correlation_id, result.detail, False)
        except Exception:
            result = BrokerResult(Outcome.UNKNOWN, envelope.correlation_id, "provider outcome is unknown; reconcile before any new approval", False)
        self._journal.append("result", envelope.correlation_id, {"outcome": result.outcome.value, "retryable": False})
        return result
