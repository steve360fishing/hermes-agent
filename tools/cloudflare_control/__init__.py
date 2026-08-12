"""Typed, approval-gated Cloudflare control primitives.

Provider and account shapes are intentionally UNVERIFIED.  This package does
not authenticate, call a provider, or expose a generic execution surface.
"""

from .approvals import ApprovalStore, ApprovalToken
from .audit import JsonlAuditJournal
from .broker import BrokerExecutor, BrokerResult, Outcome, ProductionState, TypedEnvelope
from .schemas import CAPABILITY_MANIFEST, OperationRequest, SecretValue
from .vps_adapter import VpsControlAdapter, VpsControlRequest, VpsControlStatus

__all__ = [
    "ApprovalStore", "ApprovalToken", "BrokerExecutor", "BrokerResult",
    "CAPABILITY_MANIFEST", "JsonlAuditJournal", "OperationRequest", "Outcome",
    "ProductionState", "SecretValue", "TypedEnvelope", "VpsControlAdapter", "VpsControlRequest",
    "VpsControlStatus",
]
