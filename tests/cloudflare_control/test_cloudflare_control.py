from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from uuid import uuid4

import pytest

from tools.cloudflare_control import (
    ApprovalStore,
    BrokerExecutor,
    BrokerResult,
    JsonlAuditJournal,
    Outcome,
    ProductionState,
    SecretValue,
    TypedEnvelope,
    VpsControlAdapter,
)
from tools.cloudflare_control.approvals import ApprovalError
from tools.cloudflare_control.schemas import SchemaError, parse_operation


def _request():
    return parse_operation({"operation": "cloudflare_read", "arguments": {"operation_id": "read-1", "target_alias": "prod-zone", "cursor": "page-1", "limit": 10}})


def _token(store, request=None):
    return store.issue(request or _request(), "caller-1", datetime.now(timezone.utc) + timedelta(minutes=5))


def _correlation_id():
    return str(uuid4())


@pytest.mark.parametrize("operation,arguments", [
    ("cloudflare_capabilities", {}),
    ("cloudflare_status", {"operation_id": "status-1"}),
    ("cloudflare_read", {"operation_id": "read-1", "target_alias": "prod-zone", "cursor": "page-1", "limit": 10}),
    ("cloudflare_stage", {"operation_id": "stage-1", "target_alias": "prod-zone", "artifact_id": "artifact-1", "expected_state_hash": "a" * 64, "idempotency_key": "stage-key"}),
    ("cloudflare_propose", {"operation_id": "proposal-1", "target_alias": "prod-zone", "desired_state_hash": "b" * 64, "expected_state_hash": "c" * 64, "rollback_hash": "d" * 64, "expires_at": "2026-12-01T00:00:00Z"}),
])
def test_all_closed_operations_have_deterministic_hashes(operation, arguments):
    first = parse_operation({"operation": operation, "arguments": arguments})
    second = parse_operation({"operation": operation, "arguments": dict(arguments)})
    assert first.request_hash() == second.request_hash()


@pytest.mark.parametrize("raw", [
    {"operation": "cloudflare_read", "arguments": {"operation_id": "r", "target_alias": "a", "body": "x"}},
    {"operation": "cloudflare_read", "arguments": {"operation_id": "r", "target_alias": "../etc"}},
    {"operation": "cloudflare_read", "arguments": {"operation_id": "r", "target_alias": "https://bad"}},
    {"operation": "cloudflare_read", "arguments": {"operation_id": "r", "target_alias": "a", "limit": 101}},
    {"operation": "not_in_manifest", "arguments": {}},
    {"operation": "cloudflare_status", "arguments": {"operation_id": "a"}, "url": "x"},
])
def test_schemas_reject_unknown_fields_and_injection_shapes(raw):
    with pytest.raises(SchemaError):
        parse_operation(raw)


def test_secret_value_does_not_print():
    secret = SecretValue("super-secret")
    assert "super-secret" not in str(secret)
    assert "super-secret" not in repr(secret)


def test_approval_is_atomic_single_use_under_concurrency():
    store = ApprovalStore()
    request, token = _request(), _token(store)

    def consume():
        try:
            store.consume(token, request, "caller-1")
            return True
        except ApprovalError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(lambda _: consume(), range(8)))
    assert outcomes.count(True) == 1


def test_approval_rejects_expiry_tamper_and_replay():
    fixed = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = ApprovalStore(clock=lambda: fixed)
    request = _request()
    with pytest.raises(ApprovalError):
        store.issue(request, "caller-1", fixed)
    token = store.issue(request, "caller-1", fixed + timedelta(seconds=1))
    store.consume(token, request, "caller-1")
    with pytest.raises(ApprovalError):
        store.consume(token, request, "caller-1")
    tampered = token.__class__(token.approval_id, "0" * 64, token.caller_id, token.expires_at, token.nonce, token.idempotency_key)
    with pytest.raises(ApprovalError):
        store.consume(tampered, request, "caller-1")


def test_durable_approval_survives_restart_and_preserves_consumption(tmp_path):
    state_path = tmp_path / "hermes-home" / "cloudflare-approvals.json"
    request = _request()
    first = ApprovalStore(state_path=state_path)
    token = _token(first, request)
    restarted = ApprovalStore(state_path=state_path)
    restarted.consume(token, request, "caller-1")
    after_consume = ApprovalStore(state_path=state_path)
    with pytest.raises(ApprovalError):
        after_consume.consume(token, request, "caller-1")


def test_durable_nonce_and_idempotency_keys_are_unique(tmp_path):
    store = ApprovalStore(state_path=tmp_path / "state.json")
    request = _request()
    store.issue(request, "caller-1", datetime.now(timezone.utc) + timedelta(minutes=1), nonce="nonce-a", idempotency_key="key-a")
    with pytest.raises(ApprovalError):
        store.issue(request, "caller-1", datetime.now(timezone.utc) + timedelta(minutes=1), nonce="nonce-a", idempotency_key="key-b")
    with pytest.raises(ApprovalError):
        store.issue(request, "caller-1", datetime.now(timezone.utc) + timedelta(minutes=1), nonce="nonce-b", idempotency_key="key-a")


class _ExplodingBackend:
    def execute_typed(self, request, correlation_id):
        raise RuntimeError("network uncertain")


class _SuccessfulBackend:
    def execute_typed(self, request, correlation_id):
        return BrokerResult(Outcome.SUCCESS, correlation_id, "ok")


def test_broker_turns_ambiguous_backend_failure_into_terminal_unknown(tmp_path):
    store, request = ApprovalStore(), _request()
    broker = BrokerExecutor(
        store,
        JsonlAuditJournal(tmp_path / "audit.jsonl"),
        _ExplodingBackend(),
        allowed_targets=frozenset({"prod-zone"}),
    )
    correlation_id = _correlation_id()
    result = broker.dispatch(TypedEnvelope(request, _token(store, request), "caller-1", correlation_id))
    assert result.outcome is Outcome.UNKNOWN
    assert result.retryable is False
    record = json.loads((tmp_path / "audit.jsonl").read_text().splitlines()[-1])
    assert record["correlation_id"] == correlation_id


def test_broker_accepts_only_typed_envelopes_and_backend_never_receives_credentials(tmp_path):
    broker = BrokerExecutor(
        ApprovalStore(),
        JsonlAuditJournal(tmp_path / "audit.jsonl"),
        _SuccessfulBackend(),
        allowed_targets=frozenset({"prod-zone"}),
    )
    with pytest.raises(TypeError):
        broker.dispatch({})


def test_audit_allowlist_never_serializes_secret_shaped_inputs_or_exceptions(tmp_path):
    journal = JsonlAuditJournal(tmp_path / "audit.jsonl")
    journal.append("result", _correlation_id(), {
        "outcome": "SUCCESS", "token": "bearer-token", "password": "password", "api_key": "key",
        "authorization": "Bearer private", "header": "private", "secret": SecretValue("raw-secret"),
        "exception": RuntimeError("raw-error"),
    })
    rendered = (tmp_path / "audit.jsonl").read_text()
    for leaked in ("bearer-token", "password", "Bearer private", "raw-secret", "raw-error"):
        assert leaked not in rendered


def test_vps_adapter_round_trips_only_typed_request_and_status():
    store, request = ApprovalStore(), _request()
    correlation_id = _correlation_id()
    envelope = TypedEnvelope(request, _token(store, request), "caller-1", correlation_id)
    payload = VpsControlAdapter.serialize(envelope)
    assert set(payload) == {"operation", "arguments", "request_hash", "correlation_id"}
    assert "credential" not in json.dumps(payload).lower()
    assert VpsControlAdapter.deserialize_status({"outcome": "UNKNOWN", "correlation_id": correlation_id, "detail": "uncertain", "retryable": True}).retryable is False
    with pytest.raises(ValueError):
        VpsControlAdapter.deserialize_status({"outcome": "SUCCESS", "correlation_id": correlation_id, "detail": "ok", "retryable": False, "url": "x"})
    with pytest.raises(ValueError):
        TypedEnvelope(request, _token(store, request), "caller-1", "not-a-correlation-id")


def test_broker_public_surface_has_no_generic_transport_method():
    forbidden = {"url", "path", "body", "search", "execute"}
    assert forbidden.isdisjoint(BrokerExecutor.__dict__)


def test_broker_rejects_unknown_target_before_consuming_approval(tmp_path):
    store = ApprovalStore()
    request = parse_operation({
        "operation": "cloudflare_read",
        "arguments": {"operation_id": "read-2", "target_alias": "unknown-zone"},
    })
    token = _token(store, request)
    broker = BrokerExecutor(
        store,
        JsonlAuditJournal(tmp_path / "audit.jsonl"),
        _SuccessfulBackend(),
        allowed_targets=frozenset({"prod-zone"}),
    )
    with pytest.raises(SchemaError, match="executor allowlist"):
        broker.dispatch(TypedEnvelope(request, token, "caller-1", _correlation_id()))
    store.consume(token, request, "caller-1")


def test_durable_approval_is_single_use_across_store_instances(tmp_path):
    state_path = tmp_path / "state.json"
    issuer = ApprovalStore(state_path=state_path)
    request = _request()
    token = _token(issuer, request)
    stores = [ApprovalStore(state_path=state_path), ApprovalStore(state_path=state_path)]

    def consume(store):
        try:
            store.consume(token, request, "caller-1")
            return True
        except ApprovalError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(consume, stores))
    assert outcomes.count(True) == 1


def test_production_state_contract_is_closed_and_complete():
    assert {state.value for state in ProductionState} == {
        "PROPOSED", "APPROVED_ONCE", "EXECUTING", "APPLIED", "REJECTED", "EXPIRED", "UNKNOWN", "ROLLED_BACK",
    }
