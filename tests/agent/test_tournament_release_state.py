from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from agent.tournament_release_state import (
    PendingPublicationPacket,
    ReleaseApprovalIntake,
    ReleaseState,
    TournamentReleaseStore,
)
from agent.turn_origin import TurnProvenance
from gateway.run import _mint_gateway_turn_provenance


_DIRECT = _mint_gateway_turn_provenance(
    SimpleNamespace(text="test direct request", message_id="message-1"),
    SimpleNamespace(
        user_id="steve", platform="telegram", profile="test",
        chat_id="chat-1", thread_id=None, scope_id="session-1",
    ),
    is_internal=False,
)
_UNTRUSTED = TurnProvenance.unknown()


def _packet() -> PendingPublicationPacket:
    return PendingPublicationPacket(
        task_id="task-1",
        session_id="session-1",
        destination="instagram:sportfish-hub",
        external_publication_sink="instagram:sportfish-hub",
        private_delivery_surface="telegram:steve-private",
        candidate_sha256="a" * 64,
        actor_identity="steve",
        idempotency_key="release-1",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )


def _approval(packet: PendingPublicationPacket, **overrides) -> ReleaseApprovalIntake:
    values = {
        "task_id": packet.task_id,
        "session_id": packet.session_id,
        "destination": packet.destination,
        "candidate_sha256": packet.candidate_sha256,
        "authenticated_identity": packet.actor_identity,
        "idempotency_key": packet.idempotency_key,
        "pending_action_id": packet.pending_action_id,
        "packet_checksum": packet.checksum(),
    }
    values.update(overrides)
    return ReleaseApprovalIntake(**values)


def test_authenticated_exact_approval_resolves_current_pending_packet_without_dispatch():
    store = TournamentReleaseStore()
    packet = _packet()
    assert store.prepare(packet, provenance=_DIRECT) is packet

    approval = _approval(packet)
    result = store.approve_current(approval, provenance=_DIRECT)

    assert result.accepted is True
    assert result.packet is packet
    assert packet.state is ReleaseState.APPROVED
    assert store.current("task-1", "session-1", provenance=_DIRECT) is packet


def test_blanket_or_mismatched_approval_cannot_approve_pending_packet():
    store = TournamentReleaseStore()
    packet = _packet()
    store.prepare(packet, provenance=_DIRECT)
    result = store.approve_current(
        _approval(packet, idempotency_key=""), provenance=_DIRECT
    )
    assert result.accepted is False
    assert result.code == "approval_binding_missing"
    assert packet.state is ReleaseState.PREPARED


def test_approved_packet_survives_restart_with_exact_binding(tmp_path):
    state_path = tmp_path / "state" / "tournament-release-state.json"
    first = TournamentReleaseStore(state_path=state_path)
    packet = _packet()
    first.prepare(packet, provenance=_DIRECT)
    assert first.approve_current(
        _approval(packet), provenance=_DIRECT
    ).accepted

    restarted = TournamentReleaseStore(state_path=state_path)
    restored = restarted.current(packet.task_id, packet.session_id, provenance=_DIRECT)
    assert restored is not None
    assert restored.state is ReleaseState.APPROVED
    assert restarted.approved_for(
        session_id=packet.session_id, destination=packet.destination,
        candidate_sha256=packet.candidate_sha256, identity=packet.actor_identity,
        idempotency_key=packet.idempotency_key,
        provenance=_DIRECT,
    ) is restored


def test_tampered_persisted_state_fails_closed(tmp_path):
    state_path = tmp_path / "state.json"
    store = TournamentReleaseStore(state_path=state_path)
    store.prepare(_packet(), provenance=_DIRECT)
    state_path.write_text('{"packets":[],"checksum":"tampered"}', encoding="utf-8")

    restarted = TournamentReleaseStore(state_path=state_path)
    assert restarted.current("task-1", "session-1", provenance=_DIRECT) is None
    assert restarted.approve_current(
        _approval(_packet(), pending_action_id="missing", packet_checksum="0" * 64),
        provenance=_DIRECT,
    ).code == "release_state_unavailable"


def test_dispatch_transitions_are_one_way_and_survive_restart(tmp_path):
    state_path = tmp_path / "state.json"
    store = TournamentReleaseStore(state_path=state_path)
    packet = store.prepare(_packet(), provenance=_DIRECT)
    assert store.approve_current(
        _approval(packet), provenance=_DIRECT
    ).accepted
    assert store.transition(packet, expected=ReleaseState.APPROVED, target=ReleaseState.IN_FLIGHT, provenance=_DIRECT)
    assert not store.transition(packet, expected=ReleaseState.APPROVED, target=ReleaseState.CONSUMED, provenance=_DIRECT)
    assert store.transition(packet, expected=ReleaseState.IN_FLIGHT, target=ReleaseState.AMBIGUOUS, provenance=_DIRECT)
    assert TournamentReleaseStore(state_path=state_path).current("task-1", "session-1", provenance=_DIRECT).state is ReleaseState.AMBIGUOUS


def test_expired_prepared_packet_is_failed_pre_dispatch_and_persisted(tmp_path):
    state_path = tmp_path / "state.json"
    store = TournamentReleaseStore(state_path=state_path)
    packet = _packet()
    packet.expires_at = datetime.now(timezone.utc) + timedelta(seconds=1)
    store.prepare(packet, provenance=_DIRECT)
    packet.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    decision = store.approve_current(
        _approval(packet), provenance=_DIRECT
    )
    assert decision.code == "pending_publication_expired"
    assert TournamentReleaseStore(state_path=state_path).current("task-1", "session-1", provenance=_DIRECT).state is ReleaseState.FAILED_PRE_DISPATCH


def test_strict_packet_bound_intake_rejects_second_in_flight_and_preserves_external_sink(tmp_path):
    state_path = tmp_path / "state.json"
    store = TournamentReleaseStore(state_path=state_path)
    packet = _packet()
    store.prepare(packet, provenance=_DIRECT)
    intake = ReleaseApprovalIntake(
        task_id="new-intake-turn", session_id=packet.session_id, destination=packet.destination,
        candidate_sha256=packet.candidate_sha256, authenticated_identity=packet.actor_identity,
        idempotency_key=packet.idempotency_key, pending_action_id=packet.pending_action_id,
        packet_checksum=packet.checksum(),
    )
    assert store.approve_current(intake, provenance=_DIRECT).accepted
    assert packet.external_publication_sink == "instagram:sportfish-hub"
    assert store.transition(packet, expected=ReleaseState.APPROVED, target=ReleaseState.IN_FLIGHT, provenance=_DIRECT)
    assert store.approve_current(intake, provenance=_DIRECT).code == "pending_publication_not_prepared"


def test_failed_pre_dispatch_is_persisted_and_can_retry_same_binding_after_restart(tmp_path):
    state_path = tmp_path / "state.json"
    store = TournamentReleaseStore(state_path=state_path)
    packet = store.prepare(_packet(), provenance=_DIRECT)
    assert store.approve_current(_approval(packet), provenance=_DIRECT).accepted
    assert store.transition(packet, expected=ReleaseState.APPROVED, target=ReleaseState.IN_FLIGHT, provenance=_DIRECT)
    assert store.transition(packet, expected=ReleaseState.IN_FLIGHT, target=ReleaseState.FAILED_PRE_DISPATCH, provenance=_DIRECT)
    restarted = TournamentReleaseStore(state_path=state_path)
    restored = restarted.current(packet.task_id, packet.session_id, provenance=_DIRECT)
    assert restarted.approved_for(
        session_id=restored.session_id,
        destination=restored.destination,
        candidate_sha256=restored.candidate_sha256,
        identity=restored.actor_identity,
        idempotency_key=restored.idempotency_key,
        provenance=_DIRECT,
    ) is restored
    assert restarted.transition(
        restored, expected=ReleaseState.FAILED_PRE_DISPATCH, target=ReleaseState.IN_FLIGHT,
        provenance=_DIRECT,
    )


def test_session_resolution_is_unambiguous_and_authenticated_revocation_is_scoped(tmp_path):
    store = TournamentReleaseStore(state_path=tmp_path / "state.json")
    packet = store.prepare(_packet(), provenance=_DIRECT)
    assert store.current_for_session(packet.session_id, provenance=_DIRECT) is packet
    assert store.revoke_session(session_id=packet.session_id, authenticated_identity="other", provenance=_DIRECT) == 0
    assert packet.state is ReleaseState.PREPARED
    assert store.revoke_session(session_id=packet.session_id, authenticated_identity=packet.actor_identity, provenance=_DIRECT) == 1
    assert packet.state is ReleaseState.FAILED_PRE_DISPATCH
    assert store.approved_for(
        session_id=packet.session_id,
        destination=packet.destination,
        candidate_sha256=packet.candidate_sha256,
        identity=packet.actor_identity,
        idempotency_key=packet.idempotency_key,
        provenance=_DIRECT,
    ) is None
    assert store.current_for_session(packet.session_id, provenance=_DIRECT) is None


def test_restart_quarantines_crashed_in_flight_packet_without_replay(tmp_path):
    state_path = tmp_path / "state.json"
    store = TournamentReleaseStore(state_path=state_path)
    packet = store.prepare(_packet(), provenance=_DIRECT)
    assert store.approve_current(
        _approval(packet), provenance=_DIRECT
    ).accepted
    assert store.transition(packet, expected=ReleaseState.APPROVED, target=ReleaseState.IN_FLIGHT, provenance=_DIRECT)

    restarted = TournamentReleaseStore(state_path=state_path)
    restored = restarted.current(packet.task_id, packet.session_id, provenance=_DIRECT)
    assert restored is not None
    assert restored.state is ReleaseState.AMBIGUOUS
    assert restarted.approved_for(
        session_id=packet.session_id, destination=packet.destination,
        candidate_sha256=packet.candidate_sha256, identity=packet.actor_identity,
        idempotency_key=packet.idempotency_key,
        provenance=_DIRECT,
    ) is None


def test_untrusted_origin_cannot_observe_or_mutate_pending_authority():
    store = TournamentReleaseStore()
    packet = store.prepare(_packet(), provenance=_DIRECT)
    assert store.current(packet.task_id, packet.session_id, provenance=_UNTRUSTED) is None
    assert store.current_for_session(packet.session_id, provenance=_UNTRUSTED) is None
    assert store.approve_current(_approval(packet), provenance=_UNTRUSTED).accepted is False
    assert store.revoke_session(
        session_id=packet.session_id,
        authenticated_identity=packet.actor_identity,
        provenance=_UNTRUSTED,
    ) == 0
    assert packet.state is ReleaseState.PREPARED
