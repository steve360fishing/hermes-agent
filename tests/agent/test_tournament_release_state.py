from __future__ import annotations

from datetime import datetime, timedelta, timezone

from agent.tournament_release_state import (
    PendingPublicationPacket,
    ReleaseApprovalIntake,
    ReleaseState,
    TournamentReleaseStore,
)


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
    assert store.prepare(packet) is packet

    approval = _approval(packet)
    result = store.approve_current(approval)

    assert result.accepted is True
    assert result.packet is packet
    assert packet.state is ReleaseState.APPROVED
    assert store.current("task-1", "session-1") is packet


def test_blanket_or_mismatched_approval_cannot_approve_pending_packet():
    store = TournamentReleaseStore()
    packet = _packet()
    store.prepare(packet)
    result = store.approve_current(
        _approval(packet, idempotency_key="")
    )
    assert result.accepted is False
    assert result.code == "approval_binding_missing"
    assert packet.state is ReleaseState.PREPARED


def test_approved_packet_survives_restart_with_exact_binding(tmp_path):
    state_path = tmp_path / "state" / "tournament-release-state.json"
    first = TournamentReleaseStore(state_path=state_path)
    packet = _packet()
    first.prepare(packet)
    assert first.approve_current(
        _approval(packet)
    ).accepted

    restarted = TournamentReleaseStore(state_path=state_path)
    restored = restarted.current(packet.task_id, packet.session_id)
    assert restored is not None
    assert restored.state is ReleaseState.APPROVED
    assert restarted.approved_for(
        session_id=packet.session_id, destination=packet.destination,
        candidate_sha256=packet.candidate_sha256, identity=packet.actor_identity,
        idempotency_key=packet.idempotency_key,
    ) is restored


def test_tampered_persisted_state_fails_closed(tmp_path):
    state_path = tmp_path / "state.json"
    store = TournamentReleaseStore(state_path=state_path)
    store.prepare(_packet())
    state_path.write_text('{"packets":[],"checksum":"tampered"}', encoding="utf-8")

    restarted = TournamentReleaseStore(state_path=state_path)
    assert restarted.current("task-1", "session-1") is None
    assert restarted.approve_current(
        _approval(_packet(), pending_action_id="missing", packet_checksum="0" * 64)
    ).code == "release_state_unavailable"


def test_dispatch_transitions_are_one_way_and_survive_restart(tmp_path):
    state_path = tmp_path / "state.json"
    store = TournamentReleaseStore(state_path=state_path)
    packet = store.prepare(_packet())
    assert store.approve_current(
        _approval(packet)
    ).accepted
    assert store.transition(packet, expected=ReleaseState.APPROVED, target=ReleaseState.IN_FLIGHT)
    assert not store.transition(packet, expected=ReleaseState.APPROVED, target=ReleaseState.CONSUMED)
    assert store.transition(packet, expected=ReleaseState.IN_FLIGHT, target=ReleaseState.AMBIGUOUS)
    assert TournamentReleaseStore(state_path=state_path).current("task-1", "session-1").state is ReleaseState.AMBIGUOUS


def test_expired_prepared_packet_is_failed_pre_dispatch_and_persisted(tmp_path):
    state_path = tmp_path / "state.json"
    store = TournamentReleaseStore(state_path=state_path)
    packet = _packet()
    packet.expires_at = datetime.now(timezone.utc) + timedelta(seconds=1)
    store.prepare(packet)
    packet.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    decision = store.approve_current(
        _approval(packet)
    )
    assert decision.code == "pending_publication_expired"
    assert TournamentReleaseStore(state_path=state_path).current("task-1", "session-1").state is ReleaseState.FAILED_PRE_DISPATCH


def test_strict_packet_bound_intake_rejects_second_in_flight_and_preserves_external_sink(tmp_path):
    state_path = tmp_path / "state.json"
    store = TournamentReleaseStore(state_path=state_path)
    packet = _packet()
    store.prepare(packet)
    intake = ReleaseApprovalIntake(
        task_id="new-intake-turn", session_id=packet.session_id, destination=packet.destination,
        candidate_sha256=packet.candidate_sha256, authenticated_identity=packet.actor_identity,
        idempotency_key=packet.idempotency_key, pending_action_id=packet.pending_action_id,
        packet_checksum=packet.checksum(),
    )
    assert store.approve_current(intake).accepted
    assert packet.external_publication_sink == "instagram:sportfish-hub"
    assert store.transition(packet, expected=ReleaseState.APPROVED, target=ReleaseState.IN_FLIGHT)
    assert store.approve_current(intake).code == "pending_publication_not_prepared"


def test_failed_pre_dispatch_is_persisted_and_can_retry_same_binding_after_restart(tmp_path):
    state_path = tmp_path / "state.json"
    store = TournamentReleaseStore(state_path=state_path)
    packet = store.prepare(_packet())
    assert store.approve_current(_approval(packet)).accepted
    assert store.transition(packet, expected=ReleaseState.APPROVED, target=ReleaseState.IN_FLIGHT)
    assert store.transition(packet, expected=ReleaseState.IN_FLIGHT, target=ReleaseState.FAILED_PRE_DISPATCH)
    restarted = TournamentReleaseStore(state_path=state_path)
    restored = restarted.current(packet.task_id, packet.session_id)
    assert restarted.approved_for(
        session_id=restored.session_id,
        destination=restored.destination,
        candidate_sha256=restored.candidate_sha256,
        identity=restored.actor_identity,
        idempotency_key=restored.idempotency_key,
    ) is restored
    assert restarted.transition(
        restored, expected=ReleaseState.FAILED_PRE_DISPATCH, target=ReleaseState.IN_FLIGHT
    )


def test_session_resolution_is_unambiguous_and_authenticated_revocation_is_scoped(tmp_path):
    store = TournamentReleaseStore(state_path=tmp_path / "state.json")
    packet = store.prepare(_packet())
    assert store.current_for_session(packet.session_id) is packet
    assert store.revoke_session(session_id=packet.session_id, authenticated_identity="other") == 0
    assert packet.state is ReleaseState.PREPARED
    assert store.revoke_session(session_id=packet.session_id, authenticated_identity=packet.actor_identity) == 1
    assert packet.state is ReleaseState.FAILED_PRE_DISPATCH
    assert store.approved_for(
        session_id=packet.session_id,
        destination=packet.destination,
        candidate_sha256=packet.candidate_sha256,
        identity=packet.actor_identity,
        idempotency_key=packet.idempotency_key,
    ) is None
    assert store.current_for_session(packet.session_id) is None


def test_restart_quarantines_crashed_in_flight_packet_without_replay(tmp_path):
    state_path = tmp_path / "state.json"
    store = TournamentReleaseStore(state_path=state_path)
    packet = store.prepare(_packet())
    assert store.approve_current(
        _approval(packet)
    ).accepted
    assert store.transition(packet, expected=ReleaseState.APPROVED, target=ReleaseState.IN_FLIGHT)

    restarted = TournamentReleaseStore(state_path=state_path)
    restored = restarted.current(packet.task_id, packet.session_id)
    assert restored is not None
    assert restored.state is ReleaseState.AMBIGUOUS
    assert restarted.approved_for(
        session_id=packet.session_id, destination=packet.destination,
        candidate_sha256=packet.candidate_sha256, identity=packet.actor_identity,
        idempotency_key=packet.idempotency_key,
    ) is None
