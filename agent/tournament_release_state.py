"""Exact, restart-durable publication approval state, never derived from prose."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import uuid

from hermes_constants import get_hermes_home
from agent.turn_origin import coerce_turn_provenance


class ReleaseState(str, Enum):
    PREPARED = "prepared"
    APPROVED = "approved"
    IN_FLIGHT = "in_flight"
    CONSUMED = "consumed"
    AMBIGUOUS = "ambiguous"
    FAILED_PRE_DISPATCH = "failed_pre_dispatch"


@dataclass
class PendingPublicationPacket:
    task_id: str
    session_id: str
    destination: str
    candidate_sha256: str
    actor_identity: str
    idempotency_key: str
    expires_at: datetime
    pending_action_id: str = ""
    action_tool: str = "send_message"
    private_delivery_surface: str = "direct_public"
    external_publication_sink: str = ""
    state: ReleaseState = ReleaseState.PREPARED
    retryable_pre_dispatch: bool = False

    def binding(self) -> tuple[str, str, str, str]:
        return (
            self.external_publication_sink,
            self.candidate_sha256,
            self.actor_identity,
            self.idempotency_key,
        )

    def checksum(self) -> str:
        return hashlib.sha256(json.dumps(self.to_record(), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()

    def to_record(self) -> dict[str, str]:
        record = asdict(self)
        record["expires_at"] = self.expires_at.astimezone(timezone.utc).isoformat()
        record["state"] = self.state.value
        return record

    @classmethod
    def from_record(cls, record: object) -> "PendingPublicationPacket":
        if not isinstance(record, dict):
            raise ValueError("packet is not an object")
        expires_at = datetime.fromisoformat(str(record["expires_at"]).replace("Z", "+00:00"))
        if expires_at.tzinfo is None:
            raise ValueError("packet expiry must be timezone-aware")
        packet = cls(
            task_id=str(record["task_id"]), session_id=str(record["session_id"]),
            destination=str(record["destination"]), candidate_sha256=str(record["candidate_sha256"]),
            actor_identity=str(record["actor_identity"]), idempotency_key=str(record["idempotency_key"]),
            expires_at=expires_at.astimezone(timezone.utc), state=ReleaseState(str(record["state"])),
            pending_action_id=str(record.get("pending_action_id") or ""),
            action_tool=str(record.get("action_tool") or "send_message"),
            private_delivery_surface=str(record.get("private_delivery_surface") or "direct_public"),
            external_publication_sink=str(record.get("external_publication_sink") or ""),
            retryable_pre_dispatch=record.get("retryable_pre_dispatch") is True,
        )
        if (
            not all(packet.binding())
            or len(packet.candidate_sha256) != 64
            or packet.destination != packet.external_publication_sink
            or packet.action_tool != "send_message"
            or not packet.private_delivery_surface
        ):
            raise ValueError("packet binding is incomplete")
        return packet


@dataclass(frozen=True)
class ReleaseApprovalIntake:
    task_id: str
    session_id: str
    destination: str
    candidate_sha256: str
    authenticated_identity: str
    idempotency_key: str
    pending_action_id: str = ""
    packet_checksum: str = ""


@dataclass(frozen=True)
class ReleaseStateDecision:
    accepted: bool
    code: str
    packet: PendingPublicationPacket | None = None


class TournamentReleaseStore:
    """Private JSON state with atomic replacement and canonical tamper detection."""

    def __init__(self, *, state_path: Path | None = None) -> None:
        self._path = state_path or (Path(get_hermes_home()) / "state" / "tournament-release-state.json")
        self._packets: dict[tuple[str, str], PendingPublicationPacket] = {}
        self._lock = threading.RLock()
        self._load_error = False
        self._load()

    @staticmethod
    def _checksum(payload: dict[str, object]) -> str:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or set(raw) != {"packets", "checksum"}:
                raise ValueError("invalid state envelope")
            payload = {"packets": raw["packets"]}
            if not isinstance(raw["checksum"], str) or raw["checksum"] != self._checksum(payload):
                raise ValueError("state checksum mismatch")
            packets = [PendingPublicationPacket.from_record(item) for item in raw["packets"]]
            self._packets = {(packet.task_id, packet.session_id): packet for packet in packets}
            if len(self._packets) != len(packets):
                raise ValueError("duplicate packet key")
            if any(packet.state is ReleaseState.IN_FLIGHT for packet in self._packets.values()):
                for packet in self._packets.values():
                    if packet.state is ReleaseState.IN_FLIGHT:
                        packet.state = ReleaseState.AMBIGUOUS
                self._persist()
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            self._packets = {}
            self._load_error = True

    def _persist(self) -> None:
        payload: dict[str, object] = {
            "packets": [packet.to_record() for _, packet in sorted(self._packets.items())],
        }
        envelope = {**payload, "checksum": self._checksum(payload)}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self._path.parent, 0o700)
        except OSError:
            pass
        fd, temporary = tempfile.mkstemp(prefix=".tournament-release-", dir=self._path.parent, text=True)
        try:
            try:
                os.chmod(temporary, 0o600)
            except OSError:
                pass
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(envelope, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _available(self) -> bool:
        return not self._load_error

    @staticmethod
    def _trusted_actor(provenance) -> str:
        trusted = coerce_turn_provenance(provenance)
        return trusted.actor_identity if trusted.is_authenticated_direct_user else ""

    def prepare(self, packet: PendingPublicationPacket, *, provenance) -> PendingPublicationPacket:
        if not self._available():
            raise ValueError("release state unavailable")
        if self._trusted_actor(provenance) != packet.actor_identity:
            raise ValueError("authenticated publication actor required")
        if (
            not all(packet.binding())
            or packet.destination != packet.external_publication_sink
            or packet.action_tool != "send_message"
            or not packet.private_delivery_surface
            or packet.expires_at <= datetime.now(timezone.utc)
        ):
            raise ValueError("pending publication packet must have a future exact binding")
        with self._lock:
            packet.pending_action_id = packet.pending_action_id or uuid.uuid4().hex
            self._packets[(packet.task_id, packet.session_id)] = packet
            self._persist()
        return packet

    def current(self, task_id: str, session_id: str, *, provenance) -> PendingPublicationPacket | None:
        with self._lock:
            actor = self._trusted_actor(provenance)
            packet = self._packets.get((str(task_id), str(session_id)))
            if not self._available() or not actor or packet is None:
                return None
            return packet if packet.actor_identity == actor else None

    def current_action(self, pending_action_id: str, session_id: str, *, provenance) -> PendingPublicationPacket | None:
        with self._lock:
            actor = self._trusted_actor(provenance)
            if not self._available() or not actor:
                return None
            return next((p for p in self._packets.values() if p.pending_action_id == pending_action_id and p.session_id == str(session_id) and p.actor_identity == actor), None)

    def current_for_session(self, session_id: str, *, provenance) -> PendingPublicationPacket | None:
        """Resolve only one live publication object; ambiguity fails closed."""
        now = datetime.now(timezone.utc)
        with self._lock:
            actor = self._trusted_actor(provenance)
            if not self._available() or not actor:
                return None
            current = [
                packet for packet in self._packets.values()
                if packet.session_id == str(session_id)
                and packet.actor_identity == actor
                and packet.expires_at > now
                and packet.state in {ReleaseState.PREPARED, ReleaseState.APPROVED}
            ]
            return current[0] if len(current) == 1 else None

    def revoke_session(self, *, session_id: str, authenticated_identity: str, provenance) -> int:
        """Cancel only non-dispatched packets owned by the authenticated actor."""
        changed = 0
        with self._lock:
            actor = self._trusted_actor(provenance)
            if not self._available() or not actor or actor != str(authenticated_identity):
                return 0
            for packet in self._packets.values():
                if (
                    packet.session_id == str(session_id)
                    and packet.actor_identity == str(authenticated_identity)
                    and packet.state in {ReleaseState.PREPARED, ReleaseState.APPROVED}
                ):
                    packet.state = ReleaseState.FAILED_PRE_DISPATCH
                    changed += 1
            if changed:
                self._persist()
        return changed

    def approved_for(self, *, session_id: str, destination: str, candidate_sha256: str, identity: str, idempotency_key: str, provenance) -> PendingPublicationPacket | None:
        expected = (destination, candidate_sha256, identity, idempotency_key)
        with self._lock:
            actor = self._trusted_actor(provenance)
            if not self._available() or not actor or actor != str(identity):
                return None
            for packet in self._packets.values():
                if (
                    packet.session_id == str(session_id)
                    and (
                        packet.state is ReleaseState.APPROVED
                        or (
                            packet.state is ReleaseState.FAILED_PRE_DISPATCH
                            and packet.retryable_pre_dispatch
                        )
                    )
                ):
                    if packet.binding() == expected and packet.expires_at > datetime.now(timezone.utc):
                        return packet
        return None

    def approve_current(self, intake: ReleaseApprovalIntake, *, provenance) -> ReleaseStateDecision:
        actor = self._trusted_actor(provenance)
        if not actor or actor != intake.authenticated_identity:
            return ReleaseStateDecision(False, "approval_actor_unauthenticated")
        if not all((
            intake.destination, intake.candidate_sha256, intake.authenticated_identity,
            intake.idempotency_key, intake.pending_action_id, intake.packet_checksum,
        )):
            return ReleaseStateDecision(False, "approval_binding_missing")
        with self._lock:
            if not self._available():
                return ReleaseStateDecision(False, "release_state_unavailable")
            packet = self.current_action(
                intake.pending_action_id, intake.session_id, provenance=provenance
            )
            if packet is None:
                return ReleaseStateDecision(False, "pending_publication_not_found")
            if packet.expires_at <= datetime.now(timezone.utc):
                packet.state = ReleaseState.FAILED_PRE_DISPATCH
                self._persist()
                return ReleaseStateDecision(False, "pending_publication_expired", packet)
            if packet.state is not ReleaseState.PREPARED:
                return ReleaseStateDecision(False, "pending_publication_not_prepared", packet)
            if (
                packet.binding() != (
                    intake.destination, intake.candidate_sha256,
                    intake.authenticated_identity, intake.idempotency_key,
                )
                or packet.pending_action_id != intake.pending_action_id
                or packet.checksum() != intake.packet_checksum
            ):
                return ReleaseStateDecision(False, "approval_binding_mismatch", packet)
            packet.state = ReleaseState.APPROVED
            self._persist()
            return ReleaseStateDecision(True, "release_approval_recorded", packet)

    def transition(self, packet: PendingPublicationPacket, *, expected: ReleaseState, target: ReleaseState, provenance) -> bool:
        """Persist a monotonic dispatch outcome; mismatched/replayed state fails closed."""
        allowed = {
            ReleaseState.PREPARED: {ReleaseState.APPROVED, ReleaseState.FAILED_PRE_DISPATCH},
            ReleaseState.APPROVED: {ReleaseState.IN_FLIGHT, ReleaseState.FAILED_PRE_DISPATCH},
            ReleaseState.FAILED_PRE_DISPATCH: {ReleaseState.IN_FLIGHT},
            ReleaseState.IN_FLIGHT: {
                ReleaseState.CONSUMED, ReleaseState.AMBIGUOUS,
                ReleaseState.FAILED_PRE_DISPATCH,
            },
            ReleaseState.CONSUMED: set(),
            ReleaseState.AMBIGUOUS: set(),
        }
        with self._lock:
            actor = self._trusted_actor(provenance)
            stored = self._packets.get((packet.task_id, packet.session_id))
            if (
                not self._available()
                or not actor
                or stored is None
                or stored.actor_identity != actor
                or stored is not packet
                or stored.state is not expected
                or target not in allowed[expected]
            ):
                return False
            stored.state = target
            stored.retryable_pre_dispatch = bool(
                target is ReleaseState.FAILED_PRE_DISPATCH
                and expected is ReleaseState.IN_FLIGHT
            )
            self._persist()
            return True
