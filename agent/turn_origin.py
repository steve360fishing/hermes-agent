"""Typed, immutable provenance for one conversation turn.

Text, message roles, session identifiers, and client-supplied metadata are not
authority.  Callers must pass a :class:`TurnProvenance` minted at a trusted
ingress; absent or malformed values fail closed to ``UNKNOWN``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
import time
from typing import Any
from weakref import ref


def _authenticated_ingress_registry() -> tuple[Any, Any]:
    """Keep gateway-minted envelopes process-local and non-serializable.

    This protects the supported gateway/plugin event contract: a plugin can
    mutate event data, but cannot recreate or transplant a registered envelope
    onto a different bound request.  Python plugins share this process, so this
    is deliberately not presented as isolation against malicious Python code.
    """
    registered: dict[int, Any] = {}

    def register(value: object) -> None:
        key = id(value)

        def discard(reference: object) -> None:
            if registered.get(key) is reference:
                registered.pop(key, None)

        registered[key] = ref(value, discard)

    def is_registered(value: object) -> bool:
        reference = registered.get(id(value))
        return reference is not None and reference() is value

    return register, is_registered


_register_authenticated_ingress, _is_registered_authenticated_ingress = (
    _authenticated_ingress_registry()
)


def _text_digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _binding_digest(*, actor: str, text_sha256: str, platform: str, profile: str,
                    chat_id: str, thread_id: str, message_id: str, event_id: str,
                    session_scope: str, captured_at_unix_ms: int) -> str:
    canonical = json.dumps(
        {
            "actor": actor, "text_sha256": text_sha256, "platform": platform,
            "profile": profile, "chat_id": chat_id, "thread_id": thread_id,
            "message_id": message_id, "event_id": event_id,
            "session_scope": session_scope, "captured_at_unix_ms": captured_at_unix_ms,
        },
        ensure_ascii=False, separators=(",", ":"), sort_keys=True,
    )
    return _text_digest(canonical)


class TurnOrigin(str, Enum):
    AUTHENTICATED_DIRECT_USER = "authenticated_direct_user"
    RUNTIME_ASYNC_COMPLETION = "runtime_async_completion"
    GOAL_MODE_CONTINUATION = "goal_mode_continuation"
    CRON = "cron"
    SKILL = "skill"
    TOOL = "tool"
    DELEGATED_AGENT = "delegated_agent"
    MODEL_GENERATED = "model_generated"
    REPLAYED_PERSISTED_CONTENT = "replayed_persisted_content"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True, weakref_slot=True)
class TurnProvenance:
    origin: TurnOrigin
    actor_identity: str = ""
    authority_text: str = ""
    authority_text_sha256: str = ""
    platform: str = ""
    profile: str = ""
    chat_id: str = ""
    thread_id: str = ""
    message_id: str = ""
    event_id: str = ""
    session_scope: str = ""
    captured_at_unix_ms: int = 0
    binding_sha256: str = ""

    @property
    def is_authenticated_direct_user(self) -> bool:
        return (
            self.origin is TurnOrigin.AUTHENTICATED_DIRECT_USER
            and bool(self.actor_identity.strip())
            and _is_registered_authenticated_ingress(self)
            and bool(self.platform)
            and bool(self.chat_id)
            and bool(self.session_scope)
            and bool(self.message_id or self.event_id)
            and self.captured_at_unix_ms > 0
            and self.authority_text_sha256 == _text_digest(self.authority_text)
            and self.binding_sha256 == _binding_digest(
                actor=self.actor_identity,
                text_sha256=self.authority_text_sha256,
                platform=self.platform,
                profile=self.profile,
                chat_id=self.chat_id,
                thread_id=self.thread_id,
                message_id=self.message_id,
                event_id=self.event_id,
                session_scope=self.session_scope,
                captured_at_unix_ms=self.captured_at_unix_ms,
            )
        )

    def matches_bound_request(
        self, *, platform: object, profile: object, chat_id: object,
        thread_id: object, message_id: object, event_id: object,
        session_scope: object, authority_text: object,
    ) -> bool:
        """Require the sealed envelope to match this exact current request."""
        return self.is_authenticated_direct_user and all((
            self.platform == str(platform or ""),
            self.profile == str(profile or ""),
            self.chat_id == str(chat_id or ""),
            self.thread_id == str(thread_id or ""),
            self.message_id == str(message_id or ""),
            self.event_id == str(event_id or ""),
            self.session_scope == str(session_scope or ""),
            isinstance(authority_text, str),
            self.authority_text_sha256 == _text_digest(authority_text),
        ))

    @classmethod
    def authenticated_direct_user(cls, actor_identity: object) -> "TurnProvenance":
        """Deprecated compatibility entry point; it never grants authority.

        There is no context-free direct-user authority.  The gateway's sole
        ingress mint site must bind a successfully authenticated platform
        event, source identity, scope, and original text bytes.
        """
        return cls.unknown()

    @classmethod
    def internal(cls, origin: TurnOrigin) -> "TurnProvenance":
        if origin is TurnOrigin.AUTHENTICATED_DIRECT_USER:
            return cls.unknown()
        return cls(origin, "")

    @classmethod
    def unknown(cls) -> "TurnProvenance":
        return cls(TurnOrigin.UNKNOWN, "")

    @classmethod
    def from_storage(cls, origin: object, actor_identity: object = "", metadata: Any = None) -> "TurnProvenance":
        try:
            parsed = TurnOrigin(str(origin or "").strip())
        except ValueError:
            return cls.unknown()
        # A persisted direct-user label proves only historical provenance, not
        # current authority.  Replays therefore remain visible for diagnostics
        # but cannot enter tournament intent or release authority.
        if parsed is TurnOrigin.REPLAYED_PERSISTED_CONTENT and not metadata:
            return cls.internal(TurnOrigin.REPLAYED_PERSISTED_CONTENT)
        if parsed in {
            TurnOrigin.AUTHENTICATED_DIRECT_USER,
            TurnOrigin.REPLAYED_PERSISTED_CONTENT,
        }:
            if not isinstance(metadata, dict):
                return cls.unknown()
            actor = str(actor_identity or "").strip()
            required = ("platform", "chat_id", "session_scope", "authority_text_sha256",
                        "captured_at_unix_ms", "binding_sha256")
            if not actor or any(not metadata.get(key) for key in required):
                return cls.unknown()
            try:
                captured_at = int(metadata["captured_at_unix_ms"])
            except (TypeError, ValueError):
                return cls.unknown()
            expected = _binding_digest(
                actor=actor, text_sha256=str(metadata["authority_text_sha256"]),
                platform=str(metadata["platform"]), profile=str(metadata.get("profile") or ""),
                chat_id=str(metadata["chat_id"]), thread_id=str(metadata.get("thread_id") or ""),
                message_id=str(metadata.get("message_id") or ""), event_id=str(metadata.get("event_id") or ""),
                session_scope=str(metadata["session_scope"]), captured_at_unix_ms=captured_at,
            )
            if not (metadata.get("message_id") or metadata.get("event_id")) or metadata["binding_sha256"] != expected:
                return cls.unknown()
            # Keep validated audit fields for persistence/forensics only.  The
            # replay origin and absent runtime seal make this permanently
            # non-executable, including after another persistence round-trip.
            return cls(
                TurnOrigin.REPLAYED_PERSISTED_CONTENT,
                actor,
                authority_text="",
                authority_text_sha256=str(metadata["authority_text_sha256"]),
                platform=str(metadata["platform"]),
                profile=str(metadata.get("profile") or ""),
                chat_id=str(metadata["chat_id"]),
                thread_id=str(metadata.get("thread_id") or ""),
                message_id=str(metadata.get("message_id") or ""),
                event_id=str(metadata.get("event_id") or ""),
                session_scope=str(metadata["session_scope"]),
                captured_at_unix_ms=captured_at,
                binding_sha256=str(metadata["binding_sha256"]),
            )
        actor = ""
        if parsed is not TurnOrigin.AUTHENTICATED_DIRECT_USER:
            actor = ""
        return cls(parsed, actor)


def coerce_turn_provenance(value: Any) -> TurnProvenance:
    """Accept only a runtime-created typed value; strings/dicts grant nothing."""
    if not isinstance(value, TurnProvenance):
        return TurnProvenance.unknown()
    if value.origin is TurnOrigin.AUTHENTICATED_DIRECT_USER:
        if value.is_authenticated_direct_user:
            return value
        return TurnProvenance.unknown()
    return TurnProvenance.from_storage(value.origin.value, value.actor_identity)


def _mint_gateway_authenticated_ingress(
    actor_identity: object, *, authority_text: object,
    platform: object = "", profile: object = "", chat_id: object = "",
    thread_id: object = "", message_id: object = "", event_id: object = "",
    session_scope: object = "",
) -> TurnProvenance:
    """Gateway-private ingress mint; gateway/run.py has the sole call site.

    Its privacy is a supported-plugin boundary, not an isolation claim against
    arbitrary malicious Python executing in this interpreter.
    """
    actor = str(actor_identity or "").strip()
    platform_value, chat_value, scope_value = str(platform or ""), str(chat_id or ""), str(session_scope or "")
    message_value, event_value = str(message_id or ""), str(event_id or "")
    if not actor or not isinstance(authority_text, str) or not platform_value or not chat_value or not scope_value or not (message_value or event_value):
        return TurnProvenance.unknown()
    text_sha256 = _text_digest(authority_text)
    captured_at = time.time_ns() // 1_000_000
    fields = dict(actor=actor, text_sha256=text_sha256, platform=platform_value,
                  profile=str(profile or ""), chat_id=chat_value, thread_id=str(thread_id or ""),
                  message_id=message_value, event_id=event_value, session_scope=scope_value,
                  captured_at_unix_ms=captured_at)
    provenance = TurnProvenance(
        TurnOrigin.AUTHENTICATED_DIRECT_USER, actor, authority_text, text_sha256,
        platform_value, fields["profile"], chat_value, fields["thread_id"], message_value,
        event_value, scope_value, captured_at,
        _binding_digest(**fields),
    )
    _register_authenticated_ingress(provenance)
    return provenance
