"""Typed, immutable provenance for one conversation turn.

Text, message roles, session identifiers, and client-supplied metadata are not
authority.  Callers must pass a :class:`TurnProvenance` minted at a trusted
ingress; absent or malformed values fail closed to ``UNKNOWN``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


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


@dataclass(frozen=True, slots=True)
class TurnProvenance:
    origin: TurnOrigin
    actor_identity: str = ""

    @property
    def is_authenticated_direct_user(self) -> bool:
        return (
            self.origin is TurnOrigin.AUTHENTICATED_DIRECT_USER
            and bool(self.actor_identity.strip())
        )

    @classmethod
    def authenticated_direct_user(cls, actor_identity: object) -> "TurnProvenance":
        actor = str(actor_identity or "").strip()
        if not actor:
            return cls.unknown()
        return cls(TurnOrigin.AUTHENTICATED_DIRECT_USER, actor)

    @classmethod
    def internal(cls, origin: TurnOrigin) -> "TurnProvenance":
        if origin is TurnOrigin.AUTHENTICATED_DIRECT_USER:
            return cls.unknown()
        return cls(origin, "")

    @classmethod
    def unknown(cls) -> "TurnProvenance":
        return cls(TurnOrigin.UNKNOWN, "")

    @classmethod
    def from_storage(cls, origin: object, actor_identity: object = "") -> "TurnProvenance":
        try:
            parsed = TurnOrigin(str(origin or "").strip())
        except ValueError:
            return cls.unknown()
        actor = str(actor_identity or "").strip()
        if parsed is TurnOrigin.AUTHENTICATED_DIRECT_USER and not actor:
            return cls.unknown()
        if parsed is not TurnOrigin.AUTHENTICATED_DIRECT_USER:
            actor = ""
        return cls(parsed, actor)


def coerce_turn_provenance(value: Any) -> TurnProvenance:
    """Accept only a runtime-created typed value; strings/dicts grant nothing."""
    if not isinstance(value, TurnProvenance):
        return TurnProvenance.unknown()
    return TurnProvenance.from_storage(value.origin.value, value.actor_identity)
