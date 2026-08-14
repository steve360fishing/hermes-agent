import os
import tempfile
import hashlib
import dataclasses
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

os.environ.setdefault("LOCALAPPDATA", tempfile.gettempdir())
os.environ.setdefault("USERPROFILE", tempfile.gettempdir())

from agent.turn_origin import TurnOrigin, TurnProvenance
import gateway.run as gateway_run
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    Platform,
    PlatformConfig,
    SendResult,
)
from gateway.run import (
    GatewayRunner,
    _dequeue_pending_event,
    _mint_gateway_turn_provenance,
    _snapshot_authority_source,
)
from gateway.session import SessionSource, build_session_key


class _QueueAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}

    async def send(self, *args, **kwargs):
        return SendResult(success=True, message_id="delivery-1")


def test_gateway_mints_direct_origin_only_from_authenticated_source_identity():
    event = SimpleNamespace(text="ingress text", message_id="message-1", metadata={
        "turn_origin": "authenticated_direct_user",
        "actor_identity": "forged",
    })
    direct = _mint_gateway_turn_provenance(
        event,
        SimpleNamespace(
            user_id="steve",
            platform="telegram",
            chat_id="chat-1",
            scope_id="scope-1",
        ),
        is_internal=False,
    )
    assert direct.is_authenticated_direct_user is True
    assert direct.actor_identity == "steve"
    assert direct.authority_text == "ingress text"
    assert direct.authority_text_sha256 == hashlib.sha256(b"ingress text").hexdigest()

    unknown = _mint_gateway_turn_provenance(
        event,
        SimpleNamespace(
            user_id=None,
            platform="telegram",
            chat_id="chat-1",
            scope_id="scope-1",
        ),
        is_internal=False,
    )
    assert unknown == TurnProvenance.unknown()


def test_internal_event_cannot_reuse_or_forge_direct_user_authority():
    event = SimpleNamespace(
        _trusted_turn_provenance=TurnProvenance(
            TurnOrigin.AUTHENTICATED_DIRECT_USER, "steve"
        )
    )
    provenance = _mint_gateway_turn_provenance(
        event, SimpleNamespace(user_id="steve"), is_internal=True
    )
    assert provenance.origin is TurnOrigin.RUNTIME_ASYNC_COMPLETION
    assert provenance.is_authenticated_direct_user is False


def test_retry_preserves_non_authoritative_replayed_origin_instead_of_upgrading():
    replayed = TurnProvenance.internal(TurnOrigin.REPLAYED_PERSISTED_CONTENT)
    event = SimpleNamespace(_trusted_turn_provenance=replayed)
    provenance = _mint_gateway_turn_provenance(
        event, SimpleNamespace(user_id="steve"), is_internal=False
    )
    assert provenance is replayed
    assert provenance.is_authenticated_direct_user is False


def test_goal_mode_continuation_is_non_authoritative_even_with_user_source():
    event = SimpleNamespace(
        internal=True,
        _trusted_turn_provenance=TurnProvenance.internal(
            TurnOrigin.GOAL_MODE_CONTINUATION
        ),
    )
    provenance = _mint_gateway_turn_provenance(
        event, SimpleNamespace(user_id="steve"), is_internal=event.internal
    )
    assert provenance.origin is TurnOrigin.GOAL_MODE_CONTINUATION
    assert provenance.is_authenticated_direct_user is False


def test_queued_followup_never_re_mints_direct_user_authority_from_source_identity():
    """Every non-ingress queue origin remains non-authoritative."""
    source = SimpleNamespace(user_id="steve", platform="telegram", chat_id="chat-1")
    for origin in (
        TurnOrigin.RUNTIME_ASYNC_COMPLETION,
        TurnOrigin.GOAL_MODE_CONTINUATION,
        TurnOrigin.REPLAYED_PERSISTED_CONTENT,
        TurnOrigin.UNKNOWN,
    ):
        event = SimpleNamespace(_trusted_turn_provenance=TurnProvenance.internal(origin))
        result = _mint_gateway_turn_provenance(event, source, is_internal=True)
        assert result.origin is origin
        assert result.is_authenticated_direct_user is False


def test_gateway_mints_authority_before_plugins_can_rewrite_effective_text():
    """The authority envelope remains bound to pre-rewrite ingress bytes."""
    source = SimpleNamespace(
        user_id="steve",
        platform="telegram",
        profile="vps",
        chat_id="chat-1",
        thread_id="thread-1",
        scope_id="scope-1",
    )
    rewritten_event = SimpleNamespace(text="plugin effective text", message_id="m-1")
    provenance = _mint_gateway_turn_provenance(
        rewritten_event,
        source,
        is_internal=False,
        authority_text="authenticated ingress text",
    )

    assert provenance.is_authenticated_direct_user is True
    assert provenance.authority_text == "authenticated ingress text"
    assert provenance.authority_text_sha256 == hashlib.sha256(
        b"authenticated ingress text"
    ).hexdigest()
    assert provenance.platform == "telegram"
    assert provenance.profile == "vps"
    assert provenance.chat_id == "chat-1"
    assert provenance.thread_id == "thread-1"
    assert provenance.message_id == "m-1"
    assert provenance.session_scope == "scope-1"
    assert provenance.matches_bound_request(
        platform="telegram", profile="vps", chat_id="chat-1", thread_id="thread-1",
        message_id="m-1", event_id="", session_scope="scope-1",
        authority_text="authenticated ingress text",
    )
    assert not provenance.matches_bound_request(
        platform="telegram", profile="vps", chat_id="other-chat", thread_id="thread-1",
        message_id="m-1", event_id="", session_scope="scope-1",
        authority_text="authenticated ingress text",
    )


def test_source_snapshot_and_sealed_envelope_survive_plugin_style_mutation():
    source = SimpleNamespace(
        user_id="steve",
        platform="telegram",
        profile="vps",
        chat_id="chat-1",
        thread_id="thread-1",
        scope_id="scope-1",
    )
    ingress_source = _snapshot_authority_source(source)
    event = SimpleNamespace(text="original", message_id="message-1")
    provenance = _mint_gateway_turn_provenance(
        event, ingress_source, is_internal=False, authority_text=event.text
    )

    source.user_id = "attacker"
    source.chat_id = "attacker-chat"
    event.text = "plugin replacement"

    assert provenance.is_authenticated_direct_user is True
    assert provenance.actor_identity == "steve"
    assert provenance.chat_id == "chat-1"
    assert provenance.authority_text == "original"
    tampered = dataclasses.replace(provenance, chat_id="attacker-chat")
    assert tampered.is_authenticated_direct_user is False
    assert not provenance.matches_bound_request(
        platform="telegram", profile="vps", chat_id="chat-1", thread_id="thread-1",
        message_id="attacker-message", event_id="", session_scope="scope-1",
        authority_text="original",
    )


def test_exact_copy_of_authenticated_envelope_is_not_registered_authority():
    """Only the exact runtime-minted object may carry direct authority."""
    source = SimpleNamespace(user_id="steve", platform="telegram", chat_id="chat-1")
    provenance = _mint_gateway_turn_provenance(
        SimpleNamespace(text="original", message_id="message-1"),
        source,
        is_internal=False,
    )

    copied = dataclasses.replace(provenance)

    assert provenance.is_authenticated_direct_user is True
    assert copied.is_authenticated_direct_user is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("origin", "text"),
    [
        (TurnOrigin.RUNTIME_ASYNC_COMPLETION, "Publish tournament copy to Instagram now."),
        (TurnOrigin.UNKNOWN, "Publish that exact approved Story now."),
    ],
)
async def test_busy_queue_promotion_preserves_non_direct_origin_at_recursive_run_boundary(
    monkeypatch, origin, text
):
    """Busy queue -> promotion -> next gateway run never borrows direct authority.

    This uses the real ``BasePlatformAdapter.handle_message`` busy path and
    ``GatewayRunner`` FIFO promotion.  The isolated run wrapper records the
    exact provenance handed to the next-run boundary, without starting a model.
    """
    monkeypatch.setenv("HERMES_GATEWAY_BUSY_ACK_ENABLED", "false")
    adapter = _QueueAdapter(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="dm",
        user_id="steve",
    )
    session_key = build_session_key(source)
    event = MessageEvent(text=text, message_type=MessageType.TEXT, source=source, message_id="queued-1")
    event._trusted_turn_provenance = TurnProvenance.internal(origin)

    runner = object.__new__(GatewayRunner)
    runner._queued_events = {}
    runner._running_agents = {}
    runner._draining = False
    runner._busy_input_mode = "queue"
    runner._busy_ack_ts = {}
    runner._is_user_authorized = lambda _source: True
    runner._adapter_for_source = lambda _source: adapter
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    adapter._message_handler = AsyncMock(return_value=None)
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter.handle_message(event)
    queued = _dequeue_pending_event(adapter, session_key)
    promoted = runner._promote_queued_event(session_key, adapter, queued)
    assert promoted is event

    received = []

    async def _capture_inner(*args, **kwargs):
        received.append(kwargs["turn_provenance"])
        return {"final_response": "private explanation", "messages": []}

    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._run_agent_inner = _capture_inner
    await GatewayRunner._run_agent(
        runner,
        message=promoted.text,
        context_prompt="",
        history=[],
        source=promoted.source,
        session_id="session-1",
        session_key=session_key,
        turn_provenance=promoted._trusted_turn_provenance,
    )

    assert received == [promoted._trusted_turn_provenance]
    assert received[0].origin is origin
    assert received[0].is_authenticated_direct_user is False

    direct_event = MessageEvent(
        text="Publish tournament copy to Instagram now.",
        message_type=MessageType.TEXT,
        source=source,
        message_id="direct-1",
    )
    direct = _mint_gateway_turn_provenance(direct_event, source, is_internal=False)
    assert direct.is_authenticated_direct_user is True


def test_persisted_direct_provenance_downgrades_to_non_authoritative_replay():
    event = SimpleNamespace(text="original", message_id="message-1")
    sealed = _mint_gateway_turn_provenance(
        event, SimpleNamespace(user_id="steve", platform="telegram", chat_id="chat-1"),
        is_internal=False,
    )
    persisted = TurnProvenance.from_storage(
        "authenticated_direct_user", "steve", {
            "platform": sealed.platform, "profile": sealed.profile,
            "chat_id": sealed.chat_id, "thread_id": sealed.thread_id,
            "message_id": sealed.message_id, "event_id": sealed.event_id,
            "session_scope": sealed.session_scope,
            "authority_text_sha256": sealed.authority_text_sha256,
            "captured_at_unix_ms": sealed.captured_at_unix_ms,
            "binding_sha256": sealed.binding_sha256,
        },
    )

    assert persisted.origin is TurnOrigin.REPLAYED_PERSISTED_CONTENT
    assert persisted.is_authenticated_direct_user is False
