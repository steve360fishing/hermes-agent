"""Regression coverage for Telegram final delivery after streamed edit failure."""

from datetime import datetime
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
)
from gateway.run import AttachmentDeliveryOutcome, GatewayRunner
from gateway.session import SessionEntry, SessionSource
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _reconstructed_txt_fixture() -> Path:
    fixture = (
        Path(__file__).parents[1]
        / "fixtures"
        / "telegram"
        / "vps_txt_delivery_acceptance.txt"
    )
    content = fixture.read_bytes()
    assert len(content) == 16643
    assert hashlib.sha256(content).hexdigest() == (
        "e86bbce10cfbba20b7619e7b8dc9bfd892df01e5f6e0a35c972fb27afecfd111"
    )
    assert content.startswith(b"NON-ORIGINAL SYNTHETIC RECONSTRUCTION\n")
    return fixture


def _adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.REQUIRES_EDIT_FINALIZE = True
    adapter.FALLBACK_ON_FINAL_EDIT_FLOOD = True
    adapter.RESEND_FINAL_ON_EMPTY_STREAM_FALLBACK = True
    adapter.MAX_MESSAGE_LENGTH = 4096
    adapter.edit_message = AsyncMock()
    adapter.send = AsyncMock()
    adapter.delete_message = AsyncMock(return_value=True)
    return adapter


def _media_event() -> MessageEvent:
    return MessageEvent(
        text="",
        message_type=MessageType.TEXT,
        source=SimpleNamespace(
            chat_id="chat-1", platform="telegram", thread_id=None, chat_type="dm"
        ),
    )


def _media_runner() -> SimpleNamespace:
    return SimpleNamespace(
        _thread_metadata_for_source=lambda _source, _anchor=None: {"notify": True},
        _reply_anchor_for_event=lambda _event: None,
    )


def _real_gateway_runner(monkeypatch, tmp_path):
    runner = GatewayRunner(GatewayConfig())
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    runner._session_db = MagicMock()
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:main:telegram:dm:12345",
        session_id="sess-attachment-outcome",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100_000,
    )
    return runner


@pytest.mark.asyncio
async def test_real_already_streamed_caller_retains_ambiguous_attachment_outcome(
    monkeypatch, tmp_path
):
    runner = _real_gateway_runner(monkeypatch, tmp_path)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="steve",
    )
    event = MessageEvent(
        text="give me the file",
        source=source,
        message_id="91",
    )
    agent_result = {
        "final_response": "MEDIA:/tmp/evidence.txt",
        "messages": [
            {"role": "user", "content": "give me the file"},
            {"role": "assistant", "content": "MEDIA:/tmp/evidence.txt"},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": False,
        "already_sent": True,
    }
    runner._run_agent = AsyncMock(return_value=agent_result)
    config = PlatformConfig(enabled=True, token="test-token")
    config.typing_indicator = False
    delivery_adapter = TelegramAdapter(config)
    delivery_adapter._run_processing_hook = AsyncMock()
    runner._adapter_for_source = lambda _source: delivery_adapter
    ambiguous = AttachmentDeliveryOutcome(
        path="/tmp/evidence.txt",
        dispatch_attempted=True,
        delivered=False,
        message_id_confirmed=False,
        state="ambiguous",
        error_code="document_dispatch_exception",
    )
    runner._deliver_media_from_response = AsyncMock(return_value=(ambiguous,))

    async def _gateway_handler(inbound_event):
        return await runner._handle_message_with_agent(
            inbound_event,
            source,
            "agent:main:telegram:dm:12345",
            1,
        )

    delivery_adapter._message_handler = _gateway_handler

    await delivery_adapter._process_message_background(
        event,
        "agent:main:telegram:dm:12345",
    )

    assert event._attachment_delivery_outcomes == (ambiguous,)
    assert agent_result["attachment_delivery_state"] == "ambiguous"
    runner._deliver_media_from_response.assert_awaited_once()
    complete_calls = [
        call
        for call in delivery_adapter._run_processing_hook.await_args_list
        if call.args and call.args[0] == "on_processing_complete"
    ]
    assert complete_calls[-1].args[-1] is ProcessingOutcome.FAILURE


@pytest.mark.asyncio
async def test_real_already_streamed_caller_retains_retryable_preflight_state(
    monkeypatch, tmp_path
):
    runner = _real_gateway_runner(monkeypatch, tmp_path)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="steve",
    )
    event = MessageEvent(
        text="give me the file",
        source=source,
        message_id="91-preflight",
    )
    agent_result = {
        "final_response": "MEDIA:/tmp/evidence.txt",
        "messages": [
            {"role": "user", "content": "give me the file"},
            {"role": "assistant", "content": "MEDIA:/tmp/evidence.txt"},
        ],
        "tools": [],
        "history_offset": 0,
        "last_prompt_tokens": 0,
        "api_calls": 1,
        "failed": False,
        "already_sent": True,
    }
    runner._run_agent = AsyncMock(return_value=agent_result)
    config = PlatformConfig(enabled=True, token="test-token")
    config.typing_indicator = False
    delivery_adapter = TelegramAdapter(config)
    delivery_adapter._run_processing_hook = AsyncMock()
    runner._adapter_for_source = lambda _source: delivery_adapter
    preflight = AttachmentDeliveryOutcome(
        path="/tmp/evidence.txt",
        dispatch_attempted=False,
        delivered=False,
        message_id_confirmed=False,
        state="failed_pre_dispatch",
        error_code="telegram_not_connected",
    )
    runner._deliver_media_from_response = AsyncMock(return_value=(preflight,))

    async def _gateway_handler(inbound_event):
        return await runner._handle_message_with_agent(
            inbound_event,
            source,
            "agent:main:telegram:dm:12345",
            1,
        )

    delivery_adapter._message_handler = _gateway_handler
    await delivery_adapter._process_message_background(
        event,
        "agent:main:telegram:dm:12345",
    )

    assert event._attachment_delivery_outcomes == (preflight,)
    assert agent_result["attachment_delivery_state"] == "failed_pre_dispatch"
    complete_calls = [
        call
        for call in delivery_adapter._run_processing_hook.await_args_list
        if call.args and call.args[0] == "on_processing_complete"
    ]
    assert complete_calls[-1].args[-1] is ProcessingOutcome.FAILURE


@pytest.mark.asyncio
async def test_real_nonstream_text_success_cannot_mask_unconfirmed_document(
    monkeypatch, tmp_path
):
    config = PlatformConfig(enabled=True, token="test-token")
    config.typing_indicator = False
    adapter = TelegramAdapter(config)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="steve",
    )
    event = MessageEvent(text="request", source=source, message_id="92")
    attachment = _reconstructed_txt_fixture()
    adapter._message_handler = AsyncMock(
        return_value=f"Visible explanation\nMEDIA:{attachment}"
    )
    adapter._run_processing_hook = AsyncMock()
    adapter._send_with_retry = AsyncMock(
        return_value=SendResult(success=True, message_id="501")
    )
    adapter.send_document = AsyncMock(
        return_value=SendResult(success=True, message_id="")
    )
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="502"))
    adapter._get_human_delay = lambda: 0
    monkeypatch.setattr(
        adapter,
        "filter_media_delivery_paths",
        lambda media: list(media),
    )

    await adapter._process_message_background(event, "telegram:test:12345")

    complete_calls = [
        call
        for call in adapter._run_processing_hook.await_args_list
        if call.args and call.args[0] == "on_processing_complete"
    ]
    assert complete_calls
    assert complete_calls[-1].args[-1] is ProcessingOutcome.FAILURE
    adapter._send_with_retry.assert_awaited_once()
    adapter.send_document.assert_awaited_once()


@pytest.mark.asyncio
async def test_post_stream_document_without_message_id_is_not_recorded_delivered(
    monkeypatch, tmp_path
):
    """Text streaming success cannot turn an unconfirmed attachment into delivery."""
    attachment = str(_reconstructed_txt_fixture())
    attempted_dispatches = []
    monkeypatch.setattr(
        "agent.task_execution_contract.record_artifact_dispatch",
        lambda _path, **kwargs: attempted_dispatches.append(kwargs),
    )
    monkeypatch.setattr(
        BasePlatformAdapter,
        "filter_media_delivery_paths",
        staticmethod(lambda media: list(media)),
    )
    adapter = SimpleNamespace(
        name="telegram-test",
        extract_media=lambda _response: ([(attachment, False)], ""),
        extract_images=lambda text: ([], text),
        extract_local_files=lambda text: ([], text),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="")),
        send=AsyncMock(return_value=SendResult(success=True, message_id="notice-1")),
        send_voice=AsyncMock(),
        send_video=AsyncMock(),
        send_multiple_images=AsyncMock(),
    )

    outcomes = await GatewayRunner._deliver_media_from_response(
        _media_runner(),
        f"MEDIA:{attachment}",
        _media_event(),
        adapter,
    )

    assert attempted_dispatches == [
        {"state": "ambiguous", "error_code": "document_message_id_missing"}
    ]
    adapter.send.assert_awaited_once()
    assert len(outcomes) == 1
    assert outcomes[0].dispatch_attempted is True
    assert outcomes[0].delivered is False
    assert outcomes[0].message_id_confirmed is False
    assert outcomes[0].state == "ambiguous"
    assert outcomes[0].error_code == "document_message_id_missing"


@pytest.mark.asyncio
@pytest.mark.parametrize("message_id", ["abc", "12.5", "-1", "0", True])
async def test_post_stream_document_requires_integer_telegram_message_id(
    monkeypatch, tmp_path, message_id
):
    attachment = str(tmp_path / "evidence.txt")
    attempted_dispatches = []
    monkeypatch.setattr(
        "agent.task_execution_contract.record_artifact_dispatch",
        lambda _path, **kwargs: attempted_dispatches.append(kwargs),
    )
    monkeypatch.setattr(
        BasePlatformAdapter,
        "filter_media_delivery_paths",
        staticmethod(lambda media: list(media)),
    )
    adapter = SimpleNamespace(
        name="telegram-test",
        extract_media=lambda _response: ([(attachment, False)], ""),
        extract_images=lambda text: ([], text),
        extract_local_files=lambda text: ([], text),
        send_document=AsyncMock(
            return_value=SendResult(success=True, message_id=message_id)
        ),
        send=AsyncMock(return_value=SendResult(success=True, message_id="91")),
        send_voice=AsyncMock(),
        send_video=AsyncMock(),
        send_multiple_images=AsyncMock(),
    )

    outcomes = await GatewayRunner._deliver_media_from_response(
        _media_runner(),
        f"MEDIA:{attachment}",
        _media_event(),
        adapter,
    )

    assert attempted_dispatches == [
        {"state": "ambiguous", "error_code": "document_message_id_invalid"}
    ]
    assert outcomes[0].state == "ambiguous"
    assert outcomes[0].message_id_confirmed is False


@pytest.mark.asyncio
async def test_post_stream_document_records_confirmed_integer_id(monkeypatch):
    attachment = str(_reconstructed_txt_fixture())
    attempted_dispatches = []
    monkeypatch.setattr(
        "agent.task_execution_contract.record_artifact_dispatch",
        lambda _path, **kwargs: attempted_dispatches.append(kwargs),
    )
    monkeypatch.setattr(
        BasePlatformAdapter,
        "filter_media_delivery_paths",
        staticmethod(lambda media: list(media)),
    )
    adapter = SimpleNamespace(
        name="telegram-test",
        extract_media=lambda _response: ([(attachment, False)], ""),
        extract_images=lambda text: ([], text),
        extract_local_files=lambda text: ([], text),
        send_document=AsyncMock(
            return_value=SendResult(success=True, message_id="8123")
        ),
        send=AsyncMock(),
        send_voice=AsyncMock(),
        send_video=AsyncMock(),
        send_multiple_images=AsyncMock(),
    )

    outcomes = await GatewayRunner._deliver_media_from_response(
        _media_runner(),
        f"MEDIA:{attachment}",
        _media_event(),
        adapter,
    )

    assert attempted_dispatches == [
        {"state": "delivered", "message_id": "8123"}
    ]
    assert outcomes[0].state == "delivered"
    assert outcomes[0].delivered is True
    assert outcomes[0].message_id_confirmed is True
    assert outcomes[0].message_id == "8123"
    adapter.send_document.assert_awaited_once()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_stream_local_preflight_failure_remains_retryable(
    monkeypatch, tmp_path
):
    attachment = str(tmp_path / "evidence.txt")
    attempted_dispatches = []
    monkeypatch.setattr(
        "agent.task_execution_contract.record_artifact_dispatch",
        lambda _path, **kwargs: attempted_dispatches.append(kwargs),
    )
    monkeypatch.setattr(
        BasePlatformAdapter,
        "filter_media_delivery_paths",
        staticmethod(lambda media: list(media)),
    )
    adapter = SimpleNamespace(
        name="telegram-test",
        extract_media=lambda _response: ([(attachment, False)], ""),
        extract_images=lambda text: ([], text),
        extract_local_files=lambda text: ([], text),
        send_document=AsyncMock(
            return_value=SendResult(
                success=False,
                error="telegram_not_connected",
                retryable=True,
            )
        ),
        send=AsyncMock(return_value=SendResult(success=True, message_id="93")),
        send_voice=AsyncMock(),
        send_video=AsyncMock(),
        send_multiple_images=AsyncMock(),
    )

    outcomes = await GatewayRunner._deliver_media_from_response(
        _media_runner(),
        f"MEDIA:{attachment}",
        _media_event(),
        adapter,
    )

    assert attempted_dispatches == [
        {
            "state": "failed_pre_dispatch",
            "error_code": "telegram_not_connected",
        }
    ]
    assert outcomes[0].state == "failed_pre_dispatch"
    assert outcomes[0].dispatch_attempted is False
    assert outcomes[0].message_id_confirmed is False
    assert outcomes[0].error_code == "telegram_not_connected"


@pytest.mark.asyncio
async def test_streamed_text_delivery_is_not_attachment_delivery():
    """A clean streamed MEDIA directive never counts as document confirmation."""
    adapter = _adapter()
    adapter.send.return_value = SendResult(success=True, message_id="text-1")
    consumer = GatewayStreamConsumer(adapter, "chat-1")

    await consumer._send_or_edit("Visible explanation\nMEDIA:/tmp/evidence.txt")

    assert consumer.final_response_sent is False
    assert consumer.final_content_delivered is False


@pytest.mark.asyncio
async def test_turn_final_flood_immediately_delivers_missing_tail():
    """A short visible preview must not suppress the completed answer."""
    adapter = _adapter()
    adapter.edit_message.return_value = SendResult(
        success=False,
        error="Flood control exceeded. Retry in 180 seconds",
        retry_after=180.0,
    )
    adapter.send.return_value = SendResult(success=True, message_id="tail-1")

    consumer = GatewayStreamConsumer(
        adapter,
        "chat-1",
        StreamConsumerConfig(cursor=" ▉"),
        metadata={"thread_id": "77"},
    )
    consumer._message_id = "preview-1"
    consumer._last_sent_text = ":("
    consumer._already_sent = True

    ok = await consumer._send_or_edit(
        ":( The completed answer follows here.",
        finalize=True,
        is_turn_final=True,
    )

    assert ok is False
    assert consumer._flood_strikes == 1
    assert consumer._fallback_final_send is True
    assert consumer.final_content_delivered is False
    assert adapter.edit_message.await_count == 1

    await consumer._send_fallback_final(":( The completed answer follows here.")

    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.kwargs["content"] == "The completed answer follows here."
    assert adapter.send.await_args.kwargs["metadata"] == {
        "thread_id": "77",
        "notify": True,
    }
    adapter.delete_message.assert_not_awaited()
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_non_opt_in_adapter_keeps_adaptive_final_edit_retry():
    """Immediate final fallback remains scoped to opted-in adapters."""
    adapter = _adapter()
    adapter.FALLBACK_ON_FINAL_EDIT_FLOOD = False
    adapter.edit_message.return_value = SendResult(
        success=False,
        error="Flood control exceeded. Retry in 30 seconds",
        retry_after=30.0,
    )

    consumer = GatewayStreamConsumer(adapter, "chat-1")
    consumer._message_id = "preview-1"
    consumer._last_sent_text = "partial"
    consumer._already_sent = True

    ok = await consumer._send_or_edit(
        "partial plus final",
        finalize=True,
        is_turn_final=True,
    )

    assert ok is False
    assert consumer._flood_strikes == 1
    assert consumer._fallback_final_send is False


@pytest.mark.asyncio
async def test_turn_final_flood_commits_empty_tail_as_fresh_message():
    """Telegram gets a durable final even when the internal tail is empty."""
    adapter = _adapter()
    adapter.edit_message.return_value = SendResult(
        success=False,
        error="Flood control exceeded. Retry in 30 seconds",
        retry_after=30.0,
    )
    adapter.send.return_value = SendResult(success=True, message_id="final-1")

    consumer = GatewayStreamConsumer(
        adapter,
        "chat-1",
        StreamConsumerConfig(cursor=" ▉"),
    )
    final_text = "The complete answer"
    consumer._message_id = "preview-1"
    consumer._preview_message_ids = {"preview-1"}
    consumer._last_sent_text = f"{final_text} ▉"
    consumer._already_sent = True

    ok = await consumer._send_or_edit(
        final_text,
        finalize=True,
        is_turn_final=True,
    )

    assert ok is False
    assert consumer._fallback_final_send is True
    assert consumer.final_content_delivered is True
    assert adapter.edit_message.await_count == 1

    await consumer._send_fallback_final(final_text)

    adapter.send.assert_awaited_once()
    assert adapter.send.await_args.kwargs["content"] == final_text
    assert adapter.send.await_args.kwargs["metadata"] == {"notify": True}
    adapter.delete_message.assert_awaited_once_with("chat-1", "preview-1")
    assert consumer.message_id == "final-1"
    assert consumer.final_response_sent is True
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_empty_tail_commit_honors_retry_after(monkeypatch):
    adapter = _adapter()
    adapter.send.side_effect = [
        SendResult(
            success=False,
            error="Flood control exceeded",
            retry_after=3.0,
        ),
        SendResult(success=True, message_id="final-1"),
    ]
    sleep = AsyncMock()
    monkeypatch.setattr("gateway.stream_consumer.asyncio.sleep", sleep)

    consumer = GatewayStreamConsumer(adapter, "chat-1")
    consumer._message_id = "preview-1"
    consumer._last_sent_text = "Final answer"
    consumer._fallback_final_send = True

    await consumer._send_fallback_final("Final answer")

    assert adapter.send.await_count == 2
    sleep.assert_awaited_once_with(3.0)
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_empty_tail_recovery_keeps_prior_segment_messages():
    """Recovery replaces only its current preview, not earlier preambles."""
    adapter = _adapter()
    adapter.send.return_value = SendResult(success=True, message_id="final-1")
    consumer = GatewayStreamConsumer(adapter, "chat-1")

    consumer._track_preview_id("preamble-1")
    consumer._reset_segment_state()
    consumer._track_preview_id("preview-1")
    consumer._message_id = "preview-1"
    consumer._last_sent_text = "Final answer"
    consumer._fallback_final_send = True

    await consumer._send_fallback_final("Final answer")

    adapter.delete_message.assert_awaited_once_with("chat-1", "preview-1")
    assert "preamble-1" in consumer._preview_message_ids


@pytest.mark.asyncio
async def test_empty_tail_commit_skips_long_flood_retry(monkeypatch):
    adapter = _adapter()
    adapter.send.return_value = SendResult(
        success=False,
        error="flood_control:30.0",
        retry_after=30.0,
    )
    sleep = AsyncMock()
    monkeypatch.setattr("gateway.stream_consumer.asyncio.sleep", sleep)

    consumer = GatewayStreamConsumer(adapter, "chat-1")
    consumer._message_id = "preview-1"
    consumer._last_sent_text = "Final answer"
    consumer._fallback_final_send = True

    await consumer._send_fallback_final("Final answer")

    adapter.send.assert_awaited_once()
    sleep.assert_not_awaited()
    assert consumer.final_response_sent is False
    assert consumer.final_content_delivered is False


@pytest.mark.asyncio
async def test_telegram_long_flood_result_keeps_retry_after():
    """The real adapter contract preserves the server delay for consumers."""
    class FloodError(Exception):
        retry_after = 30.0

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = MagicMock()
    adapter._bot.edit_message_text = AsyncMock(side_effect=FloodError("Retry after 30"))

    result = await adapter.edit_message("123", "456", "Final answer", finalize=False)

    assert result.success is False
    assert result.error == "flood_control:30.0"
    assert result.retry_after == 30.0


@pytest.mark.asyncio
async def test_ambiguous_empty_tail_timeout_preserves_duplicate_suppression():
    adapter = _adapter()
    adapter.send.return_value = SimpleNamespace(
        success=False,
        error="Timed out",
        retryable=False,
    )

    consumer = GatewayStreamConsumer(adapter, "chat-1")
    consumer._message_id = "preview-1"
    consumer._last_sent_text = "Final answer"
    consumer._fallback_final_send = True

    await consumer._send_fallback_final("Final answer")

    adapter.delete_message.assert_not_awaited()
    assert consumer.final_response_sent is False
    assert consumer.final_content_delivered is True


@pytest.mark.asyncio
async def test_confirmed_empty_tail_send_failure_allows_gateway_retry():
    adapter = _adapter()
    adapter.send.return_value = SendResult(
        success=False,
        error="network unavailable",
        retryable=False,
    )

    consumer = GatewayStreamConsumer(adapter, "chat-1")
    consumer._message_id = "preview-1"
    consumer._last_sent_text = "Final answer"
    consumer._fallback_final_send = True
    consumer._final_content_delivered = True

    await consumer._send_fallback_final("Final answer")

    adapter.delete_message.assert_not_awaited()
    assert consumer.final_response_sent is False
    assert consumer.final_content_delivered is False


def test_timeout_exception_is_treated_as_ambiguous_delivery():
    class TimedOut(Exception):
        pass

    assert GatewayStreamConsumer._send_failure_may_have_delivered(
        TimedOut("request timed out")
    ) is True
