"""
Tests for cross-platform audio/voice media routing.

These tests pin the expected delivery path for audio media files across
Telegram (where Bot-API sendAudio only accepts MP3/M4A and .ogg/.opus
only renders as a voice bubble when explicitly flagged) and via
``GatewayRunner._deliver_media_from_response``.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


class _MediaRoutingAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content=None, **kwargs):
        return SendResult(success=True, message_id="text")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


def _event(thread_id=None):
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="dm",
        thread_id=thread_id,
    )
    return MessageEvent(
        text="make speech",
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg-1",
    )


def _allowed_media_path(tmp_path, monkeypatch, name):
    root = tmp_path / "media-cache"
    media_file = root / name
    media_file.parent.mkdir(parents=True, exist_ok=True)
    media_file.write_bytes(b"media")
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
        (root,),
    )
    return media_file.resolve()


@pytest.mark.asyncio
async def test_base_adapter_routes_telegram_flac_media_tag_to_document_sender(tmp_path, monkeypatch):
    adapter = _MediaRoutingAdapter()
    event = _event()
    media_file = _allowed_media_path(tmp_path, monkeypatch, "speech.flac")
    adapter._message_handler = AsyncMock(return_value=f"MEDIA:{media_file}")
    adapter.send_voice = AsyncMock(return_value=SendResult(success=True, message_id="voice"))
    adapter.send_document = AsyncMock(return_value=SendResult(success=True, message_id="doc"))

    await adapter._process_message_background(event, build_session_key(event.source))

    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1",
        file_path=str(media_file),
        metadata={"notify": True},
    )
    adapter.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_document_false_send_result_emits_safe_visible_failure(tmp_path, monkeypatch):
    adapter = _MediaRoutingAdapter()
    event = _event()
    media_file = _allowed_media_path(tmp_path, monkeypatch, "report.txt")
    adapter._message_handler = AsyncMock(return_value=f"MEDIA:{media_file}")
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="failure"))
    adapter.send_document = AsyncMock(return_value=SendResult(success=False, error="rejected"))

    await adapter._process_message_background(event, build_session_key(event.source))

    assert adapter.send.await_count == 1
    safe_failure = adapter.send.await_args.kwargs["content"]
    assert "Couldn't deliver the requested attachment" in safe_failure
    assert str(media_file) not in safe_failure


@pytest.mark.asyncio
async def test_document_exception_emits_safe_visible_failure(tmp_path, monkeypatch):
    adapter = _MediaRoutingAdapter()
    event = _event()
    media_file = _allowed_media_path(tmp_path, monkeypatch, "report.txt")
    adapter._message_handler = AsyncMock(return_value=f"MEDIA:{media_file}")
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="failure"))
    adapter.send_document = AsyncMock(side_effect=RuntimeError("provider timeout"))

    await adapter._process_message_background(event, build_session_key(event.source))

    safe_failure = adapter.send.await_args.kwargs["content"]
    assert "Couldn't deliver the requested attachment" in safe_failure
    assert str(media_file) not in safe_failure


@pytest.mark.asyncio
async def test_failed_failure_notice_is_not_treated_as_visible_success(tmp_path, monkeypatch):
    from agent.task_execution_contract import (
        _ARTIFACT_RECEIPTS,
        build_task_execution_contract,
        record_artifact_written,
    )

    root = tmp_path / "media-cache"
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(root))
    contract = build_task_execution_contract(
        "Create and deliver report.txt containing safe text.",
        task_id="failed-notice",
        platform="telegram",
    )
    Path(contract.artifact_output_path).write_bytes(b"media")
    assert record_artifact_written(contract) is True
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (root,))

    adapter = _MediaRoutingAdapter()
    event = _event()
    adapter._message_handler = AsyncMock(return_value=f"MEDIA:{contract.artifact_output_path}")
    adapter.send_document = AsyncMock(return_value=SendResult(success=False, error="rejected"))
    adapter.send = AsyncMock(return_value=SendResult(success=False, error="notice rejected"))

    await adapter._process_message_background(event, build_session_key(event.source))

    receipt = json.loads(Path(contract.artifact_receipt_path).read_text(encoding="utf-8"))
    assert receipt["state"] == "ambiguous"
    assert receipt["error_code"] == "failure_notice_undelivered"
    assert adapter.send.await_count == 1
    assert os.path.abspath(contract.artifact_output_path) not in _ARTIFACT_RECEIPTS
    assert not Path(contract.artifact_root).exists()


@pytest.mark.asyncio
async def test_failure_notice_exception_is_attempted_once(tmp_path, monkeypatch):
    from agent.task_execution_contract import build_task_execution_contract, record_artifact_written

    root = tmp_path / "media-cache"
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(root))
    contract = build_task_execution_contract(
        "Create and deliver report.txt containing safe text.",
        task_id="failed-notice-exception",
        platform="telegram",
    )
    Path(contract.artifact_output_path).write_bytes(b"media")
    assert record_artifact_written(contract) is True
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (root,))
    adapter = _MediaRoutingAdapter()
    adapter._message_handler = AsyncMock(return_value=f"MEDIA:{contract.artifact_output_path}")
    adapter.send_document = AsyncMock(return_value=SendResult(success=False, error="rejected"))
    adapter.send = AsyncMock(side_effect=RuntimeError("notice transport failed"))

    await adapter._process_message_background(_event(), build_session_key(_event().source))

    receipt = json.loads(Path(contract.artifact_receipt_path).read_text(encoding="utf-8"))
    assert receipt["error_code"] == "failure_notice_undelivered"
    assert adapter.send.await_count == 1


@pytest.mark.asyncio
async def test_base_adapter_routes_non_voice_telegram_ogg_media_tag_to_document_sender(tmp_path, monkeypatch):
    adapter = _MediaRoutingAdapter()
    event = _event()
    media_file = _allowed_media_path(tmp_path, monkeypatch, "speech.ogg")
    adapter._message_handler = AsyncMock(return_value=f"MEDIA:{media_file}")
    adapter.send_voice = AsyncMock(return_value=SendResult(success=True, message_id="voice"))
    adapter.send_document = AsyncMock(return_value=SendResult(success=True, message_id="doc"))

    await adapter._process_message_background(event, build_session_key(event.source))

    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1",
        file_path=str(media_file),
        metadata={"notify": True},
    )
    adapter.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_base_adapter_routes_voice_tagged_telegram_ogg_media_tag_to_voice_sender(tmp_path, monkeypatch):
    adapter = _MediaRoutingAdapter()
    event = _event()
    media_file = _allowed_media_path(tmp_path, monkeypatch, "speech.ogg")
    adapter._message_handler = AsyncMock(
        return_value=f"[[audio_as_voice]]\nMEDIA:{media_file}"
    )
    adapter.send_voice = AsyncMock(return_value=SendResult(success=True, message_id="voice"))
    adapter.send_document = AsyncMock(return_value=SendResult(success=True, message_id="doc"))

    await adapter._process_message_background(event, build_session_key(event.source))

    adapter.send_voice.assert_awaited_once_with(
        chat_id="chat-1",
        audio_path=str(media_file),
        metadata={"notify": True},
    )
    adapter.send_document.assert_not_awaited()


def _fake_runner(thread_meta):
    """Build a fake GatewayRunner-like object with the helper methods needed by
    _deliver_media_from_response."""
    runner = SimpleNamespace(
        _thread_metadata_for_source=lambda source, anchor=None: thread_meta,
        _reply_anchor_for_event=lambda event: None,
    )
    return runner


@pytest.mark.asyncio
async def test_streaming_failed_failure_notice_records_undelivered(tmp_path, monkeypatch):
    from agent.task_execution_contract import build_task_execution_contract, record_artifact_written

    root = tmp_path / "media-cache"
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(root))
    contract = build_task_execution_contract(
        "Create and deliver report.txt containing safe text.",
        task_id="stream-failed-notice",
        platform="telegram",
    )
    Path(contract.artifact_output_path).write_bytes(b"media")
    assert record_artifact_written(contract) is True
    monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (root,))
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send=AsyncMock(return_value=SendResult(success=False, error="notice rejected")),
        send_voice=AsyncMock(),
        send_document=AsyncMock(return_value=SendResult(success=False, error="rejected")),
        send_image_file=AsyncMock(),
        send_video=AsyncMock(),
    )

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({"thread_id": "topic-1"}),
        f"MEDIA:{contract.artifact_output_path}",
        _event(thread_id="topic-1"),
        adapter,
    )

    receipt = json.loads(Path(contract.artifact_receipt_path).read_text(encoding="utf-8"))
    assert receipt["state"] == "ambiguous"
    assert receipt["error_code"] == "failure_notice_undelivered"
    adapter.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_streaming_delivery_routes_telegram_flac_media_tag_to_document_sender(tmp_path, monkeypatch):
    event = _event(thread_id="topic-1")
    media_file = _allowed_media_path(tmp_path, monkeypatch, "speech.flac")
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({"thread_id": "topic-1"}),
        f"MEDIA:{media_file}",
        event,
        adapter,
    )

    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1",
        file_path=str(media_file),
        metadata={"thread_id": "topic-1"},
    )
    adapter.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_streaming_delivery_routes_non_voice_telegram_ogg_media_tag_to_document_sender(tmp_path, monkeypatch):
    event = _event(thread_id="topic-1")
    media_file = _allowed_media_path(tmp_path, monkeypatch, "speech.ogg")
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({"thread_id": "topic-1"}),
        f"MEDIA:{media_file}",
        event,
        adapter,
    )

    adapter.send_document.assert_awaited_once_with(
        chat_id="chat-1",
        file_path=str(media_file),
        metadata={"thread_id": "topic-1"},
    )
    adapter.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_streaming_delivery_routes_telegram_mp3_media_tag_to_voice_sender(tmp_path, monkeypatch):
    """MP3 audio on Telegram must go through send_voice (which routes to
    sendAudio internally); Telegram accepts MP3 for the audio player."""
    event = _event(thread_id="topic-1")
    media_file = _allowed_media_path(tmp_path, monkeypatch, "speech.mp3")
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({"thread_id": "topic-1"}),
        f"MEDIA:{media_file}",
        event,
        adapter,
    )

    adapter.send_voice.assert_awaited_once_with(
        chat_id="chat-1",
        audio_path=str(media_file),
        metadata={"thread_id": "topic-1"},
    )
    adapter.send_document.assert_not_awaited()


@pytest.mark.asyncio
async def test_streaming_delivery_blocks_media_path_outside_allowed_roots(tmp_path, monkeypatch):
    event = _event(thread_id="topic-1")
    allowed_root = tmp_path / "media-cache"
    allowed_root.mkdir()
    secret = tmp_path / "outside.pdf"
    secret.write_bytes(b"%PDF secret")
    monkeypatch.setattr(
        "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS",
        (allowed_root,),
    )
    # This test exercises the strict-allowlist path; force strict mode on
    # and disable recency trust so the freshly-written tmp_path file is not
    # auto-accepted by the trust window. (Recency trust is covered separately
    # in test_platform_base.py. The public default flipped to non-strict in
    # 2026-05; this test pins strict on explicitly.)
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1")
    monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "0")
    adapter = SimpleNamespace(
        name="test",
        extract_media=BasePlatformAdapter.extract_media,
        extract_images=BasePlatformAdapter.extract_images,
        extract_local_files=BasePlatformAdapter.extract_local_files,
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )

    await GatewayRunner._deliver_media_from_response(
        _fake_runner({"thread_id": "topic-1"}),
        f"MEDIA:{secret}",
        event,
        adapter,
    )

    adapter.send_document.assert_not_awaited()
    adapter.send_voice.assert_not_awaited()
