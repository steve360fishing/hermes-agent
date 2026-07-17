import asyncio
import json
import os
import stat
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from plugins.platforms.telegram import adapter as tg_adapter  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402
from plugins.platforms.telegram import liveness  # noqa: E402


SHA = "a" * 40


@pytest.fixture(autouse=True)
def baked_artifact_sha(monkeypatch):
    monkeypatch.setattr(liveness, "_read_baked_build_sha", lambda: SHA)
    monkeypatch.delenv("HERMES_REVISION", raising=False)


def test_marker_success_is_atomic_schema_v1_and_mode_0600(tmp_path, monkeypatch):
    marker = tmp_path / liveness.MARKER_NAME
    monkeypatch.setattr(liveness.time, "time", lambda: 1_000.0)
    monkeypatch.setattr(liveness.time, "monotonic", lambda: 200.0)

    assert liveness.write_polling_liveness_marker(path=marker, source_sha=SHA)

    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": 1,
        "producer_source_sha": SHA,
        "generated_at_unix": 1_000.0,
        "generated_at_monotonic": 200.0,
        "telegram_last_success_unix": 1_000.0,
        "telegram_last_success_monotonic": 200.0,
    }
    if os.name != "nt":
        assert stat.S_IMODE(marker.stat().st_mode) == 0o600
    assert not list(tmp_path.glob("*.tmp"))


def test_marker_replaces_stale_value_only_after_new_success(tmp_path, monkeypatch):
    marker = tmp_path / liveness.MARKER_NAME
    marker.write_text('{"generated_at_unix":1}', encoding="utf-8")
    os.chmod(marker, 0o600)
    monkeypatch.setattr(liveness.time, "time", lambda: 2_000.0)
    monkeypatch.setattr(liveness.time, "monotonic", lambda: 300.0)

    assert liveness.write_polling_liveness_marker(path=marker, source_sha=SHA)
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["telegram_last_success_unix"] == 2_000.0
    assert payload["telegram_last_success_monotonic"] == 300.0


def test_marker_fails_closed_without_a_full_source_sha(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_REVISION", raising=False)
    monkeypatch.setattr(liveness, "_read_baked_build_sha", lambda: None)
    marker = tmp_path / liveness.MARKER_NAME

    assert not liveness.write_polling_liveness_marker(path=marker)
    assert not marker.exists()


def test_marker_requires_environment_revision_to_match_baked_artifact(tmp_path, monkeypatch):
    marker = tmp_path / liveness.MARKER_NAME
    monkeypatch.setenv("HERMES_REVISION", "b" * 40)

    assert not liveness.write_polling_liveness_marker(path=marker)
    assert not liveness.write_polling_liveness_marker(path=marker, source_sha=SHA)
    assert not marker.exists()


def test_marker_rejects_caller_sha_that_does_not_match_baked_artifact(tmp_path):
    marker = tmp_path / liveness.MARKER_NAME

    assert not liveness.write_polling_liveness_marker(path=marker, source_sha="b" * 40)
    assert not marker.exists()


def test_failed_marker_replace_keeps_existing_marker_and_returns_false(tmp_path, monkeypatch):
    marker = tmp_path / liveness.MARKER_NAME
    marker.write_text("old", encoding="utf-8")
    os.chmod(marker, 0o600)
    monkeypatch.setattr(liveness.os, "replace", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))

    assert not liveness.write_polling_liveness_marker(path=marker, source_sha=SHA)
    assert marker.read_text(encoding="utf-8") == "old"


def test_marker_refuses_symlinked_target_and_parent(tmp_path):
    target = tmp_path / "outside.json"
    target.write_text("outside", encoding="utf-8")
    marker = tmp_path / liveness.MARKER_NAME
    try:
        marker.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    assert not liveness.write_polling_liveness_marker(path=marker, source_sha=SHA)
    assert target.read_text(encoding="utf-8") == "outside"

    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(tmp_path, target_is_directory=True)
    assert not liveness.write_polling_liveness_marker(path=linked_parent / liveness.MARKER_NAME, source_sha=SHA)


@pytest.mark.asyncio
async def test_failed_provider_probe_does_not_write_false_heartbeat(monkeypatch):
    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter._fatal_error_code = None
    adapter._app = MagicMock()
    adapter._app.updater.running = True
    adapter._app.bot.get_me = AsyncMock(side_effect=OSError("provider unavailable"))
    adapter._polling_error_task = None
    wrote = []
    adapter._record_polling_liveness = lambda: wrote.append(True)
    adapter._handle_polling_network_error = AsyncMock()

    calls = 0

    async def sleep_once(_seconds):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise asyncio.CancelledError()

    monkeypatch.setattr(tg_adapter.asyncio, "sleep", sleep_once)
    await adapter._polling_heartbeat_loop()

    assert wrote == []
    adapter._handle_polling_network_error.assert_not_awaited()


@pytest.mark.asyncio
async def test_initial_start_polling_does_not_write_false_green():
    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter._app = MagicMock()
    adapter._app.updater.start_polling = AsyncMock(return_value=None)
    adapter._record_polling_liveness = MagicMock()

    assert await adapter._start_polling_resilient(
        drop_pending_updates=False, error_callback=None
    )

    adapter._record_polling_liveness.assert_not_called()


@pytest.mark.asyncio
async def test_polling_bootstrap_failure_stays_unhealthy_until_recovery_receives():
    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter._webhook_mode = False
    adapter._polling_receive_evidence = False
    adapter._polling_error_task = None
    adapter._app = MagicMock()
    adapter._app.updater.running = True
    adapter._record_polling_liveness = MagicMock()
    adapter._publish_safe_mode_receive_readiness = MagicMock()

    assert adapter.has_healthy_polling_receive is False
    adapter._record_completed_polling_receive()
    assert adapter.has_healthy_polling_receive is True
    adapter._publish_safe_mode_receive_readiness.assert_called_once_with()

    recovery = MagicMock()
    recovery.done.return_value = False
    adapter._polling_error_task = recovery
    assert adapter.has_healthy_polling_receive is False


@pytest.mark.asyncio
async def test_completed_empty_polling_receive_refreshes_marker(monkeypatch):
    class FakeGetUpdatesRequest:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def post(self, *args, **kwargs):
            return []

    monkeypatch.setattr(tg_adapter, "HTTPXRequest", FakeGetUpdatesRequest)
    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter._webhook_mode = False
    adapter._app = MagicMock()
    adapter._app.updater.running = True
    adapter._polling_error_task = None
    adapter._record_polling_liveness = MagicMock()

    request = tg_adapter._new_polling_liveness_request(
        on_completed_receive=adapter._record_completed_polling_receive
    )
    assert await request.post("https://api.telegram.org/bot/token/getUpdates") == []

    adapter._record_polling_liveness.assert_called_once_with()


@pytest.mark.asyncio
async def test_wedged_receive_does_not_refresh_marker(monkeypatch):
    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter._fatal_error_code = None
    adapter._app = MagicMock()
    adapter._app.updater.running = True
    adapter._app.bot.get_me = AsyncMock(return_value=MagicMock())
    adapter._polling_error_task = None
    adapter._polling_pending_stuck_count = 0
    adapter._polling_not_running_count = 0
    adapter._record_polling_liveness = MagicMock()
    adapter._handle_polling_network_error = AsyncMock()
    adapter.platform = tg_adapter.Platform.TELEGRAM

    async def sleep_once(_seconds):
        raise asyncio.CancelledError()

    monkeypatch.setattr(tg_adapter.asyncio, "sleep", sleep_once)
    await adapter._polling_heartbeat_loop()

    adapter._record_polling_liveness.assert_not_called()


@pytest.mark.asyncio
async def test_pending_consumer_health_probe_cannot_refresh_marker():
    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter._webhook_mode = False
    adapter._app = MagicMock()
    adapter._app.updater.running = True
    adapter._polling_error_task = None
    adapter._polling_pending_stuck_count = 0
    adapter._polling_not_running_count = 0
    adapter._handle_polling_network_error = AsyncMock()
    adapter.platform = tg_adapter.Platform.TELEGRAM
    bot = MagicMock()
    bot.get_webhook_info = AsyncMock(return_value=MagicMock(pending_update_count=1))

    adapter._record_polling_liveness = MagicMock()
    if await adapter._probe_pending_updates(bot, 1):
        adapter._record_polling_liveness()

    adapter._record_polling_liveness.assert_not_called()


@pytest.mark.asyncio
async def test_inbound_update_refreshes_marker_when_active_poller_delivers_it():
    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter._webhook_mode = False
    adapter._app = MagicMock()
    adapter._app.updater.running = True
    adapter._polling_error_task = None
    adapter._record_polling_liveness = MagicMock()
    update = MagicMock()
    update.effective_message = None
    update.message = None

    await adapter._handle_text_message(update, MagicMock())

    adapter._record_polling_liveness.assert_called_once_with()


@pytest.mark.asyncio
async def test_reconnect_or_stopped_updater_cannot_refresh_inbound_marker():
    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter._webhook_mode = False
    adapter._app = MagicMock()
    adapter._app.updater.running = False
    adapter._polling_error_task = MagicMock()
    adapter._polling_error_task.done.return_value = False
    adapter._record_polling_liveness = MagicMock()

    adapter._record_inbound_polling_liveness()

    adapter._record_polling_liveness.assert_not_called()
