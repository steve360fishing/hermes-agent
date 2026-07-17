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
