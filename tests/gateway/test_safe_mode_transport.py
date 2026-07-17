from types import SimpleNamespace

from gateway.config import Platform, PlatformConfig


def _runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(
        group_sessions_per_user=False,
        thread_sessions_per_user=False,
    )
    return runner


def test_safe_mode_keeps_only_bundled_telegram_transport(monkeypatch):
    from plugins.platforms.telegram import adapter as telegram_adapter

    sentinel = object()
    monkeypatch.setenv("HERMES_SAFE_MODE", "1")
    monkeypatch.setattr(telegram_adapter, "check_telegram_requirements", lambda: True)
    monkeypatch.setattr(telegram_adapter, "_build_adapter", lambda _config: sentinel)

    result = _runner()._create_adapter(
        Platform.TELEGRAM,
        PlatformConfig(enabled=True, token="configured"),
    )

    assert result is sentinel


def test_safe_mode_rejects_non_recovery_transports(monkeypatch):
    monkeypatch.setenv("HERMES_SAFE_MODE", "1")

    result = _runner()._create_adapter(
        Platform.SIGNAL,
        PlatformConfig(enabled=True),
    )

    assert result is None
