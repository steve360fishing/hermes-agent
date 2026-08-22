"""Durable rescue telemetry spool regression tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest


linux_only = pytest.mark.skipif(os.name != "posix", reason="POSIX identity required")


def _spool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import agent.rescue_plane_core as core

    root = tmp_path / "client-spool"
    pending = root / "pending"
    quarantine = root / "quarantine"
    root.mkdir(mode=0o750)
    root.chmod(0o750)
    pending.mkdir(mode=0o770)
    pending.chmod(0o770)
    quarantine.mkdir(mode=0o750)
    quarantine.chmod(0o750)
    monkeypatch.setattr(core, "RESCUE_REPORTER_UID", os.getuid())
    return root


def _event(event_id: str, event: str = "tool_start") -> dict[str, object]:
    payload: dict[str, object] = {
        "event": event,
        "event_id": event_id,
        "turn_id": "turn-spool",
        "work_id": "tool-spool",
    }
    return payload


@linux_only
def test_offline_reporter_durably_spools_and_tool_scope_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent.rescue_plane_core as core

    spool = _spool(tmp_path, monkeypatch)
    client = core.RescueTelemetryClient(
        tmp_path / "missing.sock",
        spool_path=spool,
    )
    executed: list[str] = []

    with core.rescue_tool_execution_scope("turn-spool", client=client):
        executed.append("handler")

    assert executed == ["handler"]
    queued = sorted((spool / "pending").glob("*.json"))
    assert len(queued) == 2
    envelopes = [json.loads(path.read_text(encoding="ascii")) for path in queued]
    assert [item["event"]["event"] for item in envelopes] == ["tool_start", "tool_end"]
    assert all(item["schema_version"] == "hermes-rescue-spooled-event-v1" for item in envelopes)
    assert all(path.stat().st_mode & 0o777 == 0o640 for path in queued)


@linux_only
def test_required_telemetry_accepts_secure_spool_when_socket_is_down(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent.rescue_plane_core as core

    continuity = tmp_path / "continuity"
    continuity.mkdir(mode=0o750)
    continuity.chmod(0o750)
    marker = continuity / "telemetry-required-v1.json"
    marker.write_bytes(core.TELEMETRY_REQUIRED_MARKER)
    marker.chmod(0o440)
    spool = _spool(tmp_path, monkeypatch)
    monkeypatch.setattr(core, "RESCUE_TELEMETRY_REQUIRED_PATH", marker)
    monkeypatch.setattr(core, "RESCUE_EVENT_SPOOL_PATH", spool)
    monkeypatch.setattr(core, "RESCUE_EVENT_SOCKET_PATH", tmp_path / "missing.sock")

    client = core.get_rescue_telemetry_client()

    assert client is not None
    client.emit(_event("offline-1"))
    assert len(list((spool / "pending").glob("*.json"))) == 1


@linux_only
@pytest.mark.parametrize("failure", ["low_bytes", "low_inodes", "full_files"])
def test_spool_capacity_failures_still_block_execution(
    failure: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent.rescue_plane_core as core

    spool = _spool(tmp_path, monkeypatch)
    if failure == "full_files":
        monkeypatch.setattr(core, "_SPOOL_MAX_FILES", 0)
    else:
        values = SimpleNamespace(
            f_bavail=0 if failure == "low_bytes" else 1024,
            f_frsize=4096,
            f_favail=0 if failure == "low_inodes" else 1024,
        )
        monkeypatch.setattr(core.os, "statvfs", lambda _path: values)
    client = core.RescueTelemetryClient(tmp_path / "missing.sock", spool_path=spool)

    with pytest.raises(core.RescueTelemetryUnavailable, match="capacity exhausted"):
        client.emit(_event("capacity-1"))


@linux_only
def test_concurrent_spool_writes_are_unique_and_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent.rescue_plane_core as core

    spool = _spool(tmp_path, monkeypatch)
    client = core.RescueTelemetryClient(tmp_path / "missing.sock", spool_path=spool)
    failures: list[BaseException] = []

    def emit(index: int) -> None:
        try:
            client.emit(_event(f"concurrent-{index}"))
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    threads = [threading.Thread(target=emit, args=(index,)) for index in range(24)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    queued = list((spool / "pending").glob("*.json"))
    assert failures == []
    assert len(queued) == 24
    assert len({path.name for path in queued}) == 24


@linux_only
def test_secret_bearing_or_unknown_event_fields_are_never_spooled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent.rescue_plane_core as core

    spool = _spool(tmp_path, monkeypatch)
    client = core.RescueTelemetryClient(tmp_path / "missing.sock", spool_path=spool)
    unsafe = _event("secret-1")
    unsafe["secret"] = "must-not-persist"

    with pytest.raises(core.RescueTelemetryUnavailable, match="invalid rescue telemetry event"):
        client.emit(unsafe)
    assert not list((spool / "pending").iterdir())


@linux_only
def test_slow_persistence_remains_fail_closed_after_durable_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent.rescue_plane_core as core

    spool = _spool(tmp_path, monkeypatch)
    client = core.RescueTelemetryClient(tmp_path / "missing.sock", spool_path=spool)
    clock = iter([10.0, 12.0])
    monkeypatch.setattr(core.time, "monotonic", lambda: next(clock))

    with pytest.raises(core.RescueTelemetryUnavailable, match="latency exceeded"):
        client.emit(_event("slow-1"))
    assert len(list((spool / "pending").glob("*.json"))) == 1


@linux_only
def test_reporter_drains_in_order_and_deduplicates_ack_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent.rescue_plane_core as core
    from agent.rescue_quiescence_reporter import QuiescenceReporter

    runtime = tmp_path / "runtime"
    continuity = tmp_path / "continuity"
    runtime.mkdir(mode=0o750)
    runtime.chmod(0o750)
    continuity.mkdir(mode=0o750)
    continuity.chmod(0o750)
    spool = _spool(tmp_path, monkeypatch)
    client = core.RescueTelemetryClient(tmp_path / "missing.sock", spool_path=spool)
    client.emit(
        {
            "event": "turn_start",
            "event_id": "turn-start",
            "turn_id": "turn-spool",
            "lane": "normal",
            "artifact_requested": False,
        }
    )
    client.emit(
        {
            "event": "turn_end",
            "event_id": "turn-end",
            "turn_id": "turn-spool",
        }
    )
    queued = sorted((spool / "pending").glob("*.json"))
    duplicate = spool / "pending" / f"{queued[0].stem}-duplicate.json"
    duplicate.write_bytes(queued[0].read_bytes())
    duplicate.chmod(0o640)
    reporter = QuiescenceReporter(
        runtime_dir=runtime,
        continuity_dir=continuity,
        keyring=core.KeyRing(current=core.KeySlot("current", b"k" * 32)),
        source_sha="a" * 40,
        image_id="sha256:" + "b" * 64,
        expected_hermes_uid=os.getuid(),
        spool_dir=spool,
        recovery_authorization_path=None,
    )
    monkeypatch.setattr(reporter, "emit_snapshot", lambda **_kwargs: {})

    assert reporter.drain_spool() == 3
    assert reporter.state.active_counts() == (0, 0, 0)
    assert not list((spool / "pending").glob("*.json"))


@linux_only
def test_corrupt_spool_record_is_quarantined_and_degrades_health(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent.rescue_plane_core as core
    from agent.rescue_quiescence_reporter import QuiescenceReporter

    runtime = tmp_path / "runtime"
    continuity = tmp_path / "continuity"
    runtime.mkdir(mode=0o750)
    runtime.chmod(0o750)
    continuity.mkdir(mode=0o750)
    continuity.chmod(0o750)
    spool = _spool(tmp_path, monkeypatch)
    corrupt = spool / "pending" / "00000000000000000001-corrupt.json"
    corrupt.write_bytes(b'{"schema_version":"wrong"}')
    corrupt.chmod(0o640)
    reporter = QuiescenceReporter(
        runtime_dir=runtime,
        continuity_dir=continuity,
        keyring=core.KeyRing(current=core.KeySlot("current", b"k" * 32)),
        source_sha="a" * 40,
        image_id="sha256:" + "b" * 64,
        expected_hermes_uid=os.getuid(),
        spool_dir=spool,
        recovery_authorization_path=None,
    )

    assert reporter.drain_spool() == 0
    assert reporter.state.telemetry_health == "degraded"
    assert reporter.state.degradation_reasons == {"accounting_gap"}
    assert not corrupt.exists()
    assert len(list((spool / "quarantine").glob("*.invalid"))) == 1
