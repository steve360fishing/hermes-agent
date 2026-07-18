from __future__ import annotations

from contextlib import contextmanager
import inspect
import json
import math
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import time
import threading
from types import SimpleNamespace

import pytest


linux_only = pytest.mark.skipif(os.name == "nt", reason="Linux security boundary")


def _secure_file(path: Path, content: bytes, mode: int) -> None:
    path.write_bytes(content)
    path.chmod(mode)


def _keyring(now: float = 100.0):
    from agent.rescue_plane_core import KeyRing, KeySlot

    return KeyRing(
        current=KeySlot("old", b"a" * 32),
        next=KeySlot("new", b"b" * 32),
        cutover_at=now + 10,
    )


def _replay_state(path: Path):
    from agent.rescue_plane_core import DurableReplayState

    replay = DurableReplayState(path)
    replay.initialize()
    return replay


def _reporter(runtime_dir: Path, *, max_turns: int = 8):
    from agent.rescue_plane_core import KeyRing, KeySlot
    from agent.rescue_quiescence_reporter import QuiescenceReporter

    runtime_dir.chmod(0o750)
    return QuiescenceReporter(
        runtime_dir=runtime_dir,
        continuity_dir=runtime_dir,
        keyring=KeyRing(current=KeySlot("current", b"q" * 32)),
        source_sha="a" * 40,
        image_id="sha256:" + "b" * 64,
        expected_hermes_uid=os.getuid(),
        max_turns=max_turns,
    )


def _durable_reporter(
    runtime_dir: Path,
    continuity_dir: Path,
    *,
    max_turns: int = 8,
):
    from agent.rescue_plane_core import KeyRing, KeySlot
    from agent.rescue_quiescence_reporter import QuiescenceReporter

    runtime_dir.mkdir(mode=0o750, exist_ok=True)
    runtime_dir.chmod(0o750)
    continuity_dir.mkdir(mode=0o750, exist_ok=True)
    continuity_dir.chmod(0o750)
    return QuiescenceReporter(
        runtime_dir=runtime_dir,
        continuity_dir=continuity_dir,
        keyring=KeyRing(current=KeySlot("current", b"q" * 32)),
        source_sha="a" * 40,
        image_id="sha256:" + "b" * 64,
        expected_hermes_uid=os.getuid(),
        max_turns=max_turns,
    )


@linux_only
def test_safe_overlay_uses_descriptor_metadata_and_rejects_hardlinks(tmp_path: Path) -> None:
    from agent.rescue_plane_core import read_safe_mode_overlay

    safe_dir = tmp_path / "safe"
    safe_dir.mkdir(mode=0o750)
    safe_dir.chmod(0o750)
    overlay = safe_dir / "safe-mode-v1.json"
    _secure_file(
        overlay,
        json.dumps(
            {
                "schema_version": "hermes-safe-mode-v1",
                "enabled": True,
                "incident_id": "4faeb31c-15fe-4f14-a1e2-11892cbcb5b6",
                "issued_at": "2026-07-17T12:00:00Z",
                "disables": ["artifact_only"],
            }
        ).encode(),
        0o640,
    )
    linked = safe_dir / "linked.json"
    os.link(overlay, linked)

    decision = read_safe_mode_overlay(
        overlay,
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
    )

    assert decision.valid is False
    assert decision.reason == "invalid_hardlink"


@linux_only
def test_safe_overlay_rejects_parent_symlink_and_owner_mode_drift(tmp_path: Path) -> None:
    from agent.rescue_plane_core import read_safe_mode_overlay

    real = tmp_path / "real"
    real.mkdir(mode=0o750)
    real.chmod(0o750)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    overlay = real / "safe-mode-v1.json"
    _secure_file(overlay, b"{}", 0o640)

    assert read_safe_mode_overlay(
        linked / overlay.name,
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
    ).reason == "invalid_parent"
    real.chmod(0o755)
    assert read_safe_mode_overlay(
        overlay,
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
    ).reason == "invalid_parent_mode"


@linux_only
def test_safe_overlay_mount_must_be_a_separate_read_only_mount(tmp_path: Path) -> None:
    from agent.rescue_plane_core import path_is_on_readonly_mount

    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "1 0 8:1 / / rw,relatime - ext4 /dev/root rw\n"
        "2 1 8:1 /rescue /run/hermes-rescue ro,nosuid,nodev - ext4 /dev/root ro\n",
        encoding="utf-8",
    )

    assert path_is_on_readonly_mount(
        Path("/var/run/hermes-rescue/safe-mode-v1.json"),
        mountinfo_path=mountinfo,
    )
    assert not path_is_on_readonly_mount(
        Path("/run/hermes-rescue-reporter/quiescence-v1.json"),
        mountinfo_path=mountinfo,
    )


@linux_only
def test_atomic_output_rejects_symlink_and_hardlink_targets(tmp_path: Path) -> None:
    from agent.rescue_plane_core import atomic_write_secure

    tmp_path.chmod(0o700)
    target = tmp_path / "state.json"
    victim = tmp_path / "victim"
    victim.write_bytes(b"do-not-touch")
    target.symlink_to(victim)
    with pytest.raises(OSError):
        atomic_write_secure(target, b"safe")
    assert victim.read_bytes() == b"do-not-touch"

    target.unlink()
    os.link(victim, target)
    with pytest.raises(PermissionError, match="unsafe output"):
        atomic_write_secure(target, b"safe")


@linux_only
def test_reporter_outputs_keep_runtime_gid_for_three_emissions_and_s6_style_restart(
    tmp_path: Path,
) -> None:
    candidate_gids = [gid for gid in os.getgroups() if gid != os.getgid()]
    if candidate_gids:
        runtime_gid = candidate_gids[0]
    elif os.geteuid() == 0:
        runtime_gid = os.getgid() + 1
    else:
        pytest.skip("requires a supplementary group or root")
    os.chown(tmp_path, os.getuid(), runtime_gid)
    tmp_path.chmod(0o750)

    reporter = _reporter(tmp_path)
    snapshots = [reporter.emit_snapshot(now=float(value)) for value in (10, 11, 12)]
    restarted = _reporter(tmp_path)
    snapshots.append(restarted.emit_snapshot(now=13.0))

    assert [snapshot["sequence"] for snapshot in snapshots] == [1, 2, 3, 4]
    assert len({snapshot["producer_epoch"] for snapshot in snapshots}) == 1
    for output in (
        reporter.state_path,
        reporter.output_path,
        reporter.producer_state_path,
    ):
        assert output.stat().st_gid == runtime_gid


@linux_only
def test_reporter_recovers_missing_signed_continuity_but_rejects_corruption(
    tmp_path: Path,
) -> None:
    reporter = _reporter(tmp_path)
    before = reporter.emit_snapshot(now=10.0)
    reporter.producer_state_path.unlink()
    recovered = _reporter(tmp_path)
    after = recovered.emit_snapshot(now=11.0)
    assert after["producer_epoch"] == before["producer_epoch"]
    assert after["sequence"] == before["sequence"] + 1
    assert after["telemetry_health"] == "degraded"

    recovered.continuity_state_path.write_text("{}", encoding="ascii")
    recovered.continuity_state_path.chmod(0o600)
    with pytest.raises(ValueError, match="corrupt continuity state"):
        _reporter(tmp_path)


@linux_only
def test_partial_aggregate_loss_with_live_provider_recovers_sticky_degraded(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime"
    continuity = tmp_path / "continuity"
    reporter = _durable_reporter(runtime, continuity)
    reporter.process_event(
        json.dumps(
            {
                "schema_version": "hermes-rescue-event-v1",
                "event": "provider_start",
                "event_id": "provider-start",
                "turn_id": "turn-live",
                "work_id": "provider-live",
            }
        ).encode("ascii"),
        peer_pid=os.getpid(),
        peer_uid=os.getuid(),
        now=10.0,
    )
    live = reporter.emit_snapshot(now=11.0)
    reporter.continuity_state_path.unlink()

    restarted = _durable_reporter(runtime, continuity)
    recovered = restarted.emit_snapshot(now=12.0)

    assert recovered["producer_epoch"] == live["producer_epoch"]
    assert recovered["sequence"] == live["sequence"] + 1
    assert recovered["active_provider_action_count"] >= 1
    assert recovered["telemetry_health"] == "degraded"
    restarted_again = _durable_reporter(runtime, continuity)
    sticky = restarted_again.emit_snapshot(now=13.0)
    assert sticky["producer_epoch"] == live["producer_epoch"]
    assert sticky["sequence"] == recovered["sequence"] + 1
    assert sticky["active_provider_action_count"] >= 1
    assert sticky["telemetry_health"] == "degraded"


@linux_only
def test_full_runtime_loss_preserves_epoch_sequence_and_host_replay(
    tmp_path: Path,
) -> None:
    import shutil

    from agent.rescue_plane_core import validate_quiescence_snapshot

    runtime = tmp_path / "runtime"
    continuity = tmp_path / "continuity"
    reporter = _durable_reporter(runtime, continuity)
    before = reporter.emit_snapshot(now=10.0)
    replay = _replay_state(tmp_path / "host-replay.json")
    assert validate_quiescence_snapshot(
        before,
        keyring=reporter.keyring,
        now=10.1,
        replay_state=replay,
    ).valid

    shutil.rmtree(runtime)
    runtime.mkdir(mode=0o750)
    runtime.chmod(0o750)
    restarted = _durable_reporter(runtime, continuity)
    after = restarted.emit_snapshot(now=11.0)

    assert after["producer_epoch"] == before["producer_epoch"]
    assert after["sequence"] == before["sequence"] + 1
    assert validate_quiescence_snapshot(
        after,
        keyring=reporter.keyring,
        now=11.1,
        replay_state=replay,
    ).valid


@linux_only
def test_key_read_is_descriptor_pinned_and_requires_exact_permissions(tmp_path: Path) -> None:
    from agent.rescue_plane_core import read_hmac_key

    key = tmp_path / "key"
    _secure_file(key, b"k" * 32, 0o400)
    assert read_hmac_key(key, expected_uid=os.getuid(), expected_gid=os.getgid()) == b"k" * 32
    key.chmod(0o440)
    with pytest.raises(PermissionError, match="mode"):
        read_hmac_key(key, expected_uid=os.getuid(), expected_gid=os.getgid())


def test_reporter_derives_multiprocess_aggregate_and_does_not_last_writer_zero(tmp_path: Path) -> None:
    from agent.rescue_quiescence_reporter import ReporterState

    state = ReporterState(max_turns=8)
    state.apply_event(
        {"schema_version": "hermes-rescue-event-v1", "event": "turn_start", "event_id": "e1",
         "turn_id": "t1", "lane": "normal", "artifact_requested": False},
        peer_pid=101, peer_uid=10000, now=10,
    )
    state.apply_event(
        {"schema_version": "hermes-rescue-event-v1", "event": "tool_start", "event_id": "e2",
         "turn_id": "t1", "work_id": "w1"},
        peer_pid=201, peer_uid=10000, now=11,
    )
    state.apply_event(
        {"schema_version": "hermes-rescue-event-v1", "event": "provider_start", "event_id": "e3",
         "turn_id": "t2", "work_id": "w2"},
        peer_pid=202, peer_uid=10000, now=12,
    )
    state.apply_event(
        {"schema_version": "hermes-rescue-event-v1", "event": "tool_end", "event_id": "e4",
         "turn_id": "t1", "work_id": "w1"},
        peer_pid=201, peer_uid=10000, now=13,
    )

    assert state.active_counts() == (1, 0, 1)
    evidence = state.policy_evidence()
    assert evidence[0]["lane"] == "normal"
    assert evidence[0]["artifact_requested"] is False


def test_surviving_worker_stays_active_and_dead_worker_degrades(tmp_path: Path) -> None:
    from agent.rescue_quiescence_reporter import ReporterState

    state = ReporterState(max_turns=8)
    state.apply_event(
        {"schema_version": "hermes-rescue-event-v1", "event": "tool_start", "event_id": "e1",
         "turn_id": "t1", "work_id": "w1"},
        peer_pid=501, peer_uid=10000, now=1,
    )
    state.reconcile(lambda pid: pid == 501, now=500)
    assert state.active_counts()[1] == 1
    assert state.telemetry_health == "healthy"

    state.reconcile(lambda _pid: False, now=501)
    state.reconcile(lambda _pid: False, now=502)
    state.reconcile(lambda _pid: False, now=503)
    assert state.active_counts()[1] == 1
    assert state.telemetry_health == "degraded"
    assert state.reconciled_crashes == 1


def test_reporter_state_is_bounded_and_deduplicates_events() -> None:
    from agent.rescue_quiescence_reporter import ReporterState

    state = ReporterState(max_turns=2)
    for index in range(3):
        started = float(index * 120)
        event = {
            "schema_version": "hermes-rescue-event-v1",
            "event": "turn_start",
            "event_id": f"event-{index}",
            "turn_id": f"turn-{index}",
            "lane": "normal",
            "artifact_requested": False,
        }
        state.apply_event(event, peer_pid=100 + index, peer_uid=10000, now=started)
        state.apply_event(event, peer_pid=100 + index, peer_uid=10000, now=started)
        state.apply_event(
            {
                "schema_version": "hermes-rescue-event-v1",
                "event": "turn_end",
                "event_id": f"end-{index}",
                "turn_id": f"turn-{index}",
            },
            peer_pid=100 + index,
            peer_uid=10000,
            now=started + 0.5,
        )

    assert len(state.policy_evidence()) == 1
    assert state.active_counts() == (0, 0, 0)


def test_policy_evidence_retains_120_seconds_and_capacity_fails_closed() -> None:
    from agent.rescue_quiescence_reporter import ReporterState

    state = ReporterState(max_turns=2)
    for index, started in enumerate((0.0, 2.0), start=1):
        turn_id = f"turn-{index}"
        state.apply_event(
            {
                "schema_version": "hermes-rescue-event-v1",
                "event": "turn_start",
                "event_id": f"start-{index}",
                "turn_id": turn_id,
                "lane": "normal",
                "artifact_requested": False,
            },
            peer_pid=500 + index,
            peer_uid=10000,
            now=started,
        )
        state.apply_event(
            {
                "schema_version": "hermes-rescue-event-v1",
                "event": "turn_end",
                "event_id": f"end-{index}",
                "turn_id": turn_id,
            },
            peer_pid=500 + index,
            peer_uid=10000,
            now=started + 1,
        )

    third = {
        "schema_version": "hermes-rescue-event-v1",
        "event": "turn_start",
        "event_id": "start-3",
        "turn_id": "turn-3",
        "lane": "normal",
        "artifact_requested": False,
    }
    with pytest.raises(ValueError, match="capacity"):
        state.apply_event(third, peer_pid=503, peer_uid=10000, now=100.0)
    assert [item["turn_id"] for item in state.policy_evidence()] == [
        "turn-1",
        "turn-2",
    ]
    assert state.telemetry_health == "degraded"

    state.telemetry_health = "healthy"
    state.apply_event(third, peer_pid=503, peer_uid=10000, now=122.0)
    assert [item["turn_id"] for item in state.policy_evidence()] == [
        "turn-2",
        "turn-3",
    ]


def test_completed_turn_id_cannot_overwrite_retained_policy_evidence() -> None:
    from agent.rescue_quiescence_reporter import ReporterState

    state = ReporterState(max_turns=2)
    original = {
        "schema_version": "hermes-rescue-event-v1",
        "event": "turn_start",
        "event_id": "original-start",
        "turn_id": "retained-turn",
        "lane": "artifact_only",
        "artifact_requested": True,
    }
    state.apply_event(original, peer_pid=501, peer_uid=10000, now=0.0)
    state.apply_event(
        {
            "schema_version": "hermes-rescue-event-v1",
            "event": "turn_end",
            "event_id": "original-end",
            "turn_id": "retained-turn",
        },
        peer_pid=501,
        peer_uid=10000,
        now=1.0,
    )

    with pytest.raises(ValueError, match="retained turn id"):
        state.apply_event(
            {
                **original,
                "event_id": "replacement-start",
                "lane": "normal",
                "artifact_requested": False,
            },
            peer_pid=502,
            peer_uid=10000,
            now=100.0,
        )

    assert state.policy_evidence() == [
        {
            "turn_id": "retained-turn",
            "lane": "artifact_only",
            "artifact_requested": True,
            "started_at": 0.0,
            "completed_at": 1.0,
        }
    ]


@linux_only
def test_start_event_publishes_active_snapshot_before_process_event_returns(
    tmp_path: Path,
) -> None:
    reporter = _reporter(tmp_path)
    idle = reporter.emit_snapshot(now=10.0)
    reporter.process_event(
        json.dumps(
            {
                "schema_version": "hermes-rescue-event-v1",
                "event": "tool_start",
                "event_id": "start",
                "turn_id": "turn",
                "work_id": "work",
            }
        ).encode("ascii"),
        peer_pid=os.getpid(),
        peer_uid=os.getuid(),
        now=11.0,
    )

    published = json.loads(reporter.output_path.read_text(encoding="ascii"))
    assert published["sequence"] == idle["sequence"] + 1
    assert published["active_tool_count"] == 1
    assert published["timestamp"] == 11.0


@linux_only
def test_failed_start_snapshot_publish_stays_invalid_and_conservatively_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reporter = _reporter(tmp_path)
    reporter.emit_snapshot(now=10.0)
    original_emit = reporter.emit_snapshot

    def fail_start_publish(*, now: float | None = None):
        if now == 11.0:
            raise OSError("snapshot disk failure")
        return original_emit(now=now)

    monkeypatch.setattr(reporter, "emit_snapshot", fail_start_publish)
    with pytest.raises(OSError, match="snapshot disk failure"):
        reporter.process_event(
            json.dumps(
                {
                    "schema_version": "hermes-rescue-event-v1",
                    "event": "provider_start",
                    "event_id": "provider-start",
                    "turn_id": "turn",
                    "work_id": "provider",
                }
            ).encode("ascii"),
            peer_pid=os.getpid(),
            peer_uid=os.getuid(),
            now=11.0,
        )

    assert not reporter.output_path.exists()
    recovered = reporter.emit_snapshot(now=12.0)
    assert recovered["active_provider_action_count"] == 1


@linux_only
def test_capacity_exhaustion_publishes_degraded_snapshot_and_rejects_start(
    tmp_path: Path,
) -> None:
    reporter = _reporter(tmp_path, max_turns=2)
    for index in range(2):
        turn_id = f"turn-{index}"
        reporter.process_event(
            json.dumps(
                {
                    "schema_version": "hermes-rescue-event-v1",
                    "event": "turn_start",
                    "event_id": f"start-{index}",
                    "turn_id": turn_id,
                    "lane": "normal",
                    "artifact_requested": False,
                }
            ).encode("ascii"),
            peer_pid=os.getpid(),
            peer_uid=os.getuid(),
            now=float(index * 2),
        )
        reporter.process_event(
            json.dumps(
                {
                    "schema_version": "hermes-rescue-event-v1",
                    "event": "turn_end",
                    "event_id": f"end-{index}",
                    "turn_id": turn_id,
                }
            ).encode("ascii"),
            peer_pid=os.getpid(),
            peer_uid=os.getuid(),
            now=float(index * 2 + 1),
        )

    with pytest.raises(ValueError, match="capacity"):
        reporter.process_event(
            json.dumps(
                {
                    "schema_version": "hermes-rescue-event-v1",
                    "event": "turn_start",
                    "event_id": "start-overflow",
                    "turn_id": "overflow",
                    "lane": "normal",
                    "artifact_requested": False,
                }
            ).encode("ascii"),
            peer_pid=os.getpid(),
            peer_uid=os.getuid(),
            now=100.0,
        )

    snapshot = json.loads(reporter.output_path.read_text(encoding="ascii"))
    assert snapshot["telemetry_health"] == "degraded"
    assert [item["turn_id"] for item in snapshot["policy_evidence"]] == [
        "turn-0",
        "turn-1",
    ]


def test_reporter_rolls_back_exit_when_durable_persist_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent.rescue_quiescence_reporter import QuiescenceReporter, ReporterState

    reporter = QuiescenceReporter.__new__(QuiescenceReporter)
    reporter.expected_hermes_uid = 10000
    reporter.state = ReporterState(max_turns=8)
    reporter._lock = threading.RLock()
    reporter.state.apply_event(
        {
            "schema_version": "hermes-rescue-event-v1",
            "event": "tool_start",
            "event_id": "start",
            "turn_id": "turn",
            "work_id": "work",
        },
        peer_pid=501,
        peer_uid=10000,
        now=1,
    )
    monkeypatch.setattr(
        reporter,
        "_persist_state",
        lambda: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        reporter.process_event(
            json.dumps(
                {
                    "schema_version": "hermes-rescue-event-v1",
                    "event": "tool_end",
                    "event_id": "end",
                    "turn_id": "turn",
                    "work_id": "work",
                }
            ).encode("ascii"),
            peer_pid=501,
            peer_uid=10000,
            now=2,
        )

    assert reporter.state.active_counts()[1] == 1


def test_reporter_rejects_duplicate_json_fields() -> None:
    from agent.rescue_quiescence_reporter import QuiescenceReporter, ReporterState

    reporter = QuiescenceReporter.__new__(QuiescenceReporter)
    reporter.expected_hermes_uid = 10000
    reporter.state = ReporterState(max_turns=8)
    reporter._lock = threading.RLock()

    with pytest.raises(ValueError, match="duplicate"):
        reporter.process_event(
            (
                b'{"schema_version":"hermes-rescue-event-v1","event":"turn_start",'
                b'"event_id":"one","event_id":"two","turn_id":"turn","lane":"normal",'
                b'"artifact_requested":false}'
            ),
            peer_pid=501,
            peer_uid=10000,
            now=2,
        )


def test_reporter_reloads_active_state_without_converting_crash_to_idle() -> None:
    from agent.rescue_quiescence_reporter import ReporterState

    original = ReporterState(max_turns=8)
    original.apply_event(
        {
            "schema_version": "hermes-rescue-event-v1",
            "event": "provider_start",
            "event_id": "persist-1",
            "turn_id": "turn-1",
            "work_id": "provider-1",
        },
        peer_pid=700,
        peer_uid=10000,
        now=10.0,
    )

    restored = ReporterState.from_payload(original.to_payload())
    restored.reconcile(lambda _pid: False, now=20.0)

    assert restored.active_counts() == (0, 0, 1)
    assert restored.telemetry_health == "degraded"


def test_corrupt_reporter_and_replay_state_fail_closed(tmp_path: Path) -> None:
    from agent.rescue_plane_core import DurableReplayState
    from agent.rescue_quiescence_reporter import ReporterState

    with pytest.raises(ValueError, match="corrupt"):
        ReporterState.from_payload({"schema_version": "wrong"})
    replay = tmp_path / "replay.json"
    replay.write_text('{"key":true}', encoding="utf-8")
    replay.chmod(0o600)
    with pytest.raises(ValueError, match="corrupt"):
        DurableReplayState(replay).accept("1" * 32, "key", 1)


def test_snapshot_strict_types_finite_values_policy_and_durable_replay(tmp_path: Path) -> None:
    from agent.rescue_plane_core import (
        DurableReplayState,
        sign_quiescence_snapshot,
        validate_quiescence_snapshot,
    )

    ring = _keyring()
    replay = _replay_state(tmp_path / "replay.json")
    unsigned = {
        "schema_version": "hermes-quiescence-snapshot-v1",
        "producer_epoch": "1" * 32,
        "key_id": "old",
        "sequence": 1,
        "timestamp": 100.0,
        "gateway_pids": [123],
        "gateway_state": "active",
        "active_turn_count": 1,
        "active_tool_count": 0,
        "active_provider_action_count": 0,
        "source_sha": "a" * 40,
        "image_id": "sha256:" + "b" * 64,
        "telemetry_health": "healthy",
        "policy_evidence": [
            {
                "turn_id": "t1",
                "lane": "normal",
                "artifact_requested": False,
                "started_at": 99.0,
                "completed_at": None,
            }
        ],
    }
    snapshot = sign_quiescence_snapshot(unsigned, ring, now=100.0)
    assert validate_quiescence_snapshot(snapshot, keyring=ring, now=100.1, replay_state=replay).valid
    assert validate_quiescence_snapshot(snapshot, keyring=ring, now=100.2, replay_state=replay).reason == "replayed"

    malformed = dict(snapshot)
    malformed["timestamp"] = math.nan
    assert validate_quiescence_snapshot(
        malformed,
        keyring=ring,
        now=100.2,
        replay_state=_replay_state(tmp_path / "replay2.json"),
    ).reason == "malformed"
    malformed_sequence = dict(snapshot)
    malformed_sequence["sequence"] = 2**63
    assert validate_quiescence_snapshot(
        malformed_sequence,
        keyring=ring,
        now=100.2,
        replay_state=_replay_state(tmp_path / "replay3.json"),
    ).reason == "malformed"
    for invalid_now in (math.nan, math.inf, -math.inf):
        assert validate_quiescence_snapshot(
            snapshot,
            keyring=ring,
            now=invalid_now,
            replay_state=DurableReplayState(
                tmp_path / f"now-{str(invalid_now).replace('-', 'neg')}.json"
            ),
        ).reason == "invalid_verifier_time"


def test_host_replay_pins_producer_epoch_without_auto_reset(tmp_path: Path) -> None:
    from agent.rescue_plane_core import (
        DurableReplayState,
        sign_quiescence_snapshot,
        validate_quiescence_snapshot,
    )

    ring = _keyring()
    base = {
        "schema_version": "hermes-quiescence-snapshot-v1",
        "producer_epoch": "a" * 32,
        "key_id": "old",
        "sequence": 1,
        "timestamp": 100.0,
        "gateway_pids": [],
        "gateway_state": "dead",
        "active_turn_count": 0,
        "active_tool_count": 0,
        "active_provider_action_count": 0,
        "source_sha": "a" * 40,
        "image_id": "sha256:" + "b" * 64,
        "telemetry_health": "healthy",
        "policy_evidence": [],
    }
    replay = _replay_state(tmp_path / "epoch-replay.json")
    first = sign_quiescence_snapshot(base, ring, now=100.0)
    changed = sign_quiescence_snapshot(
        {**base, "producer_epoch": "b" * 32, "sequence": 2, "timestamp": 101.0},
        ring,
        now=101.0,
    )

    assert validate_quiescence_snapshot(
        first, keyring=ring, now=100.1, replay_state=replay
    ).valid
    assert validate_quiescence_snapshot(
        changed, keyring=ring, now=101.1, replay_state=replay
    ).reason == "producer_epoch_changed"


def test_missing_host_replay_state_does_not_auto_reset(tmp_path: Path) -> None:
    from agent.rescue_plane_core import (
        DurableReplayState,
        sign_quiescence_snapshot,
        validate_quiescence_snapshot,
    )

    ring = _keyring()
    snapshot = sign_quiescence_snapshot(
        {
            "schema_version": "hermes-quiescence-snapshot-v1",
            "producer_epoch": "a" * 32,
            "key_id": "old",
            "sequence": 50,
            "timestamp": 100.0,
            "gateway_pids": [],
            "gateway_state": "dead",
            "active_turn_count": 0,
            "active_tool_count": 0,
            "active_provider_action_count": 0,
            "source_sha": "a" * 40,
            "image_id": "sha256:" + "b" * 64,
            "telemetry_health": "healthy",
            "policy_evidence": [],
        },
        ring,
        now=100.0,
    )
    replay_path = tmp_path / "missing-replay.json"

    assert validate_quiescence_snapshot(
        snapshot,
        keyring=ring,
        now=100.1,
        replay_state=DurableReplayState(replay_path),
    ).reason == "replay_state_unavailable"
    assert not replay_path.exists()


def test_key_rotation_cutover_accepts_only_correct_slot_and_window(tmp_path: Path) -> None:
    from agent.rescue_plane_core import (
        DurableReplayState,
        sign_quiescence_snapshot,
        validate_quiescence_snapshot,
    )

    ring = _keyring(now=100.0)
    base = {
        "schema_version": "hermes-quiescence-snapshot-v1",
        "producer_epoch": "1" * 32,
        "sequence": 1,
        "timestamp": 109.0,
        "gateway_pids": [],
        "gateway_state": "dead",
        "active_turn_count": 0,
        "active_tool_count": 0,
        "active_provider_action_count": 0,
        "source_sha": "a" * 40,
        "image_id": "sha256:" + "b" * 64,
        "telemetry_health": "healthy",
        "policy_evidence": [],
    }
    old = sign_quiescence_snapshot({**base, "key_id": "old"}, ring, now=109.0)
    cross_key_replay = _replay_state(tmp_path / "cross-key.json")
    assert validate_quiescence_snapshot(
        old,
        keyring=ring,
        now=120.0,
        replay_state=cross_key_replay,
    ).valid
    new_same_sequence = sign_quiescence_snapshot(
        {**base, "key_id": "new", "timestamp": 111.0},
        ring,
        now=111.0,
    )
    assert validate_quiescence_snapshot(
        new_same_sequence,
        keyring=ring,
        now=112.0,
        replay_state=cross_key_replay,
    ).reason == "replayed"
    assert validate_quiescence_snapshot(
        old,
        keyring=ring,
        now=140.0,
        replay_state=_replay_state(tmp_path / "expired.json"),
    ).reason == "old_key_expired"

    new = sign_quiescence_snapshot({**base, "key_id": "new", "sequence": 2, "timestamp": 111.0}, ring, now=111.0)
    assert validate_quiescence_snapshot(
        new,
        keyring=ring,
        now=112.0,
        replay_state=_replay_state(tmp_path / "new.json"),
    ).valid


@linux_only
def test_sequence_recovery_requires_valid_previous_signature(tmp_path: Path) -> None:
    from agent.rescue_plane_core import recover_snapshot_sequence, sign_quiescence_snapshot

    ring = _keyring()
    path = tmp_path / "snapshot.json"
    payload = {
        "schema_version": "hermes-quiescence-snapshot-v1",
        "producer_epoch": "1" * 32,
        "key_id": "old",
        "sequence": 41,
        "timestamp": 100.0,
        "gateway_pids": [],
        "gateway_state": "dead",
        "active_turn_count": 0,
        "active_tool_count": 0,
        "active_provider_action_count": 0,
        "source_sha": "a" * 40,
        "image_id": "sha256:" + "b" * 64,
        "telemetry_health": "healthy",
        "policy_evidence": [],
    }
    path.write_text(json.dumps(sign_quiescence_snapshot(payload, ring, now=100.0)), encoding="utf-8")
    path.chmod(0o600)
    assert recover_snapshot_sequence(path, keyring=ring, expected_uid=os.getuid(), expected_gid=os.getgid()) == 41
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["sequence"] = 900
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="signature"):
        recover_snapshot_sequence(path, keyring=ring, expected_uid=os.getuid(), expected_gid=os.getgid())


@linux_only
def test_gateway_discovery_reports_actual_alive_and_dead_state(tmp_path: Path) -> None:
    from agent.rescue_quiescence_reporter import discover_gateway_state

    proc = tmp_path / "proc"
    process = proc / "321"
    process.mkdir(parents=True)
    (process / "cmdline").write_bytes(b"/opt/hermes/.venv/bin/python\0hermes\0gateway\0run\0")
    (process / "status").write_text(f"Uid:\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\t{os.getuid()}\n", encoding="ascii")
    assert discover_gateway_state(proc_root=proc, expected_uid=os.getuid()) == ([321], "active")
    (process / "cmdline").unlink()
    assert discover_gateway_state(proc_root=proc, expected_uid=os.getuid()) == ([], "dead")


@linux_only
def test_unreadable_proc_is_unknown_and_degraded_restart_ineligible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent.rescue_quiescence_reporter as reporter_module
    from agent.rescue_plane_core import (
        DurableReplayState,
        validate_quiescence_snapshot,
    )

    proc = tmp_path / "proc"
    proc.mkdir()
    original_iterdir = Path.iterdir

    def unreadable(path: Path):
        if path == proc:
            raise PermissionError("blocked proc")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", unreadable)
    assert reporter_module.discover_gateway_state(
        proc_root=proc, expected_uid=os.getuid()
    ) == ([], "unknown")

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    reporter = _reporter(runtime)
    monkeypatch.setattr(
        reporter_module,
        "discover_gateway_state",
        lambda **_kwargs: ([], "unknown"),
    )
    snapshot = reporter.emit_snapshot(now=100.0)
    assert snapshot["telemetry_health"] == "degraded"
    assert validate_quiescence_snapshot(
        snapshot,
        keyring=reporter.keyring,
        now=100.1,
        replay_state=_replay_state(tmp_path / "unknown-replay.json"),
    ).reason == "gateway_unknown"


@linux_only
def test_reporter_single_writer_lock_rejects_overlap(tmp_path: Path) -> None:
    from agent.rescue_quiescence_reporter import acquire_reporter_lock

    first = acquire_reporter_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            acquire_reporter_lock(tmp_path)
    finally:
        os.close(first)


@linux_only
def test_kernel_peer_events_aggregate_across_processes_and_crash_degrades(tmp_path: Path) -> None:
    from agent.rescue_plane_core import KeyRing, KeySlot, RescueTelemetryClient
    from agent.rescue_quiescence_reporter import QuiescenceReporter

    tmp_path.chmod(0o750)
    reporter = QuiescenceReporter(
        runtime_dir=tmp_path,
        continuity_dir=tmp_path,
        keyring=KeyRing(current=KeySlot("current", b"q" * 32)),
        source_sha="a" * 40,
        image_id="sha256:" + "b" * 64,
        expected_hermes_uid=os.getuid(),
    )
    stop = threading.Event()
    thread = threading.Thread(target=reporter.serve, kwargs={"stop_event": stop}, daemon=True)
    thread.start()
    deadline = time.time() + 5
    while not reporter.socket_path.exists() and time.time() < deadline:
        time.sleep(0.01)

    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path;"
                "from agent.rescue_plane_core import RescueTelemetryClient;"
                "RescueTelemetryClient(Path(__import__('sys').argv[1])).emit({"
                "'event':'tool_start','event_id':'child-start',"
                "'turn_id':'turn-child','work_id':'work-child'})"
            ),
            str(reporter.socket_path),
        ],
        check=True,
    )
    RescueTelemetryClient(reporter.socket_path).emit(
        {
            "event": "provider_start",
            "event_id": "parent-start",
            "turn_id": "turn-parent",
            "work_id": "work-parent",
        }
    )
    RescueTelemetryClient(reporter.socket_path).emit(
        {
            "event": "provider_end",
            "event_id": "parent-end",
            "turn_id": "turn-parent",
            "work_id": "work-parent",
        }
    )
    reporter.state.reconcile(lambda pid: pid != next(iter(reporter.state.tools.values()))["peer_pid"], now=time.time())
    stop.set()
    thread.join(timeout=5)

    assert reporter.state.active_counts() == (0, 1, 0)
    assert reporter.state.telemetry_health == "degraded"


def test_reporter_checks_actual_effective_uid() -> None:
    from agent.rescue_quiescence_reporter import require_effective_uid

    require_effective_uid(10001, actual_uid=10001)
    with pytest.raises(PermissionError, match="effective uid"):
        require_effective_uid(10001, actual_uid=0)


def test_s6_service_is_executable_drops_uid_and_has_no_failure_mask() -> None:
    from agent import rescue_plane_core

    root = Path(__file__).parents[2]
    run = root / "docker" / "s6-rc.d" / "rescue-quiescence-reporter" / "run"
    text = run.read_text(encoding="utf-8")
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    stage2 = (root / "docker" / "stage2-hook.sh").read_text(encoding="utf-8")
    index_mode = subprocess.run(
        ["git", "ls-files", "--stage", "docker/s6-rc.d/rescue-quiescence-reporter/run"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()[0]

    assert index_mode == "100755"
    assert "s6-setuidgid hermes-rescue" in text
    assert rescue_plane_core.RESCUE_REPORTER_UID == 10002
    assert "useradd -u 10002" in dockerfile
    assert "useradd -u 10001" not in dockerfile
    assert "--expected-reporter-uid" in text
    assert "|| true" not in text
    assert "gateway run" not in text
    assert "/var/run/hermes-rescue" not in text
    assert "--runtime-dir /run/hermes-rescue-reporter" in text
    assert "--continuity-dir /var/lib/hermes-rescue" in text
    assert "key=/run/hermes-rescue-secrets/hmac-current" in text
    assert "chmod 0755 /etc/s6-overlay/s6-rc.d/rescue-quiescence-reporter/run" in dockerfile
    assert "requested Hermes GID collides with hermes-rescue" in stage2
    assert '[ -L /run/hermes-rescue-reporter ]' in stage2
    assert "telemetry-required-v1.json" in stage2
    assert "hermes-rescue-telemetry-required-v1" in stage2
    assert "install -d -o hermes-rescue -g hermes -m 0750" in stage2
    assert "/var/run/hermes-rescue" not in stage2


def test_wrappers_preserve_introspection_contract() -> None:
    from agent.chat_completion_helpers import (
        interruptible_api_call,
        interruptible_streaming_api_call,
    )
    from agent.auxiliary_client import async_call_llm, call_llm
    from model_tools import handle_function_call

    assert inspect.unwrap(interruptible_api_call) is not interruptible_api_call
    assert (
        inspect.unwrap(interruptible_streaming_api_call)
        is not interruptible_streaming_api_call
    )
    assert inspect.unwrap(call_llm) is not call_llm
    assert inspect.unwrap(async_call_llm) is not async_call_llm
    assert "api_kwargs" in inspect.signature(interruptible_api_call).parameters
    assert "api_kwargs" in inspect.signature(
        interruptible_streaming_api_call
    ).parameters
    assert "function_name" in inspect.signature(handle_function_call).parameters


def test_provider_worker_accounting_lasts_until_daemon_worker_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent.rescue_plane_core as core
    from agent.chat_completion_helpers import _rescue_account_provider_worker

    active = {"count": 0}
    entered = threading.Event()
    release = threading.Event()

    class FakeClient:
        @contextmanager
        def active_work(self, kind: str, *, turn_id: str):
            assert kind == "provider"
            assert turn_id == "turn-1"
            active["count"] += 1
            try:
                yield
            finally:
                active["count"] -= 1

    class Agent:
        _current_turn_id = "turn-1"

    monkeypatch.setattr(core, "get_rescue_telemetry_client", lambda: FakeClient())

    def worker() -> None:
        entered.set()
        release.wait(timeout=5)

    thread = threading.Thread(
        target=_rescue_account_provider_worker(Agent(), worker),
        daemon=True,
    )
    thread.start()
    assert entered.wait(timeout=2)
    assert active["count"] == 1
    assert thread.is_alive()
    release.set()
    thread.join(timeout=2)
    assert active["count"] == 0


def test_telemetry_is_dormant_without_linux_socket(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import agent.rescue_plane_core as core

    monkeypatch.setattr(core, "RESCUE_EVENT_SOCKET_PATH", tmp_path / "missing.sock")
    assert core.get_rescue_telemetry_client() is None


@linux_only
def test_required_telemetry_outage_blocks_provider_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import agent.rescue_plane_core as core
    from agent.chat_completion_helpers import _rescue_account_provider_call

    continuity = tmp_path / "continuity"
    continuity.mkdir(mode=0o750)
    continuity.chmod(0o750)
    marker = continuity / "telemetry-required-v1.json"
    marker.write_text(
        '{"required":true,"schema_version":'
        '"hermes-rescue-telemetry-required-v1"}',
        encoding="ascii",
    )
    marker.chmod(0o440)
    monkeypatch.setattr(core, "RESCUE_TELEMETRY_REQUIRED_PATH", marker, raising=False)
    monkeypatch.setattr(core, "RESCUE_REPORTER_UID", os.getuid(), raising=False)
    monkeypatch.setattr(core, "RESCUE_EVENT_SOCKET_PATH", tmp_path / "missing.sock")
    calls: list[str] = []

    class Agent:
        _current_turn_id = "turn-outage"

    @_rescue_account_provider_call
    def provider(_agent, _kwargs):
        calls.append("provider")

    with pytest.raises(core.RescueTelemetryUnavailable):
        provider(Agent(), {})
    assert calls == []


@linux_only
@pytest.mark.parametrize(
    "tool_name",
    [
        "todo",
        "session_search",
        "memory",
        "hindsight_retain",
        "clarify",
        "read_terminal",
        "delegate_task",
        "lcm_grep",
        "terminal",
        "write_file",
    ],
)
def test_common_tool_boundary_blocks_every_path_during_required_outage(
    tool_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import agent.rescue_plane_core as core
    from agent.agent_runtime_helpers import invoke_tool

    continuity = tmp_path / "continuity"
    continuity.mkdir(mode=0o750)
    continuity.chmod(0o750)
    marker = continuity / "telemetry-required-v1.json"
    marker.write_bytes(core.TELEMETRY_REQUIRED_MARKER)
    marker.chmod(0o440)
    monkeypatch.setattr(core, "RESCUE_TELEMETRY_REQUIRED_PATH", marker)
    monkeypatch.setattr(core, "RESCUE_REPORTER_UID", os.getuid())
    monkeypatch.setattr(core, "RESCUE_EVENT_SOCKET_PATH", tmp_path / "outage.sock")
    agent = SimpleNamespace(
        session_id="session-outage",
        _current_turn_id="turn-outage",
        _current_api_request_id="request-outage",
        _memory_manager=None,
        valid_tool_names=[],
        enabled_toolsets=None,
        disabled_toolsets=None,
    )

    with pytest.raises(core.RescueTelemetryUnavailable):
        invoke_tool(
            agent,
            tool_name,
            {},
            "task-outage",
        )


@linux_only
def test_sequential_tool_dispatch_uses_common_required_telemetry_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import agent.rescue_plane_core as core
    from agent.tool_executor import _run_agent_tool_execution_middleware

    continuity = tmp_path / "continuity"
    continuity.mkdir(mode=0o750)
    continuity.chmod(0o750)
    marker = continuity / "telemetry-required-v1.json"
    marker.write_bytes(core.TELEMETRY_REQUIRED_MARKER)
    marker.chmod(0o440)
    monkeypatch.setattr(core, "RESCUE_TELEMETRY_REQUIRED_PATH", marker)
    monkeypatch.setattr(core, "RESCUE_REPORTER_UID", os.getuid())
    monkeypatch.setattr(core, "RESCUE_EVENT_SOCKET_PATH", tmp_path / "outage.sock")
    executed: list[str] = []

    with pytest.raises(core.RescueTelemetryUnavailable):
        _run_agent_tool_execution_middleware(
            SimpleNamespace(
                session_id="session-outage",
                _current_turn_id="turn-outage",
                _current_api_request_id="request-outage",
            ),
            function_name="clarify",
            function_args={},
            effective_task_id="task-outage",
            tool_call_id="call-outage",
            execute=lambda _args: executed.append("clarify"),
        )
    assert executed == []


@linux_only
@pytest.mark.parametrize("tool_name", ["tool_search", "tool_describe", "tool_call"])
def test_direct_catalog_paths_use_common_required_telemetry_boundary(
    tool_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import agent.rescue_plane_core as core
    import model_tools
    from model_tools import handle_function_call

    continuity = tmp_path / "continuity"
    continuity.mkdir(mode=0o750)
    continuity.chmod(0o750)
    marker = continuity / "telemetry-required-v1.json"
    marker.write_bytes(core.TELEMETRY_REQUIRED_MARKER)
    marker.chmod(0o440)
    monkeypatch.setattr(core, "RESCUE_TELEMETRY_REQUIRED_PATH", marker)
    monkeypatch.setattr(core, "RESCUE_REPORTER_UID", os.getuid())
    monkeypatch.setattr(core, "RESCUE_EVENT_SOCKET_PATH", tmp_path / "outage.sock")
    catalog_reads: list[str] = []
    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **_kwargs: catalog_reads.append(tool_name),
    )

    with pytest.raises(core.RescueTelemetryUnavailable):
        handle_function_call(tool_name, {}, turn_id="turn-catalog-outage")
    assert catalog_reads == []


def test_background_work_survives_turn_end_and_gateway_death_until_completion() -> None:
    from agent.rescue_quiescence_reporter import ReporterState

    state = ReporterState(max_turns=4)
    state.apply_event(
        {
            "schema_version": "hermes-rescue-event-v1",
            "event": "turn_start",
            "event_id": "turn-start",
            "turn_id": "turn-background",
            "lane": "normal",
            "artifact_requested": False,
        },
        peer_pid=501,
        peer_uid=10000,
        now=1.0,
    )
    state.apply_event(
        {
            "schema_version": "hermes-rescue-event-v1",
            "event": "background_start",
            "event_id": "background-start",
            "turn_id": "turn-background",
            "work_id": "terminal:proc-1",
        },
        peer_pid=501,
        peer_uid=10000,
        now=2.0,
    )
    state.apply_event(
        {
            "schema_version": "hermes-rescue-event-v1",
            "event": "turn_end",
            "event_id": "turn-end",
            "turn_id": "turn-background",
        },
        peer_pid=501,
        peer_uid=10000,
        now=3.0,
    )

    assert state.active_counts() == (0, 1, 0)
    state.reconcile(lambda _pid: False, now=4.0)
    assert state.active_counts() == (0, 1, 0)
    assert state.telemetry_health == "degraded"

    state.apply_event(
        {
            "schema_version": "hermes-rescue-event-v1",
            "event": "background_end",
            "event_id": "background-end",
            "turn_id": "turn-background",
            "work_id": "terminal:proc-1",
        },
        peer_pid=501,
        peer_uid=10000,
        now=5.0,
    )
    assert state.active_counts() == (0, 0, 0)


def test_background_unknown_degrades_without_removing_active_work() -> None:
    from agent.rescue_quiescence_reporter import ReporterState

    state = ReporterState(max_turns=4)
    state.apply_event(
        {
            "schema_version": "hermes-rescue-event-v1",
            "event": "background_start",
            "event_id": "background-start-unknown",
            "turn_id": "turn-background-unknown",
            "work_id": "terminal:proc-unknown",
        },
        peer_pid=502,
        peer_uid=10000,
        now=1.0,
    )
    state.apply_event(
        {
            "schema_version": "hermes-rescue-event-v1",
            "event": "background_unknown",
            "event_id": "background-status-unknown",
            "turn_id": "turn-background-unknown",
            "work_id": "terminal:proc-unknown",
        },
        peer_pid=502,
        peer_uid=10000,
        now=2.0,
    )

    assert state.telemetry_health == "degraded"
    assert state.active_counts() == (0, 1, 0)


def test_registry_dispatch_enforces_required_telemetry_before_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent.rescue_plane_core as core
    from tools.registry import ToolRegistry

    entered: list[bool] = []
    registry = ToolRegistry()
    registry.register(
        name="required_probe",
        toolset="debugging",
        schema={
            "name": "required_probe",
            "description": "probe",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=lambda _args, **_kwargs: entered.append(True) or "{}",
    )
    monkeypatch.setattr(
        core,
        "get_rescue_telemetry_client",
        lambda: (_ for _ in ()).throw(
            core.RescueTelemetryUnavailable("required reporter outage")
        ),
    )

    with pytest.raises(core.RescueTelemetryUnavailable):
        registry.dispatch("required_probe", {}, turn_id="turn-registry-outage")
    assert entered == []


def test_registry_dispatch_deduplicates_existing_tool_middleware_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent.rescue_plane_core as core
    from hermes_cli.middleware import run_tool_execution_middleware
    from tools.registry import ToolRegistry

    events: list[tuple[str, str]] = []

    class FakeClient:
        @contextmanager
        def active_work(self, kind: str, *, turn_id: str):
            events.append((f"{kind}_start", turn_id))
            try:
                yield
            finally:
                events.append((f"{kind}_end", turn_id))

    registry = ToolRegistry()
    registry.register(
        name="dedupe_probe",
        toolset="debugging",
        schema={
            "name": "dedupe_probe",
            "description": "probe",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=lambda _args, **_kwargs: "{}",
    )
    monkeypatch.setattr(core, "get_rescue_telemetry_client", lambda: FakeClient())

    assert (
        run_tool_execution_middleware(
            "dedupe_probe",
            {},
            lambda args: registry.dispatch(
                "dedupe_probe", args, turn_id="turn-deduplicated"
            ),
            turn_id="turn-deduplicated",
        )
        == "{}"
    )
    assert events == [
        ("tool_start", "turn-deduplicated"),
        ("tool_end", "turn-deduplicated"),
    ]


@linux_only
def test_terminal_background_registry_accounts_until_process_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from agent.rescue_plane_core import rescue_tool_execution_scope
    import tools.process_registry as process_module

    events: list[dict[str, object]] = []

    class FakeClient:
        @contextmanager
        def active_work(self, kind: str, *, turn_id: str):
            assert kind == "tool"
            yield

        def emit(self, event):
            events.append(dict(event))

    monkeypatch.setattr(process_module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    registry = process_module.ProcessRegistry()
    with rescue_tool_execution_scope("turn-terminal", client=FakeClient()):
        session = registry.spawn_local(
            "sleep 0.3",
            cwd=str(tmp_path),
            task_id="task-terminal",
        )

    assert [event["event"] for event in events] == ["background_start"]
    registry.wait(session.id, timeout=5)
    deadline = time.time() + 5
    while len(events) < 2 and time.time() < deadline:
        time.sleep(0.01)
    assert [event["event"] for event in events] == [
        "background_start",
        "background_end",
    ]
    assert events[0]["work_id"] == events[1]["work_id"]


@linux_only
def test_terminal_background_descendant_keeps_work_active_until_group_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from agent.rescue_plane_core import rescue_tool_execution_scope
    import tools.process_registry as process_module

    events: list[dict[str, object]] = []

    class FakeClient:
        @contextmanager
        def active_work(self, kind: str, *, turn_id: str):
            assert kind == "tool"
            yield

        def emit(self, event):
            events.append(dict(event))

    monkeypatch.setattr(process_module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    registry = process_module.ProcessRegistry()
    descendant_pid_path = tmp_path / "descendant.pid"
    launcher = tmp_path / "spawn_descendant.py"
    launcher.write_text(
        "import pathlib, subprocess\n"
        "child = subprocess.Popen(\n"
        "    ['sleep', '1.5'],\n"
        "    stdout=subprocess.DEVNULL,\n"
        "    stderr=subprocess.DEVNULL,\n"
        ")\n"
        f"pathlib.Path({str(descendant_pid_path)!r}).write_text(str(child.pid))\n",
        encoding="utf-8",
    )
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(launcher))}"
    with rescue_tool_execution_scope("turn-descendant", client=FakeClient()):
        session = registry.spawn_local(command, cwd=str(tmp_path))

    deadline = time.time() + 5
    while (
        (session.process.poll() is None or not descendant_pid_path.exists())
        and time.time() < deadline
    ):
        time.sleep(0.01)
    assert session.process.poll() == 0
    descendant_pid = int(descendant_pid_path.read_text(encoding="utf-8"))
    os.kill(descendant_pid, 0)

    registry.poll(session.id)
    time.sleep(0.1)
    os.kill(descendant_pid, 0)
    assert [event["event"] for event in events] == [
        "background_start",
        "background_unknown",
    ]

    deadline = time.time() + 5
    while len(events) < 3 and time.time() < deadline:
        time.sleep(0.02)
    assert [event["event"] for event in events] == [
        "background_start",
        "background_unknown",
        "background_end",
    ]


@linux_only
def test_uncertain_environment_launch_never_publishes_false_background_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from agent.rescue_plane_core import rescue_tool_execution_scope
    import tools.process_registry as process_module

    events: list[dict[str, object]] = []

    class FakeClient:
        @contextmanager
        def active_work(self, kind: str, *, turn_id: str):
            assert kind == "tool"
            yield

        def emit(self, event):
            events.append(dict(event))

    class UncertainEnvironment:
        def get_temp_dir(self):
            return "/tmp"

        def execute(self, *_args, **_kwargs):
            raise TimeoutError("backend response lost after possible launch")

    monkeypatch.setattr(process_module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    registry = process_module.ProcessRegistry()
    with rescue_tool_execution_scope("turn-uncertain", client=FakeClient()):
        session = registry.spawn_via_env(
            UncertainEnvironment(),
            "sleep 30",
            task_id="task-uncertain",
        )

    assert session.completion_reason == "launch_unknown"
    assert [event["event"] for event in events] == [
        "background_start",
        "background_unknown",
    ]


@linux_only
def test_async_delegation_accounts_worker_lifetime_and_completion_cleanup() -> None:
    from agent.rescue_plane_core import rescue_tool_execution_scope
    from tools.async_delegation import (
        dispatch_async_delegation,
        list_async_delegations,
    )

    events: list[dict[str, object]] = []
    release = threading.Event()

    class FakeClient:
        @contextmanager
        def active_work(self, kind: str, *, turn_id: str):
            assert kind == "tool"
            yield

        def emit(self, event):
            events.append(dict(event))

    def runner() -> dict[str, object]:
        release.wait(timeout=5)
        return {"status": "completed", "summary": "done"}

    with rescue_tool_execution_scope("turn-delegate", client=FakeClient()):
        result = dispatch_async_delegation(
            goal="wait for cleanup proof",
            context=None,
            toolsets=None,
            role="worker",
            model=None,
            session_key="session",
            runner=runner,
            max_async_children=32,
        )
    delegation_id = result["delegation_id"]
    assert [event["event"] for event in events] == ["background_start"]
    release.set()
    deadline = time.time() + 5
    while time.time() < deadline:
        record = next(
            item
            for item in list_async_delegations()
            if item["delegation_id"] == delegation_id
        )
        if record["status"] != "running":
            break
        time.sleep(0.01)
    assert [event["event"] for event in events] == [
        "background_start",
        "background_end",
    ]
    assert events[0]["work_id"] == events[1]["work_id"]
