from __future__ import annotations

import inspect
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import threading

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
        "2 1 8:1 /rescue /var/run/hermes-rescue ro,nosuid,nodev - ext4 /dev/root ro\n",
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
        event = {
            "schema_version": "hermes-rescue-event-v1",
            "event": "turn_start",
            "event_id": f"event-{index}",
            "turn_id": f"turn-{index}",
            "lane": "normal",
            "artifact_requested": False,
        }
        state.apply_event(event, peer_pid=100 + index, peer_uid=10000, now=float(index))
        state.apply_event(event, peer_pid=100 + index, peer_uid=10000, now=float(index))
        state.apply_event(
            {
                "schema_version": "hermes-rescue-event-v1",
                "event": "turn_end",
                "event_id": f"end-{index}",
                "turn_id": f"turn-{index}",
            },
            peer_pid=100 + index,
            peer_uid=10000,
            now=float(index) + 0.5,
        )

    assert len(state.policy_evidence()) == 2
    assert state.active_counts() == (0, 0, 0)


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
        DurableReplayState(replay).accept("key", 1)


def test_snapshot_strict_types_finite_values_policy_and_durable_replay(tmp_path: Path) -> None:
    from agent.rescue_plane_core import (
        DurableReplayState,
        sign_quiescence_snapshot,
        validate_quiescence_snapshot,
    )

    ring = _keyring()
    replay = DurableReplayState(tmp_path / "replay.json")
    unsigned = {
        "schema_version": "hermes-quiescence-snapshot-v1",
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
        replay_state=DurableReplayState(tmp_path / "replay2.json"),
    ).reason == "malformed"
    malformed_sequence = dict(snapshot)
    malformed_sequence["sequence"] = 2**63
    assert validate_quiescence_snapshot(
        malformed_sequence,
        keyring=ring,
        now=100.2,
        replay_state=DurableReplayState(tmp_path / "replay3.json"),
    ).reason == "malformed"


def test_key_rotation_cutover_accepts_only_correct_slot_and_window(tmp_path: Path) -> None:
    from agent.rescue_plane_core import (
        DurableReplayState,
        sign_quiescence_snapshot,
        validate_quiescence_snapshot,
    )

    ring = _keyring(now=100.0)
    base = {
        "schema_version": "hermes-quiescence-snapshot-v1",
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
    cross_key_replay = DurableReplayState(tmp_path / "cross-key.json")
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
        replay_state=DurableReplayState(tmp_path / "expired.json"),
    ).reason == "old_key_expired"

    new = sign_quiescence_snapshot({**base, "key_id": "new", "sequence": 2, "timestamp": 111.0}, ring, now=111.0)
    assert validate_quiescence_snapshot(
        new,
        keyring=ring,
        now=112.0,
        replay_state=DurableReplayState(tmp_path / "new.json"),
    ).valid


@linux_only
def test_sequence_recovery_requires_valid_previous_signature(tmp_path: Path) -> None:
    from agent.rescue_plane_core import recover_snapshot_sequence, sign_quiescence_snapshot

    ring = _keyring()
    path = tmp_path / "snapshot.json"
    payload = {
        "schema_version": "hermes-quiescence-snapshot-v1",
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
    assert "--expected-reporter-uid" in text
    assert "|| true" not in text
    assert "gateway run" not in text
    assert "/var/run/hermes-rescue" not in text
    assert "--runtime-dir /run/hermes-rescue-reporter" in text
    assert "key=/run/hermes-rescue-secrets/hmac-current" in text
    assert "chmod 0755 /etc/s6-overlay/s6-rc.d/rescue-quiescence-reporter/run" in dockerfile
    assert "requested Hermes GID collides with hermes-rescue" in stage2
    assert '[ -L /run/hermes-rescue-reporter ]' in stage2
    assert "/var/run/hermes-rescue" not in stage2


def test_wrappers_preserve_introspection_contract() -> None:
    from agent.chat_completion_helpers import interruptible_api_call
    from agent.auxiliary_client import async_call_llm, call_llm
    from model_tools import handle_function_call

    assert inspect.unwrap(interruptible_api_call) is not interruptible_api_call
    assert inspect.unwrap(handle_function_call) is not handle_function_call
    assert inspect.unwrap(call_llm) is not call_llm
    assert inspect.unwrap(async_call_llm) is not async_call_llm
    assert "api_kwargs" in inspect.signature(interruptible_api_call).parameters
    assert "function_name" in inspect.signature(handle_function_call).parameters


def test_telemetry_is_dormant_without_linux_socket(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import agent.rescue_plane_core as core

    monkeypatch.setattr(core, "RESCUE_EVENT_SOCKET_PATH", tmp_path / "missing.sock")
    assert core.get_rescue_telemetry_client() is None
