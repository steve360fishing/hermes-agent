"""Source-only contracts for the Hermes Rescue Plane V1 core slice."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

FIXTURE_ID = "HERMES-ARTIFACT-STICKY-20260717-v1"
FIXTURE_SHA256 = "d8d190a659e77d714b555dfbccf49f15222171db8bccaa37d38d31373c81f420"


def test_sticky_artifact_fixture_is_frozen_and_identifiable() -> None:
    """The incident replay is an immutable, sanitized source fixture."""
    from agent.rescue_plane_core import load_sticky_artifact_fixture

    fixture = Path(__file__).parents[1] / "fixtures" / "rescue_plane" / "artifact-sticky-20260717-v1.json"
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == FIXTURE_SHA256
    assert load_sticky_artifact_fixture(fixture) == {
        "fixture_id": FIXTURE_ID,
        "schema_version": "hermes-rescue-artifact-replay-v1",
        "turns": [
            {
                "artifact_requested": False,
                "filesystem_artifact_observed": True,
                "lane": "normal",
                "turn_id": "sanitized-turn-001",
            },
            {
                "artifact_requested": False,
                "delivery_receipt_artifact_observed": True,
                "lane": "normal",
                "turn_id": "sanitized-turn-002",
            },
        ],
    }


def _overlay(path: Path, **overrides: object) -> None:
    payload = {
        "schema_version": "hermes-safe-mode-v1",
        "enabled": True,
        "incident_id": "4faeb31c-15fe-4f14-a1e2-11892cbcb5b6",
        "issued_at": "2026-07-17T12:00:00Z",
        "disables": ["artifact_only"],
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o640)


def test_valid_overlay_disables_only_artifact_only(tmp_path: Path) -> None:
    from agent.rescue_plane_core import read_safe_mode_overlay

    overlay = tmp_path / "safe-mode-v1.json"
    _overlay(overlay)

    decision = read_safe_mode_overlay(overlay)

    assert decision.valid is True
    assert decision.disables == frozenset({"artifact_only"})
    assert decision.disables_artifact_only is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"unexpected": True},
        {"disables": ["artifact_only", "tools"]},
        {"enabled": False},
        {"incident_id": "not-a-uuid"},
    ],
)
def test_unknown_or_malformed_overlay_fails_closed(tmp_path: Path, overrides: dict[str, object]) -> None:
    from agent.rescue_plane_core import read_safe_mode_overlay

    overlay = tmp_path / "safe-mode-v1.json"
    _overlay(overlay, **overrides)

    decision = read_safe_mode_overlay(overlay)

    assert decision.valid is False
    assert decision.disables_artifact_only is True
    assert decision.reason.startswith("invalid_")


def test_missing_overlay_preserves_normal_policy(tmp_path: Path) -> None:
    from agent.rescue_plane_core import read_safe_mode_overlay

    decision = read_safe_mode_overlay(tmp_path / "absent.json")

    assert decision.valid is True
    assert decision.disables_artifact_only is False
    assert decision.reason == "absent"


def test_overlay_symlink_and_insecure_mode_fail_closed(tmp_path: Path) -> None:
    from agent.rescue_plane_core import read_safe_mode_overlay

    target = tmp_path / "target.json"
    _overlay(target)
    linked = tmp_path / "safe-mode-v1.json"
    try:
        linked.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    assert read_safe_mode_overlay(linked).reason == "invalid_symlink"
    linked.unlink()
    _overlay(linked)
    linked.chmod(0o644)
    assert read_safe_mode_overlay(linked).reason == "invalid_mode"


def test_overlay_is_applied_per_turn_without_sticky_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import agent.rescue_plane_core as rescue_core
    from agent.task_execution_contract import ARTIFACT_ONLY, NORMAL, build_task_execution_contract

    overlay = tmp_path / "safe-mode-v1.json"
    monkeypatch.setattr(rescue_core, "SAFE_MODE_OVERLAY_PATH", overlay)
    first = build_task_execution_contract(
        "Return only one paste-ready GPT Image prompt.", task_id="first", platform="telegram"
    )
    _overlay(overlay)
    second = build_task_execution_contract(
        "Return only one paste-ready GPT Image prompt.", task_id="second", platform="telegram"
    )
    overlay.unlink()
    third = build_task_execution_contract(
        "Return only one paste-ready GPT Image prompt.", task_id="third", platform="telegram"
    )

    assert first.lane == ARTIFACT_ONLY
    assert second.lane == NORMAL
    assert second.decision_reason == "artifact_only_disabled_by_rescue_overlay"
    assert third.lane == ARTIFACT_ONLY


def test_execution_accounting_tracks_turn_tool_and_provider_lifetimes() -> None:
    from agent.rescue_plane_core import RescueExecutionTelemetry

    telemetry = RescueExecutionTelemetry()
    telemetry.start_turn("turn-1", lane="normal", artifact_requested=False, now=10.0)
    with telemetry.active_tool():
        with telemetry.active_provider_action():
            assert telemetry.active_counts() == (1, 1, 1)
    telemetry.finish_turn("turn-1", now=11.0)

    assert telemetry.active_counts() == (0, 0, 0)
    assert telemetry.turn_record("turn-1") == {
        "artifact_requested": False,
        "completed_at": 11.0,
        "lane": "normal",
        "started_at": 10.0,
        "turn_id": "turn-1",
    }


def test_signed_quiescence_snapshots_reject_stale_replay_and_wrong_key() -> None:
    from agent.rescue_plane_core import RescueExecutionTelemetry, validate_quiescence_snapshot

    telemetry = RescueExecutionTelemetry()
    key = b"a" * 32
    snapshot = telemetry.quiescence_snapshot(
        key=key,
        key_id="current-v1",
        gateway_pid=123,
        gateway_state="dead",
        source_sha="a" * 40,
        image_id="sha256:" + "b" * 64,
        now=100.0,
    )
    seen: dict[str, int] = {}

    assert validate_quiescence_snapshot(snapshot, keys={"current-v1": key}, now=114.9, seen_sequences=seen).valid
    assert validate_quiescence_snapshot(snapshot, keys={"current-v1": key}, now=115.0, seen_sequences={}).reason == "stale"
    assert validate_quiescence_snapshot(snapshot, keys={"current-v1": key}, now=101.0, seen_sequences=seen).reason == "replayed"
    assert validate_quiescence_snapshot(snapshot, keys={"other": key}, now=101.0, seen_sequences={}).reason == "unknown_key"
    tampered = dict(snapshot)
    tampered["active_tool_count"] = 1
    assert validate_quiescence_snapshot(tampered, keys={"current-v1": key}, now=101.0, seen_sequences={}).reason == "invalid_signature"


def test_reporter_source_is_independent_of_gateway_and_preserves_active_counts(tmp_path: Path) -> None:
    from agent.rescue_plane_core import RescueExecutionTelemetry
    from agent.rescue_quiescence_reporter import QuiescenceReporter

    telemetry_path = tmp_path / "turn-telemetry-v1.json"
    output_path = tmp_path / "quiescence-v1.json"
    key_path = tmp_path / "quiescence.key"
    key_path.write_bytes(b"q" * 32)
    key_path.chmod(0o400)
    telemetry = RescueExecutionTelemetry()
    telemetry.start_turn("turn-1", lane="normal", artifact_requested=False, now=10.0)
    with telemetry.active_tool():
        telemetry.write(telemetry_path)
        reporter = QuiescenceReporter(
            telemetry_path=telemetry_path,
            output_path=output_path,
            key_path=key_path,
            key_id="current-v1",
            source_sha="a" * 40,
            image_id="sha256:" + "b" * 64,
            gateway_state=lambda: (None, "dead"),
        )
        snapshot = reporter.emit_once(now=20.0)

    assert snapshot["gateway_state"] == "dead"
    assert snapshot["active_turn_count"] == 1
    assert snapshot["active_tool_count"] == 1
    assert snapshot["active_provider_action_count"] == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["signature"] == snapshot["signature"]
    next_snapshot = QuiescenceReporter(
        telemetry_path=telemetry_path,
        output_path=output_path,
        key_path=key_path,
        key_id="current-v1",
        source_sha="a" * 40,
        image_id="sha256:" + "b" * 64,
        gateway_state=lambda: (None, "dead"),
    ).emit_once(now=21.0)
    assert next_snapshot["sequence"] == snapshot["sequence"] + 1


def test_s6_reporter_service_is_separate_from_gateway() -> None:
    root = Path(__file__).parents[2]
    run = (root / "docker" / "s6-rc.d" / "rescue-quiescence-reporter" / "run").read_text(encoding="utf-8")
    bundle_entry = root / "docker" / "s6-rc.d" / "user" / "contents.d" / "rescue-quiescence-reporter"

    assert "agent.rescue_quiescence_reporter" in run
    assert "gateway run" not in run
    assert "HERMES_RESCUE" not in run
    assert "sleep 10" in run
    assert bundle_entry.is_file()
