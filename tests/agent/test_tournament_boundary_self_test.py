from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

import agent.tournament_boundary_self_test as boundary_self_test
from agent.tournament_boundary_self_test import main, run_boundary_self_test
from agent.tournament_intent_contract import ContractDecision, TournamentIntentState


IDENTITY = "a" * 64
ROUTING = "b" * 64


def _run(**overrides):
    values = {
        "source_revision": IDENTITY,
        "image_digest": f"sha256:{IDENTITY}",
        "expected_source_revision": IDENTITY,
        "expected_image_digest": f"sha256:{IDENTITY}",
        "routing_fingerprint": ROUTING,
        "expected_routing_fingerprint": ROUTING,
        "canonical_image_digest": f"sha256:{IDENTITY}",
        "live_image_digest": f"sha256:{IDENTITY}",
        "now": datetime(2026, 8, 11, 19, 0, tzinfo=timezone.utc),
    }
    values.update(overrides)
    return run_boundary_self_test(**values)


def test_self_test_passes_without_network_provider_subprocess_or_writes(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError("external I/O is forbidden")

    monkeypatch.setattr("socket.create_connection", blocked)
    monkeypatch.setattr("subprocess.run", blocked)
    monkeypatch.setattr("subprocess.Popen", blocked)
    monkeypatch.setattr("pathlib.Path.write_bytes", blocked)
    monkeypatch.setattr("pathlib.Path.write_text", blocked)

    receipt = _run()

    assert receipt["overall"] == "PASS"
    assert all(case["passed"] for case in receipt["cases"])
    serialized = json.dumps(receipt, sort_keys=True)
    for forbidden in (
        "prompt",
        "output_text",
        "user_id",
        "chat_id",
        "task_id",
        "session_id",
        "destination",
        "receipt_path",
        "credential",
        "environment",
        "url",
        "tool_args",
    ):
        assert forbidden not in serialized.lower()


@pytest.mark.parametrize(
    ("override", "case_id"),
    (
        ({"source_revision": "c" * 64}, "source_provenance"),
        ({"image_digest": "sha256:" + "c" * 64}, "image_provenance"),
        ({"routing_fingerprint": "c" * 64}, "routing_identity"),
        ({"live_image_digest": "sha256:" + "c" * 64}, "canonical_live_image"),
    ),
)
def test_seeded_identity_failures_hold(override, case_id):
    receipt = _run(**override)

    assert receipt["overall"] == "HOLD"
    failed = {case["id"] for case in receipt["cases"] if not case["passed"]}
    assert failed == {case_id}


def test_seeded_quoted_data_regression_holds(monkeypatch):
    original = boundary_self_test.classify_tournament_intent

    def regressed(message):
        if "quoted text privately" in str(message):
            return TournamentIntentState.PUBLICATION_REQUEST
        return original(message)

    monkeypatch.setattr(boundary_self_test, "classify_tournament_intent", regressed)

    receipt = _run()

    assert receipt["overall"] == "HOLD"
    assert {case["id"] for case in receipt["cases"] if not case["passed"]} == {
        "private_quoted_data"
    }


def test_seeded_cron_contract_installation_regression_holds(monkeypatch):
    monkeypatch.setattr(
        boundary_self_test,
        "platform_bypasses_tournament_contract",
        lambda _platform: False,
    )

    receipt = _run()

    assert receipt["overall"] == "HOLD"
    assert {case["id"] for case in receipt["cases"] if not case["passed"]} == {
        "cron_whitespace_bypass"
    }


def test_seeded_ungated_public_release_regression_holds(monkeypatch):
    monkeypatch.setattr(
        boundary_self_test.TournamentIntentContract,
        "authorize_tool",
        lambda *_args, **_kwargs: ContractDecision(True, "regressed"),
    )

    receipt = _run()

    assert receipt["overall"] == "HOLD"
    assert {case["id"] for case in receipt["cases"] if not case["passed"]} == {
        "ungated_public_release"
    }


def test_cli_returns_nonzero_for_a_seeded_failure(capsys):
    exit_code = main(
        [
            "--source-revision",
            "c" * 64,
            "--expected-source-revision",
            IDENTITY,
            "--image-digest",
            f"sha256:{IDENTITY}",
            "--expected-image-digest",
            f"sha256:{IDENTITY}",
            "--routing-fingerprint",
            ROUTING,
            "--expected-routing-fingerprint",
            ROUTING,
            "--canonical-image-digest",
            f"sha256:{IDENTITY}",
            "--live-image-digest",
            f"sha256:{IDENTITY}",
        ]
    )

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out)["overall"] == "HOLD"
