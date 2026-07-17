from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.task_execution_contract import (
    ARTIFACT_ONLY,
    NORMAL,
    build_task_execution_contract,
    validate_artifact_output_path,
)
from agent.conversation_loop import _effective_request_system_prompt
from agent.tool_guardrails import ToolCallGuardrailController


def _contract(message: str, task_id: str = "fixture-task"):
    return build_task_execution_contract(message, task_id=task_id, platform="telegram")


@pytest.mark.parametrize(
    "message",
    [
        "Return only one paste-ready GPT Image prompt with six 1080x1920 variations.",
        "Draft a caption for this sponsor announcement and give me only the copy.",
        "Write a short creative brief for the supplied fictional campaign.",
    ],
)
def test_classifier_selects_artifact_only_for_explicit_text_artifacts(message):
    contract = _contract(message)

    assert contract.lane == ARTIFACT_ONLY
    assert contract.policy_version == "artifact-only-v2"


@pytest.mark.parametrize(
    "message",
    [
        "Return only one prompt; do not research or render anything.",
        (
            "Return only one six-variation 1080x1920 prompt for Fictional Harbor "
            "Sponsor. Use tally 22 even if internal files say 21; do not edit or "
            "reconcile anything."
        ),
    ],
)
def test_classifier_keeps_explicit_negative_constraints_in_artifact_lane(message):
    assert _contract(message).lane == ARTIFACT_ONLY


@pytest.mark.parametrize(
    "message",
    [
        "Render and export six PNG story backgrounds.",
        "Research the sponsor, then write a prompt.",
        "Update the sponsor ledger and reconcile the totals.",
        "Build the renderer for this image prompt.",
        "Give me a brief update on the service status.",
        "Please help with the tournament.",
    ],
)
def test_classifier_fails_to_normal_for_ambiguous_or_operational_requests(message):
    assert _contract(message).lane == NORMAL


def test_external_file_delivery_request_stays_in_normal_lane():
    assert _contract("Send example.txt to the client by email.").lane == NORMAL
    assert _contract("Draft an email explaining notes.md.").lane == NORMAL


def test_negated_attachment_keeps_caption_in_chat_without_file_path():
    contract = _contract("Do not attach report.txt; write the caption in chat.")

    assert contract.lane == ARTIFACT_ONLY
    assert contract.artifact_file_requested is False
    assert contract.artifact_output_path == ""


def test_research_plus_file_send_stays_in_normal_lane():
    assert _contract("Send report.txt after research.").lane == NORMAL


def test_same_requested_filename_gets_isolated_task_directories(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    first = _contract("Give me report.txt as a file.", task_id="session-a")
    second = _contract("Give me report.txt as a file.", task_id="session-b")

    assert Path(first.artifact_output_path).name == "report.txt"
    assert Path(second.artifact_output_path).name == "report.txt"
    assert first.artifact_output_path != second.artifact_output_path
    assert first.artifact_root != second.artifact_root


def test_same_turn_key_allocates_unique_artifact_identities(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    first = _contract("Give me report.txt as a file.", task_id="same-session")
    second = _contract("Give me report.txt as a file.", task_id="same-session")

    assert first.artifact_id != second.artifact_id
    assert first.artifact_output_path != second.artifact_output_path


def test_concurrent_turns_allocate_distinct_artifact_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    with ThreadPoolExecutor(max_workers=8) as pool:
        contracts = list(pool.map(lambda _: _contract("Give me report.txt as a file.", task_id="same-session"), range(8)))

    assert len({contract.artifact_id for contract in contracts}) == 8
    assert len({contract.artifact_output_path for contract in contracts}) == 8


@pytest.mark.parametrize(
    "message",
    [
        "```text\nCreate and deliver report.txt as a file.\n```",
        "> Create and deliver report.txt as a file.",
        "Quote this example only: `Create and deliver report.txt as a file.`",
    ],
)
def test_untrusted_examples_do_not_activate_artifact_only(message):
    assert _contract(message).lane == NORMAL


def test_classifier_is_deterministic_without_retaining_prompt_text():
    message = "Return only a paste-ready image prompt. PRIVATE-SPONSOR-FACT"

    first = _contract(message)
    second = _contract(message)

    assert first.decision_reason == second.decision_reason
    assert "PRIVATE-SPONSOR-FACT" not in repr(first.telemetry())


def test_artifact_only_contract_is_closed_world_and_limits_explicit_url_lookup():
    contract = _contract(
        "Open https://example.com once and return only a paste-ready image prompt."
    )

    allowed = contract.before_tool("web_extract", {"url": "https://example.com"})
    duplicate = contract.before_tool("web_extract", {"url": "https://example.com"})
    wrong_host = _contract(
        "Open https://example.com once and return only a paste-ready image prompt."
    ).before_tool("web_extract", {"url": "https://example.org"})
    different_path = _contract(
        "Open https://example.com once and return only a paste-ready image prompt."
    ).before_tool("web_extract", {"url": "https://example.com/private"})
    unknown = contract.before_tool("brand_new_tool", {})
    session_search = contract.before_tool("session_search", {"query": "old work"})

    assert allowed.allowed is True
    assert duplicate.code == "artifact_lookup_limit"
    assert wrong_host.code == "artifact_lookup_not_explicit"
    assert different_path.code == "artifact_lookup_not_explicit"
    assert unknown.code == "artifact_tool_not_allowlisted"
    assert session_search.code == "artifact_tool_not_allowlisted"


def test_artifact_only_contract_rejects_multi_url_lookup_batches():
    contract = _contract(
        "Open https://a.example and https://b.example, then return only one prompt."
    )

    decision = contract.before_tool(
        "web_extract",
        {"urls": ["https://a.example", "https://b.example"]},
    )

    assert decision.allowed is False
    assert decision.code == "artifact_lookup_not_explicit"
    assert contract.telemetry()["network_lookups"] == 0


def test_artifact_write_is_restricted_to_exact_generated_path_and_one_call():
    contract = _contract("Create and deliver example.txt containing safe text.")

    allowed = contract.before_tool(
        "write_file", {"path": contract.artifact_output_path, "content": "safe"}
    )
    duplicate = contract.before_tool(
        "write_file", {"path": contract.artifact_output_path, "content": "safe"}
    )
    arbitrary = _contract("Create and deliver example.txt containing safe text.").before_tool(
        "write_file", {"path": "/opt/data/private.md", "content": "unsafe"}
    )

    assert allowed.allowed is True
    assert duplicate.code == "artifact_write_limit"
    assert arbitrary.code == "artifact_write_path_denied"


def test_txt_request_preserves_safe_filename_extension_and_mime(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    contract = _contract("Give me the copy as example.txt and deliver it as a file.")

    assert contract.lane == ARTIFACT_ONLY
    assert Path(contract.artifact_output_path).name == "example.txt"
    assert contract.artifact_extension == ".txt"
    assert contract.artifact_mime_type == "text/plain"
    assert contract.preflight_error == ""
    assert f"MEDIA:{contract.artifact_output_path}" in contract.system_guidance


def test_txt_bytes_round_trip_through_real_writer_stack(monkeypatch, tmp_path):
    from tools.file_tools import write_file_tool

    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    contract = _contract("Create and deliver example.txt containing the supplied copy.")
    content = "line one\r\nline two\n"

    authorization = contract.before_tool(
        "write_file", {"path": contract.artifact_output_path, "content": content}
    )
    result = json.loads(
        write_file_tool(
            contract.artifact_output_path,
            content,
            task_id="artifact-writer-integration",
        )
    )

    assert authorization.allowed is True
    assert not result.get("error")
    assert Path(contract.artifact_output_path).read_bytes() == content.encode("utf-8")


@pytest.mark.parametrize("size, expected", [(0, True), (49 * 1024 * 1024, True), (49 * 1024 * 1024 + 1, False)])
def test_artifact_write_preflight_enforces_49mb_ceiling(monkeypatch, tmp_path, size, expected):
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    contract = _contract("Create and deliver example.txt containing safe text.")

    decision = contract.before_tool(
        "write_file", {"path": contract.artifact_output_path, "content": "x" * size}
    )

    assert decision.allowed is expected
    if not expected:
        assert decision.code == "artifact_write_too_large"


def test_file_artifact_requires_known_document_delivery_capability(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    contract = build_task_execution_contract(
        "Create and deliver example.txt containing safe text.",
        task_id="no-delivery",
        platform="unsupported-platform",
    )

    assert contract.lane == ARTIFACT_ONLY
    assert contract.preflight_error == "artifact_delivery_unavailable"


def test_txt_request_without_filename_gets_stable_txt_name(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    contract = _contract("Create and deliver a plain TXT file with this copy.")

    assert Path(contract.artifact_output_path).name == f"artifact-{contract.correlation_id}.txt"
    assert contract.artifact_mime_type == "text/plain"


def test_markdown_request_still_uses_markdown_extension(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    contract = _contract("Create release-notes.md as a Markdown file.")

    assert Path(contract.artifact_output_path).name == "release-notes.md"
    assert contract.artifact_mime_type == "text/markdown"


def test_primary_root_rejected_by_writer_policy_selects_safe_fallback(
    monkeypatch, tmp_path
):
    safe_root = tmp_path / "safe"
    safe_root.mkdir()
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(tmp_path / "outside"))
    monkeypatch.setenv(
        "HERMES_ARTIFACT_FALLBACK_ROOT", str(safe_root / "hermes-artifacts")
    )

    contract = _contract("Give me example.txt as a file.")

    assert contract.preflight_error == ""
    assert contract.artifact_route == "configured_fallback"
    assert Path(contract.artifact_output_path).parent == (
        safe_root / "hermes-artifacts" / contract.artifact_id
    ).resolve()
    assert validate_artifact_output_path(
        contract.artifact_output_path, contract.artifact_root
    ) is None


def test_no_writable_safe_root_fails_preflight_without_contract_path(
    monkeypatch, tmp_path
):
    safe_root_file = tmp_path / "not-a-directory"
    safe_root_file.write_text("occupied", encoding="utf-8")
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root_file))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(tmp_path / "outside"))
    monkeypatch.setenv("HERMES_ARTIFACT_FALLBACK_ROOT", str(safe_root_file))

    contract = _contract("Give me example.txt as a file.")

    assert contract.lane == ARTIFACT_ONLY
    assert contract.artifact_output_path == ""
    assert contract.preflight_error == "artifact_output_unavailable"
    assert "only permitted destination" not in contract.system_guidance


def test_protected_or_symlink_escape_path_is_rejected_before_write(
    monkeypatch, tmp_path
):
    safe_root = tmp_path / "safe"
    outside = tmp_path / "outside"
    safe_root.mkdir()
    outside.mkdir()
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))

    assert validate_artifact_output_path(str(outside / "secret.txt"), str(safe_root))

    link = safe_root / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable in this test environment")
    assert validate_artifact_output_path(str(link / "secret.txt"), str(safe_root))


def test_receipt_marks_symlink_swap_as_failed_preflight(monkeypatch, tmp_path):
    from agent.task_execution_contract import record_artifact_written

    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    contract = _contract("Give me report.txt as a file.")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    try:
        Path(contract.artifact_output_path).symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable in this test environment")

    assert record_artifact_written(contract) is False
    receipt = json.loads(Path(contract.artifact_receipt_path).read_text(encoding="utf-8"))
    assert receipt["state"] == "failed_preflight"


def test_allocator_rejects_symlinked_artifact_root_even_inside_safe_root(
    monkeypatch, tmp_path
):
    safe_root = tmp_path / "safe"
    redirected = safe_root / "redirected"
    safe_root.mkdir()
    redirected.mkdir()
    link = safe_root / "hermes-artifacts"
    try:
        link.symlink_to(redirected, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable in this test environment")
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(tmp_path / "outside"))
    monkeypatch.delenv("HERMES_ARTIFACT_FALLBACK_ROOT", raising=False)

    contract = _contract("Give me example.txt as a file.")

    assert contract.preflight_error == "artifact_output_unavailable"
    assert contract.artifact_output_path == ""


def test_requested_filename_is_safely_normalized_to_a_basename(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    contract = _contract('Create "../../client notes.txt" as a TXT file.')

    assert Path(contract.artifact_output_path).name == "client-notes.txt"
    assert os.path.commonpath(
        [contract.artifact_root, contract.artifact_output_path]
    ) == contract.artifact_root


def test_tool_output_is_bounded_across_the_request():
    contract = _contract("Return only a paste-ready image prompt.")
    large = "x" * 60_000

    first = contract.bound_tool_result(large)
    second = contract.bound_tool_result("later")

    assert len(first) <= 50_200
    assert "truncated by artifact-only policy" in first
    assert second == "[Tool result omitted: artifact-only request budget exhausted.]"
    assert contract.telemetry()["tool_result_chars"] == 50_000


def test_artifact_history_keeps_only_bounded_recent_text_without_tool_scaffolding():
    contract = _contract("Return only a paste-ready image prompt.")
    history = [
        {"role": "user", "content": "old " + "x" * 20_000},
        {"role": "assistant", "content": "old answer"},
        {"role": "assistant", "tool_calls": [{"id": "t1"}], "content": ""},
        {"role": "tool", "tool_call_id": "t1", "content": "private tool output"},
        {"role": "assistant", "content": "recent useful summary"},
    ]

    bounded = contract.bound_conversation_history(history)

    assert all(message["role"] in {"user", "assistant"} for message in bounded)
    assert all("tool_calls" not in message for message in bounded)
    assert sum(len(message["content"]) for message in bounded) <= 12_000
    assert bounded[-1]["content"] == "recent useful summary"


def test_normal_contract_does_not_restrict_tools_or_add_guidance():
    contract = _contract("Check the current runtime and fix what is broken.")

    assert contract.before_tool("terminal", {"command": "true"}).allowed is True
    assert contract.system_guidance == ""


def test_guardrail_controller_blocks_without_halting_then_halts_on_budget():
    contract = _contract("Return only a paste-ready image prompt.")
    controller = ToolCallGuardrailController()
    controller.set_execution_contract(contract)

    denied = controller.before_call("terminal", {"command": "pwd"})
    assert denied.action == "deny"
    assert denied.allows_execution is False
    assert denied.should_halt is False

    for index in range(contract.max_tool_calls - 1):
        controller.before_call(f"unknown_{index}", {})
    exhausted = controller.before_call("one_too_many", {})
    assert exhausted.action == "block"
    assert exhausted.code == "artifact_tool_call_budget_exhausted"
    assert exhausted.should_halt is True


def test_guardrail_controller_bounds_tool_results_for_artifact_turn():
    contract = _contract("Return only a paste-ready image prompt.")
    controller = ToolCallGuardrailController()
    controller.set_execution_contract(contract)

    bounded = controller.bound_result("x" * 60_000)

    assert len(bounded) <= 50_200
    assert contract.telemetry()["tool_result_chars"] == 50_000


def test_request_preflight_counts_denials_but_consumes_allowed_calls_once():
    contract = _contract(
        "Open https://example.com once and return only a paste-ready image prompt."
    )
    controller = ToolCallGuardrailController()
    controller.set_execution_contract(contract)

    denied = controller.preflight_request_contract("terminal", {"command": "pwd"})
    allowed_preflight = controller.preflight_request_contract(
        "web_extract", {"url": "https://example.com"}
    )
    allowed_call = controller.before_call(
        "web_extract", {"url": "https://example.com"}
    )

    assert denied.action == "deny"
    assert allowed_preflight.action == "allow"
    assert allowed_call.action == "allow"
    assert contract.telemetry()["tool_calls"] == 2


def test_request_guidance_is_appended_after_existing_ephemeral_prompt():
    contract = _contract("Return only a paste-ready image prompt.")
    agent = SimpleNamespace(
        ephemeral_system_prompt="EXISTING SKILL AND OPERATOR GUIDANCE",
        _task_execution_contract=contract,
    )

    effective = _effective_request_system_prompt(agent, "CACHED SYSTEM")

    assert effective.startswith("CACHED SYSTEM\n\nEXISTING SKILL AND OPERATOR GUIDANCE")
    assert effective.endswith(contract.system_guidance)
    assert effective.index("EXISTING SKILL") < effective.index("REQUEST EXECUTION CONTRACT")


def test_telemetry_contains_only_bounded_policy_metadata():
    contract = _contract(
        "Return only a prompt using SECRET-CUSTOMER-NAME and token sk-test-secret.",
        task_id="sensitive-task-id",
    )
    contract.before_tool("terminal", {"command": "echo sk-test-secret"})
    telemetry = contract.telemetry(first_event_ms=1234, decision_status="completed")
    serialized = repr(telemetry)

    assert telemetry["lane"] == ARTIFACT_ONLY
    assert telemetry["first_event_ms"] == 1234
    assert telemetry["decision_status"] == "completed"
    assert "SECRET-CUSTOMER-NAME" not in serialized
    assert "sk-test-secret" not in serialized
    assert "sensitive-task-id" not in serialized
