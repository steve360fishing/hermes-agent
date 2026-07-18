from __future__ import annotations

import json
import os
import time
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
import agent.conversation_loop as conversation_loop
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
    assert contract.policy_version == "artifact-only-v3"


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


@pytest.mark.parametrize(
    "message",
    [
        (
            "Alright so we got cut off there. This was my last prompt before we "
            "got cut off and I want to make sure you don't miss anything."
        ),
        (
            "I've been trying to send you copies of our conversation even as a "
            "text file but you keep saying there is no safe artifact destination."
        ),
        (
            "I don't know why you're giving me an artifact.txt file. We have work "
            "to do here and need to start following tournaments."
        ),
    ],
)
def test_incident_recovery_language_never_activates_artifact_only(message):
    assert _contract(message).lane == NORMAL


def test_emergency_disable_keeps_explicit_artifact_request_normal(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"agent": {"artifact_only_enabled": False}},
    )

    contract = _contract("Create and deliver recovery.txt containing safe text.")

    assert contract.lane == NORMAL
    assert contract.decision_reason == "artifact_only_disabled"
    assert contract.artifact_output_path == ""
    assert contract.preflight_error == ""


def test_emergency_disable_fails_closed_when_config_cannot_be_read(monkeypatch):
    def fail_config_read():
        raise OSError("config unavailable")

    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        fail_config_read,
    )

    contract = _contract("Create and deliver recovery.txt containing safe text.")

    assert contract.lane == NORMAL
    assert contract.decision_reason == "artifact_only_disabled"
    assert contract.artifact_output_path == ""


@pytest.mark.parametrize(
    "message",
    [
        "Could you please create recovery.txt containing safe text?",
        "Make a copy for the sponsor.",
        "Can you make a creative brief?",
    ],
)
def test_common_direct_artifact_requests_remain_supported(message):
    assert _contract(message).lane == ARTIFACT_ONLY


def test_normal_txt_normal_sequence_has_no_sticky_lane(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"agent": {"artifact_only_enabled": True}},
    )

    before = _contract(
        "Inspect the runtime status and investigate any failures.",
        task_id="normal-before",
    )
    txt = _contract(
        "Create and deliver recovery.txt containing safe text.",
        task_id="txt-middle",
    )
    after = _contract("Continue with normal analysis.", task_id="normal-after")

    assert [before.lane, txt.lane, after.lane] == [
        NORMAL,
        ARTIFACT_ONLY,
        NORMAL,
    ]
    assert after.artifact_output_path == ""
    assert after.artifact_file_requested is False


@pytest.mark.parametrize(
    "separator",
    [
        ". ",
        ", ",
        ": ",
        "; ",
        " - ",
        " \u2013 ",
        " \u2014 ",
        " (",
        " [",
        ' "',
        " \u201c",
    ],
)
def test_file_request_verb_and_filename_must_be_in_the_same_clause(separator):
    message = f"Send me the runtime status{separator}the old artifact.txt was wrong."
    if separator == " (":
        message += ")"

    assert _contract(message).lane == NORMAL


@pytest.mark.parametrize(
    "message",
    [
        "Create pre-flight report.txt",
        "Create concise, client-ready report.txt",
        r"Create C:\reports\release-notes.txt",
    ],
)
def test_ambiguous_punctuation_falls_back_to_normal_file_handling(message):
    contract = _contract(message)

    assert contract.lane == NORMAL
    assert contract.before_tool("write_file", {"path": "safe.txt"}).allowed


@pytest.mark.parametrize(
    "separator",
    [" and ", "\x85", "\u2028", "\u2029"],
)
def test_conjunctions_and_all_splitline_boundaries_do_not_bridge_to_old_files(
    separator,
):
    message = (
        "Send me the runtime status"
        + separator
        + "the old artifact.txt was wrong."
    )

    assert _contract(message).lane == NORMAL


def test_bounded_direct_file_grammar_still_accepts_a_simple_request():
    assert _contract("Send me the final report.txt").lane == ARTIFACT_ONLY


@pytest.mark.parametrize(
    "message",
    [
        (
            "Create report.txt was the bad instruction, not my request. "
            "Diagnose the gateway instead."
        ),
        'The phrase "Create report.txt" was the old broken instruction.',
        "Create report.txt is an example, not a request.",
        "Create report.txt caused the gateway failure; this is historical context.",
        "Create report.txt \u2014 that instruction caused the outage.",
    ],
)
def test_discussed_or_historical_file_instructions_stay_normal(message):
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

    assert contract.lane == NORMAL
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


def test_terminal_receipt_cannot_regress_and_registry_is_cleaned(monkeypatch, tmp_path):
    from agent.task_execution_contract import (
        _ARTIFACT_RECEIPTS,
        record_artifact_dispatch,
        record_artifact_written,
    )

    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    contract = _contract("Give me report.txt as a file.")
    Path(contract.artifact_output_path).write_bytes(b"payload")
    assert record_artifact_written(contract) is True

    assert record_artifact_dispatch(contract.artifact_output_path, state="dispatching")
    assert record_artifact_dispatch(
        contract.artifact_output_path, state="delivered", message_id="msg-1"
    )
    record_artifact_dispatch(contract.artifact_output_path, state="dispatching")

    receipt = json.loads(Path(contract.artifact_receipt_path).read_text(encoding="utf-8"))
    assert receipt["state"] == "delivered"
    assert receipt["attempt_count"] == 0
    assert os.path.abspath(contract.artifact_output_path) not in _ARTIFACT_RECEIPTS
    assert not Path(contract.artifact_root).exists()


def test_concurrent_receipt_transitions_are_serialized(monkeypatch, tmp_path):
    from agent.task_execution_contract import record_artifact_dispatch, record_artifact_written

    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    contract = _contract("Give me report.txt as a file.")
    Path(contract.artifact_output_path).write_bytes(b"payload")
    assert record_artifact_written(contract) is True

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(
            lambda _: record_artifact_dispatch(contract.artifact_output_path, state="dispatching"),
            range(24),
        ))
    record_artifact_dispatch(contract.artifact_output_path, state="delivered", message_id="msg-1")

    receipt = json.loads(Path(contract.artifact_receipt_path).read_text(encoding="utf-8"))
    assert receipt["state"] == "delivered"
    assert receipt["attempt_count"] == 0


def test_receipt_write_failure_keeps_registry_and_artifact(monkeypatch, tmp_path):
    import agent.task_execution_contract as contract_module
    from agent.task_execution_contract import (
        _ARTIFACT_RECEIPTS,
        record_artifact_dispatch,
        record_artifact_written,
    )

    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    contract = _contract("Give me report.txt as a file.", task_id="receipt-io-failure")
    artifact = Path(contract.artifact_output_path)
    artifact.write_bytes(b"payload")
    assert record_artifact_written(contract) is True
    before = Path(contract.artifact_receipt_path).read_bytes()
    real_replace = contract_module.os.replace
    real_rename = contract_module.os.rename

    def fail_receipt_replace(source, destination, *args, **kwargs):
        if os.path.basename(destination) == os.path.basename(contract.artifact_receipt_path):
            raise OSError("simulated durable receipt failure")
        return real_replace(source, destination, *args, **kwargs)

    def fail_receipt_rename(source, destination, *args, **kwargs):
        if os.path.basename(destination) == os.path.basename(contract.artifact_receipt_path):
            raise OSError("simulated durable receipt failure")
        return real_rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(contract_module.os, "replace", fail_receipt_replace)
    monkeypatch.setattr(contract_module.os, "rename", fail_receipt_rename)

    with pytest.raises(OSError, match="receipt"):
        record_artifact_dispatch(
            contract.artifact_output_path,
            state="ambiguous",
            error_code="document_dispatch_exception",
        )

    assert Path(contract.artifact_receipt_path).read_bytes() == before
    assert artifact.read_bytes() == b"payload"
    assert os.path.normcase(os.path.abspath(contract.artifact_output_path)) in _ARTIFACT_RECEIPTS


def test_written_artifact_backlog_is_bounded_deterministically(monkeypatch, tmp_path):
    from agent.task_execution_contract import _ARTIFACT_RECEIPTS, record_artifact_written

    artifact_base = tmp_path / "artifacts"
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(artifact_base))
    contracts = []
    for index in range(24):
        contract = _contract(
            f"Give me report-{index}.txt as a file.",
            task_id=f"abandoned-{index:02d}",
        )
        Path(contract.artifact_output_path).write_bytes(f"payload-{index}".encode())
        assert record_artifact_written(contract) is True
        contracts.append(contract)

    pending_receipts = []
    for receipt_path in (artifact_base / ".receipts").glob("*.json"):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["state"] == "written":
            pending_receipts.append(receipt)

    assert len(pending_receipts) <= 16
    assert sum(int(item["bytes"]) for item in pending_receipts) <= 128 * 1024 * 1024
    assert len(
        [
            path
            for path in _ARTIFACT_RECEIPTS
            if os.path.commonpath([artifact_base, Path(path)]) == str(artifact_base)
        ]
    ) <= 16
    assert not Path(contracts[0].artifact_root).exists()
    oldest = json.loads(Path(contracts[0].artifact_receipt_path).read_text(encoding="utf-8"))
    assert oldest["state"] == "failed_preflight"
    assert oldest["error_code"] == "artifact_dispatch_abandoned"


def test_written_artifact_backlog_enforces_byte_cap(monkeypatch, tmp_path):
    import agent.task_execution_contract as contract_module
    from agent.task_execution_contract import record_artifact_written

    artifact_base = tmp_path / "artifacts"
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(artifact_base))
    monkeypatch.setattr(contract_module, "MAX_PENDING_ARTIFACT_BYTES", 10)
    for index in range(2):
        contract = _contract(
            f"Give me bytes-{index}.txt as a file.",
            task_id=f"byte-cap-{index}",
        )
        Path(contract.artifact_output_path).write_bytes(b"12345678")
        assert record_artifact_written(contract) is True

    pending = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (artifact_base / ".receipts").glob("*.json")
    ]
    pending = [receipt for receipt in pending if receipt["state"] == "written"]
    assert sum(receipt["bytes"] for receipt in pending) <= 10


def test_dispatching_artifact_backlog_enforces_count_and_byte_caps(monkeypatch, tmp_path):
    import agent.task_execution_contract as contract_module
    from agent.task_execution_contract import record_artifact_dispatch, record_artifact_written

    artifact_base = tmp_path / "artifacts"
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(artifact_base))
    monkeypatch.setattr(contract_module, "MAX_PENDING_ARTIFACTS", 2)
    monkeypatch.setattr(contract_module, "MAX_PENDING_ARTIFACT_BYTES", 10)
    contracts = []
    for index in range(3):
        contract = _contract(
            f"Give me dispatch-{index}.txt as a file.",
            task_id=f"dispatch-cap-{index}",
        )
        Path(contract.artifact_output_path).write_bytes(b"12345678")
        assert record_artifact_written(contract) is True
        assert record_artifact_dispatch(contract.artifact_output_path, state="dispatching")
        contracts.append(contract)
        contract_module._reconcile_artifact_store(str(artifact_base))

    receipts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (artifact_base / ".receipts").glob("*.json")
    ]
    pending = [item for item in receipts if item["state"] in {"written", "dispatching"}]
    assert len(pending) <= 2
    assert sum(int(item["bytes"]) for item in pending) <= 10
    oldest = json.loads(Path(contracts[0].artifact_receipt_path).read_text(encoding="utf-8"))
    assert oldest["state"] == "ambiguous"
    assert oldest["error_code"] == "artifact_dispatch_abandoned"


def test_startup_reconciliation_expires_durable_orphan(monkeypatch, tmp_path):
    import agent.task_execution_contract as contract_module
    from agent.task_execution_contract import _ARTIFACT_RECEIPTS, record_artifact_written

    artifact_base = tmp_path / "artifacts"
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(artifact_base))
    old = _contract("Give me old.txt as a file.", task_id="old-orphan")
    Path(old.artifact_output_path).write_bytes(b"old")
    assert record_artifact_written(old) is True
    old_time = time.time() - 7200
    os.utime(old.artifact_receipt_path, (old_time, old_time))
    _ARTIFACT_RECEIPTS.pop(
        os.path.normcase(os.path.abspath(old.artifact_output_path)),
        None,
    )
    monkeypatch.setattr(contract_module, "ARTIFACT_WRITTEN_TTL_SECONDS", 3600)

    _contract("Give me new.txt as a file.", task_id="startup-reconcile")

    receipt = json.loads(Path(old.artifact_receipt_path).read_text(encoding="utf-8"))
    assert receipt["state"] == "failed_preflight"
    assert receipt["error_code"] == "artifact_dispatch_abandoned"
    assert not Path(old.artifact_root).exists()


def test_startup_reconciliation_expires_unadopted_allocated_contract(
    monkeypatch,
    tmp_path,
):
    import agent.task_execution_contract as contract_module
    from agent.task_execution_contract import _ARTIFACT_RECEIPTS, reconcile_artifact_receipts

    artifact_base = tmp_path / "artifacts"
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(artifact_base))
    contract = _contract("Create abandoned.txt", task_id="allocated-orphan")
    old_time = time.time() - 7200
    os.utime(contract.artifact_receipt_path, (old_time, old_time))
    _ARTIFACT_RECEIPTS.pop(
        os.path.normcase(os.path.abspath(contract.artifact_output_path)),
        None,
    )
    monkeypatch.setattr(contract_module, "ARTIFACT_WRITTEN_TTL_SECONDS", 3600)

    reconcile_artifact_receipts()

    receipt = json.loads(
        Path(contract.artifact_receipt_path).read_text(encoding="utf-8")
    )
    assert receipt["state"] == "failed_preflight"
    assert receipt["error_code"] == "artifact_dispatch_abandoned"
    assert not Path(contract.artifact_root).exists()


def test_startup_reconciliation_terminalizes_crash_after_dispatching(monkeypatch, tmp_path):
    import agent.task_execution_contract as contract_module
    from agent.task_execution_contract import (
        _ARTIFACT_RECEIPTS,
        reconcile_artifact_receipts,
        record_artifact_dispatch,
        record_artifact_written,
    )

    artifact_base = tmp_path / "artifacts"
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(artifact_base))
    contract = _contract("Give me crash.txt as a file.", task_id="dispatch-crash")
    Path(contract.artifact_output_path).write_bytes(b"payload")
    assert record_artifact_written(contract) is True
    assert record_artifact_dispatch(contract.artifact_output_path, state="dispatching")
    old_time = time.time() - 7200
    os.utime(contract.artifact_receipt_path, (old_time, old_time))
    _ARTIFACT_RECEIPTS.pop(os.path.normcase(os.path.abspath(contract.artifact_output_path)), None)
    monkeypatch.setattr(contract_module, "ARTIFACT_WRITTEN_TTL_SECONDS", 3600)

    reconcile_artifact_receipts()

    receipt = json.loads(Path(contract.artifact_receipt_path).read_text(encoding="utf-8"))
    assert receipt["state"] == "ambiguous"
    assert receipt["error_code"] == "artifact_dispatch_abandoned"
    assert not Path(contract.artifact_root).exists()


def test_periodic_reconciliation_leaves_receipt_before_ttl(monkeypatch, tmp_path):
    import agent.task_execution_contract as contract_module
    from agent.task_execution_contract import reconcile_artifact_receipts, record_artifact_written

    artifact_base = tmp_path / "artifacts"
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(artifact_base))
    contract = _contract("Give me fresh.txt as a file.", task_id="periodic-fresh")
    Path(contract.artifact_output_path).write_bytes(b"payload")
    assert record_artifact_written(contract) is True
    monkeypatch.setattr(contract_module, "ARTIFACT_WRITTEN_TTL_SECONDS", 3600)

    reconcile_artifact_receipts()

    receipt = json.loads(Path(contract.artifact_receipt_path).read_text(encoding="utf-8"))
    assert receipt["state"] == "written"
    assert Path(contract.artifact_root).exists()


def test_file_artifact_mode_rejects_missing_platform(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(tmp_path / "artifacts"))

    contract = build_task_execution_contract(
        "Create and deliver report.txt containing safe text.",
        task_id="missing-platform",
        platform=None,
    )

    assert contract.lane == NORMAL
    assert contract.preflight_error == "artifact_delivery_unavailable"
    assert contract.artifact_output_path == ""


def test_conversation_contract_prepares_txt_artifact_only_for_telegram(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    captured = {}

    def stop_after_turn_preparation(*args, **kwargs):
        contract = kwargs["task_execution_contract"]
        captured["contract"] = contract
        assert contract.lane == ARTIFACT_ONLY
        assert contract.artifact_extension == ".txt"
        assert contract.artifact_output_path.endswith("report.txt")
        raise RuntimeError("turn-preparation-complete")

    monkeypatch.setattr(conversation_loop, "build_turn_context", stop_after_turn_preparation)
    telegram_agent = SimpleNamespace(platform="telegram", _task_execution_contract=None)

    with pytest.raises(RuntimeError, match="turn-preparation-complete"):
        conversation_loop.run_conversation(
            telegram_agent,
            "Create and deliver report.txt containing safe text.",
            task_id="telegram-conversation-artifact",
        )

    assert captured["contract"].artifact_route != "none"
    receipt = json.loads(
        Path(captured["contract"].artifact_receipt_path).read_text(encoding="utf-8")
    )
    assert receipt["state"] == "failed_preflight"
    assert receipt["error_code"] == "artifact_turn_setup_failed_before_adoption"
    assert not Path(captured["contract"].artifact_root).exists()

    other_agent = SimpleNamespace(
        platform="discord",
        _task_execution_contract=None,
        model="test-model",
        provider="test-provider",
    )
    result = conversation_loop.run_conversation(
        other_agent,
        "Create and deliver report.txt containing safe text.",
        task_id="discord-conversation-artifact",
    )
    assert result["turn_exit_reason"] == "artifact_output_preflight_failed"
    assert result["task_execution"]["decision_reason"] == "artifact_delivery_unavailable"


def test_conversation_emergency_disable_prepares_a_normal_turn(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"agent": {"artifact_only_enabled": False}},
    )
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    captured = {}

    def stop_after_turn_preparation(*args, **kwargs):
        contract = kwargs["task_execution_contract"]
        captured["contract"] = contract
        assert contract.lane == NORMAL
        assert contract.preflight_error == ""
        assert contract.artifact_output_path == ""
        raise RuntimeError("normal-turn-prepared")

    monkeypatch.setattr(conversation_loop, "build_turn_context", stop_after_turn_preparation)
    agent = SimpleNamespace(platform="telegram", _task_execution_contract=None)

    with pytest.raises(RuntimeError, match="normal-turn-prepared"):
        conversation_loop.run_conversation(
            agent,
            "Create and deliver recovery.txt containing safe text.",
            task_id="kill-switch-normal-turn",
        )

    assert captured["contract"].decision_reason == "artifact_only_disabled"


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


def test_normal_contract_removes_expired_artifact_tool_trace_from_model_history(
    monkeypatch,
):
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", "/opt/data/hermes-artifacts")
    artifact_path = (
        "/opt/data/hermes-artifacts/"
        "68fc2177cc474858a2c9b998f3b8be6f/recovery.txt"
    )
    history = [
        {"role": "user", "content": "Give me recovery.txt as a file."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "artifact-call",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps(
                            {"path": artifact_path, "content": "payload"}
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "artifact-call",
            "tool_name": "write_file",
            "content": json.dumps({"resolved_path": artifact_path}),
        },
        {"role": "assistant", "content": f"MEDIA:{artifact_path}"},
        {"role": "user", "content": "Now continue the tournament planning."},
        {"role": "assistant", "content": "A normal planning response."},
    ]
    contract = _contract("Check the current runtime and fix what is broken.")

    bounded = contract.bound_conversation_history(history)

    assert [message["role"] for message in bounded] == ["user", "assistant"]
    assert bounded[0] is history[4]
    assert bounded[1] is history[5]
    assert all(not _contains_path(message, artifact_path) for message in bounded)
    assert "normal capabilities" in contract.system_guidance
    assert "normal guarded file workflow and a new safe path" in contract.system_guidance
    denied = contract.before_tool(
        "write_file",
        {"path": artifact_path, "content": "must not overwrite prior artifact"},
    )
    assert denied.allowed is False
    assert denied.code == "expired_artifact_path_reuse"
    assert contract.before_tool(
        "write_file",
        {"path": "/opt/data/new-normal-file.txt", "content": "safe"},
    ).allowed is True


def test_normal_history_filter_removes_stale_system_execution_contract(monkeypatch):
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", "/opt/data/hermes-artifacts")
    history = [
        {
            "role": "system",
            "content": (
                "REQUEST EXECUTION CONTRACT (artifact_only, fail closed):\n"
                "Use only the prior request-local artifact path."
            ),
        },
        {"role": "user", "content": "Continue normal work."},
    ]
    contract = _contract("Continue normal work.")

    assert contract.bound_conversation_history(history) == [
        {"role": "user", "content": "Continue normal work."}
    ]
    assert "REQUEST RECOVERY NOTICE" in contract.system_guidance


def test_normal_history_filter_supports_configured_root_and_tool_call_only(
    monkeypatch, tmp_path
):
    configured_root = tmp_path / "configured-artifacts"
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(configured_root))
    artifact_path = str(
        configured_root
        / "68fc2177cc474858a2c9b998f3b8be6f"
        / "recovery.txt"
    )
    history = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "artifact-call",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps(
                            {"path": artifact_path, "content": "payload"}
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "artifact-call",
            "tool_name": "write_file",
            "content": json.dumps({"resolved_path": artifact_path}),
        },
    ]
    contract = _contract("Continue normal work.")

    assert contract.bound_conversation_history(history) == []
    assert contract.references_expired_artifact({"path": artifact_path})
    assert (
        contract.before_tool(
            "write_file", {"path": artifact_path, "content": "overwrite"}
        ).code
        == "expired_artifact_path_reuse"
    )


def test_normal_history_filter_supports_flat_persisted_tool_call_shape(
    monkeypatch, tmp_path
):
    configured_root = tmp_path / "configured-artifacts"
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(configured_root))
    artifact_path = str(
        configured_root
        / "68fc2177cc474858a2c9b998f3b8be6f"
        / "recovery.txt"
    )
    history = [
        {"role": "user", "content": "Create recovery.txt."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "artifact-call",
                    "name": "write_file",
                    "arguments": {
                        "path": artifact_path,
                        "content": "payload",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "artifact-call",
            "name": "write_file",
            "content": {"resolved_path": artifact_path},
        },
        {"role": "assistant", "content": f"MEDIA:{artifact_path}"},
        {"role": "user", "content": "Continue normally."},
    ]
    contract = _contract("Continue normally.")

    assert contract.bound_conversation_history(history) == [history[-1]]
    assert contract.references_expired_artifact({"path": artifact_path})


def test_unanchored_summary_that_mentions_artifact_path_is_preserved(
    monkeypatch,
):
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", "/opt/data/hermes-artifacts")
    artifact_path = (
        "/opt/data/hermes-artifacts/"
        "68fc2177cc474858a2c9b998f3b8be6f/recovery.txt"
    )
    history = [
        {
            "role": "assistant",
            "content": f"Compression summary: the old file was {artifact_path}.",
        },
        {"role": "user", "content": "Continue normally."},
    ]
    contract = _contract("Continue normally.")

    bounded = contract.bound_conversation_history(history)

    assert bounded[0] is history[0]
    assert bounded[1] is history[1]
    assert contract._expired_artifact_history_messages == 0


def test_normal_history_filter_recognizes_artifact_from_previous_root(
    monkeypatch, tmp_path
):
    old_root = tmp_path / "old-artifacts"
    current_root = tmp_path / "current-artifacts"
    artifact_path = str(
        old_root
        / "68fc2177cc474858a2c9b998f3b8be6f"
        / "recovery.txt"
    )
    history = [
        {"role": "user", "content": "Create recovery.txt."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "artifact-call",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps(
                            {"path": artifact_path, "content": "payload"}
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "artifact-call",
            "tool_name": "write_file",
            "content": json.dumps({"resolved_path": artifact_path}),
        },
        {"role": "assistant", "content": f"MEDIA:{artifact_path}"},
        {"role": "user", "content": "Continue normally."},
    ]
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(current_root))
    contract = _contract("Continue normally.")

    bounded = contract.bound_conversation_history(history)

    assert bounded == [history[-1]]
    assert contract.references_expired_artifact({"path": artifact_path})
    denied = contract.before_tool(
        "write_file",
        {"path": artifact_path, "content": "must not overwrite"},
    )
    assert denied.allowed is False
    assert denied.code == "expired_artifact_path_reuse"


def test_normal_user_path_outside_artifact_root_is_not_marked_expired(monkeypatch):
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", "/opt/data")
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", "/opt/data/hermes-artifacts")
    ordinary_path = (
        "/opt/data/projects/"
        "68fc2177cc474858a2c9b998f3b8be6f/notes.txt"
    )
    history = [{"role": "user", "content": f"Review {ordinary_path} later."}]
    contract = _contract("Continue normal work.")

    bounded = contract.bound_conversation_history(history)

    assert bounded[0] is history[0]
    assert not contract.references_expired_artifact({"path": ordinary_path})
    assert contract.before_tool(
        "write_file", {"path": ordinary_path, "content": "normal"}
    ).allowed


def test_normal_history_filter_removes_complete_mixed_artifact_turn(
    monkeypatch, tmp_path
):
    configured_root = tmp_path / "artifacts"
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(configured_root))
    artifact_path = str(
        configured_root
        / "68fc2177cc474858a2c9b998f3b8be6f"
        / "recovery.txt"
    )
    history = [
        {"role": "user", "content": "Create recovery.txt after reading the notes."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "artifact-call",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps({"path": artifact_path, "content": "x"}),
                    },
                },
                {
                    "id": "read-call",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"/opt/data/normal.txt"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "artifact-call",
            "tool_name": "write_file",
            "content": json.dumps({"resolved_path": artifact_path}),
        },
        {
            "role": "tool",
            "tool_call_id": "read-call",
            "tool_name": "read_file",
            "content": "normal contents",
        },
    ]
    contract = _contract("Continue normal work.")

    bounded = contract.bound_conversation_history(history)

    assert bounded == []


def _contains_path(message, path):
    return path in json.dumps(message, sort_keys=True)


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
