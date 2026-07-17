"""Regression test for #8049.

When the post-loop cleanup chain in ``finalize_turn`` raises — trajectory
save (file I/O), resource teardown (remote VM/browser), or session
persistence (SQLite) — the partial ``final_response`` the caller is waiting
for must still be returned.  Previously any of those raised straight out of
``run_conversation``, so a subprocess wrapper saw an empty stdout with no
traceback and lost the whole turn.
"""

import os
import json
from pathlib import Path
import pytest
from unittest.mock import patch

from agent.turn_finalizer import finalize_turn
from agent.task_execution_contract import build_task_execution_contract


class _StubBudget:
    used = 5
    max_total = 3
    remaining = 0


class _StubCompressor:
    last_prompt_tokens = 0


class _StubAgent:
    """Minimal agent surface that ``finalize_turn`` reads from."""

    def __init__(self, *, raise_in):
        self._raise_in = set(raise_in)
        self.max_iterations = 3
        self.iteration_budget = _StubBudget()
        self.context_compressor = _StubCompressor()
        self.model = "stub/model"
        self.provider = "stub"
        self.base_url = "http://stub"
        self.session_id = "sess-1"
        self.quiet_mode = True
        self.platform = "cli"
        self._interrupt_requested = False
        self._interrupt_message = None
        self._tool_guardrail_halt_decision = None
        self._task_execution_contract = None
        self._tool_guardrails = type(
            "Guardrails",
            (),
            {
                "contract": None,
                "set_execution_contract": lambda guard, value: setattr(
                    guard, "contract", value
                ),
            },
        )()
        self.sync_calls = 0
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = set()
        self.background_review_calls = 0
        for attr in (
            "session_input_tokens",
            "session_output_tokens",
            "session_cache_read_tokens",
            "session_cache_write_tokens",
            "session_reasoning_tokens",
            "session_prompt_tokens",
            "session_completion_tokens",
            "session_total_tokens",
            "session_estimated_cost_usd",
        ):
            setattr(self, attr, 0)
        self.session_cost_status = "ok"
        self.session_cost_source = "stub"

    # --- fallible cleanup surfaces -------------------------------------
    def _save_trajectory(self, *a, **k):
        if "save_trajectory" in self._raise_in:
            raise RuntimeError("trajectory disk full")

    def _cleanup_task_resources(self, *a, **k):
        if "cleanup_task_resources" in self._raise_in:
            raise RuntimeError("docker teardown EOF")

    def _drop_trailing_empty_response_scaffolding(self, *a, **k):
        pass

    def _persist_session(self, *a, **k):
        if "persist_session" in self._raise_in:
            raise RuntimeError("sqlite database is locked")

    # --- harmless no-ops ------------------------------------------------
    def _emit_status(self, *a, **k):
        pass

    def _safe_print(self, *a, **k):
        pass

    def _handle_max_iterations(self, messages, n):
        return "PARTIAL SUMMARY FROM MODEL"

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **k):
        self.sync_calls += 1

    def _spawn_background_review(self, **k):
        self.background_review_calls += 1


def _run(
    agent,
    *,
    final_response=None,
    api_call_count=3,
    turn_exit_reason="unknown",
):
    messages = [
        {"role": "user", "content": "do a thing"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "file contents"},
    ]
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=api_call_count,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="do a thing",
        original_user_message="do a thing",
        _should_review_memory=False,
        _turn_exit_reason=turn_exit_reason,
    )


def test_all_cleanup_steps_raise_response_still_returned():
    agent = _StubAgent(
        raise_in=("save_trajectory", "cleanup_task_resources", "persist_session")
    )
    result = _run(agent)
    assert result["final_response"] == "PARTIAL SUMMARY FROM MODEL"
    labels = [e.split(":")[0] for e in result["cleanup_errors"]]
    assert labels == ["save_trajectory", "cleanup_task_resources", "persist_session"]


@pytest.mark.parametrize(
    "step", ["save_trajectory", "cleanup_task_resources", "persist_session"]
)
def test_single_cleanup_step_raises_does_not_skip_others(step):
    agent = _StubAgent(raise_in=(step,))
    result = _run(agent)
    # Response survives.
    assert result["final_response"] == "PARTIAL SUMMARY FROM MODEL"
    # Exactly the failing step is recorded; the others ran without error.
    assert result["cleanup_errors"] == [
        next(
            e
            for e in result["cleanup_errors"]
            if e.startswith(step)
        )
    ]
    assert len(result["cleanup_errors"]) == 1


def test_clean_turn_has_no_cleanup_errors_key():
    agent = _StubAgent(raise_in=())
    result = _run(agent)
    assert result["final_response"] == "PARTIAL SUMMARY FROM MODEL"
    assert result["completed"] is False
    assert "cleanup_errors" not in result


def test_text_response_on_last_allowed_call_is_completed():
    agent = _StubAgent(raise_in=())
    result = _run(
        agent,
        final_response="final report",
        api_call_count=agent.max_iterations,
        turn_exit_reason="text_response(finish_reason=stop)",
    )
    assert result["final_response"] == "final report"
    assert result["completed"] is True


def test_artifact_contract_telemetry_is_returned_without_prompt_content():
    agent = _StubAgent(raise_in=())
    agent._task_execution_contract = build_task_execution_contract(
        "Return only a prompt using PRIVATE-SPONSOR-FACT.",
        task_id="private-task-id",
    )

    with patch("hermes_cli.plugins.invoke_hook", return_value=[]) as invoke_hook:
        result = _run(
            agent,
            final_response="final prompt",
            api_call_count=1,
            turn_exit_reason="text_response(finish_reason=stop)",
        )

    serialized = repr(result["task_execution"])
    assert result["task_execution"]["lane"] == "artifact_only"
    assert result["task_execution"]["decision_status"] == "completed"
    assert "PRIVATE-SPONSOR-FACT" not in serialized
    assert "private-task-id" not in serialized
    assert agent.sync_calls == 0
    assert agent._task_execution_contract is None
    assert agent._tool_guardrails.contract is None
    invoke_hook.assert_not_called()


def test_artifact_contract_is_cleared_when_finalizer_cleanup_reports_failures():
    agent = _StubAgent(raise_in=("persist_session",))
    contract = build_task_execution_contract(
        "Return only a paste-ready prompt.", task_id="artifact-failure"
    )
    agent._task_execution_contract = contract
    agent._tool_guardrails.set_execution_contract(contract)

    result = _run(agent, final_response="partial prompt", api_call_count=1)

    assert result["cleanup_errors"]
    assert agent._task_execution_contract is None
    assert agent._tool_guardrails.contract is None
    assert contract.active is False


def test_contract_is_cleared_even_if_finalizer_itself_raises():
    agent = _StubAgent(raise_in=())
    contract = build_task_execution_contract(
        "Return only a paste-ready prompt.", task_id="artifact-finalizer-exception"
    )
    agent._task_execution_contract = contract
    agent._tool_guardrails.set_execution_contract(contract)
    agent._drain_pending_steer = lambda: (_ for _ in ()).throw(
        RuntimeError("finalizer failure")
    )

    with pytest.raises(RuntimeError, match="finalizer failure"):
        _run(agent, final_response="prompt", api_call_count=1)

    assert contract.active is False
    assert agent._task_execution_contract is None
    assert agent._tool_guardrails.contract is None


def test_successful_artifact_write_forces_exact_media_delivery_response(
    monkeypatch, tmp_path
):
    agent = _StubAgent(raise_in=())
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    contract = build_task_execution_contract(
        "Create and deliver example.txt containing the supplied copy.",
        task_id="artifact-delivery",
    )
    with open(contract.artifact_output_path, "wb") as artifact:
        artifact.write(b"exact bytes\n")
    agent._task_execution_contract = contract
    agent._tool_guardrails.set_execution_contract(contract)
    agent._turn_file_mutation_paths = {contract.artifact_output_path}

    result = _run(
        agent,
        final_response="I wrote the file but forgot the attachment tag.",
        api_call_count=1,
        turn_exit_reason="text_response(finish_reason=stop)",
    )

    assert result["final_response"] == f"MEDIA:{contract.artifact_output_path}"
    assert agent._task_execution_contract is None


def test_successful_artifact_write_creates_non_secret_written_receipt(monkeypatch, tmp_path):
    agent = _StubAgent(raise_in=())
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    contract = build_task_execution_contract(
        "Create and deliver example.txt containing the supplied copy.",
        task_id="artifact-receipt",
        platform="telegram",
    )
    Path(contract.artifact_output_path).write_bytes(b"exact bytes\n")
    agent._task_execution_contract = contract
    agent._tool_guardrails.set_execution_contract(contract)
    agent._turn_file_mutation_paths = {contract.artifact_output_path}

    _run(agent, final_response="done", api_call_count=1)

    receipt = json.loads(Path(contract.artifact_receipt_path).read_text(encoding="utf-8"))
    assert receipt["state"] == "written"
    assert receipt["bytes"] == len(b"exact bytes\n")
    assert receipt["sha256"]
    assert "content" not in receipt
    assert "chat_id" not in receipt


def test_artifact_contract_suppresses_background_skill_review():
    agent = _StubAgent(raise_in=())
    agent._task_execution_contract = build_task_execution_contract(
        "Return only a paste-ready prompt.",
        task_id="artifact-background-review",
    )
    agent._skill_nudge_interval = 1
    agent._iters_since_skill = 1
    agent.valid_tool_names = {"skill_manage"}

    _run(
        agent,
        final_response="final prompt",
        api_call_count=1,
        turn_exit_reason="text_response(finish_reason=stop)",
    )

    assert agent.background_review_calls == 0


def test_artifact_max_iteration_does_not_mutate_kanban_state():
    agent = _StubAgent(raise_in=())
    agent._task_execution_contract = build_task_execution_contract(
        "Return only a paste-ready prompt.",
        task_id="artifact-kanban",
    )

    with (
        patch.dict(os.environ, {"HERMES_KANBAN_TASK": "t_artifact"}),
        patch("hermes_cli.kanban_db.connect") as connect,
        patch("hermes_cli.kanban_db._record_task_failure") as record_failure,
    ):
        _run(agent)

    connect.assert_not_called()
    record_failure.assert_not_called()
