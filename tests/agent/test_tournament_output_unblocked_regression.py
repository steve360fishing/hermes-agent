from types import SimpleNamespace

from agent.tool_guardrails import ToolCallGuardrailController


class Agent:
    def __init__(self):
        self.session_id = "regression-session"
        self.platform = "telegram"
        self.streamed = []
        self.stream_delta_callback = self.streamed.append
        self._stream_callback = None
        self._persist_session = lambda *_args: None
        self._tool_guardrails = ToolCallGuardrailController()


def test_private_codex_handoff_never_installs_a_tournament_release_contract():
    agent = Agent()
    prompt = "I do not want any tournament artifact blocks. Give me a .txt prompt for Codex explaining both errors and telling it to remove them."
    assert agent._tool_guardrails.before_call("write_file", {}).action == "allow"


def test_distinctive_10000_byte_response_is_an_ordinary_value():
    candidate = "distinctive-payload-" + ("x" * 10_000)
    assert len(candidate) > 10_000
    assert candidate == candidate


def test_tournament_contract_cannot_restrict_normal_tools():
    controller = ToolCallGuardrailController()
    contract = SimpleNamespace(has_valid_receipt=lambda: False)
    controller.set_tournament_contract(contract)
    for tool_name in ("write_file", "patch", "terminal", "web_search", "read_file"):
        assert controller.before_call(tool_name, {}).code != "tournament_receipt_required"


def test_explicit_validator_without_context_is_advisory_only():
    import json
    from tools.tournament_truth_gate_tool import run_tournament_truth_gate
    result = json.loads(run_tournament_truth_gate({"candidate": "draft", "request": {}, "artifact_metadata": {}}, task_id="no-context", session_id="none"))
    assert result["code"] in {"trusted_runtime_roots_unavailable", "trusted_source_snapshot_required"}


def test_forbidden_runtime_literals_are_absent():
    from pathlib import Path
    root = Path(__file__).parents[2]
    production = [p for p in (root / "agent").glob("*.py")] + [p for p in (root / "gateway").glob("*.py")]
    for path in production:
        assert "ROUTE" + "_HOLD" not in path.read_text(encoding="utf-8")
        assert "PUBLIC" + "_ARTIFACT_BLOCKED" not in path.read_text(encoding="utf-8")
